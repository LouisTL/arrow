#!/usr/bin/env python3
"""
Arrow Lang — LLVM Compiler Backend

Compiles Arrow Lang source to native executables via LLVM IR.

Usage:
    python compiler.py example.arrow                  # compile & run
    python compiler.py example.arrow --emit-ir        # print LLVM IR
    python compiler.py example.arrow -o example       # compile to executable

Requires: llvmlite, clang (for linking)
    pip install llvmlite
"""

import sys
import argparse
import subprocess
import tempfile
import os

from llvmlite import ir, binding
from lang import (
    Lexer, Parser, LexerError, ParseError,
    Program, Assignment, PrintStmt, IfStmt, WhileStmt, Block,
    NumberLit, StringLit, BoolLit, Identifier, BinOp, UnaryOp,
)


# ─────────────────────────────────────────────
#  TYPE SYSTEM
# ─────────────────────────────────────────────
# Arrow Lang has 4 runtime types that map to LLVM types:
#   int    -> i64
#   float  -> double
#   bool   -> i1
#   string -> i8* (pointer to null-terminated C string)

class ArrowType:
    """Tracks the Arrow Lang type alongside an LLVM value."""
    INT    = "int"
    FLOAT  = "float"
    BOOL   = "bool"
    STRING = "string"


class TypedValue:
    """An LLVM IR value paired with its Arrow Lang type."""
    __slots__ = ("value", "type")

    def __init__(self, value: ir.Value, arrow_type: str):
        self.value = value
        self.type = arrow_type

    def __repr__(self):
        return f"TypedValue({self.type}, {self.value})"


# ─────────────────────────────────────────────
#  LLVM TYPE CONSTANTS
# ─────────────────────────────────────────────
i1  = ir.IntType(1)
i8  = ir.IntType(8)
i32 = ir.IntType(32)
i64 = ir.IntType(64)
double = ir.DoubleType()
i8_ptr = i8.as_pointer()
void = ir.VoidType()


# ─────────────────────────────────────────────
#  COMPILER ERROR
# ─────────────────────────────────────────────
class CompileError(Exception):
    pass


# ─────────────────────────────────────────────
#  CODEGEN — AST → LLVM IR
# ─────────────────────────────────────────────
class Compiler:
    """
    Walks the AST and emits LLVM IR using llvmlite's IR builder.

    Strategy:
    - All variables are stack-allocated (alloca) and accessed via load/store.
      LLVM's mem2reg optimization pass promotes these to SSA registers.
    - Numbers are i64 by default, floats are double.
    - Strings are global constants (i8*).
    - print() calls C's printf via extern declaration.
    - String concatenation uses snprintf + malloc.
    """

    def __init__(self):
        self.module = ir.Module(name="arrow_lang")
        self.module.triple = binding.get_default_triple()

        # Detect Windows for format string differences
        self._is_windows = "windows" in self.module.triple or "msvc" in self.module.triple
        # Windows msvcrt uses %I64d for 64-bit ints, UCRT supports %lld
        # Use %I64d for maximum compatibility on Windows
        self._int_fmt = "%I64d" if self._is_windows else "%lld"

        # Variable storage: name -> (alloca ptr, ArrowType)
        self.variables: dict[str, tuple[ir.AllocaInstr, str]] = {}

        # String constant counter
        self._str_counter = 0

        # Declare external C functions
        self._declare_externals()

        # Create main function with two blocks:
        #   - entry: only allocas live here (no terminators until compile() finalizes)
        #   - code:  all real instructions start here
        func_type = ir.FunctionType(i32, [])
        self.main_func = ir.Function(self.module, func_type, name="main")
        self.alloca_block = self.main_func.append_basic_block(name="entry")
        self.code_block = self.main_func.append_basic_block(name="code")

        # Builder for emitting allocas into the entry block
        self.alloca_builder = ir.IRBuilder(self.alloca_block)

        # Main builder starts in the code block
        self.builder = ir.IRBuilder(self.code_block)

    # ── External C functions ─────────────────

    def _declare_externals(self):
        """Declare C standard library functions we'll call."""
        # int puts(const char*) — prints string + newline, auto-flushes
        puts_type = ir.FunctionType(i32, [i8_ptr])
        self.puts = ir.Function(self.module, puts_type, name="puts")

        # int printf(const char* fmt, ...) — for formatting numbers
        printf_type = ir.FunctionType(i32, [i8_ptr], var_arg=True)
        self.printf = ir.Function(self.module, printf_type, name="printf")

        # int fflush(FILE*) — pass NULL to flush all streams
        fflush_type = ir.FunctionType(i32, [i8_ptr])
        self.fflush = ir.Function(self.module, fflush_type, name="fflush")

        # char* malloc(size_t)
        malloc_type = ir.FunctionType(i8_ptr, [i64])
        self.malloc = ir.Function(self.module, malloc_type, name="malloc")

        # int sprintf(char* buf, const char* fmt, ...)
        sprintf_type = ir.FunctionType(i32, [i8_ptr, i8_ptr], var_arg=True)
        self.sprintf = ir.Function(self.module, sprintf_type, name="sprintf")

        # size_t strlen(const char*)
        strlen_type = ir.FunctionType(i64, [i8_ptr])
        self.strlen = ir.Function(self.module, strlen_type, name="strlen")

        # char* strcpy(char* dest, const char* src)
        strcpy_type = ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr])
        self.strcpy = ir.Function(self.module, strcpy_type, name="strcpy")

        # char* strcat(char* dest, const char* src)
        strcat_type = ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr])
        self.strcat = ir.Function(self.module, strcat_type, name="strcat")

    # ── String helpers ───────────────────────

    def _global_string(self, text: str) -> ir.Value:
        """Create a global string constant and return an i8* pointer to it."""
        encoded = (text + "\0").encode("utf-8")
        str_type = ir.ArrayType(i8, len(encoded))
        name = f".str.{self._str_counter}"
        self._str_counter += 1

        global_var = ir.GlobalVariable(self.module, str_type, name=name)
        global_var.linkage = "private"
        global_var.global_constant = True
        global_var.initializer = ir.Constant(str_type, bytearray(encoded))

        # Return pointer to first element (i8*)
        zero = ir.Constant(i64, 0)
        return self.builder.gep(global_var, [zero, zero], inbounds=True, name="str_ptr")

    # ── Format value as string (for print & concat) ──

    def _value_to_string(self, tv: TypedValue) -> ir.Value:
        """Convert any typed value to an i8* string representation."""
        if tv.type == ArrowType.STRING:
            return tv.value

        # Allocate a buffer and sprintf into it
        buf_size = ir.Constant(i64, 64)
        buf = self.builder.call(self.malloc, [buf_size], name="fmt_buf")

        if tv.type == ArrowType.INT:
            fmt = self._global_string(self._int_fmt)
            self.builder.call(self.sprintf, [buf, fmt, tv.value])
        elif tv.type == ArrowType.FLOAT:
            fmt = self._global_string("%g")
            self.builder.call(self.sprintf, [buf, fmt, tv.value])
        elif tv.type == ArrowType.BOOL:
            true_str = self._global_string("true")
            false_str = self._global_string("false")
            result = self.builder.select(tv.value, true_str, false_str, name="bool_str")
            return result

        return buf

    # ── Compile entry point ──────────────────

    def compile(self, program: Program) -> str:
        """Compile a program and return the LLVM IR as a string."""
        for stmt in program.statements:
            self._compile_stmt(stmt)

        # Return 0 from main
        self.builder.ret(ir.Constant(i32, 0))

        # Finalize: entry block jumps to code block
        # (must be done last so all allocas are already in the entry block)
        self.alloca_builder.branch(self.code_block)

        # Verify the module
        llvm_ir = str(self.module)
        return llvm_ir

    # ── Statement codegen ────────────────────

    def _compile_stmt(self, node):
        match node:
            case Assignment(name, expr):
                self._compile_assignment(name, expr)

            case PrintStmt(expr):
                self._compile_print(expr)

            case IfStmt(cond, then_body, else_body):
                self._compile_if(cond, then_body, else_body)

            case WhileStmt(cond, body):
                self._compile_while(cond, body)

            case Block(stmts):
                for s in stmts:
                    self._compile_stmt(s)

            case _:
                # Expression statement — compile and discard
                self._compile_expr(node)

    def _compile_assignment(self, name: str, expr):
        """Compile: name <- expr;"""
        tv = self._compile_expr(expr)

        if name in self.variables:
            # Existing variable — store into its alloca
            ptr, existing_type = self.variables[name]
            if existing_type != tv.type:
                raise CompileError(
                    f"Type mismatch: variable '{name}' is {existing_type}, "
                    f"cannot assign {tv.type}"
                )
            self.builder.store(tv.value, ptr)
        else:
            # New variable — create alloca at function entry
            llvm_type = self._arrow_type_to_llvm(tv.type)
            ptr = self._create_entry_alloca(name, llvm_type)
            self.builder.store(tv.value, ptr)
            self.variables[name] = (ptr, tv.type)

    def _compile_print(self, expr):
        """Compile: print(expr);
        
        Uses printf directly with the right format specifier per type,
        avoiding malloc/snprintf which can fail to resolve in Windows JIT.
        """
        tv = self._compile_expr(expr)
        null_ptr = ir.Constant(i8_ptr, None)

        if tv.type == ArrowType.STRING:
            fmt = self._global_string("%s\n")
            self.builder.call(self.printf, [fmt, tv.value])
        elif tv.type == ArrowType.INT:
            fmt = self._global_string(self._int_fmt + "\n")
            self.builder.call(self.printf, [fmt, tv.value])
        elif tv.type == ArrowType.FLOAT:
            fmt = self._global_string("%g\n")
            self.builder.call(self.printf, [fmt, tv.value])
        elif tv.type == ArrowType.BOOL:
            true_str = self._global_string("true")
            false_str = self._global_string("false")
            str_val = self.builder.select(tv.value, true_str, false_str, name="bool_str")
            fmt = self._global_string("%s\n")
            self.builder.call(self.printf, [fmt, str_val])

        self.builder.call(self.fflush, [null_ptr])

    def _compile_if(self, cond, then_body, else_body):
        """Compile: if (cond) { ... } else { ... }"""
        cond_tv = self._compile_expr(cond)
        cond_val = self._to_bool(cond_tv)

        then_bb  = self.builder.append_basic_block("if_then")
        else_bb  = self.builder.append_basic_block("if_else") if else_body else None
        merge_bb = self.builder.append_basic_block("if_end")

        # Conditional branch
        if else_bb:
            self.builder.cbranch(cond_val, then_bb, else_bb)
        else:
            self.builder.cbranch(cond_val, then_bb, merge_bb)

        # Then block
        self.builder.position_at_start(then_bb)
        for s in then_body:
            self._compile_stmt(s)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        # Else block
        if else_bb:
            self.builder.position_at_start(else_bb)
            for s in else_body:
                self._compile_stmt(s)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        # Continue after if
        self.builder.position_at_start(merge_bb)

    def _compile_while(self, cond, body):
        """Compile: while (cond) { ... }"""
        # Create basic blocks
        cond_bb = self.builder.append_basic_block("while_cond")
        body_bb = self.builder.append_basic_block("while_body")
        end_bb  = self.builder.append_basic_block("while_end")

        # Jump to condition check
        self.builder.branch(cond_bb)

        # Condition block
        self.builder.position_at_start(cond_bb)
        cond_tv = self._compile_expr(cond)
        cond_val = self._to_bool(cond_tv)
        self.builder.cbranch(cond_val, body_bb, end_bb)

        # Body block
        self.builder.position_at_start(body_bb)
        for s in body:
            self._compile_stmt(s)
        # Loop back to condition (only if block isn't already terminated)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)

        # Continue after loop
        self.builder.position_at_start(end_bb)

    # ── Expression codegen ───────────────────

    def _compile_expr(self, node) -> TypedValue:
        match node:
            case NumberLit(value):
                if isinstance(value, float):
                    return TypedValue(ir.Constant(double, value), ArrowType.FLOAT)
                else:
                    return TypedValue(ir.Constant(i64, value), ArrowType.INT)

            case StringLit(value):
                ptr = self._global_string(value)
                return TypedValue(ptr, ArrowType.STRING)

            case BoolLit(value):
                return TypedValue(ir.Constant(i1, int(value)), ArrowType.BOOL)

            case Identifier(name):
                if name not in self.variables:
                    raise CompileError(f"Undefined variable '{name}'")
                ptr, arrow_type = self.variables[name]
                val = self.builder.load(ptr, name=name)
                return TypedValue(val, arrow_type)

            case UnaryOp(op, operand):
                return self._compile_unary(op, operand)

            case BinOp(op, left, right):
                return self._compile_binop(op, left, right)

            case _:
                raise CompileError(f"Cannot compile expression: {node}")

    def _compile_unary(self, op: str, operand) -> TypedValue:
        tv = self._compile_expr(operand)

        if op == '-':
            if tv.type == ArrowType.INT:
                result = self.builder.neg(tv.value, name="neg")
                return TypedValue(result, ArrowType.INT)
            elif tv.type == ArrowType.FLOAT:
                result = self.builder.fneg(tv.value, name="fneg")
                return TypedValue(result, ArrowType.FLOAT)
            else:
                raise CompileError(f"Cannot negate type {tv.type}")

        elif op == '!':
            bool_val = self._to_bool(tv)
            result = self.builder.not_(bool_val, name="not")
            return TypedValue(result, ArrowType.BOOL)

        raise CompileError(f"Unknown unary operator: {op}")

    def _compile_binop(self, op: str, left, right) -> TypedValue:
        # String concatenation with +
        if op == '+':
            ltv = self._compile_expr(left)
            rtv = self._compile_expr(right)

            if ltv.type == ArrowType.STRING or rtv.type == ArrowType.STRING:
                return self._compile_string_concat(ltv, rtv)

            return self._compile_arithmetic(op, ltv, rtv)

        ltv = self._compile_expr(left)
        rtv = self._compile_expr(right)

        # Arithmetic operators
        if op in ('+', '-', '*', '/', '%'):
            return self._compile_arithmetic(op, ltv, rtv)

        # Comparison operators
        if op in ('<', '>', '<=', '>=', '=', '!='):
            return self._compile_comparison(op, ltv, rtv)

        # Logical operators
        if op == '&&':
            lb = self._to_bool(ltv)
            rb = self._to_bool(rtv)
            result = self.builder.and_(lb, rb, name="and")
            return TypedValue(result, ArrowType.BOOL)

        if op == '||':
            lb = self._to_bool(ltv)
            rb = self._to_bool(rtv)
            result = self.builder.or_(lb, rb, name="or")
            return TypedValue(result, ArrowType.BOOL)

        raise CompileError(f"Unknown binary operator: {op}")

    def _compile_arithmetic(self, op: str, ltv: TypedValue, rtv: TypedValue) -> TypedValue:
        """Compile arithmetic: +, -, *, /, %"""
        # Promote to float if either side is float
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            lv = self._to_float(ltv)
            rv = self._to_float(rtv)
            match op:
                case '+': result = self.builder.fadd(lv, rv, name="fadd")
                case '-': result = self.builder.fsub(lv, rv, name="fsub")
                case '*': result = self.builder.fmul(lv, rv, name="fmul")
                case '/': result = self.builder.fdiv(lv, rv, name="fdiv")
                case '%': result = self.builder.frem(lv, rv, name="frem")
                case _: raise CompileError(f"Unknown arithmetic op: {op}")
            return TypedValue(result, ArrowType.FLOAT)
        else:
            # Integer arithmetic
            lv, rv = ltv.value, rtv.value
            match op:
                case '+': result = self.builder.add(lv, rv, name="add")
                case '-': result = self.builder.sub(lv, rv, name="sub")
                case '*': result = self.builder.mul(lv, rv, name="mul")
                case '/': result = self.builder.sdiv(lv, rv, name="sdiv")
                case '%': result = self.builder.srem(lv, rv, name="srem")
                case _: raise CompileError(f"Unknown arithmetic op: {op}")
            return TypedValue(result, ArrowType.INT)

    def _compile_comparison(self, op: str, ltv: TypedValue, rtv: TypedValue) -> TypedValue:
        """Compile comparisons: <, >, <=, >=, =, !="""
        # llvmlite's icmp_signed/fcmp_ordered use symbolic operators, not LLVM asm mnemonics
        icmp_map = {'<': '<', '>': '>', '<=': '<=', '>=': '>=', '=': '==', '!=': '!='}
        fcmp_map = {'<': '<', '>': '>', '<=': '<=', '>=': '>=', '=': '==', '!=': '!='}

        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            lv = self._to_float(ltv)
            rv = self._to_float(rtv)
            result = self.builder.fcmp_ordered(fcmp_map[op], lv, rv, name="fcmp")
        elif ltv.type == ArrowType.STRING and rtv.type == ArrowType.STRING:
            raise CompileError("String comparison not yet supported")
        else:
            lv = self._to_int(ltv)
            rv = self._to_int(rtv)
            result = self.builder.icmp_signed(icmp_map[op], lv, rv, name="icmp")

        return TypedValue(result, ArrowType.BOOL)

    def _compile_string_concat(self, ltv: TypedValue, rtv: TypedValue) -> TypedValue:
        """Compile string concatenation: convert both sides to strings, concat."""
        left_str  = self._value_to_string(ltv)
        right_str = self._value_to_string(rtv)

        # Calculate total length: strlen(left) + strlen(right) + 1
        left_len  = self.builder.call(self.strlen, [left_str],  name="llen")
        right_len = self.builder.call(self.strlen, [right_str], name="rlen")
        total = self.builder.add(left_len, right_len, name="total_len")
        total = self.builder.add(total, ir.Constant(i64, 1), name="with_null")

        # Allocate buffer
        buf = self.builder.call(self.malloc, [total], name="concat_buf")

        # strcpy(buf, left); strcat(buf, right);
        self.builder.call(self.strcpy, [buf, left_str])
        self.builder.call(self.strcat, [buf, right_str])

        return TypedValue(buf, ArrowType.STRING)

    # ── Type conversion helpers ──────────────

    def _to_bool(self, tv: TypedValue) -> ir.Value:
        """Convert any typed value to an i1 boolean."""
        if tv.type == ArrowType.BOOL:
            return tv.value
        if tv.type == ArrowType.INT:
            return self.builder.icmp_signed('!=', tv.value, ir.Constant(i64, 0), name="tobool")
        if tv.type == ArrowType.FLOAT:
            return self.builder.fcmp_ordered('one', tv.value, ir.Constant(double, 0.0), name="tobool")
        if tv.type == ArrowType.STRING:
            # Non-empty string is truthy: strlen(s) != 0
            length = self.builder.call(self.strlen, [tv.value], name="slen")
            return self.builder.icmp_signed('!=', length, ir.Constant(i64, 0), name="tobool")
        raise CompileError(f"Cannot convert {tv.type} to bool")

    def _to_float(self, tv: TypedValue) -> ir.Value:
        """Promote a value to double."""
        if tv.type == ArrowType.FLOAT:
            return tv.value
        if tv.type == ArrowType.INT:
            return self.builder.sitofp(tv.value, double, name="tofloat")
        if tv.type == ArrowType.BOOL:
            int_val = self.builder.zext(tv.value, i64, name="btoi")
            return self.builder.sitofp(int_val, double, name="tofloat")
        raise CompileError(f"Cannot convert {tv.type} to float")

    def _to_int(self, tv: TypedValue) -> ir.Value:
        """Convert a value to i64."""
        if tv.type == ArrowType.INT:
            return tv.value
        if tv.type == ArrowType.BOOL:
            return self.builder.zext(tv.value, i64, name="btoi")
        raise CompileError(f"Cannot convert {tv.type} to int")

    def _arrow_type_to_llvm(self, arrow_type: str) -> ir.Type:
        """Map Arrow type string to LLVM IR type."""
        match arrow_type:
            case ArrowType.INT:    return i64
            case ArrowType.FLOAT:  return double
            case ArrowType.BOOL:   return i1
            case ArrowType.STRING: return i8_ptr
            case _: raise CompileError(f"Unknown type: {arrow_type}")

    def _create_entry_alloca(self, name: str, llvm_type: ir.Type) -> ir.AllocaInstr:
        """Create an alloca in the dedicated entry block."""
        return self.alloca_builder.alloca(llvm_type, name=name)


# ─────────────────────────────────────────────
#  OPTIMIZATION
# ─────────────────────────────────────────────
def optimize_ir(llvm_ir: str) -> str:
    """Run LLVM optimization passes on the IR."""
    # Parse the IR
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    # Create a pass manager and add optimizations
    pm = binding.ModulePassManager()

    # Standard optimization passes
    pmb = binding.PassManagerBuilder()
    pmb.opt_level = 2  # -O2
    pmb.populate_module_pass_manager(pm)

    # Run passes
    pm.run(mod)

    return str(mod)


# ─────────────────────────────────────────────
#  COMPILE TO EXECUTABLE
# ─────────────────────────────────────────────
def compile_to_executable(llvm_ir: str, output_path: str):
    """Use clang to compile LLVM IR to a native executable."""
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir)
        ir_path = f.name

    try:
        subprocess.run(
            ["clang", ir_path, "-o", output_path, "-O2"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Compiled to: {output_path}")
    except FileNotFoundError:
        print("Error: 'clang' not found. Install it with:")
        print("  Windows:       https://github.com/llvm/llvm-project/releases (LLVM installer)")
        print("  Ubuntu/Debian: sudo apt install clang")
        print("  macOS:         xcode-select --install")
        print("\nAlternatively, use --jit mode which needs no external compiler:")
        print(f"  python compiler.py {output_path}.arrow --jit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Clang error:\n{e.stderr}")
        sys.exit(1)
    finally:
        os.unlink(ir_path)


# ─────────────────────────────────────────────
#  JIT EXECUTION
# ─────────────────────────────────────────────
def jit_execute(llvm_ir: str):
    """JIT-compile and execute the IR in-process.
    
    Manually resolves C runtime symbols via ctypes and registers them
    with the MCJIT engine, since automatic symbol resolution is unreliable
    on Windows.
    """
    import ctypes
    import ctypes.util
    import platform

    # Register native target for JIT
    try:
        binding.initialize_native_target()
    except Exception:
        pass
    try:
        binding.initialize_native_asmprinter()
    except Exception:
        pass

    system = platform.system()

    # ── Helper: find a C function address across multiple libraries ──
    def _find_func(name: str, libs: list) -> int:
        """Search multiple libraries for a function, return its address."""
        for lib in libs:
            try:
                func = getattr(lib, name)
                return ctypes.cast(func, ctypes.c_void_p).value
            except (AttributeError, OSError):
                continue
        raise RuntimeError(f"Could not find C function '{name}' in any loaded library")

    # ── Load candidate C runtime libraries ──
    libs = []
    if system == "Windows":
        # On Windows, functions are spread across multiple DLLs:
        #   - msvcrt.dll has printf, puts, malloc, etc. (legacy CRT, always present)
        #   - ucrtbase.dll has some but not all (modern UCRT)
        #   - api-ms-win-crt-stdio-l1-1-0.dll has stdio functions
        #   - Python's own DLL also links the CRT
        for dll_name in [
            "msvcrt",
            "ucrtbase",
            "api-ms-win-crt-stdio-l1-1-0",
            "api-ms-win-crt-string-l1-1-0",
            "api-ms-win-crt-heap-l1-1-0",
        ]:
            try:
                libs.append(ctypes.CDLL(dll_name))
            except OSError:
                continue
    elif system == "Darwin":
        libs.append(ctypes.CDLL("libSystem.B.dylib"))
    else:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            libs.append(ctypes.CDLL(libc_name))

    if not libs:
        raise RuntimeError("Could not load any C runtime library")

    # ── Resolve all needed C functions ──
    needed = ["printf", "puts", "fflush", "malloc", "sprintf", "strlen", "strcpy", "strcat"]
    c_functions = {}
    for name in needed:
        c_functions[name] = _find_func(name, libs)

    # ── Parse and compile the IR ──
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = binding.create_mcjit_compiler(mod, target_machine)

    # ── Register each C function address with the JIT engine ──
    for name, addr in c_functions.items():
        engine.add_global_mapping(mod.get_function(name), addr)

    engine.finalize_object()

    # ── Run main() ──
    func_ptr = engine.get_function_address("main")
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(func_ptr)
    result = cfunc()
    return result


# ─────────────────────────────────────────────
#  MAIN — CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Arrow Lang Compiler — Compiles .arrow files via LLVM"
    )
    parser.add_argument("file", help="Source file to compile (.arrow)")
    parser.add_argument("--emit-ir", action="store_true",
                        help="Print the generated LLVM IR and exit")
    parser.add_argument("--emit-ir-opt", action="store_true",
                        help="Print the optimized LLVM IR and exit")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Compile to native executable at FILE")
    parser.add_argument("--jit", action="store_true",
                        help="JIT compile and execute (default if no -o)")

    args = parser.parse_args()

    # Read source
    try:
        with open(args.file) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Lex & parse (reuse the existing frontend)
    try:
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
    except (LexerError, ParseError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Compile to LLVM IR
    try:
        compiler = Compiler()
        llvm_ir = compiler.compile(program)
    except CompileError as e:
        print(f"Compile Error: {e}")
        sys.exit(1)

    # Handle output modes
    if args.emit_ir:
        print(llvm_ir)
    elif args.emit_ir_opt:
        optimized = optimize_ir(llvm_ir)
        print(optimized)
    elif args.output:
        compile_to_executable(llvm_ir, args.output)
    else:
        # Default: JIT execute
        jit_execute(llvm_ir)


if __name__ == "__main__":
    main()