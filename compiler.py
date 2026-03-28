#!/usr/bin/env python3
"""
Arrow Lang — LLVM Compiler Backend

Compiles Arrow Lang source to native executables via LLVM IR.
Supports: variables, arithmetic, strings, if/else, while, functions,
          arrow functions, recursion, first-class function pointers.

Usage:
    python compiler.py example.arrow                  # compile & run (JIT)
    python compiler.py example.arrow --emit-ir        # print LLVM IR
    python compiler.py example.arrow -o example       # compile to executable

Requires: llvmlite, clang (for -o mode only)
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
    FnDecl, ArrowFn, ReturnStmt, CallExpr,
)


# ─────────────────────────────────────────────
#  TYPE SYSTEM
# ─────────────────────────────────────────────
class ArrowType:
    INT    = "int"
    FLOAT  = "float"
    BOOL   = "bool"
    STRING = "string"
    FUNC   = "func"   # function pointer


class TypedValue:
    __slots__ = ("value", "type", "func_info")

    def __init__(self, value: ir.Value, arrow_type: str, func_info=None):
        self.value = value
        self.type = arrow_type
        # For FUNC type: (ir.Function, param_types, return_type)
        self.func_info = func_info


# ─────────────────────────────────────────────
#  LLVM TYPE CONSTANTS
# ─────────────────────────────────────────────
i1     = ir.IntType(1)
i8     = ir.IntType(8)
i32    = ir.IntType(32)
i64    = ir.IntType(64)
double = ir.DoubleType()
i8_ptr = i8.as_pointer()
void   = ir.VoidType()


class CompileError(Exception):
    pass


# ─────────────────────────────────────────────
#  FUNCTION SCOPE — tracks per-function state
# ─────────────────────────────────────────────
class FunctionScope:
    """State for the function currently being compiled."""
    def __init__(self, llvm_func, builder, alloca_builder, alloca_block, code_block):
        self.llvm_func = llvm_func
        self.builder = builder
        self.alloca_builder = alloca_builder
        self.alloca_block = alloca_block
        self.code_block = code_block
        # Variables local to this function: name -> (alloca_ptr, ArrowType)
        self.variables: dict[str, tuple[ir.AllocaInstr, str]] = {}


# ─────────────────────────────────────────────
#  FUNCTION INFO — metadata about a compiled fn
# ─────────────────────────────────────────────
class FuncInfo:
    """Metadata about a compiled Arrow Lang function."""
    def __init__(self, llvm_func: ir.Function, param_types: list[str], return_type: str):
        self.llvm_func = llvm_func
        self.param_types = param_types
        self.return_type = return_type


# ─────────────────────────────────────────────
#  CODEGEN — AST to LLVM IR
# ─────────────────────────────────────────────
class Compiler:
    """
    Walks the AST and emits LLVM IR using llvmlite's IR builder.

    Function strategy:
    - All Arrow Lang functions compile to LLVM functions.
    - Since Arrow Lang is dynamically typed but LLVM needs static types,
      we use i64 for all function parameters and return values.
      This is a simplification — it means functions can only pass/return
      integers. Strings, bools, and floats are cast through i64.
    - Arrow functions assigned to variables are compiled identically
      to named functions (no closures in compiled mode).
    - Function pointers use a uniform type: i64(i64, i64, ...) so that
      first-class function passing works through a single pointer type.
    """

    # All Arrow Lang functions use i64 params and i64 return
    # This uniform ABI lets us pass function pointers freely.
    PARAM_LLVM_TYPE = i64
    RET_LLVM_TYPE   = i64

    def __init__(self):
        self.module = ir.Module(name="arrow_lang")
        self.module.triple = binding.get_default_triple()

        self._is_windows = "windows" in self.module.triple or "msvc" in self.module.triple
        self._int_fmt = "%I64d" if self._is_windows else "%lld"

        self._str_counter = 0
        self._func_counter = 0

        # Global function registry: name -> FuncInfo
        self.functions: dict[str, FuncInfo] = {}

        # Stack of function scopes (innermost = current)
        self._scope_stack: list[FunctionScope] = []

        # Declare external C functions
        self._declare_externals()

        # Create main function
        self._init_main()

    @property
    def scope(self) -> FunctionScope:
        return self._scope_stack[-1]

    @property
    def builder(self) -> ir.IRBuilder:
        return self.scope.builder

    @property
    def variables(self) -> dict:
        return self.scope.variables

    def _init_main(self):
        """Create the main() function and push its scope."""
        func_type = ir.FunctionType(i32, [])
        main_func = ir.Function(self.module, func_type, name="main")
        alloca_block = main_func.append_basic_block(name="entry")
        code_block = main_func.append_basic_block(name="code")
        alloca_builder = ir.IRBuilder(alloca_block)
        builder = ir.IRBuilder(code_block)
        scope = FunctionScope(main_func, builder, alloca_builder, alloca_block, code_block)
        self._scope_stack.append(scope)

    # ── External C functions ─────────────────

    def _declare_externals(self):
        puts_type = ir.FunctionType(i32, [i8_ptr])
        self.puts = ir.Function(self.module, puts_type, name="puts")

        printf_type = ir.FunctionType(i32, [i8_ptr], var_arg=True)
        self.printf = ir.Function(self.module, printf_type, name="printf")

        fflush_type = ir.FunctionType(i32, [i8_ptr])
        self.fflush = ir.Function(self.module, fflush_type, name="fflush")

        malloc_type = ir.FunctionType(i8_ptr, [i64])
        self.malloc = ir.Function(self.module, malloc_type, name="malloc")

        sprintf_type = ir.FunctionType(i32, [i8_ptr, i8_ptr], var_arg=True)
        self.sprintf = ir.Function(self.module, sprintf_type, name="sprintf")

        strlen_type = ir.FunctionType(i64, [i8_ptr])
        self.strlen = ir.Function(self.module, strlen_type, name="strlen")

        strcpy_type = ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr])
        self.strcpy = ir.Function(self.module, strcpy_type, name="strcpy")

        strcat_type = ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr])
        self.strcat = ir.Function(self.module, strcat_type, name="strcat")

    # ── String helpers ───────────────────────

    def _global_string(self, text: str) -> ir.Value:
        encoded = (text + "\0").encode("utf-8")
        str_type = ir.ArrayType(i8, len(encoded))
        name = f".str.{self._str_counter}"
        self._str_counter += 1

        global_var = ir.GlobalVariable(self.module, str_type, name=name)
        global_var.linkage = "private"
        global_var.global_constant = True
        global_var.initializer = ir.Constant(str_type, bytearray(encoded))

        zero = ir.Constant(i64, 0)
        return self.builder.gep(global_var, [zero, zero], inbounds=True, name="str_ptr")

    def _value_to_string(self, tv: TypedValue) -> ir.Value:
        if tv.type == ArrowType.STRING:
            return tv.value

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
            return self.builder.select(tv.value, true_str, false_str, name="bool_str")

        return buf

    # ── Compile entry point ──────────────────

    def compile(self, program: Program) -> str:
        # First pass: compile all top-level fn declarations so they can
        # reference each other (mutual recursion)
        for stmt in program.statements:
            if isinstance(stmt, FnDecl):
                self._declare_function(stmt.name, stmt.params)

        # Second pass: compile everything
        for stmt in program.statements:
            self._compile_stmt(stmt)

        # Return 0 from main
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(i32, 0))

        # Finalize main's entry block
        self.scope.alloca_builder.branch(self.scope.code_block)

        return str(self.module)

    # ── Function declaration & compilation ───

    def _declare_function(self, name: str, params: list[str]) -> ir.Function:
        """Forward-declare a function (so it can be called before its body is compiled)."""
        if name in self.functions:
            return self.functions[name].llvm_func

        param_types = [self.PARAM_LLVM_TYPE] * len(params)
        func_type = ir.FunctionType(self.RET_LLVM_TYPE, param_types)
        llvm_func = ir.Function(self.module, func_type, name=f"arrow_{name}")

        # Set parameter names
        for i, p in enumerate(params):
            llvm_func.args[i].name = p

        info = FuncInfo(llvm_func, [ArrowType.INT] * len(params), ArrowType.INT)
        self.functions[name] = info

        # Also register as a variable in current scope (function pointer)
        ptr = self._create_entry_alloca(name, llvm_func.type)
        self.builder.store(llvm_func, ptr)
        self.variables[name] = (ptr, ArrowType.FUNC)

        return llvm_func

    def _compile_function_body(self, name: str, params: list[str], body: list):
        """Compile the body of a function into its LLVM function."""
        info = self.functions[name]
        llvm_func = info.llvm_func

        # Create blocks for the function
        alloca_block = llvm_func.append_basic_block("entry")
        code_block = llvm_func.append_basic_block("code")
        alloca_builder = ir.IRBuilder(alloca_block)
        builder = ir.IRBuilder(code_block)

        # Push a new scope
        scope = FunctionScope(llvm_func, builder, alloca_builder, alloca_block, code_block)
        self._scope_stack.append(scope)

        # Create allocas for parameters and store the argument values
        for i, param_name in enumerate(params):
            ptr = self._create_entry_alloca(param_name, self.PARAM_LLVM_TYPE)
            self.builder.store(llvm_func.args[i], ptr)
            self.variables[param_name] = (ptr, ArrowType.INT)

        # Make all global functions visible in this scope
        for fn_name, fn_info in self.functions.items():
            if fn_name not in self.variables:
                ptr = self._create_entry_alloca(fn_name, fn_info.llvm_func.type)
                self.builder.store(fn_info.llvm_func, ptr)
                self.variables[fn_name] = (ptr, ArrowType.FUNC)

        # Compile body statements
        for stmt in body:
            self._compile_stmt(stmt)
            if self.builder.block.is_terminated:
                break

        # Add implicit return 0 if no explicit return
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))

        # Finalize entry block
        alloca_builder.branch(code_block)

        # Pop scope
        self._scope_stack.pop()

    # ── Statement codegen ────────────────────

    def _compile_stmt(self, node):
        match node:
            case FnDecl(name, params, body):
                # Declaration was already done in first pass; compile body
                if name not in self.functions:
                    self._declare_function(name, params)
                self._compile_function_body(name, params, body)

            case Assignment(name, expr):
                # Check if RHS is an arrow function
                if isinstance(expr, ArrowFn):
                    self._compile_arrow_fn_assignment(name, expr)
                else:
                    self._compile_assignment(name, expr)

            case PrintStmt(expr):
                self._compile_print(expr)

            case ReturnStmt(expr):
                self._compile_return(expr)

            case IfStmt(cond, then_body, else_body):
                self._compile_if(cond, then_body, else_body)

            case WhileStmt(cond, body):
                self._compile_while(cond, body)

            case Block(stmts):
                for s in stmts:
                    self._compile_stmt(s)

            case _:
                self._compile_expr(node)

    def _compile_arrow_fn_assignment(self, name: str, arrow: ArrowFn):
        """Compile: name <- (params) => expr_or_body;"""
        # Wrap expression body in a return statement
        if isinstance(arrow.body, list):
            body = arrow.body
        else:
            body = [ReturnStmt(arrow.body)]

        # Declare and compile like a named function
        self._declare_function(name, arrow.params)
        self._compile_function_body(name, arrow.params, body)

    def _compile_assignment(self, name: str, expr):
        tv = self._compile_expr(expr)

        if name in self.variables:
            ptr, existing_type = self.variables[name]
            if existing_type != tv.type:
                raise CompileError(
                    f"Type mismatch: variable '{name}' is {existing_type}, "
                    f"cannot assign {tv.type}"
                )
            self.builder.store(tv.value, ptr)
        else:
            llvm_type = self._arrow_type_to_llvm(tv.type)
            ptr = self._create_entry_alloca(name, llvm_type)
            self.builder.store(tv.value, ptr)
            self.variables[name] = (ptr, tv.type)

    def _compile_print(self, expr):
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

    def _compile_return(self, expr):
        if expr is not None:
            tv = self._compile_expr(expr)
            # Convert to i64 return type
            ret_val = self._to_i64(tv)
            self.builder.ret(ret_val)
        else:
            self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))

    def _compile_if(self, cond, then_body, else_body):
        cond_tv = self._compile_expr(cond)
        cond_val = self._to_bool(cond_tv)

        then_bb  = self.builder.append_basic_block("if_then")
        else_bb  = self.builder.append_basic_block("if_else") if else_body else None
        merge_bb = self.builder.append_basic_block("if_end")

        if else_bb:
            self.builder.cbranch(cond_val, then_bb, else_bb)
        else:
            self.builder.cbranch(cond_val, then_bb, merge_bb)

        self.builder.position_at_start(then_bb)
        for s in then_body:
            self._compile_stmt(s)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        if else_bb:
            self.builder.position_at_start(else_bb)
            for s in else_body:
                self._compile_stmt(s)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        self.builder.position_at_start(merge_bb)

    def _compile_while(self, cond, body):
        cond_bb = self.builder.append_basic_block("while_cond")
        body_bb = self.builder.append_basic_block("while_body")
        end_bb  = self.builder.append_basic_block("while_end")

        self.builder.branch(cond_bb)

        self.builder.position_at_start(cond_bb)
        cond_tv = self._compile_expr(cond)
        cond_val = self._to_bool(cond_tv)
        self.builder.cbranch(cond_val, body_bb, end_bb)

        self.builder.position_at_start(body_bb)
        for s in body:
            self._compile_stmt(s)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)

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
                return self._compile_identifier(name)

            case UnaryOp(op, operand):
                return self._compile_unary(op, operand)

            case BinOp(op, left, right):
                return self._compile_binop(op, left, right)

            case CallExpr(callee, args):
                return self._compile_call(callee, args)

            case ArrowFn(params, body):
                # Anonymous arrow function as expression — compile as named
                fn_name = f"_lambda_{self._func_counter}"
                self._func_counter += 1
                if isinstance(body, list):
                    fn_body = body
                else:
                    fn_body = [ReturnStmt(body)]
                self._declare_function(fn_name, params)
                self._compile_function_body(fn_name, params, fn_body)
                info = self.functions[fn_name]
                return TypedValue(info.llvm_func, ArrowType.FUNC, func_info=info)

            case _:
                raise CompileError(f"Cannot compile expression: {node}")

    def _compile_identifier(self, name: str) -> TypedValue:
        # Check local variables first
        if name in self.variables:
            ptr, arrow_type = self.variables[name]
            val = self.builder.load(ptr, name=name)
            if arrow_type == ArrowType.FUNC and name in self.functions:
                return TypedValue(val, ArrowType.FUNC, func_info=self.functions[name])
            return TypedValue(val, arrow_type)

        # Check global functions
        if name in self.functions:
            info = self.functions[name]
            return TypedValue(info.llvm_func, ArrowType.FUNC, func_info=info)

        raise CompileError(f"Undefined variable '{name}'")

    def _compile_call(self, callee, args) -> TypedValue:
        """Compile a function call: callee(args...)"""
        # Compile arguments
        arg_tvs = [self._compile_expr(a) for a in args]
        arg_vals = [self._to_i64(tv) for tv in arg_tvs]

        # Get the function to call
        if isinstance(callee, Identifier) and callee.name in self.functions:
            # Direct call — most common case
            info = self.functions[callee.name]
            result = self.builder.call(info.llvm_func, arg_vals, name="call")
            return TypedValue(result, info.return_type)

        # Indirect call through a value (function pointer or i64 holding one)
        callee_tv = self._compile_expr(callee)

        # Build the function type for indirect call
        param_types = [self.PARAM_LLVM_TYPE] * len(args)
        func_type = ir.FunctionType(self.RET_LLVM_TYPE, param_types)
        func_ptr_type = func_type.as_pointer()

        if callee_tv.type == ArrowType.FUNC:
            fn_ptr = self.builder.bitcast(callee_tv.value, func_ptr_type, name="fn_ptr")
        elif callee_tv.type == ArrowType.INT:
            # Parameter passed as i64 through uniform ABI — convert back to fn ptr
            fn_ptr = self.builder.inttoptr(callee_tv.value, func_ptr_type, name="fn_ptr")
        else:
            raise CompileError("Cannot call a non-function value")

        result = self.builder.call(fn_ptr, arg_vals, name="icall")
        return TypedValue(result, ArrowType.INT)

    def _compile_unary(self, op: str, operand) -> TypedValue:
        tv = self._compile_expr(operand)

        if op == '-':
            if tv.type == ArrowType.INT:
                return TypedValue(self.builder.neg(tv.value, name="neg"), ArrowType.INT)
            elif tv.type == ArrowType.FLOAT:
                return TypedValue(self.builder.fneg(tv.value, name="fneg"), ArrowType.FLOAT)
            else:
                raise CompileError(f"Cannot negate type {tv.type}")
        elif op == '!':
            bool_val = self._to_bool(tv)
            return TypedValue(self.builder.not_(bool_val, name="not"), ArrowType.BOOL)

        raise CompileError(f"Unknown unary operator: {op}")

    def _compile_binop(self, op: str, left, right) -> TypedValue:
        if op == '+':
            ltv = self._compile_expr(left)
            rtv = self._compile_expr(right)
            if ltv.type == ArrowType.STRING or rtv.type == ArrowType.STRING:
                return self._compile_string_concat(ltv, rtv)
            return self._compile_arithmetic(op, ltv, rtv)

        ltv = self._compile_expr(left)
        rtv = self._compile_expr(right)

        if op in ('+', '-', '*', '/', '%'):
            return self._compile_arithmetic(op, ltv, rtv)
        if op in ('<', '>', '<=', '>=', '=', '!='):
            return self._compile_comparison(op, ltv, rtv)
        if op == '&&':
            return TypedValue(
                self.builder.and_(self._to_bool(ltv), self._to_bool(rtv), name="and"),
                ArrowType.BOOL)
        if op == '||':
            return TypedValue(
                self.builder.or_(self._to_bool(ltv), self._to_bool(rtv), name="or"),
                ArrowType.BOOL)

        raise CompileError(f"Unknown binary operator: {op}")

    def _compile_arithmetic(self, op, ltv, rtv):
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            lv, rv = self._to_float(ltv), self._to_float(rtv)
            ops = {'+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv', '%': 'frem'}
            result = getattr(self.builder, ops[op])(lv, rv, name=ops[op])
            return TypedValue(result, ArrowType.FLOAT)
        else:
            lv, rv = ltv.value, rtv.value
            ops = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv', '%': 'srem'}
            result = getattr(self.builder, ops[op])(lv, rv, name=ops[op])
            return TypedValue(result, ArrowType.INT)

    def _compile_comparison(self, op, ltv, rtv):
        cmp_map = {'<': '<', '>': '>', '<=': '<=', '>=': '>=', '=': '==', '!=': '!='}
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            lv, rv = self._to_float(ltv), self._to_float(rtv)
            result = self.builder.fcmp_ordered(cmp_map[op], lv, rv, name="fcmp")
        else:
            lv, rv = self._to_int(ltv), self._to_int(rtv)
            result = self.builder.icmp_signed(cmp_map[op], lv, rv, name="icmp")
        return TypedValue(result, ArrowType.BOOL)

    def _compile_string_concat(self, ltv, rtv):
        left_str  = self._value_to_string(ltv)
        right_str = self._value_to_string(rtv)

        left_len  = self.builder.call(self.strlen, [left_str],  name="llen")
        right_len = self.builder.call(self.strlen, [right_str], name="rlen")
        total = self.builder.add(left_len, right_len, name="total_len")
        total = self.builder.add(total, ir.Constant(i64, 1), name="with_null")

        buf = self.builder.call(self.malloc, [total], name="concat_buf")
        self.builder.call(self.strcpy, [buf, left_str])
        self.builder.call(self.strcat, [buf, right_str])

        return TypedValue(buf, ArrowType.STRING)

    # ── Type conversion helpers ──────────────

    def _to_bool(self, tv: TypedValue) -> ir.Value:
        if tv.type == ArrowType.BOOL:
            return tv.value
        if tv.type == ArrowType.INT:
            return self.builder.icmp_signed('!=', tv.value, ir.Constant(i64, 0), name="tobool")
        if tv.type == ArrowType.FLOAT:
            return self.builder.fcmp_ordered('!=', tv.value, ir.Constant(double, 0.0), name="tobool")
        if tv.type == ArrowType.STRING:
            length = self.builder.call(self.strlen, [tv.value], name="slen")
            return self.builder.icmp_signed('!=', length, ir.Constant(i64, 0), name="tobool")
        raise CompileError(f"Cannot convert {tv.type} to bool")

    def _to_float(self, tv: TypedValue) -> ir.Value:
        if tv.type == ArrowType.FLOAT:
            return tv.value
        if tv.type == ArrowType.INT:
            return self.builder.sitofp(tv.value, double, name="tofloat")
        if tv.type == ArrowType.BOOL:
            v = self.builder.zext(tv.value, i64, name="btoi")
            return self.builder.sitofp(v, double, name="tofloat")
        raise CompileError(f"Cannot convert {tv.type} to float")

    def _to_int(self, tv: TypedValue) -> ir.Value:
        if tv.type == ArrowType.INT:
            return tv.value
        if tv.type == ArrowType.BOOL:
            return self.builder.zext(tv.value, i64, name="btoi")
        raise CompileError(f"Cannot convert {tv.type} to int")

    def _to_i64(self, tv: TypedValue) -> ir.Value:
        """Convert any value to i64 for passing to/from functions."""
        if tv.type == ArrowType.INT:
            return tv.value
        if tv.type == ArrowType.BOOL:
            return self.builder.zext(tv.value, i64, name="btoi64")
        if tv.type == ArrowType.STRING:
            return self.builder.ptrtoint(tv.value, i64, name="stoi64")
        if tv.type == ArrowType.FLOAT:
            return self.builder.bitcast(tv.value, i64, name="ftoi64")
        if tv.type == ArrowType.FUNC:
            return self.builder.ptrtoint(tv.value, i64, name="fntoi64")
        raise CompileError(f"Cannot convert {tv.type} to i64")

    def _arrow_type_to_llvm(self, arrow_type: str) -> ir.Type:
        match arrow_type:
            case ArrowType.INT:    return i64
            case ArrowType.FLOAT:  return double
            case ArrowType.BOOL:   return i1
            case ArrowType.STRING: return i8_ptr
            case ArrowType.FUNC:   return i8_ptr  # function pointer stored as opaque ptr
            case _: raise CompileError(f"Unknown type: {arrow_type}")

    def _create_entry_alloca(self, name: str, llvm_type: ir.Type) -> ir.AllocaInstr:
        return self.scope.alloca_builder.alloca(llvm_type, name=name)


# ─────────────────────────────────────────────
#  OPTIMIZATION
# ─────────────────────────────────────────────
def optimize_ir(llvm_ir: str) -> str:
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()
    pm = binding.ModulePassManager()
    pmb = binding.PassManagerBuilder()
    pmb.opt_level = 2
    pmb.populate_module_pass_manager(pm)
    pm.run(mod)
    return str(mod)


# ─────────────────────────────────────────────
#  COMPILE TO EXECUTABLE
# ─────────────────────────────────────────────
def compile_to_executable(llvm_ir: str, output_path: str):
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir)
        ir_path = f.name
    try:
        subprocess.run(
            ["clang", ir_path, "-o", output_path, "-O2"],
            check=True, capture_output=True, text=True,
        )
        print(f"Compiled to: {output_path}")
    except FileNotFoundError:
        print("Error: 'clang' not found. Install it with:")
        print("  Windows:       https://github.com/llvm/llvm-project/releases")
        print("  Ubuntu/Debian: sudo apt install clang")
        print("  macOS:         xcode-select --install")
        print(f"\nAlternatively, use JIT mode: python compiler.py {output_path}.arrow")
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
    import ctypes
    import ctypes.util
    import platform

    try:
        binding.initialize_native_target()
    except Exception:
        pass
    try:
        binding.initialize_native_asmprinter()
    except Exception:
        pass

    system = platform.system()

    def _find_func(name, libs):
        for lib in libs:
            try:
                func = getattr(lib, name)
                return ctypes.cast(func, ctypes.c_void_p).value
            except (AttributeError, OSError):
                continue
        raise RuntimeError(f"Could not find C function '{name}'")

    libs = []
    if system == "Windows":
        for dll in ["msvcrt", "ucrtbase", "api-ms-win-crt-stdio-l1-1-0",
                     "api-ms-win-crt-string-l1-1-0", "api-ms-win-crt-heap-l1-1-0"]:
            try:
                libs.append(ctypes.CDLL(dll))
            except OSError:
                continue
    elif system == "Darwin":
        libs.append(ctypes.CDLL("libSystem.B.dylib"))
    else:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            libs.append(ctypes.CDLL(libc_name))

    needed = ["printf", "puts", "fflush", "malloc", "sprintf", "strlen", "strcpy", "strcat"]
    c_functions = {}
    for name in needed:
        c_functions[name] = _find_func(name, libs)

    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = binding.create_mcjit_compiler(mod, target_machine)

    for name, addr in c_functions.items():
        engine.add_global_mapping(mod.get_function(name), addr)

    engine.finalize_object()

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
                        help="Print optimized LLVM IR and exit")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Compile to native executable")
    parser.add_argument("--jit", action="store_true",
                        help="JIT compile and execute (default)")

    args = parser.parse_args()

    try:
        with open(args.file, encoding="utf-8") as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    try:
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
    except (LexerError, ParseError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        compiler = Compiler()
        llvm_ir = compiler.compile(program)
    except CompileError as e:
        print(f"Compile Error: {e}")
        sys.exit(1)

    if args.emit_ir:
        print(llvm_ir)
    elif args.emit_ir_opt:
        print(optimize_ir(llvm_ir))
    elif args.output:
        compile_to_executable(llvm_ir, args.output)
    else:
        jit_execute(llvm_ir)


if __name__ == "__main__":
    main()