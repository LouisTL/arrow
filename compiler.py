#!/usr/bin/env python3
"""
Arrow Lang — LLVM Compiler Backend (v0.4)

Supports: variables, arithmetic, strings, if/else, while, functions,
          arrow functions, recursion, first-class function pointers,
          closures, arrays, index access/assign, len/push/pop builtins.

Closure strategy:
    Every callable is a closure: { i64 fn_ptr, i64 env_ptr }
    - All functions take an extra first param: i8* env (may be null)
    - At call sites: load fn_ptr and env from closure, call fn(env, args...)
    - Captured variables are stored in a malloc'd env struct
    - Non-capturing functions get env=null, which they ignore
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
    ArrayLit, IndexExpr, IndexAssign,
)


# ─────────────────────────────────────────────
#  TYPE SYSTEM
# ─────────────────────────────────────────────
class ArrowType:
    INT    = "int"
    FLOAT  = "float"
    BOOL   = "bool"
    STRING = "string"
    FUNC   = "func"    # closure pointer (i64-packed)
    ARRAY  = "array"


class TypedValue:
    __slots__ = ("value", "type", "func_info")
    def __init__(self, value, arrow_type, func_info=None):
        self.value = value
        self.type = arrow_type
        self.func_info = func_info


# ─────────────────────────────────────────────
#  LLVM TYPE CONSTANTS
# ─────────────────────────────────────────────
i1      = ir.IntType(1)
i8      = ir.IntType(8)
i32     = ir.IntType(32)
i64     = ir.IntType(64)
double  = ir.DoubleType()
i8_ptr  = i8.as_pointer()
i64_ptr = i64.as_pointer()
void    = ir.VoidType()

# Array struct: { i64 length, i64 capacity, i64* data }
array_struct = ir.LiteralStructType([i64, i64, i64_ptr])
array_ptr = array_struct.as_pointer()

# Closure struct: { i8* fn_ptr, i8* env_ptr }
# Stored as two i64 values for simplicity (fn_ptr as i64, env_ptr as i64)
closure_struct = ir.LiteralStructType([i64, i64])
closure_ptr = closure_struct.as_pointer()


class CompileError(Exception):
    pass


# ─────────────────────────────────────────────
#  FREE VARIABLE ANALYSIS
# ─────────────────────────────────────────────
def find_free_variables(params: list[str], body, outer_scope_vars: set[str]) -> list[str]:
    """Find variables referenced in body that are defined in outer_scope_vars
    but not in params or locally defined."""
    defined = set(params)
    referenced = set()

    def scan_expr(node):
        match node:
            case Identifier(name):
                referenced.add(name)
            case BinOp(_, left, right):
                scan_expr(left); scan_expr(right)
            case UnaryOp(_, operand):
                scan_expr(operand)
            case CallExpr(callee, args):
                scan_expr(callee)
                for a in args: scan_expr(a)
            case ArrayLit(elements):
                for e in elements: scan_expr(e)
            case IndexExpr(obj, index):
                scan_expr(obj); scan_expr(index)
            case ArrowFn(p, b):
                # Recursively scan, but arrow fn params shadow
                if isinstance(b, list):
                    for s in b: scan_stmt(s)
                else:
                    scan_expr(b)
            case NumberLit(_) | StringLit(_) | BoolLit(_):
                pass

    def scan_stmt(node):
        match node:
            case Assignment(name, expr):
                scan_expr(expr)
                defined.add(name)
            case IndexAssign(obj, index, value):
                scan_expr(obj); scan_expr(index); scan_expr(value)
            case PrintStmt(expr):
                scan_expr(expr)
            case ReturnStmt(expr):
                if expr: scan_expr(expr)
            case IfStmt(cond, then_body, else_body):
                scan_expr(cond)
                for s in then_body: scan_stmt(s)
                if else_body:
                    for s in else_body: scan_stmt(s)
            case WhileStmt(cond, body):
                scan_expr(cond)
                for s in body: scan_stmt(s)
            case Block(stmts):
                for s in stmts: scan_stmt(s)
            case FnDecl(name, _, fbody):
                defined.add(name)
                for s in fbody: scan_stmt(s)
            case _:
                scan_expr(node)

    if isinstance(body, list):
        for s in body: scan_stmt(s)
    else:
        scan_expr(body)

    # Free vars = referenced in outer scope but not locally defined
    free = [v for v in referenced if v in outer_scope_vars and v not in defined]
    return sorted(set(free))


# ─────────────────────────────────────────────
#  SCOPES
# ─────────────────────────────────────────────
class FunctionScope:
    def __init__(self, llvm_func, builder, alloca_builder, alloca_block, code_block):
        self.llvm_func = llvm_func
        self.builder = builder
        self.alloca_builder = alloca_builder
        self.alloca_block = alloca_block
        self.code_block = code_block
        self.variables: dict[str, tuple[ir.AllocaInstr, str]] = {}
        self.captures: list[str] = []
        self.env_ptr = None
        # Track what type this function actually returns
        self.observed_return_type: str | None = None


class FuncInfo:
    def __init__(self, llvm_func, n_params, return_type):
        self.llvm_func = llvm_func
        self.n_params = n_params
        self.return_type = return_type


# ─────────────────────────────────────────────
#  CODEGEN
# ─────────────────────────────────────────────
class Compiler:
    PARAM_LLVM_TYPE = i64
    RET_LLVM_TYPE   = i64

    def __init__(self):
        self.module = ir.Module(name="arrow_lang")
        self.module.triple = binding.get_default_triple()

        self._is_windows = "windows" in self.module.triple or "msvc" in self.module.triple
        self._int_fmt = "%I64d" if self._is_windows else "%lld"

        self._str_counter = 0
        self._func_counter = 0
        self.functions: dict[str, FuncInfo] = {}
        self._scope_stack: list[FunctionScope] = []
        self._array_helpers_built = False

        self._declare_externals()
        self._init_main()

    @property
    def scope(self): return self._scope_stack[-1]
    @property
    def builder(self): return self.scope.builder
    @property
    def variables(self): return self.scope.variables

    def _init_main(self):
        func_type = ir.FunctionType(i32, [])
        main_func = ir.Function(self.module, func_type, name="main")
        ab = main_func.append_basic_block("entry")
        cb = main_func.append_basic_block("code")
        self._scope_stack.append(
            FunctionScope(main_func, ir.IRBuilder(cb), ir.IRBuilder(ab), ab, cb))

    # ── External C functions ─────────────────

    def _declare_externals(self):
        self.puts    = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr]), name="puts")
        self.printf  = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr], var_arg=True), name="printf")
        self.fflush  = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr]), name="fflush")
        self.malloc  = ir.Function(self.module, ir.FunctionType(i8_ptr, [i64]), name="malloc")
        self.sprintf = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr, i8_ptr], var_arg=True), name="sprintf")
        self.strlen  = ir.Function(self.module, ir.FunctionType(i64, [i8_ptr]), name="strlen")
        self.strcpy  = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr]), name="strcpy")
        self.strcat  = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr]), name="strcat")

    # ── Closure helpers ──────────────────────

    def _make_closure(self, llvm_func, env_value=None) -> TypedValue:
        """Create a heap-allocated closure struct { fn_ptr_as_i64, env_as_i64 }.
        Returns a TypedValue of type FUNC whose value is a closure_ptr."""
        clos_raw = self.builder.call(self.malloc, [ir.Constant(i64, 16)], name="clos_raw")
        clos = self.builder.bitcast(clos_raw, closure_ptr, name="clos")

        # Store fn_ptr as i64
        fn_as_i64 = self.builder.ptrtoint(llvm_func, i64, name="fn_i64")
        fn_slot = self.builder.gep(clos, [ir.Constant(i32, 0), ir.Constant(i32, 0)], name="fn_slot")
        self.builder.store(fn_as_i64, fn_slot)

        # Store env as i64 (null if no captures)
        if env_value is not None:
            env_as_i64 = self.builder.ptrtoint(env_value, i64, name="env_i64")
        else:
            env_as_i64 = ir.Constant(i64, 0)
        env_slot = self.builder.gep(clos, [ir.Constant(i32, 0), ir.Constant(i32, 1)], name="env_slot")
        self.builder.store(env_as_i64, env_slot)

        return TypedValue(clos, ArrowType.FUNC)

    def _unpack_closure(self, clos_val):
        """Given a closure_ptr, return (fn_as_i64, env_as_i8_ptr)."""
        # clos_val might be i64 (from ABI) or closure_ptr
        if str(clos_val.type) != str(closure_ptr):
            clos_val = self.builder.inttoptr(clos_val, closure_ptr, name="clos_p")

        fn_slot = self.builder.gep(clos_val, [ir.Constant(i32, 0), ir.Constant(i32, 0)], name="fn_slot")
        fn_i64 = self.builder.load(fn_slot, name="fn_i64")

        env_slot = self.builder.gep(clos_val, [ir.Constant(i32, 0), ir.Constant(i32, 1)], name="env_slot")
        env_i64 = self.builder.load(env_slot, name="env_i64")
        env_ptr = self.builder.inttoptr(env_i64, i8_ptr, name="env_ptr")

        return fn_i64, env_ptr

    # ── Array helpers (lazy) ─────────────────

    def _ensure_array_helpers(self):
        if self._array_helpers_built:
            return
        self._array_helpers_built = True
        self.realloc = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i64]), name="realloc")
        self._build_array_helpers()

    def _build_array_helpers(self):
        # All array helper functions use the closure ABI: first param is i8* env (unused)
        # This way they can be wrapped in closures too if needed.
        # But for simplicity, array helpers are called directly, not through closures.

        # ── array_new(capacity) -> array_ptr ──
        ft = ir.FunctionType(array_ptr, [i64])
        fn = ir.Function(self.module, ft, name="arrow_array_new")
        bb = fn.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        cap = fn.args[0]; cap.name = "cap"
        raw = b.call(self.malloc, [ir.Constant(i64, 24)], name="raw")
        ptr = b.bitcast(raw, array_ptr, name="arr")
        data_bytes = b.mul(cap, ir.Constant(i64, 8), name="db")
        data_raw = b.call(self.malloc, [data_bytes], name="dr")
        data = b.bitcast(data_raw, i64_ptr, name="data")
        b.store(ir.Constant(i64, 0), b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 0)]))
        b.store(cap, b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 1)]))
        b.store(data, b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 2)]))
        b.ret(ptr)
        self._array_new = fn

        # ── array_len ──
        ft = ir.FunctionType(i64, [array_ptr])
        fn = ir.Function(self.module, ft, name="arrow_array_len")
        bb = fn.append_basic_block("entry"); b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"
        b.ret(b.load(b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 0)])))
        self._array_len = fn

        # ── array_get ──
        ft = ir.FunctionType(i64, [array_ptr, i64])
        fn = ir.Function(self.module, ft, name="arrow_array_get")
        bb = fn.append_basic_block("entry"); b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"; fn.args[1].name = "idx"
        data = b.load(b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)]))
        b.ret(b.load(b.gep(data, [fn.args[1]])))
        self._array_get = fn

        # ── array_set ──
        ft = ir.FunctionType(void, [array_ptr, i64, i64])
        fn = ir.Function(self.module, ft, name="arrow_array_set")
        bb = fn.append_basic_block("entry"); b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"; fn.args[1].name = "idx"; fn.args[2].name = "val"
        data = b.load(b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)]))
        b.store(fn.args[2], b.gep(data, [fn.args[1]]))
        b.ret_void()
        self._array_set = fn

        # ── array_push ──
        ft = ir.FunctionType(i64, [array_ptr, i64])
        fn = ir.Function(self.module, ft, name="arrow_array_push")
        entry = fn.append_basic_block("entry")
        grow_bb = fn.append_basic_block("grow")
        done_bb = fn.append_basic_block("done")
        b = ir.IRBuilder(entry)
        fn.args[0].name = "arr"; fn.args[1].name = "val"
        arr = fn.args[0]
        lp = b.gep(arr, [ir.Constant(i32, 0), ir.Constant(i32, 0)])
        cp = b.gep(arr, [ir.Constant(i32, 0), ir.Constant(i32, 1)])
        dp = b.gep(arr, [ir.Constant(i32, 0), ir.Constant(i32, 2)])
        length = b.load(lp); cap = b.load(cp)
        b.cbranch(b.icmp_signed('>=', length, cap), grow_bb, done_bb)
        b.position_at_start(grow_bb)
        nc = b.mul(cap, ir.Constant(i64, 2))
        od = b.load(dp)
        nd = b.bitcast(b.call(self.realloc, [b.bitcast(od, i8_ptr), b.mul(nc, ir.Constant(i64, 8))]), i64_ptr)
        b.store(nc, cp); b.store(nd, dp); b.branch(done_bb)
        b.position_at_start(done_bb)
        data = b.load(dp)
        b.store(fn.args[1], b.gep(data, [length]))
        nl = b.add(length, ir.Constant(i64, 1))
        b.store(nl, lp); b.ret(nl)
        self._array_push = fn

        # ── array_pop ──
        ft = ir.FunctionType(i64, [array_ptr])
        fn = ir.Function(self.module, ft, name="arrow_array_pop")
        bb = fn.append_basic_block("entry"); b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"
        lp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 0)])
        dp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)])
        nl = b.sub(b.load(lp), ir.Constant(i64, 1))
        b.store(nl, lp)
        b.ret(b.load(b.gep(b.load(dp), [nl])))
        self._array_pop = fn

    # ── String helpers ───────────────────────

    def _global_string(self, text):
        encoded = (text + "\0").encode("utf-8")
        str_type = ir.ArrayType(i8, len(encoded))
        name = f".str.{self._str_counter}"
        self._str_counter += 1
        gv = ir.GlobalVariable(self.module, str_type, name=name)
        gv.linkage = "private"; gv.global_constant = True
        gv.initializer = ir.Constant(str_type, bytearray(encoded))
        return self.builder.gep(gv, [ir.Constant(i64, 0), ir.Constant(i64, 0)], inbounds=True, name="str_ptr")

    def _value_to_string(self, tv):
        if tv.type == ArrowType.STRING: return tv.value
        buf = self.builder.call(self.malloc, [ir.Constant(i64, 64)], name="fmt_buf")
        if tv.type == ArrowType.INT:
            self.builder.call(self.sprintf, [buf, self._global_string(self._int_fmt), tv.value])
        elif tv.type == ArrowType.FLOAT:
            self.builder.call(self.sprintf, [buf, self._global_string("%g"), tv.value])
        elif tv.type == ArrowType.BOOL:
            return self.builder.select(tv.value, self._global_string("true"), self._global_string("false"))
        return buf

    # ── Current scope variable names ─────────

    def _current_scope_var_names(self) -> set[str]:
        """Get all variable names visible in the current scope."""
        names = set(self.variables.keys())
        names.update(self.functions.keys())
        return names

    # ── Compile entry point ──────────────────

    def compile(self, program):
        # First pass: forward-declare top-level functions
        for stmt in program.statements:
            if isinstance(stmt, FnDecl):
                self._declare_function(stmt.name, stmt.params)
        # Second pass: compile
        for stmt in program.statements:
            self._compile_stmt(stmt)
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(i32, 0))
        self.scope.alloca_builder.branch(self.scope.code_block)
        return str(self.module)

    # ── Function declaration & compilation ───

    def _declare_function(self, name, params):
        """Forward-declare a function. All functions take an extra env param."""
        if name in self.functions:
            return self.functions[name].llvm_func
        # ABI: i64 fn(i8* env, i64 arg1, i64 arg2, ...)
        param_types = [i8_ptr] + [self.PARAM_LLVM_TYPE] * len(params)
        ft = ir.FunctionType(self.RET_LLVM_TYPE, param_types)
        lf = ir.Function(self.module, ft, name=f"arrow_{name}")
        lf.args[0].name = "env"
        for i, p in enumerate(params):
            lf.args[i + 1].name = p

        info = FuncInfo(lf, len(params), ArrowType.INT)
        self.functions[name] = info

        # Create a closure for this function (env=null) and store as variable
        clos_tv = self._make_closure(lf, env_value=None)
        ptr = self._create_entry_alloca(name, closure_ptr)
        self.builder.store(clos_tv.value, ptr)
        self.variables[name] = (ptr, ArrowType.FUNC)

        return lf

    def _compile_function_body(self, name, params, body, captures=None):
        """Compile a function's body. captures = list of (var_name, TypedValue) from outer scope."""
        if captures is None:
            captures = []

        info = self.functions[name]
        lf = info.llvm_func
        ab = lf.append_basic_block("entry")
        cb = lf.append_basic_block("code")
        scope = FunctionScope(lf, ir.IRBuilder(cb), ir.IRBuilder(ab), ab, cb)
        scope.captures = [c[0] for c in captures]
        self._scope_stack.append(scope)

        # Store user params (skip env at index 0)
        for i, pn in enumerate(params):
            ptr = self._create_entry_alloca(pn, self.PARAM_LLVM_TYPE)
            self.builder.store(lf.args[i + 1], ptr)
            self.variables[pn] = (ptr, ArrowType.INT)

        # Load captured variables from env struct
        if captures:
            env_raw = lf.args[0]  # i8* env
            # Env is a block of i64 values
            env_as_i64_ptr = self.builder.bitcast(env_raw, i64_ptr, name="env_data")
            for idx, (cap_name, _cap_tv) in enumerate(captures):
                elem_ptr = self.builder.gep(env_as_i64_ptr, [ir.Constant(i64, idx)], name=f"cap_{cap_name}")
                val = self.builder.load(elem_ptr, name=cap_name)
                ptr = self._create_entry_alloca(cap_name, i64)
                self.builder.store(val, ptr)
                # Determine type — for simplicity, captured vars are INT
                # (they were converted to i64 when captured)
                self.variables[cap_name] = (ptr, ArrowType.INT)

        # Make all global functions visible via closures
        for fn_name, fn_info in self.functions.items():
            if fn_name not in self.variables:
                clos_tv = self._make_closure(fn_info.llvm_func, env_value=None)
                ptr = self._create_entry_alloca(fn_name, closure_ptr)
                self.builder.store(clos_tv.value, ptr)
                self.variables[fn_name] = (ptr, ArrowType.FUNC)

        for stmt in body:
            self._compile_stmt(stmt)
            if self.builder.block.is_terminated: break

        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))

        # Update return type based on what was actually returned
        if scope.observed_return_type is not None:
            info.return_type = scope.observed_return_type

        ir.IRBuilder(ab).branch(cb)
        self._scope_stack.pop()

    # ── Statement codegen ────────────────────

    def _compile_stmt(self, node):
        match node:
            case FnDecl(name, params, body):
                if name not in self.functions:
                    self._declare_function(name, params)
                self._compile_function_body(name, params, body)
            case Assignment(name, expr):
                if isinstance(expr, ArrowFn):
                    self._compile_arrow_fn_assignment(name, expr)
                else:
                    self._compile_assignment(name, expr)
            case IndexAssign(obj, index, value):
                self._compile_index_assign(obj, index, value)
            case PrintStmt(expr):
                self._compile_print(expr)
            case ReturnStmt(expr):
                self._compile_return(expr)
            case IfStmt(cond, then_body, else_body):
                self._compile_if(cond, then_body, else_body)
            case WhileStmt(cond, body):
                self._compile_while(cond, body)
            case Block(stmts):
                for s in stmts: self._compile_stmt(s)
            case _:
                self._compile_expr(node)

    def _compile_arrow_fn_assignment(self, name, arrow):
        """Compile: name <- (params) => body;  with closure capture."""
        body = arrow.body if isinstance(arrow.body, list) else [ReturnStmt(arrow.body)]

        # Find free variables
        outer_vars = self._current_scope_var_names()
        free_vars = find_free_variables(arrow.params, body, outer_vars)

        # Evaluate captures from current scope
        captures = []
        for var_name in free_vars:
            if var_name in self.variables:
                ptr, at = self.variables[var_name]
                val = self.builder.load(ptr, name=f"cap_{var_name}")
                tv = TypedValue(val, at)
                captures.append((var_name, tv))
            elif var_name in self.functions:
                # Capturing a function — load its closure pointer as i64
                info = self.functions[var_name]
                clos_tv = self._make_closure(info.llvm_func, env_value=None)
                captures.append((var_name, TypedValue(
                    self.builder.ptrtoint(clos_tv.value, i64), ArrowType.INT)))

        # Declare and compile the function
        self._declare_function(name, arrow.params)
        self._compile_function_body(name, arrow.params, body, captures=captures)

        # If there are captures, build an env and create a new closure
        if captures:
            info = self.functions[name]
            env_size = ir.Constant(i64, len(captures) * 8)
            env_raw = self.builder.call(self.malloc, [env_size], name="env_raw")
            env_data = self.builder.bitcast(env_raw, i64_ptr, name="env_data")

            for idx, (cap_name, cap_tv) in enumerate(captures):
                val_i64 = self._to_i64(cap_tv)
                slot = self.builder.gep(env_data, [ir.Constant(i64, idx)], name=f"env_{cap_name}")
                self.builder.store(val_i64, slot)

            # Create closure with env
            clos_tv = self._make_closure(info.llvm_func, env_value=env_raw)
            # Update the variable to point to the new closure
            ptr, _ = self.variables[name]
            self.builder.store(clos_tv.value, ptr)

    def _compile_assignment(self, name, expr):
        tv = self._compile_expr(expr)
        if name in self.variables:
            ptr, et = self.variables[name]
            if et == ArrowType.FUNC and tv.type == ArrowType.FUNC:
                self.builder.store(tv.value, ptr)
            elif et != tv.type:
                if et == ArrowType.INT and tv.type == ArrowType.ARRAY:
                    tv = TypedValue(self.builder.ptrtoint(tv.value, i64), ArrowType.INT)
                elif et == ArrowType.ARRAY and tv.type == ArrowType.INT:
                    pass  # allow
                else:
                    raise CompileError(f"Type mismatch: '{name}' is {et}, cannot assign {tv.type}")
            self.builder.store(tv.value, ptr)
        else:
            lt = self._arrow_type_to_llvm(tv.type)
            ptr = self._create_entry_alloca(name, lt)
            self.builder.store(tv.value, ptr)
            self.variables[name] = (ptr, tv.type)

    def _compile_index_assign(self, obj, index, value):
        self._ensure_array_helpers()
        arr_tv = self._compile_expr(obj)
        idx_tv = self._compile_expr(index)
        val_tv = self._compile_expr(value)
        self.builder.call(self._array_set, [self._to_array_ptr(arr_tv), idx_tv.value, self._to_i64(val_tv)])

    def _compile_print(self, expr):
        tv = self._compile_expr(expr)
        null_ptr = ir.Constant(i8_ptr, None)
        if tv.type == ArrowType.ARRAY:
            self._compile_print_array(tv)
        elif tv.type == ArrowType.STRING:
            self.builder.call(self.printf, [self._global_string("%s\n"), tv.value])
        elif tv.type == ArrowType.INT:
            self.builder.call(self.printf, [self._global_string(self._int_fmt + "\n"), tv.value])
        elif tv.type == ArrowType.FLOAT:
            self.builder.call(self.printf, [self._global_string("%g\n"), tv.value])
        elif tv.type == ArrowType.BOOL:
            sv = self.builder.select(tv.value, self._global_string("true"), self._global_string("false"))
            self.builder.call(self.printf, [self._global_string("%s\n"), sv])
        self.builder.call(self.fflush, [null_ptr])

    def _compile_print_array(self, tv):
        self._ensure_array_helpers()
        arr_p = self._to_array_ptr(tv)
        length = self.builder.call(self._array_len, [arr_p], name="alen")
        self.builder.call(self.printf, [self._global_string("[")])
        idx_ptr = self._create_entry_alloca("_pidx", i64)
        self.builder.store(ir.Constant(i64, 0), idx_ptr)
        cond_bb = self.builder.append_basic_block("pa_cond")
        body_bb = self.builder.append_basic_block("pa_body")
        end_bb  = self.builder.append_basic_block("pa_end")
        self.builder.branch(cond_bb)
        self.builder.position_at_start(cond_bb)
        idx = self.builder.load(idx_ptr)
        self.builder.cbranch(self.builder.icmp_signed('<', idx, length), body_bb, end_bb)
        self.builder.position_at_start(body_bb)
        idx = self.builder.load(idx_ptr)
        sep = self.builder.select(
            self.builder.icmp_signed('==', idx, ir.Constant(i64, 0)),
            self._global_string(""), self._global_string(", "))
        self.builder.call(self.printf, [sep])
        val = self.builder.call(self._array_get, [arr_p, idx])
        self.builder.call(self.printf, [self._global_string(self._int_fmt), val])
        self.builder.store(self.builder.add(idx, ir.Constant(i64, 1)), idx_ptr)
        self.builder.branch(cond_bb)
        self.builder.position_at_start(end_bb)
        self.builder.call(self.printf, [self._global_string("]\n")])

    def _compile_return(self, expr):
        if expr is not None:
            tv = self._compile_expr(expr)
            # Track the type being returned so callers know
            self.scope.observed_return_type = tv.type
            self.builder.ret(self._to_i64(tv))
        else:
            self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))

    def _compile_if(self, cond, then_body, else_body):
        cv = self._to_bool(self._compile_expr(cond))
        then_bb = self.builder.append_basic_block("if_then")
        else_bb = self.builder.append_basic_block("if_else") if else_body else None
        merge_bb = self.builder.append_basic_block("if_end")
        self.builder.cbranch(cv, then_bb, else_bb or merge_bb)
        self.builder.position_at_start(then_bb)
        for s in then_body: self._compile_stmt(s)
        if not self.builder.block.is_terminated: self.builder.branch(merge_bb)
        if else_bb:
            self.builder.position_at_start(else_bb)
            for s in else_body: self._compile_stmt(s)
            if not self.builder.block.is_terminated: self.builder.branch(merge_bb)
        self.builder.position_at_start(merge_bb)

    def _compile_while(self, cond, body):
        cond_bb = self.builder.append_basic_block("while_cond")
        body_bb = self.builder.append_basic_block("while_body")
        end_bb  = self.builder.append_basic_block("while_end")
        self.builder.branch(cond_bb)
        self.builder.position_at_start(cond_bb)
        cv = self._to_bool(self._compile_expr(cond))
        self.builder.cbranch(cv, body_bb, end_bb)
        self.builder.position_at_start(body_bb)
        for s in body: self._compile_stmt(s)
        if not self.builder.block.is_terminated: self.builder.branch(cond_bb)
        self.builder.position_at_start(end_bb)

    # ── Expression codegen ───────────────────

    def _compile_expr(self, node):
        match node:
            case NumberLit(value):
                if isinstance(value, float):
                    return TypedValue(ir.Constant(double, value), ArrowType.FLOAT)
                return TypedValue(ir.Constant(i64, value), ArrowType.INT)
            case StringLit(value):
                return TypedValue(self._global_string(value), ArrowType.STRING)
            case BoolLit(value):
                return TypedValue(ir.Constant(i1, int(value)), ArrowType.BOOL)
            case Identifier(name):
                return self._compile_identifier(name)
            case ArrayLit(elements):
                return self._compile_array_lit(elements)
            case IndexExpr(obj, index):
                return self._compile_index_expr(obj, index)
            case UnaryOp(op, operand):
                return self._compile_unary(op, operand)
            case BinOp(op, left, right):
                return self._compile_binop(op, left, right)
            case CallExpr(callee, args):
                return self._compile_call(callee, args)
            case ArrowFn(params, body):
                return self._compile_anonymous_arrow(params, body)
            case _:
                raise CompileError(f"Cannot compile expression: {node}")

    def _compile_anonymous_arrow(self, params, body):
        """Compile an anonymous arrow function expression with closure capture."""
        fn_name = f"_lambda_{self._func_counter}"
        self._func_counter += 1
        fb = body if isinstance(body, list) else [ReturnStmt(body)]

        # Find free variables
        outer_vars = self._current_scope_var_names()
        free_vars = find_free_variables(params, fb, outer_vars)

        captures = []
        for var_name in free_vars:
            if var_name in self.variables:
                ptr, at = self.variables[var_name]
                val = self.builder.load(ptr, name=f"cap_{var_name}")
                captures.append((var_name, TypedValue(val, at)))
            elif var_name in self.functions:
                info = self.functions[var_name]
                clos_tv = self._make_closure(info.llvm_func, env_value=None)
                captures.append((var_name, TypedValue(
                    self.builder.ptrtoint(clos_tv.value, i64), ArrowType.INT)))

        self._declare_function(fn_name, params)
        self._compile_function_body(fn_name, params, fb, captures=captures)

        info = self.functions[fn_name]

        if captures:
            env_size = ir.Constant(i64, len(captures) * 8)
            env_raw = self.builder.call(self.malloc, [env_size], name="env_raw")
            env_data = self.builder.bitcast(env_raw, i64_ptr, name="env_data")
            for idx, (cn, ctv) in enumerate(captures):
                slot = self.builder.gep(env_data, [ir.Constant(i64, idx)], name=f"env_{cn}")
                self.builder.store(self._to_i64(ctv), slot)
            return self._make_closure(info.llvm_func, env_value=env_raw)
        else:
            return self._make_closure(info.llvm_func, env_value=None)

    def _compile_array_lit(self, elements):
        self._ensure_array_helpers()
        cap = max(len(elements), 4)
        arr_p = self.builder.call(self._array_new, [ir.Constant(i64, cap)], name="arr")
        for elem in elements:
            tv = self._compile_expr(elem)
            self.builder.call(self._array_push, [arr_p, self._to_i64(tv)])
        return TypedValue(arr_p, ArrowType.ARRAY)

    def _compile_index_expr(self, obj, index):
        self._ensure_array_helpers()
        obj_tv = self._compile_expr(obj)
        idx_tv = self._compile_expr(index)
        if obj_tv.type == ArrowType.STRING:
            buf = self.builder.call(self.malloc, [ir.Constant(i64, 2)], name="ch_buf")
            cp = self.builder.gep(obj_tv.value, [idx_tv.value], name="cp")
            self.builder.store(self.builder.load(cp), buf)
            self.builder.store(ir.Constant(i8, 0), self.builder.gep(buf, [ir.Constant(i64, 1)]))
            return TypedValue(buf, ArrowType.STRING)
        arr_p = self._to_array_ptr(obj_tv)
        return TypedValue(self.builder.call(self._array_get, [arr_p, idx_tv.value], name="aget"), ArrowType.INT)

    def _compile_identifier(self, name):
        if name in self.variables:
            ptr, at = self.variables[name]
            val = self.builder.load(ptr, name=name)
            return TypedValue(val, at)
        if name in self.functions:
            info = self.functions[name]
            return self._make_closure(info.llvm_func, env_value=None)
        raise CompileError(f"Undefined variable '{name}'")

    def _compile_call(self, callee, args):
        if isinstance(callee, Identifier) and callee.name in ("len", "push", "pop"):
            return self._compile_builtin(callee.name, args)

        arg_tvs = [self._compile_expr(a) for a in args]
        arg_vals = [self._to_i64(tv) for tv in arg_tvs]

        # Determine the known return type (if we compiled the callee)
        known_ret_type = ArrowType.INT
        if isinstance(callee, Identifier) and callee.name in self.functions:
            known_ret_type = self.functions[callee.name].return_type

        # Compile the callee — always get a closure
        callee_tv = self._compile_expr(callee)

        # Unpack the closure
        clos_val = callee_tv.value
        fn_i64, env_ptr = self._unpack_closure(clos_val)

        # Build function type: i64(i8* env, i64, i64, ...)
        param_types = [i8_ptr] + [self.PARAM_LLVM_TYPE] * len(args)
        ft = ir.FunctionType(self.RET_LLVM_TYPE, param_types)
        fn_ptr = self.builder.inttoptr(fn_i64, ft.as_pointer(), name="fn_ptr")

        # Call: fn(env, arg1, arg2, ...)
        result = self.builder.call(fn_ptr, [env_ptr] + arg_vals, name="call")

        # Convert i64 result back to the proper type
        if known_ret_type == ArrowType.ARRAY:
            arr_p = self.builder.inttoptr(result, array_ptr, name="ret_arr")
            return TypedValue(arr_p, ArrowType.ARRAY)
        elif known_ret_type == ArrowType.STRING:
            str_p = self.builder.inttoptr(result, i8_ptr, name="ret_str")
            return TypedValue(str_p, ArrowType.STRING)
        elif known_ret_type == ArrowType.FUNC:
            clos_p = self.builder.inttoptr(result, closure_ptr, name="ret_clos")
            return TypedValue(clos_p, ArrowType.FUNC)
        return TypedValue(result, ArrowType.INT)

    def _compile_builtin(self, name, args):
        self._ensure_array_helpers()
        if name == "len":
            tv = self._compile_expr(args[0])
            if tv.type == ArrowType.STRING:
                return TypedValue(self.builder.call(self.strlen, [tv.value], name="slen"), ArrowType.INT)
            arr_p = self._to_array_ptr(tv)
            return TypedValue(self.builder.call(self._array_len, [arr_p], name="alen"), ArrowType.INT)
        elif name == "push":
            arr_tv = self._compile_expr(args[0]); val_tv = self._compile_expr(args[1])
            return TypedValue(self.builder.call(self._array_push, [self._to_array_ptr(arr_tv), self._to_i64(val_tv)], name="push"), ArrowType.INT)
        elif name == "pop":
            arr_tv = self._compile_expr(args[0])
            return TypedValue(self.builder.call(self._array_pop, [self._to_array_ptr(arr_tv)], name="pop"), ArrowType.INT)
        raise CompileError(f"Unknown builtin: {name}")

    def _compile_unary(self, op, operand):
        tv = self._compile_expr(operand)
        if op == '-':
            if tv.type == ArrowType.INT: return TypedValue(self.builder.neg(tv.value), ArrowType.INT)
            if tv.type == ArrowType.FLOAT: return TypedValue(self.builder.fneg(tv.value), ArrowType.FLOAT)
            raise CompileError(f"Cannot negate {tv.type}")
        if op == '!': return TypedValue(self.builder.not_(self._to_bool(tv)), ArrowType.BOOL)
        raise CompileError(f"Unknown unary: {op}")

    def _compile_binop(self, op, left, right):
        if op == '+':
            ltv = self._compile_expr(left); rtv = self._compile_expr(right)
            if ltv.type == ArrowType.STRING or rtv.type == ArrowType.STRING:
                return self._compile_string_concat(ltv, rtv)
            return self._compile_arithmetic(op, ltv, rtv)
        ltv = self._compile_expr(left); rtv = self._compile_expr(right)
        if op in ('+','-','*','/','%'): return self._compile_arithmetic(op, ltv, rtv)
        if op in ('<','>','<=','>=','=','!='): return self._compile_comparison(op, ltv, rtv)
        if op == '&&': return TypedValue(self.builder.and_(self._to_bool(ltv), self._to_bool(rtv)), ArrowType.BOOL)
        if op == '||': return TypedValue(self.builder.or_(self._to_bool(ltv), self._to_bool(rtv)), ArrowType.BOOL)
        raise CompileError(f"Unknown op: {op}")

    def _compile_arithmetic(self, op, ltv, rtv):
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            lv, rv = self._to_float(ltv), self._to_float(rtv)
            ops = {'+':'fadd','-':'fsub','*':'fmul','/':'fdiv','%':'frem'}
            return TypedValue(getattr(self.builder, ops[op])(lv, rv), ArrowType.FLOAT)
        ops = {'+':'add','-':'sub','*':'mul','/':'sdiv','%':'srem'}
        return TypedValue(getattr(self.builder, ops[op])(ltv.value, rtv.value), ArrowType.INT)

    def _compile_comparison(self, op, ltv, rtv):
        cm = {'<':'<','>':'>','<=':'<=','>=':'>=','=':'==','!=':'!='}
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            return TypedValue(self.builder.fcmp_ordered(cm[op], self._to_float(ltv), self._to_float(rtv)), ArrowType.BOOL)
        return TypedValue(self.builder.icmp_signed(cm[op], self._to_int(ltv), self._to_int(rtv)), ArrowType.BOOL)

    def _compile_string_concat(self, ltv, rtv):
        ls, rs = self._value_to_string(ltv), self._value_to_string(rtv)
        ll = self.builder.call(self.strlen, [ls]); rl = self.builder.call(self.strlen, [rs])
        t = self.builder.add(self.builder.add(ll, rl), ir.Constant(i64, 1))
        buf = self.builder.call(self.malloc, [t], name="cbuf")
        self.builder.call(self.strcpy, [buf, ls]); self.builder.call(self.strcat, [buf, rs])
        return TypedValue(buf, ArrowType.STRING)

    # ── Type conversions ─────────────────────

    def _to_array_ptr(self, tv):
        if tv.type == ArrowType.ARRAY: return tv.value
        if tv.type == ArrowType.INT: return self.builder.inttoptr(tv.value, array_ptr, name="arr_p")
        raise CompileError(f"Cannot convert {tv.type} to array")

    def _to_bool(self, tv):
        if tv.type == ArrowType.BOOL: return tv.value
        if tv.type == ArrowType.INT: return self.builder.icmp_signed('!=', tv.value, ir.Constant(i64, 0))
        if tv.type == ArrowType.FLOAT: return self.builder.fcmp_ordered('!=', tv.value, ir.Constant(double, 0.0))
        if tv.type == ArrowType.STRING: return self.builder.icmp_signed('!=', self.builder.call(self.strlen, [tv.value]), ir.Constant(i64, 0))
        raise CompileError(f"Cannot convert {tv.type} to bool")

    def _to_float(self, tv):
        if tv.type == ArrowType.FLOAT: return tv.value
        if tv.type == ArrowType.INT: return self.builder.sitofp(tv.value, double)
        if tv.type == ArrowType.BOOL: return self.builder.sitofp(self.builder.zext(tv.value, i64), double)
        raise CompileError(f"Cannot convert {tv.type} to float")

    def _to_int(self, tv):
        if tv.type == ArrowType.INT: return tv.value
        if tv.type == ArrowType.BOOL: return self.builder.zext(tv.value, i64)
        raise CompileError(f"Cannot convert {tv.type} to int")

    def _to_i64(self, tv):
        if tv.type == ArrowType.INT: return tv.value
        if tv.type == ArrowType.BOOL: return self.builder.zext(tv.value, i64)
        if tv.type == ArrowType.STRING: return self.builder.ptrtoint(tv.value, i64)
        if tv.type == ArrowType.FLOAT: return self.builder.bitcast(tv.value, i64)
        if tv.type == ArrowType.FUNC: return self.builder.ptrtoint(tv.value, i64)
        if tv.type == ArrowType.ARRAY: return self.builder.ptrtoint(tv.value, i64)
        raise CompileError(f"Cannot convert {tv.type} to i64")

    def _arrow_type_to_llvm(self, at):
        m = {ArrowType.INT: i64, ArrowType.FLOAT: double, ArrowType.BOOL: i1,
             ArrowType.STRING: i8_ptr, ArrowType.FUNC: closure_ptr, ArrowType.ARRAY: array_ptr}
        return m.get(at) or (_ for _ in ()).throw(CompileError(f"Unknown type: {at}"))

    def _create_entry_alloca(self, name, lt):
        return self.scope.alloca_builder.alloca(lt, name=name)


# ─────────────────────────────────────────────
#  OPTIMIZATION / COMPILE / JIT
# ─────────────────────────────────────────────
def optimize_ir(llvm_ir):
    mod = binding.parse_assembly(llvm_ir); mod.verify()
    pm = binding.ModulePassManager()
    pmb = binding.PassManagerBuilder(); pmb.opt_level = 2
    pmb.populate_module_pass_manager(pm); pm.run(mod)
    return str(mod)

def compile_to_executable(llvm_ir, output_path):
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir); ir_path = f.name
    try:
        subprocess.run(["clang", ir_path, "-o", output_path, "-O2"],
                       check=True, capture_output=True, text=True)
        print(f"Compiled to: {output_path}")
    except FileNotFoundError:
        print("Error: 'clang' not found."); sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Clang error:\n{e.stderr}"); sys.exit(1)
    finally:
        os.unlink(ir_path)

def jit_execute(llvm_ir):
    import ctypes, ctypes.util, platform
    try: binding.initialize_native_target()
    except Exception: pass
    try: binding.initialize_native_asmprinter()
    except Exception: pass

    system = platform.system()
    def _find_func(name, libs):
        for lib in libs:
            try: return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
            except (AttributeError, OSError): continue
        return None

    libs = []
    if system == "Windows":
        for dll in ["msvcrt", "ucrtbase", "api-ms-win-crt-stdio-l1-1-0",
                     "api-ms-win-crt-string-l1-1-0", "api-ms-win-crt-heap-l1-1-0"]:
            try: libs.append(ctypes.CDLL(dll))
            except OSError: continue
    elif system == "Darwin":
        libs.append(ctypes.CDLL("libSystem.B.dylib"))
    else:
        ln = ctypes.util.find_library("c")
        if ln: libs.append(ctypes.CDLL(ln))

    all_c = ["printf","puts","fflush","malloc","realloc","sprintf","strlen","strcpy","strcat"]
    needed = [n for n in all_c if f'@"{n}"' in llvm_ir or f'@{n}(' in llvm_ir]
    c_funcs = {}
    for n in needed:
        a = _find_func(n, libs)
        if a: c_funcs[n] = a

    mod = binding.parse_assembly(llvm_ir); mod.verify()
    target = binding.Target.from_default_triple()
    engine = binding.create_mcjit_compiler(mod, target.create_target_machine())
    for n, a in c_funcs.items():
        try: engine.add_global_mapping(mod.get_function(n), a)
        except Exception: pass
    engine.finalize_object()
    return ctypes.CFUNCTYPE(ctypes.c_int)(engine.get_function_address("main"))()

def main():
    parser = argparse.ArgumentParser(description="Arrow Lang Compiler")
    parser.add_argument("file")
    parser.add_argument("--emit-ir", action="store_true")
    parser.add_argument("--emit-ir-opt", action="store_true")
    parser.add_argument("-o", "--output", metavar="FILE")
    args = parser.parse_args()

    try:
        with open(args.file, encoding="utf-8") as f: source = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}"); sys.exit(1)
    try:
        tokens = Lexer(source).tokenize()
        program = Parser(tokens).parse()
    except (LexerError, ParseError) as e:
        print(f"Error: {e}"); sys.exit(1)
    try:
        compiler = Compiler()
        llvm_ir = compiler.compile(program)
    except CompileError as e:
        print(f"Compile Error: {e}"); sys.exit(1)

    if args.emit_ir: print(llvm_ir)
    elif args.emit_ir_opt: print(optimize_ir(llvm_ir))
    elif args.output: compile_to_executable(llvm_ir, args.output)
    else: jit_execute(llvm_ir)

if __name__ == "__main__":
    main()