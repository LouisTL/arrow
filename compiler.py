#!/usr/bin/env python3
"""
Arrow Lang — LLVM Compiler Backend (v0.3)

Supports: variables, arithmetic, strings, if/else, while, functions,
          arrow functions, recursion, first-class function pointers,
          arrays, index access/assign, len/push/pop builtins.

Usage:
    python compiler.py example.arrow                  # compile & run (JIT)
    python compiler.py example.arrow --emit-ir        # print LLVM IR
    python compiler.py example.arrow -o example       # compile to executable
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
    FUNC   = "func"
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
i1     = ir.IntType(1)
i8     = ir.IntType(8)
i32    = ir.IntType(32)
i64    = ir.IntType(64)
double = ir.DoubleType()
i8_ptr = i8.as_pointer()
i64_ptr = i64.as_pointer()
void   = ir.VoidType()

# Array struct: { i64 length, i64 capacity, i64* data }
array_struct = ir.LiteralStructType([i64, i64, i64_ptr])
array_ptr = array_struct.as_pointer()


class CompileError(Exception):
    pass


class FunctionScope:
    def __init__(self, llvm_func, builder, alloca_builder, alloca_block, code_block):
        self.llvm_func = llvm_func
        self.builder = builder
        self.alloca_builder = alloca_builder
        self.alloca_block = alloca_block
        self.code_block = code_block
        self.variables: dict[str, tuple[ir.AllocaInstr, str]] = {}


class FuncInfo:
    def __init__(self, llvm_func, param_types, return_type):
        self.llvm_func = llvm_func
        self.param_types = param_types
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

        self._declare_externals()
        self._array_helpers_built = False
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
        self._scope_stack.append(FunctionScope(main_func, ir.IRBuilder(cb), ir.IRBuilder(ab), ab, cb))

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

    # ── Array runtime helpers ────────────────

    def _ensure_array_helpers(self):
        """Lazily build array helper functions on first use."""
        if self._array_helpers_built:
            return
        self._array_helpers_built = True
        # Declare realloc (only needed for arrays)
        self.realloc = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i64]), name="realloc")
        self._build_array_helpers()

    def _build_array_helpers(self):
        """Emit LLVM functions for array operations."""

        # ── array_new(capacity) -> array_ptr ──
        ft = ir.FunctionType(array_ptr, [i64])
        fn = ir.Function(self.module, ft, name="arrow_array_new")
        bb = fn.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        cap = fn.args[0]
        cap.name = "cap"
        # malloc the struct
        struct_size = ir.Constant(i64, 24)  # 3 x i64 = 24 bytes
        raw = b.call(self.malloc, [struct_size], name="raw")
        ptr = b.bitcast(raw, array_ptr, name="arr")
        # malloc the data buffer
        data_bytes = b.mul(cap, ir.Constant(i64, 8), name="data_bytes")
        data_raw = b.call(self.malloc, [data_bytes], name="data_raw")
        data = b.bitcast(data_raw, i64_ptr, name="data")
        # Store fields: length=0, capacity=cap, data=data
        len_ptr = b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 0)], name="len_ptr")
        cap_ptr = b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 1)], name="cap_ptr")
        dat_ptr = b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 2)], name="dat_ptr")
        b.store(ir.Constant(i64, 0), len_ptr)
        b.store(cap, cap_ptr)
        b.store(data, dat_ptr)
        b.ret(ptr)
        self._array_new = fn

        # ── array_len(arr_ptr) -> i64 ──
        ft = ir.FunctionType(i64, [array_ptr])
        fn = ir.Function(self.module, ft, name="arrow_array_len")
        bb = fn.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"
        lp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 0)], name="lp")
        b.ret(b.load(lp, name="len"))
        self._array_len = fn

        # ── array_get(arr_ptr, index) -> i64 ──
        ft = ir.FunctionType(i64, [array_ptr, i64])
        fn = ir.Function(self.module, ft, name="arrow_array_get")
        bb = fn.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"
        fn.args[1].name = "idx"
        dp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)], name="dp")
        data = b.load(dp, name="data")
        elem_ptr = b.gep(data, [fn.args[1]], name="ep")
        b.ret(b.load(elem_ptr, name="val"))
        self._array_get = fn

        # ── array_set(arr_ptr, index, value) -> void ──
        ft = ir.FunctionType(void, [array_ptr, i64, i64])
        fn = ir.Function(self.module, ft, name="arrow_array_set")
        bb = fn.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"
        fn.args[1].name = "idx"
        fn.args[2].name = "val"
        dp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)], name="dp")
        data = b.load(dp, name="data")
        elem_ptr = b.gep(data, [fn.args[1]], name="ep")
        b.store(fn.args[2], elem_ptr)
        b.ret_void()
        self._array_set = fn

        # ── array_push(arr_ptr, value) -> i64 (new length) ──
        ft = ir.FunctionType(i64, [array_ptr, i64])
        fn = ir.Function(self.module, ft, name="arrow_array_push")
        entry = fn.append_basic_block("entry")
        grow_bb = fn.append_basic_block("grow")
        done_bb = fn.append_basic_block("done")
        b = ir.IRBuilder(entry)
        fn.args[0].name = "arr"
        fn.args[1].name = "val"
        arr = fn.args[0]
        lp = b.gep(arr, [ir.Constant(i32, 0), ir.Constant(i32, 0)], name="lp")
        cp = b.gep(arr, [ir.Constant(i32, 0), ir.Constant(i32, 1)], name="cp")
        dp = b.gep(arr, [ir.Constant(i32, 0), ir.Constant(i32, 2)], name="dp")
        length = b.load(lp, name="len")
        cap = b.load(cp, name="cap")
        need_grow = b.icmp_signed('>=', length, cap, name="full")
        b.cbranch(need_grow, grow_bb, done_bb)

        # grow block: double capacity, realloc
        b.position_at_start(grow_bb)
        new_cap = b.mul(cap, ir.Constant(i64, 2), name="newcap")
        new_bytes = b.mul(new_cap, ir.Constant(i64, 8), name="newbytes")
        old_data = b.load(dp, name="olddata")
        old_raw = b.bitcast(old_data, i8_ptr, name="oldraw")
        new_raw = b.call(self.realloc, [old_raw, new_bytes], name="newraw")
        new_data = b.bitcast(new_raw, i64_ptr, name="newdata")
        b.store(new_cap, cp)
        b.store(new_data, dp)
        b.branch(done_bb)

        # done block: store value, increment length
        b.position_at_start(done_bb)
        data = b.load(dp, name="data")
        elem_ptr = b.gep(data, [length], name="ep")
        b.store(fn.args[1], elem_ptr)
        new_len = b.add(length, ir.Constant(i64, 1), name="newlen")
        b.store(new_len, lp)
        b.ret(new_len)
        self._array_push = fn

        # ── array_pop(arr_ptr) -> i64 ──
        ft = ir.FunctionType(i64, [array_ptr])
        fn = ir.Function(self.module, ft, name="arrow_array_pop")
        bb = fn.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        fn.args[0].name = "arr"
        lp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 0)], name="lp")
        dp = b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)], name="dp")
        length = b.load(lp, name="len")
        new_len = b.sub(length, ir.Constant(i64, 1), name="newlen")
        b.store(new_len, lp)
        data = b.load(dp, name="data")
        elem_ptr = b.gep(data, [new_len], name="ep")
        val = b.load(elem_ptr, name="val")
        b.ret(val)
        self._array_pop = fn

    # ── String helpers ───────────────────────

    def _global_string(self, text):
        encoded = (text + "\0").encode("utf-8")
        str_type = ir.ArrayType(i8, len(encoded))
        name = f".str.{self._str_counter}"
        self._str_counter += 1
        gv = ir.GlobalVariable(self.module, str_type, name=name)
        gv.linkage = "private"
        gv.global_constant = True
        gv.initializer = ir.Constant(str_type, bytearray(encoded))
        zero = ir.Constant(i64, 0)
        return self.builder.gep(gv, [zero, zero], inbounds=True, name="str_ptr")

    def _value_to_string(self, tv):
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
            ts = self._global_string("true")
            fs = self._global_string("false")
            return self.builder.select(tv.value, ts, fs, name="bool_str")
        return buf

    # ── Compile entry point ──────────────────

    def compile(self, program):
        for stmt in program.statements:
            if isinstance(stmt, FnDecl):
                self._declare_function(stmt.name, stmt.params)
        for stmt in program.statements:
            self._compile_stmt(stmt)
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(i32, 0))
        self.scope.alloca_builder.branch(self.scope.code_block)
        return str(self.module)

    # ── Function declaration & compilation ───

    def _declare_function(self, name, params):
        if name in self.functions:
            return self.functions[name].llvm_func
        pt = [self.PARAM_LLVM_TYPE] * len(params)
        ft = ir.FunctionType(self.RET_LLVM_TYPE, pt)
        lf = ir.Function(self.module, ft, name=f"arrow_{name}")
        for i, p in enumerate(params):
            lf.args[i].name = p
        info = FuncInfo(lf, [ArrowType.INT]*len(params), ArrowType.INT)
        self.functions[name] = info
        ptr = self._create_entry_alloca(name, lf.type)
        self.builder.store(lf, ptr)
        self.variables[name] = (ptr, ArrowType.FUNC)
        return lf

    def _compile_function_body(self, name, params, body):
        info = self.functions[name]
        lf = info.llvm_func
        ab = lf.append_basic_block("entry")
        cb = lf.append_basic_block("code")
        scope = FunctionScope(lf, ir.IRBuilder(cb), ir.IRBuilder(ab), ab, cb)
        self._scope_stack.append(scope)
        for i, pn in enumerate(params):
            ptr = self._create_entry_alloca(pn, self.PARAM_LLVM_TYPE)
            self.builder.store(lf.args[i], ptr)
            self.variables[pn] = (ptr, ArrowType.INT)
        for fn_name, fn_info in self.functions.items():
            if fn_name not in self.variables:
                ptr = self._create_entry_alloca(fn_name, fn_info.llvm_func.type)
                self.builder.store(fn_info.llvm_func, ptr)
                self.variables[fn_name] = (ptr, ArrowType.FUNC)
        for stmt in body:
            self._compile_stmt(stmt)
            if self.builder.block.is_terminated:
                break
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))
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
                for s in stmts:
                    self._compile_stmt(s)
            case _:
                self._compile_expr(node)

    def _compile_arrow_fn_assignment(self, name, arrow):
        body = arrow.body if isinstance(arrow.body, list) else [ReturnStmt(arrow.body)]
        self._declare_function(name, arrow.params)
        self._compile_function_body(name, arrow.params, body)

    def _compile_assignment(self, name, expr):
        tv = self._compile_expr(expr)
        if name in self.variables:
            ptr, et = self.variables[name]
            # Allow type flexibility for arrays passed as i64 through ABI
            if et != tv.type and not (et == ArrowType.ARRAY and tv.type == ArrowType.INT):
                if et == ArrowType.INT and tv.type == ArrowType.ARRAY:
                    # Storing array into what was an int slot — convert
                    tv = TypedValue(self.builder.ptrtoint(tv.value, i64, name="atoi"), ArrowType.INT)
                else:
                    raise CompileError(f"Type mismatch: '{name}' is {et}, cannot assign {tv.type}")
            if tv.type == ArrowType.ARRAY and et == ArrowType.ARRAY:
                self.builder.store(tv.value, ptr)
            else:
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
        arr_p = self._to_array_ptr(arr_tv)
        self.builder.call(self._array_set, [arr_p, idx_tv.value, self._to_i64(val_tv)])

    def _compile_print(self, expr):
        tv = self._compile_expr(expr)
        null_ptr = ir.Constant(i8_ptr, None)

        if tv.type == ArrowType.ARRAY:
            self._compile_print_array(tv)
        elif tv.type == ArrowType.STRING:
            fmt = self._global_string("%s\n")
            self.builder.call(self.printf, [fmt, tv.value])
        elif tv.type == ArrowType.INT:
            fmt = self._global_string(self._int_fmt + "\n")
            self.builder.call(self.printf, [fmt, tv.value])
        elif tv.type == ArrowType.FLOAT:
            fmt = self._global_string("%g\n")
            self.builder.call(self.printf, [fmt, tv.value])
        elif tv.type == ArrowType.BOOL:
            ts = self._global_string("true")
            fs = self._global_string("false")
            sv = self.builder.select(tv.value, ts, fs, name="bool_str")
            fmt = self._global_string("%s\n")
            self.builder.call(self.printf, [fmt, sv])
        self.builder.call(self.fflush, [null_ptr])

    def _compile_print_array(self, tv):
        """Print array as [elem, elem, ...] using a loop."""
        self._ensure_array_helpers()
        arr_p = self._to_array_ptr(tv)
        length = self.builder.call(self._array_len, [arr_p], name="alen")

        # Print opening bracket
        ob = self._global_string("[")
        self.builder.call(self.printf, [ob])

        # Loop: for i = 0; i < length; i++
        idx_ptr = self._create_entry_alloca("_pidx", i64)
        self.builder.store(ir.Constant(i64, 0), idx_ptr)

        cond_bb = self.builder.append_basic_block("print_cond")
        body_bb = self.builder.append_basic_block("print_body")
        end_bb  = self.builder.append_basic_block("print_end")
        self.builder.branch(cond_bb)

        self.builder.position_at_start(cond_bb)
        idx = self.builder.load(idx_ptr, name="pidx")
        cond = self.builder.icmp_signed('<', idx, length, name="pcond")
        self.builder.cbranch(cond, body_bb, end_bb)

        self.builder.position_at_start(body_bb)
        idx = self.builder.load(idx_ptr, name="pidx2")
        # Print ", " separator (skip for first element)
        is_first = self.builder.icmp_signed('==', idx, ir.Constant(i64, 0), name="first")
        sep = self._global_string(", ")
        nosep = self._global_string("")
        sep_str = self.builder.select(is_first, nosep, sep, name="sep")
        self.builder.call(self.printf, [sep_str])
        # Get and print element
        val = self.builder.call(self._array_get, [arr_p, idx], name="elem")
        fmt = self._global_string(self._int_fmt)
        self.builder.call(self.printf, [fmt, val])
        # i++
        next_idx = self.builder.add(idx, ir.Constant(i64, 1), name="next")
        self.builder.store(next_idx, idx_ptr)
        self.builder.branch(cond_bb)

        self.builder.position_at_start(end_bb)
        cb = self._global_string("]\n")
        self.builder.call(self.printf, [cb])

    def _compile_return(self, expr):
        if expr is not None:
            tv = self._compile_expr(expr)
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
                fn_name = f"_lambda_{self._func_counter}"
                self._func_counter += 1
                fb = body if isinstance(body, list) else [ReturnStmt(body)]
                self._declare_function(fn_name, params)
                self._compile_function_body(fn_name, params, fb)
                info = self.functions[fn_name]
                return TypedValue(info.llvm_func, ArrowType.FUNC, func_info=info)
            case _:
                raise CompileError(f"Cannot compile expression: {node}")

    def _compile_array_lit(self, elements):
        """Compile [expr, expr, ...] -> heap-allocated array."""
        self._ensure_array_helpers()
        n = len(elements)
        cap = max(n, 4)  # minimum capacity of 4
        arr_p = self.builder.call(self._array_new, [ir.Constant(i64, cap)], name="arr")
        for elem in elements:
            tv = self._compile_expr(elem)
            self.builder.call(self._array_push, [arr_p, self._to_i64(tv)])
        return TypedValue(arr_p, ArrowType.ARRAY)

    def _compile_index_expr(self, obj, index):
        """Compile expr[index]."""
        self._ensure_array_helpers()
        obj_tv = self._compile_expr(obj)
        idx_tv = self._compile_expr(index)

        if obj_tv.type == ArrowType.STRING:
            # String indexing: return single char as string
            # For simplicity, use sprintf to create a 2-byte string
            buf = self.builder.call(self.malloc, [ir.Constant(i64, 2)], name="char_buf")
            # Get char via pointer arithmetic
            char_ptr = self.builder.gep(obj_tv.value, [idx_tv.value], name="char_ptr")
            ch = self.builder.load(char_ptr, name="ch")
            self.builder.store(ch, buf)
            null_pos = self.builder.gep(buf, [ir.Constant(i64, 1)], name="null_pos")
            self.builder.store(ir.Constant(i8, 0), null_pos)
            return TypedValue(buf, ArrowType.STRING)

        # Array indexing
        arr_p = self._to_array_ptr(obj_tv)
        val = self.builder.call(self._array_get, [arr_p, idx_tv.value], name="aget")
        return TypedValue(val, ArrowType.INT)

    def _compile_identifier(self, name):
        if name in self.variables:
            ptr, at = self.variables[name]
            val = self.builder.load(ptr, name=name)
            if at == ArrowType.FUNC and name in self.functions:
                return TypedValue(val, ArrowType.FUNC, func_info=self.functions[name])
            return TypedValue(val, at)
        if name in self.functions:
            info = self.functions[name]
            return TypedValue(info.llvm_func, ArrowType.FUNC, func_info=info)
        raise CompileError(f"Undefined variable '{name}'")

    def _compile_call(self, callee, args):
        # Handle builtins
        if isinstance(callee, Identifier) and callee.name in ("len", "push", "pop"):
            return self._compile_builtin(callee.name, args)

        arg_tvs = [self._compile_expr(a) for a in args]
        arg_vals = [self._to_i64(tv) for tv in arg_tvs]

        if isinstance(callee, Identifier) and callee.name in self.functions:
            info = self.functions[callee.name]
            result = self.builder.call(info.llvm_func, arg_vals, name="call")
            return TypedValue(result, info.return_type)

        callee_tv = self._compile_expr(callee)
        pt = [self.PARAM_LLVM_TYPE] * len(args)
        ft = ir.FunctionType(self.RET_LLVM_TYPE, pt)
        fpt = ft.as_pointer()

        if callee_tv.type == ArrowType.FUNC:
            fn_ptr = self.builder.bitcast(callee_tv.value, fpt, name="fn_ptr")
        elif callee_tv.type == ArrowType.INT:
            fn_ptr = self.builder.inttoptr(callee_tv.value, fpt, name="fn_ptr")
        else:
            raise CompileError("Cannot call a non-function value")

        result = self.builder.call(fn_ptr, arg_vals, name="icall")
        return TypedValue(result, ArrowType.INT)

    def _compile_builtin(self, name, args):
        self._ensure_array_helpers()
        if name == "len":
            tv = self._compile_expr(args[0])
            if tv.type == ArrowType.STRING:
                result = self.builder.call(self.strlen, [tv.value], name="slen")
                return TypedValue(result, ArrowType.INT)
            elif tv.type == ArrowType.ARRAY:
                arr_p = self._to_array_ptr(tv)
                result = self.builder.call(self._array_len, [arr_p], name="alen")
                return TypedValue(result, ArrowType.INT)
            else:
                # Might be array passed as i64 through ABI
                arr_p = self.builder.inttoptr(tv.value, array_ptr, name="arr_p")
                result = self.builder.call(self._array_len, [arr_p], name="alen")
                return TypedValue(result, ArrowType.INT)

        elif name == "push":
            arr_tv = self._compile_expr(args[0])
            val_tv = self._compile_expr(args[1])
            arr_p = self._to_array_ptr(arr_tv)
            result = self.builder.call(self._array_push, [arr_p, self._to_i64(val_tv)], name="push")
            return TypedValue(result, ArrowType.INT)

        elif name == "pop":
            arr_tv = self._compile_expr(args[0])
            arr_p = self._to_array_ptr(arr_tv)
            result = self.builder.call(self._array_pop, [arr_p], name="pop")
            return TypedValue(result, ArrowType.INT)

        raise CompileError(f"Unknown builtin: {name}")

    def _compile_unary(self, op, operand):
        tv = self._compile_expr(operand)
        if op == '-':
            if tv.type == ArrowType.INT:
                return TypedValue(self.builder.neg(tv.value, name="neg"), ArrowType.INT)
            elif tv.type == ArrowType.FLOAT:
                return TypedValue(self.builder.fneg(tv.value, name="fneg"), ArrowType.FLOAT)
            raise CompileError(f"Cannot negate type {tv.type}")
        elif op == '!':
            return TypedValue(self.builder.not_(self._to_bool(tv), name="not"), ArrowType.BOOL)
        raise CompileError(f"Unknown unary operator: {op}")

    def _compile_binop(self, op, left, right):
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
            return TypedValue(self.builder.and_(self._to_bool(ltv), self._to_bool(rtv), name="and"), ArrowType.BOOL)
        if op == '||':
            return TypedValue(self.builder.or_(self._to_bool(ltv), self._to_bool(rtv), name="or"), ArrowType.BOOL)
        raise CompileError(f"Unknown binary operator: {op}")

    def _compile_arithmetic(self, op, ltv, rtv):
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            lv, rv = self._to_float(ltv), self._to_float(rtv)
            ops = {'+': 'fadd', '-': 'fsub', '*': 'fmul', '/': 'fdiv', '%': 'frem'}
            return TypedValue(getattr(self.builder, ops[op])(lv, rv, name=ops[op]), ArrowType.FLOAT)
        lv, rv = ltv.value, rtv.value
        ops = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv', '%': 'srem'}
        return TypedValue(getattr(self.builder, ops[op])(lv, rv, name=ops[op]), ArrowType.INT)

    def _compile_comparison(self, op, ltv, rtv):
        cm = {'<': '<', '>': '>', '<=': '<=', '>=': '>=', '=': '==', '!=': '!='}
        if ltv.type == ArrowType.FLOAT or rtv.type == ArrowType.FLOAT:
            return TypedValue(self.builder.fcmp_ordered(cm[op], self._to_float(ltv), self._to_float(rtv), name="fcmp"), ArrowType.BOOL)
        return TypedValue(self.builder.icmp_signed(cm[op], self._to_int(ltv), self._to_int(rtv), name="icmp"), ArrowType.BOOL)

    def _compile_string_concat(self, ltv, rtv):
        ls, rs = self._value_to_string(ltv), self._value_to_string(rtv)
        ll = self.builder.call(self.strlen, [ls], name="llen")
        rl = self.builder.call(self.strlen, [rs], name="rlen")
        t = self.builder.add(self.builder.add(ll, rl, name="tl"), ir.Constant(i64, 1), name="wn")
        buf = self.builder.call(self.malloc, [t], name="cbuf")
        self.builder.call(self.strcpy, [buf, ls])
        self.builder.call(self.strcat, [buf, rs])
        return TypedValue(buf, ArrowType.STRING)

    # ── Type conversions ─────────────────────

    def _to_array_ptr(self, tv):
        """Convert a TypedValue to an array_ptr."""
        if tv.type == ArrowType.ARRAY:
            return tv.value
        # Could be array passed as i64 through function ABI
        if tv.type == ArrowType.INT:
            return self.builder.inttoptr(tv.value, array_ptr, name="arr_p")
        raise CompileError(f"Cannot convert {tv.type} to array pointer")

    def _to_bool(self, tv):
        if tv.type == ArrowType.BOOL: return tv.value
        if tv.type == ArrowType.INT:
            return self.builder.icmp_signed('!=', tv.value, ir.Constant(i64, 0), name="tobool")
        if tv.type == ArrowType.FLOAT:
            return self.builder.fcmp_ordered('!=', tv.value, ir.Constant(double, 0.0), name="tobool")
        if tv.type == ArrowType.STRING:
            return self.builder.icmp_signed('!=', self.builder.call(self.strlen, [tv.value], name="sl"), ir.Constant(i64, 0), name="tobool")
        raise CompileError(f"Cannot convert {tv.type} to bool")

    def _to_float(self, tv):
        if tv.type == ArrowType.FLOAT: return tv.value
        if tv.type == ArrowType.INT: return self.builder.sitofp(tv.value, double, name="tof")
        if tv.type == ArrowType.BOOL: return self.builder.sitofp(self.builder.zext(tv.value, i64), double, name="tof")
        raise CompileError(f"Cannot convert {tv.type} to float")

    def _to_int(self, tv):
        if tv.type == ArrowType.INT: return tv.value
        if tv.type == ArrowType.BOOL: return self.builder.zext(tv.value, i64, name="btoi")
        raise CompileError(f"Cannot convert {tv.type} to int")

    def _to_i64(self, tv):
        if tv.type == ArrowType.INT: return tv.value
        if tv.type == ArrowType.BOOL: return self.builder.zext(tv.value, i64, name="btoi64")
        if tv.type == ArrowType.STRING: return self.builder.ptrtoint(tv.value, i64, name="stoi64")
        if tv.type == ArrowType.FLOAT: return self.builder.bitcast(tv.value, i64, name="ftoi64")
        if tv.type == ArrowType.FUNC: return self.builder.ptrtoint(tv.value, i64, name="fntoi64")
        if tv.type == ArrowType.ARRAY: return self.builder.ptrtoint(tv.value, i64, name="arrtoi64")
        raise CompileError(f"Cannot convert {tv.type} to i64")

    def _arrow_type_to_llvm(self, at):
        m = {ArrowType.INT: i64, ArrowType.FLOAT: double, ArrowType.BOOL: i1,
             ArrowType.STRING: i8_ptr, ArrowType.FUNC: i8_ptr, ArrowType.ARRAY: array_ptr}
        if at in m: return m[at]
        raise CompileError(f"Unknown type: {at}")

    def _create_entry_alloca(self, name, lt):
        return self.scope.alloca_builder.alloca(lt, name=name)


# ─────────────────────────────────────────────
#  OPTIMIZATION / COMPILE / JIT (unchanged)
# ─────────────────────────────────────────────
def optimize_ir(llvm_ir):
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()
    pm = binding.ModulePassManager()
    pmb = binding.PassManagerBuilder()
    pmb.opt_level = 2
    pmb.populate_module_pass_manager(pm)
    pm.run(mod)
    return str(mod)

def compile_to_executable(llvm_ir, output_path):
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir)
        ir_path = f.name
    try:
        subprocess.run(["clang", ir_path, "-o", output_path, "-O2"],
                       check=True, capture_output=True, text=True)
        print(f"Compiled to: {output_path}")
    except FileNotFoundError:
        print("Error: 'clang' not found.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Clang error:\n{e.stderr}")
        sys.exit(1)
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
            try:
                func = getattr(lib, name)
                return ctypes.cast(func, ctypes.c_void_p).value
            except (AttributeError, OSError):
                continue
        return None  # Return None instead of raising

    libs = []
    if system == "Windows":
        for dll in ["msvcrt", "ucrtbase", "api-ms-win-crt-stdio-l1-1-0",
                     "api-ms-win-crt-string-l1-1-0", "api-ms-win-crt-heap-l1-1-0",
                     "api-ms-win-crt-utility-l1-1-0"]:
            try: libs.append(ctypes.CDLL(dll))
            except OSError: continue
    elif system == "Darwin":
        libs.append(ctypes.CDLL("libSystem.B.dylib"))
    else:
        ln = ctypes.util.find_library("c")
        if ln: libs.append(ctypes.CDLL(ln))

    # Resolve C functions that are actually declared in the module
    all_c_names = ["printf", "puts", "fflush", "malloc", "realloc", "sprintf",
                   "strlen", "strcpy", "strcat", "memcpy"]
    
    # Check which functions are actually declared in the IR
    needed = []
    for name in all_c_names:
        if f'@"{name}"' in llvm_ir or f'@{name}(' in llvm_ir:
            needed.append(name)

    c_funcs = {}
    for name in needed:
        addr = _find_func(name, libs)
        if addr is not None:
            c_funcs[name] = addr

    mod = binding.parse_assembly(llvm_ir)
    mod.verify()
    target = binding.Target.from_default_triple()
    tm = target.create_target_machine()
    engine = binding.create_mcjit_compiler(mod, tm)

    # Map resolved C functions to their LLVM declarations
    for name, addr in c_funcs.items():
        try:
            engine.add_global_mapping(mod.get_function(name), addr)
        except Exception:
            pass  # Function might not be declared in this module

    engine.finalize_object()

    fp = engine.get_function_address("main")
    return ctypes.CFUNCTYPE(ctypes.c_int)(fp)()

def main():
    parser = argparse.ArgumentParser(description="Arrow Lang Compiler")
    parser.add_argument("file")
    parser.add_argument("--emit-ir", action="store_true")
    parser.add_argument("--emit-ir-opt", action="store_true")
    parser.add_argument("-o", "--output", metavar="FILE")
    parser.add_argument("--jit", action="store_true")
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