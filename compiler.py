#!/usr/bin/env python3
"""
Arrow Lang — LLVM Compiler Backend (v0.4)

Supports: variables, arithmetic, strings, if/else, while, functions,
          arrow functions, recursion, closures, arrays, structs.

Struct strategy:
    Heap-allocated as: [i64 count, i64 val0, val1, ..., i8* name0, name1, ...]
    Field access resolved at compile-time when layout is known,
    or at runtime via name lookup when passed through function ABI.
"""

import sys, argparse, subprocess, tempfile, os
from llvmlite import ir, binding
from lang import (
    Lexer, Parser, LexerError, ParseError,
    Program, Assignment, PrintStmt, IfStmt, WhileStmt, Block,
    NumberLit, StringLit, BoolLit, Identifier, BinOp, UnaryOp,
    FnDecl, ArrowFn, ReturnStmt, CallExpr,
    ArrayLit, IndexExpr, IndexAssign,
    StructLit, DotExpr, DotAssign,
)

# ─── TYPE SYSTEM ─────────────────────────
class ArrowType:
    INT = "int"; FLOAT = "float"; BOOL = "bool"; STRING = "string"
    FUNC = "func"; ARRAY = "array"; STRUCT = "struct"

class TypedValue:
    __slots__ = ("value", "type", "func_info", "struct_fields")
    def __init__(self, value, arrow_type, func_info=None, struct_fields=None):
        self.value = value
        self.type = arrow_type
        self.func_info = func_info
        # For STRUCT type: list of field name strings (compile-time layout)
        self.struct_fields = struct_fields

# ─── LLVM TYPES ──────────────────────────
i1 = ir.IntType(1); i8 = ir.IntType(8); i32 = ir.IntType(32); i64 = ir.IntType(64)
double = ir.DoubleType(); i8_ptr = i8.as_pointer(); i64_ptr = i64.as_pointer(); void = ir.VoidType()
array_struct = ir.LiteralStructType([i64, i64, i64_ptr]); array_ptr = array_struct.as_pointer()
closure_struct = ir.LiteralStructType([i64, i64]); closure_ptr = closure_struct.as_pointer()

class CompileError(Exception): pass

# ─── FREE VARIABLE ANALYSIS ─────────────
def find_free_variables(params, body, outer_scope_vars):
    defined = set(params); referenced = set()
    def scan_expr(node):
        match node:
            case Identifier(name): referenced.add(name)
            case BinOp(_, l, r): scan_expr(l); scan_expr(r)
            case UnaryOp(_, o): scan_expr(o)
            case CallExpr(c, args): scan_expr(c); [scan_expr(a) for a in args]
            case ArrayLit(elems): [scan_expr(e) for e in elems]
            case IndexExpr(o, i): scan_expr(o); scan_expr(i)
            case DotExpr(o, _): scan_expr(o)
            case StructLit(fields): [scan_expr(v) for _, v in fields]
            case ArrowFn(p, b):
                if isinstance(b, list): [scan_stmt(s) for s in b]
                else: scan_expr(b)
            case NumberLit(_) | StringLit(_) | BoolLit(_): pass
    def scan_stmt(node):
        match node:
            case Assignment(n, e): scan_expr(e); defined.add(n)
            case IndexAssign(o, i, v): scan_expr(o); scan_expr(i); scan_expr(v)
            case DotAssign(o, _, v): scan_expr(o); scan_expr(v)
            case PrintStmt(e): scan_expr(e)
            case ReturnStmt(e):
                if e: scan_expr(e)
            case IfStmt(c, t, el):
                scan_expr(c); [scan_stmt(s) for s in t]
                if el: [scan_stmt(s) for s in el]
            case WhileStmt(c, b): scan_expr(c); [scan_stmt(s) for s in b]
            case Block(ss): [scan_stmt(s) for s in ss]
            case FnDecl(n, _, fb): defined.add(n); [scan_stmt(s) for s in fb]
            case _: scan_expr(node)
    if isinstance(body, list): [scan_stmt(s) for s in body]
    else: scan_expr(body)
    return sorted(set(v for v in referenced if v in outer_scope_vars and v not in defined))

# ─── SCOPES ──────────────────────────────
class FunctionScope:
    def __init__(self, llvm_func, builder, alloca_builder, alloca_block, code_block):
        self.llvm_func = llvm_func; self.builder = builder
        self.alloca_builder = alloca_builder
        self.alloca_block = alloca_block; self.code_block = code_block
        self.variables: dict[str, tuple] = {}
        # Track struct field layouts: var_name -> [(field_name, ArrowType), ...]
        self.struct_layouts: dict[str, list] = {}
        self.captures = []; self.env_ptr = None
        self.observed_return_type = None
        self.observed_return_struct_fields = None

class FuncInfo:
    def __init__(self, llvm_func, n_params, return_type):
        self.llvm_func = llvm_func; self.n_params = n_params
        self.return_type = return_type
        self.return_struct_fields = None

# ─── COMPILER ────────────────────────────
class Compiler:
    PARAM_LLVM_TYPE = i64; RET_LLVM_TYPE = i64

    def __init__(self):
        self.module = ir.Module(name="arrow_lang")
        self.module.triple = binding.get_default_triple()
        self._is_windows = "windows" in self.module.triple or "msvc" in self.module.triple
        self._int_fmt = "%I64d" if self._is_windows else "%lld"
        self._str_counter = 0; self._func_counter = 0
        self.functions: dict[str, FuncInfo] = {}
        self._scope_stack: list[FunctionScope] = []
        self._array_helpers_built = False; self._struct_helpers_built = False
        self._io_helpers_built = False
        self._declare_externals()
        self._init_main()

    @property
    def scope(self): return self._scope_stack[-1]
    @property
    def builder(self): return self.scope.builder
    @property
    def variables(self): return self.scope.variables

    def _init_main(self):
        ft = ir.FunctionType(i32, [])
        mf = ir.Function(self.module, ft, name="main")
        ab = mf.append_basic_block("entry"); cb = mf.append_basic_block("code")
        self._scope_stack.append(FunctionScope(mf, ir.IRBuilder(cb), ir.IRBuilder(ab), ab, cb))

    def _declare_externals(self):
        self.puts = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr]), name="puts")
        self.printf = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr], var_arg=True), name="printf")
        self.fflush = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr]), name="fflush")
        self.malloc = ir.Function(self.module, ir.FunctionType(i8_ptr, [i64]), name="malloc")
        self.sprintf = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr, i8_ptr], var_arg=True), name="sprintf")
        self.strlen = ir.Function(self.module, ir.FunctionType(i64, [i8_ptr]), name="strlen")
        self.strcpy = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr]), name="strcpy")
        self.strcat = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr]), name="strcat")
        self.strcmp = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr, i8_ptr]), name="strcmp")

    # ─── Closure helpers ─────────────────
    def _make_closure(self, llvm_func, env_value=None):
        raw = self.builder.call(self.malloc, [ir.Constant(i64, 16)], name="clos_raw")
        clos = self.builder.bitcast(raw, closure_ptr, name="clos")
        fn_i64 = self.builder.ptrtoint(llvm_func, i64, name="fn_i64")
        self.builder.store(fn_i64, self.builder.gep(clos, [ir.Constant(i32, 0), ir.Constant(i32, 0)]))
        env_i64 = self.builder.ptrtoint(env_value, i64) if env_value else ir.Constant(i64, 0)
        self.builder.store(env_i64, self.builder.gep(clos, [ir.Constant(i32, 0), ir.Constant(i32, 1)]))
        return TypedValue(clos, ArrowType.FUNC)

    def _unpack_closure(self, clos_val):
        if str(clos_val.type) != str(closure_ptr):
            clos_val = self.builder.inttoptr(clos_val, closure_ptr, name="clos_p")
        fn_i64 = self.builder.load(self.builder.gep(clos_val, [ir.Constant(i32, 0), ir.Constant(i32, 0)]))
        env_i64 = self.builder.load(self.builder.gep(clos_val, [ir.Constant(i32, 0), ir.Constant(i32, 1)]))
        return fn_i64, self.builder.inttoptr(env_i64, i8_ptr, name="env_ptr")

    # ─── Array helpers (lazy) ────────────
    def _ensure_array_helpers(self):
        if self._array_helpers_built: return
        self._array_helpers_built = True
        self.realloc = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i64]), name="realloc")
        self._build_array_helpers()

    def _build_array_helpers(self):
        # array_new
        fn = ir.Function(self.module, ir.FunctionType(array_ptr, [i64]), name="arrow_array_new")
        b = ir.IRBuilder(fn.append_basic_block("entry")); fn.args[0].name = "cap"
        raw = b.call(self.malloc, [ir.Constant(i64, 24)]); ptr = b.bitcast(raw, array_ptr)
        dr = b.call(self.malloc, [b.mul(fn.args[0], ir.Constant(i64, 8))]); data = b.bitcast(dr, i64_ptr)
        b.store(ir.Constant(i64, 0), b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 0)]))
        b.store(fn.args[0], b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 1)]))
        b.store(data, b.gep(ptr, [ir.Constant(i32, 0), ir.Constant(i32, 2)])); b.ret(ptr)
        self._array_new = fn
        # array_len
        fn = ir.Function(self.module, ir.FunctionType(i64, [array_ptr]), name="arrow_array_len")
        b = ir.IRBuilder(fn.append_basic_block("entry")); fn.args[0].name = "arr"
        b.ret(b.load(b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 0)]))); self._array_len = fn
        # array_get
        fn = ir.Function(self.module, ir.FunctionType(i64, [array_ptr, i64]), name="arrow_array_get")
        b = ir.IRBuilder(fn.append_basic_block("entry")); fn.args[0].name = "arr"; fn.args[1].name = "idx"
        d = b.load(b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)]))
        b.ret(b.load(b.gep(d, [fn.args[1]]))); self._array_get = fn
        # array_set
        fn = ir.Function(self.module, ir.FunctionType(void, [array_ptr, i64, i64]), name="arrow_array_set")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn.args[0].name="arr"; fn.args[1].name="idx"; fn.args[2].name="val"
        d = b.load(b.gep(fn.args[0], [ir.Constant(i32, 0), ir.Constant(i32, 2)]))
        b.store(fn.args[2], b.gep(d, [fn.args[1]])); b.ret_void(); self._array_set = fn
        # array_push
        fn = ir.Function(self.module, ir.FunctionType(i64, [array_ptr, i64]), name="arrow_array_push")
        entry = fn.append_basic_block("entry"); grow = fn.append_basic_block("grow"); done = fn.append_basic_block("done")
        b = ir.IRBuilder(entry); fn.args[0].name="arr"; fn.args[1].name="val"; arr=fn.args[0]
        lp=b.gep(arr,[ir.Constant(i32,0),ir.Constant(i32,0)]); cp=b.gep(arr,[ir.Constant(i32,0),ir.Constant(i32,1)])
        dp=b.gep(arr,[ir.Constant(i32,0),ir.Constant(i32,2)]); ln=b.load(lp); ca=b.load(cp)
        b.cbranch(b.icmp_signed('>=',ln,ca), grow, done)
        b.position_at_start(grow); nc=b.mul(ca,ir.Constant(i64,2)); od=b.load(dp)
        nd=b.bitcast(b.call(self.realloc,[b.bitcast(od,i8_ptr),b.mul(nc,ir.Constant(i64,8))]),i64_ptr)
        b.store(nc,cp); b.store(nd,dp); b.branch(done)
        b.position_at_start(done); d=b.load(dp); b.store(fn.args[1],b.gep(d,[ln]))
        nl=b.add(ln,ir.Constant(i64,1)); b.store(nl,lp); b.ret(nl); self._array_push = fn
        # array_pop
        fn = ir.Function(self.module, ir.FunctionType(i64, [array_ptr]), name="arrow_array_pop")
        b = ir.IRBuilder(fn.append_basic_block("entry")); fn.args[0].name="arr"
        lp=b.gep(fn.args[0],[ir.Constant(i32,0),ir.Constant(i32,0)])
        dp=b.gep(fn.args[0],[ir.Constant(i32,0),ir.Constant(i32,2)])
        nl=b.sub(b.load(lp),ir.Constant(i64,1)); b.store(nl,lp)
        b.ret(b.load(b.gep(b.load(dp),[nl]))); self._array_pop = fn

    # ─── Struct helpers (lazy) ───────────
    def _ensure_struct_helpers(self):
        if self._struct_helpers_built: return
        self._struct_helpers_built = True
        self._build_struct_helpers()

    def _build_struct_helpers(self):
        """Build runtime helpers for struct operations.
        
        Struct memory layout (all i64 slots):
            [0]         = field count (N)
            [1..N]      = field values (i64 each)
            [N+1..2N]   = field name pointers (i8* cast to i64)
        
        Total size = (1 + 2*N) * 8 bytes
        """
        # struct_new(count) -> i64_ptr — allocate a struct
        fn = ir.Function(self.module, ir.FunctionType(i64_ptr, [i64]), name="arrow_struct_new")
        b = ir.IRBuilder(fn.append_basic_block("entry")); fn.args[0].name = "count"
        # total slots = 1 + 2*count
        total = b.add(ir.Constant(i64, 1), b.mul(fn.args[0], ir.Constant(i64, 2)))
        raw = b.call(self.malloc, [b.mul(total, ir.Constant(i64, 8))])
        ptr = b.bitcast(raw, i64_ptr)
        b.store(fn.args[0], b.gep(ptr, [ir.Constant(i64, 0)]))  # store count
        b.ret(ptr)
        self._struct_new = fn

        # struct_get_by_index(ptr, index) -> i64 — get field by index
        fn = ir.Function(self.module, ir.FunctionType(i64, [i64_ptr, i64]), name="arrow_struct_get")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn.args[0].name = "sptr"; fn.args[1].name = "idx"
        # value at slot [1 + idx]
        slot = b.add(fn.args[1], ir.Constant(i64, 1))
        b.ret(b.load(b.gep(fn.args[0], [slot])))
        self._struct_get = fn

        # struct_set_by_index(ptr, index, value) -> void
        fn = ir.Function(self.module, ir.FunctionType(void, [i64_ptr, i64, i64]), name="arrow_struct_set")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn.args[0].name = "sptr"; fn.args[1].name = "idx"; fn.args[2].name = "val"
        slot = b.add(fn.args[1], ir.Constant(i64, 1))
        b.store(fn.args[2], b.gep(fn.args[0], [slot]))
        b.ret_void()
        self._struct_set = fn

        # struct_get_name(ptr, index) -> i8* — get field name by index
        fn = ir.Function(self.module, ir.FunctionType(i8_ptr, [i64_ptr, i64]), name="arrow_struct_get_name")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        fn.args[0].name = "sptr"; fn.args[1].name = "idx"
        count = b.load(b.gep(fn.args[0], [ir.Constant(i64, 0)]))
        # name at slot [1 + count + idx]
        name_slot = b.add(b.add(ir.Constant(i64, 1), count), fn.args[1])
        name_i64 = b.load(b.gep(fn.args[0], [name_slot]))
        b.ret(b.inttoptr(name_i64, i8_ptr))
        self._struct_get_name = fn

        # struct_count(ptr) -> i64
        fn = ir.Function(self.module, ir.FunctionType(i64, [i64_ptr]), name="arrow_struct_count")
        b = ir.IRBuilder(fn.append_basic_block("entry")); fn.args[0].name = "sptr"
        b.ret(b.load(b.gep(fn.args[0], [ir.Constant(i64, 0)])))
        self._struct_count = fn

        # struct_find_field(ptr, name_str) -> i64 index (or -1)
        fn = ir.Function(self.module, ir.FunctionType(i64, [i64_ptr, i8_ptr]), name="arrow_struct_find")
        fn.args[0].name = "sptr"; fn.args[1].name = "name"
        entry = fn.append_basic_block("entry")
        cond_bb = fn.append_basic_block("cond"); body_bb = fn.append_basic_block("body")
        found_bb = fn.append_basic_block("found"); next_bb = fn.append_basic_block("next")
        end_bb = fn.append_basic_block("end")
        b = ir.IRBuilder(entry)
        count = b.load(b.gep(fn.args[0], [ir.Constant(i64, 0)]))
        idx_ptr = b.alloca(i64, name="idx"); b.store(ir.Constant(i64, 0), idx_ptr)
        b.branch(cond_bb)
        b = ir.IRBuilder(cond_bb); idx = b.load(idx_ptr)
        b.cbranch(b.icmp_signed('<', idx, count), body_bb, end_bb)
        b = ir.IRBuilder(body_bb); idx = b.load(idx_ptr)
        name_slot = b.add(b.add(ir.Constant(i64, 1), count), idx)
        name_i64 = b.load(b.gep(fn.args[0], [name_slot]))
        name_ptr = b.inttoptr(name_i64, i8_ptr)
        cmp = b.call(self.strcmp, [name_ptr, fn.args[1]])
        b.cbranch(b.icmp_signed('==', cmp, ir.Constant(i32, 0)), found_bb, next_bb)
        b = ir.IRBuilder(found_bb); b.ret(b.load(idx_ptr))
        b = ir.IRBuilder(next_bb)
        b.store(b.add(b.load(idx_ptr), ir.Constant(i64, 1)), idx_ptr); b.branch(cond_bb)
        b = ir.IRBuilder(end_bb); b.ret(ir.Constant(i64, -1))
        self._struct_find = fn

    # ─── String helpers ──────────────────
    def _global_string(self, text):
        encoded = (text + "\0").encode("utf-8")
        st = ir.ArrayType(i8, len(encoded)); name = f".str.{self._str_counter}"; self._str_counter += 1
        gv = ir.GlobalVariable(self.module, st, name=name)
        gv.linkage = "private"; gv.global_constant = True
        gv.initializer = ir.Constant(st, bytearray(encoded))
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

    def _current_scope_var_names(self):
        names = set(self.variables.keys()); names.update(self.functions.keys()); return names

    # ─── Compile entry ───────────────────
    def compile(self, program):
        for s in program.statements:
            if isinstance(s, FnDecl): self._declare_function(s.name, s.params)
        for s in program.statements: self._compile_stmt(s)
        if not self.builder.block.is_terminated: self.builder.ret(ir.Constant(i32, 0))
        self.scope.alloca_builder.branch(self.scope.code_block)
        return str(self.module)

    # ─── Function compilation ────────────
    def _declare_function(self, name, params):
        if name in self.functions: return self.functions[name].llvm_func
        pt = [i8_ptr] + [self.PARAM_LLVM_TYPE] * len(params)
        lf = ir.Function(self.module, ir.FunctionType(self.RET_LLVM_TYPE, pt), name=f"arrow_{name}")
        lf.args[0].name = "env"
        for i, p in enumerate(params): lf.args[i+1].name = p
        info = FuncInfo(lf, len(params), ArrowType.INT); self.functions[name] = info
        ctv = self._make_closure(lf)
        ptr = self._create_entry_alloca(name, closure_ptr)
        self.builder.store(ctv.value, ptr); self.variables[name] = (ptr, ArrowType.FUNC)
        return lf

    def _compile_function_body(self, name, params, body, captures=None):
        captures = captures or []
        info = self.functions[name]; lf = info.llvm_func
        ab = lf.append_basic_block("entry"); cb = lf.append_basic_block("code")
        scope = FunctionScope(lf, ir.IRBuilder(cb), ir.IRBuilder(ab), ab, cb)
        scope.captures = [c[0] for c in captures]; self._scope_stack.append(scope)
        for i, pn in enumerate(params):
            p = self._create_entry_alloca(pn, self.PARAM_LLVM_TYPE)
            self.builder.store(lf.args[i+1], p); self.variables[pn] = (p, ArrowType.INT)
        if captures:
            env_data = self.builder.bitcast(lf.args[0], i64_ptr, name="env_data")
            for idx, (cn, _) in enumerate(captures):
                val = self.builder.load(self.builder.gep(env_data, [ir.Constant(i64, idx)]))
                p = self._create_entry_alloca(cn, i64); self.builder.store(val, p)
                self.variables[cn] = (p, ArrowType.INT)
        for fn_name, fn_info in self.functions.items():
            if fn_name not in self.variables:
                ctv = self._make_closure(fn_info.llvm_func)
                p = self._create_entry_alloca(fn_name, closure_ptr)
                self.builder.store(ctv.value, p); self.variables[fn_name] = (p, ArrowType.FUNC)
        for stmt in body:
            self._compile_stmt(stmt)
            if self.builder.block.is_terminated: break
        if not self.builder.block.is_terminated:
            self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))
        if scope.observed_return_type:
            info.return_type = scope.observed_return_type
        if scope.observed_return_struct_fields:
            info.return_struct_fields = scope.observed_return_struct_fields
        ir.IRBuilder(ab).branch(cb); self._scope_stack.pop()

    # ─── Statement codegen ───────────────
    def _compile_stmt(self, node):
        match node:
            case FnDecl(name, params, body):
                if name not in self.functions: self._declare_function(name, params)
                self._compile_function_body(name, params, body)
            case Assignment(name, expr):
                if isinstance(expr, ArrowFn): self._compile_arrow_fn_assignment(name, expr)
                else: self._compile_assignment(name, expr)
            case IndexAssign(obj, index, value): self._compile_index_assign(obj, index, value)
            case DotAssign(obj, field, value): self._compile_dot_assign(obj, field, value)
            case PrintStmt(expr): self._compile_print(expr)
            case ReturnStmt(expr): self._compile_return(expr)
            case IfStmt(cond, tb, eb): self._compile_if(cond, tb, eb)
            case WhileStmt(cond, body): self._compile_while(cond, body)
            case Block(stmts):
                for s in stmts: self._compile_stmt(s)
            case _: self._compile_expr(node)

    def _compile_arrow_fn_assignment(self, name, arrow):
        body = arrow.body if isinstance(arrow.body, list) else [ReturnStmt(arrow.body)]
        outer_vars = self._current_scope_var_names()
        free_vars = find_free_variables(arrow.params, body, outer_vars)
        captures = []
        for vn in free_vars:
            if vn in self.variables:
                p, at = self.variables[vn]; v = self.builder.load(p)
                captures.append((vn, TypedValue(v, at)))
            elif vn in self.functions:
                info = self.functions[vn]; ct = self._make_closure(info.llvm_func)
                captures.append((vn, TypedValue(self.builder.ptrtoint(ct.value, i64), ArrowType.INT)))
        self._declare_function(name, arrow.params)
        self._compile_function_body(name, arrow.params, body, captures=captures)
        if captures:
            info = self.functions[name]
            er = self.builder.call(self.malloc, [ir.Constant(i64, len(captures)*8)], name="env_raw")
            ed = self.builder.bitcast(er, i64_ptr)
            for idx, (cn, ctv) in enumerate(captures):
                self.builder.store(self._to_i64(ctv), self.builder.gep(ed, [ir.Constant(i64, idx)]))
            ct = self._make_closure(info.llvm_func, env_value=er)
            p, _ = self.variables[name]; self.builder.store(ct.value, p)

    def _compile_assignment(self, name, expr):
        tv = self._compile_expr(expr)
        # Save struct layout if assigning a struct
        if tv.type == ArrowType.STRUCT and tv.struct_fields:
            self.scope.struct_layouts[name] = tv.struct_fields
        if name in self.variables:
            p, et = self.variables[name]
            if et != tv.type:
                if et == ArrowType.FUNC and tv.type == ArrowType.FUNC: pass
                elif et == ArrowType.INT and tv.type in (ArrowType.ARRAY, ArrowType.STRUCT):
                    tv = TypedValue(self.builder.ptrtoint(tv.value, i64), ArrowType.INT)
                elif et in (ArrowType.ARRAY, ArrowType.STRUCT) and tv.type == ArrowType.INT: pass
                else:
                    raise CompileError(f"Type mismatch: '{name}' is {et}, cannot assign {tv.type}")
            self.builder.store(tv.value, p)
        else:
            lt = self._arrow_type_to_llvm(tv.type)
            p = self._create_entry_alloca(name, lt)
            self.builder.store(tv.value, p); self.variables[name] = (p, tv.type)

    def _compile_index_assign(self, obj, index, value):
        self._ensure_array_helpers()
        at = self._compile_expr(obj); it = self._compile_expr(index); vt = self._compile_expr(value)
        self.builder.call(self._array_set, [self._to_array_ptr(at), it.value, self._to_i64(vt)])

    def _compile_dot_assign(self, obj, field, value):
        """Compile: obj.field <- value;"""
        self._ensure_struct_helpers()
        obj_tv = self._compile_expr(obj)
        val_tv = self._compile_expr(value)
        sptr = self._to_struct_ptr(obj_tv)

        if obj_tv.struct_fields:
            for idx, (fname, _) in enumerate(obj_tv.struct_fields):
                if fname == field:
                    self.builder.call(self._struct_set, [sptr, ir.Constant(i64, idx), self._to_i64(val_tv)])
                    return
        # Runtime field lookup
        name_str = self._global_string(field)
        fidx = self.builder.call(self._struct_find, [sptr, name_str], name="fidx")
        self.builder.call(self._struct_set, [sptr, fidx, self._to_i64(val_tv)])

    def _compile_print(self, expr):
        tv = self._compile_expr(expr)
        null = ir.Constant(i8_ptr, None)
        if tv.type == ArrowType.ARRAY: self._compile_print_array(tv)
        elif tv.type == ArrowType.STRUCT: self._compile_print_struct(tv)
        elif tv.type == ArrowType.STRING:
            self.builder.call(self.printf, [self._global_string("%s\n"), tv.value])
        elif tv.type == ArrowType.INT:
            self.builder.call(self.printf, [self._global_string(self._int_fmt + "\n"), tv.value])
        elif tv.type == ArrowType.FLOAT:
            self.builder.call(self.printf, [self._global_string("%g\n"), tv.value])
        elif tv.type == ArrowType.BOOL:
            sv = self.builder.select(tv.value, self._global_string("true"), self._global_string("false"))
            self.builder.call(self.printf, [self._global_string("%s\n"), sv])
        self.builder.call(self.fflush, [null])

    def _compile_print_array(self, tv):
        self._ensure_array_helpers()
        ap = self._to_array_ptr(tv); ln = self.builder.call(self._array_len, [ap])
        self.builder.call(self.printf, [self._global_string("[")])
        ip = self._create_entry_alloca("_pidx", i64); self.builder.store(ir.Constant(i64, 0), ip)
        cb = self.builder.append_basic_block("pa_c"); bb = self.builder.append_basic_block("pa_b")
        eb = self.builder.append_basic_block("pa_e"); self.builder.branch(cb)
        self.builder.position_at_start(cb); idx = self.builder.load(ip)
        self.builder.cbranch(self.builder.icmp_signed('<', idx, ln), bb, eb)
        self.builder.position_at_start(bb); idx = self.builder.load(ip)
        sep = self.builder.select(self.builder.icmp_signed('==', idx, ir.Constant(i64, 0)),
            self._global_string(""), self._global_string(", "))
        self.builder.call(self.printf, [sep])
        self.builder.call(self.printf, [self._global_string(self._int_fmt),
            self.builder.call(self._array_get, [ap, idx])])
        self.builder.store(self.builder.add(idx, ir.Constant(i64, 1)), ip); self.builder.branch(cb)
        self.builder.position_at_start(eb)
        self.builder.call(self.printf, [self._global_string("]\n")])

    def _compile_print_struct(self, tv):
        """Print struct as {key: value, key: value, ...}"""
        self._ensure_struct_helpers()
        sptr = self._to_struct_ptr(tv)

        if tv.struct_fields:
            # Compile-time unrolled print — we know each field's type
            self.builder.call(self.printf, [self._global_string("{")])
            for idx, (fname, ftype) in enumerate(tv.struct_fields):
                if idx > 0:
                    self.builder.call(self.printf, [self._global_string(", ")])
                val = self.builder.call(self._struct_get, [sptr, ir.Constant(i64, idx)])
                if ftype == ArrowType.STRING:
                    str_p = self.builder.inttoptr(val, i8_ptr)
                    self.builder.call(self.printf, [self._global_string(f"{fname}: %s"), str_p])
                elif ftype == ArrowType.BOOL:
                    bv = self.builder.trunc(val, i1)
                    sv = self.builder.select(bv, self._global_string("true"), self._global_string("false"))
                    self.builder.call(self.printf, [self._global_string(f"{fname}: %s"), sv])
                else:
                    self.builder.call(self.printf, [self._global_string(f"{fname}: " + self._int_fmt), val])
            self.builder.call(self.printf, [self._global_string("}\n")])
        else:
            # Runtime loop — no type info, print all values as ints
            count = self.builder.call(self._struct_count, [sptr], name="scount")
            self.builder.call(self.printf, [self._global_string("{")])
            ip = self._create_entry_alloca("_sidx", i64)
            self.builder.store(ir.Constant(i64, 0), ip)
            cb = self.builder.append_basic_block("ps_c")
            bb = self.builder.append_basic_block("ps_b")
            eb = self.builder.append_basic_block("ps_e")
            self.builder.branch(cb)
            self.builder.position_at_start(cb); idx = self.builder.load(ip)
            self.builder.cbranch(self.builder.icmp_signed('<', idx, count), bb, eb)
            self.builder.position_at_start(bb); idx = self.builder.load(ip)
            sep = self.builder.select(self.builder.icmp_signed('==', idx, ir.Constant(i64, 0)),
                self._global_string(""), self._global_string(", "))
            self.builder.call(self.printf, [sep])
            np = self.builder.call(self._struct_get_name, [sptr, idx])
            val = self.builder.call(self._struct_get, [sptr, idx])
            self.builder.call(self.printf, [self._global_string("%s: " + self._int_fmt), np, val])
            self.builder.store(self.builder.add(idx, ir.Constant(i64, 1)), ip)
            self.builder.branch(cb)
            self.builder.position_at_start(eb)
            self.builder.call(self.printf, [self._global_string("}\n")])

    def _compile_return(self, expr):
        if expr:
            tv = self._compile_expr(expr)
            # Track return type — prefer non-INT types (STRING, ARRAY, STRUCT, FUNC)
            # since INT could just be a value passed through the ABI
            prev = self.scope.observed_return_type
            if prev is None or prev == ArrowType.INT or tv.type != ArrowType.INT:
                self.scope.observed_return_type = tv.type
            if tv.type == ArrowType.STRUCT and tv.struct_fields:
                self.scope.observed_return_struct_fields = tv.struct_fields
            self.builder.ret(self._to_i64(tv))
        else: self.builder.ret(ir.Constant(self.RET_LLVM_TYPE, 0))

    def _compile_if(self, cond, tb, eb):
        cv = self._to_bool(self._compile_expr(cond))
        then_bb = self.builder.append_basic_block("if_then")
        else_bb = self.builder.append_basic_block("if_else") if eb else None
        merge = self.builder.append_basic_block("if_end")
        self.builder.cbranch(cv, then_bb, else_bb or merge)
        self.builder.position_at_start(then_bb)
        for s in tb: self._compile_stmt(s)
        if not self.builder.block.is_terminated: self.builder.branch(merge)
        if else_bb:
            self.builder.position_at_start(else_bb)
            for s in eb: self._compile_stmt(s)
            if not self.builder.block.is_terminated: self.builder.branch(merge)
        self.builder.position_at_start(merge)

    def _compile_while(self, cond, body):
        cc = self.builder.append_basic_block("wc"); wb = self.builder.append_basic_block("wb")
        we = self.builder.append_basic_block("we"); self.builder.branch(cc)
        self.builder.position_at_start(cc)
        self.builder.cbranch(self._to_bool(self._compile_expr(cond)), wb, we)
        self.builder.position_at_start(wb)
        for s in body: self._compile_stmt(s)
        if not self.builder.block.is_terminated: self.builder.branch(cc)
        self.builder.position_at_start(we)

    # ─── Expression codegen ──────────────
    def _compile_expr(self, node):
        match node:
            case NumberLit(v):
                return TypedValue(ir.Constant(double if isinstance(v, float) else i64, v),
                    ArrowType.FLOAT if isinstance(v, float) else ArrowType.INT)
            case StringLit(v): return TypedValue(self._global_string(v), ArrowType.STRING)
            case BoolLit(v): return TypedValue(ir.Constant(i1, int(v)), ArrowType.BOOL)
            case Identifier(name): return self._compile_identifier(name)
            case ArrayLit(elems): return self._compile_array_lit(elems)
            case StructLit(fields): return self._compile_struct_lit(fields)
            case IndexExpr(obj, idx): return self._compile_index_expr(obj, idx)
            case DotExpr(obj, field): return self._compile_dot_expr(obj, field)
            case UnaryOp(op, operand): return self._compile_unary(op, operand)
            case BinOp(op, l, r): return self._compile_binop(op, l, r)
            case CallExpr(callee, args): return self._compile_call(callee, args)
            case ArrowFn(params, body): return self._compile_anonymous_arrow(params, body)
            case _: raise CompileError(f"Cannot compile: {node}")

    def _compile_struct_lit(self, fields):
        """Compile {key: val, ...} -> heap struct."""
        self._ensure_struct_helpers()
        n = len(fields)
        sptr = self.builder.call(self._struct_new, [ir.Constant(i64, n)], name="sptr")
        field_info = []  # list of (name, ArrowType)
        for idx, (key, val_expr) in enumerate(fields):
            tv = self._compile_expr(val_expr)
            self.builder.call(self._struct_set, [sptr, ir.Constant(i64, idx), self._to_i64(tv)])
            # Store field name pointer
            name_str = self._global_string(key)
            name_i64 = self.builder.ptrtoint(name_str, i64)
            count_val = ir.Constant(i64, n)
            name_slot = self.builder.add(
                self.builder.add(ir.Constant(i64, 1), count_val),
                ir.Constant(i64, idx))
            self.builder.store(name_i64, self.builder.gep(sptr, [name_slot]))
            field_info.append((key, tv.type))
        return TypedValue(sptr, ArrowType.STRUCT, struct_fields=field_info)

    def _compile_dot_expr(self, obj, field):
        """Compile obj.field — uses compile-time index and type if known."""
        self._ensure_struct_helpers()
        obj_tv = self._compile_expr(obj)
        sptr = self._to_struct_ptr(obj_tv)

        field_type = ArrowType.INT  # default
        if obj_tv.struct_fields:
            # Look up field in compile-time layout
            for idx, (fname, ftype) in enumerate(obj_tv.struct_fields):
                if fname == field:
                    val = self.builder.call(self._struct_get, [sptr, ir.Constant(i64, idx)], name=f"f_{field}")
                    field_type = ftype
                    # Convert i64 back to the proper type
                    if field_type == ArrowType.STRING:
                        return TypedValue(self.builder.inttoptr(val, i8_ptr, name=f"{field}_str"), ArrowType.STRING)
                    elif field_type == ArrowType.ARRAY:
                        return TypedValue(self.builder.inttoptr(val, array_ptr), ArrowType.ARRAY)
                    elif field_type == ArrowType.STRUCT:
                        return TypedValue(self.builder.inttoptr(val, i64_ptr), ArrowType.STRUCT,
                                          struct_fields=None)  # nested struct layout unknown
                    elif field_type == ArrowType.FUNC:
                        return TypedValue(self.builder.inttoptr(val, closure_ptr), ArrowType.FUNC)
                    elif field_type == ArrowType.BOOL:
                        return TypedValue(self.builder.trunc(val, i1), ArrowType.BOOL)
                    return TypedValue(val, field_type)

        # Runtime field lookup (struct passed through function ABI, layout unknown)
        name_str = self._global_string(field)
        idx = self.builder.call(self._struct_find, [sptr, name_str], name="fidx")
        val = self.builder.call(self._struct_get, [sptr, idx], name=f"f_{field}")
        return TypedValue(val, ArrowType.INT)

    def _compile_anonymous_arrow(self, params, body):
        fn_name = f"_lambda_{self._func_counter}"; self._func_counter += 1
        fb = body if isinstance(body, list) else [ReturnStmt(body)]
        outer_vars = self._current_scope_var_names()
        free_vars = find_free_variables(params, fb, outer_vars)
        captures = []
        for vn in free_vars:
            if vn in self.variables:
                p, at = self.variables[vn]; captures.append((vn, TypedValue(self.builder.load(p), at)))
            elif vn in self.functions:
                ct = self._make_closure(self.functions[vn].llvm_func)
                captures.append((vn, TypedValue(self.builder.ptrtoint(ct.value, i64), ArrowType.INT)))
        self._declare_function(fn_name, params)
        self._compile_function_body(fn_name, params, fb, captures=captures)
        info = self.functions[fn_name]
        if captures:
            er = self.builder.call(self.malloc, [ir.Constant(i64, len(captures)*8)])
            ed = self.builder.bitcast(er, i64_ptr)
            for idx, (cn, ctv) in enumerate(captures):
                self.builder.store(self._to_i64(ctv), self.builder.gep(ed, [ir.Constant(i64, idx)]))
            return self._make_closure(info.llvm_func, env_value=er)
        return self._make_closure(info.llvm_func)

    def _compile_array_lit(self, elems):
        self._ensure_array_helpers()
        ap = self.builder.call(self._array_new, [ir.Constant(i64, max(len(elems), 4))])
        for e in elems:
            self.builder.call(self._array_push, [ap, self._to_i64(self._compile_expr(e))])
        return TypedValue(ap, ArrowType.ARRAY)

    def _compile_index_expr(self, obj, index):
        self._ensure_array_helpers()
        ot = self._compile_expr(obj); it = self._compile_expr(index)
        if ot.type == ArrowType.STRING:
            buf = self.builder.call(self.malloc, [ir.Constant(i64, 2)])
            self.builder.store(self.builder.load(self.builder.gep(ot.value, [it.value])), buf)
            self.builder.store(ir.Constant(i8, 0), self.builder.gep(buf, [ir.Constant(i64, 1)]))
            return TypedValue(buf, ArrowType.STRING)
        return TypedValue(self.builder.call(self._array_get, [self._to_array_ptr(ot), it.value]), ArrowType.INT)

    def _compile_identifier(self, name):
        if name in self.variables:
            p, at = self.variables[name]
            val = self.builder.load(p, name=name)
            sf = self.scope.struct_layouts.get(name) if at == ArrowType.STRUCT else None
            return TypedValue(val, at, struct_fields=sf)
        if name in self.functions:
            return self._make_closure(self.functions[name].llvm_func)
        raise CompileError(f"Undefined variable '{name}'")

    def _compile_call(self, callee, args):
        if isinstance(callee, Identifier) and callee.name in (
                "len", "push", "pop", "keys", "read_file", "write_file", "input",
                "char_code", "from_char_code", "substring", "char_at", "str_len"):
            return self._compile_builtin(callee.name, args)
        atvs = [self._compile_expr(a) for a in args]; avs = [self._to_i64(t) for t in atvs]
        known_ret = ArrowType.INT
        known_sf = None
        if isinstance(callee, Identifier) and callee.name in self.functions:
            fi = self.functions[callee.name]
            known_ret = fi.return_type
            known_sf = fi.return_struct_fields
        ctv = self._compile_expr(callee)
        fn_i64, env_ptr = self._unpack_closure(ctv.value)
        pt = [i8_ptr] + [self.PARAM_LLVM_TYPE] * len(args)
        fp = self.builder.inttoptr(fn_i64, ir.FunctionType(self.RET_LLVM_TYPE, pt).as_pointer())
        result = self.builder.call(fp, [env_ptr] + avs, name="call")
        if known_ret == ArrowType.ARRAY:
            return TypedValue(self.builder.inttoptr(result, array_ptr), ArrowType.ARRAY)
        if known_ret == ArrowType.STRUCT:
            return TypedValue(self.builder.inttoptr(result, i64_ptr), ArrowType.STRUCT,
                              struct_fields=known_sf)
        if known_ret == ArrowType.STRING:
            return TypedValue(self.builder.inttoptr(result, i8_ptr), ArrowType.STRING)
        if known_ret == ArrowType.FUNC:
            return TypedValue(self.builder.inttoptr(result, closure_ptr), ArrowType.FUNC)
        return TypedValue(result, ArrowType.INT)

    def _compile_builtin(self, name, args):
        if name == "len":
            tv = self._compile_expr(args[0])
            if tv.type == ArrowType.STRING:
                return TypedValue(self.builder.call(self.strlen, [tv.value]), ArrowType.INT)
            self._ensure_array_helpers()
            return TypedValue(self.builder.call(self._array_len, [self._to_array_ptr(tv)]), ArrowType.INT)
        elif name == "str_len":
            # Explicit string length — always uses strlen, safe through ABI
            tv = self._compile_expr(args[0])
            str_ptr = tv.value if tv.type == ArrowType.STRING else self.builder.inttoptr(tv.value, i8_ptr)
            return TypedValue(self.builder.call(self.strlen, [str_ptr]), ArrowType.INT)
        elif name == "push":
            self._ensure_array_helpers()
            at = self._compile_expr(args[0]); vt = self._compile_expr(args[1])
            return TypedValue(self.builder.call(self._array_push, [self._to_array_ptr(at), self._to_i64(vt)]), ArrowType.INT)
        elif name == "pop":
            self._ensure_array_helpers()
            at = self._compile_expr(args[0])
            return TypedValue(self.builder.call(self._array_pop, [self._to_array_ptr(at)]), ArrowType.INT)
        elif name == "keys":
            self._ensure_struct_helpers(); self._ensure_array_helpers()
            st = self._compile_expr(args[0]); sptr = self._to_struct_ptr(st)
            count = self.builder.call(self._struct_count, [sptr])
            ap = self.builder.call(self._array_new, [count])
            ip = self._create_entry_alloca("_kidx", i64); self.builder.store(ir.Constant(i64, 0), ip)
            cc = self.builder.append_basic_block("kc"); kb = self.builder.append_basic_block("kb")
            ke = self.builder.append_basic_block("ke"); self.builder.branch(cc)
            self.builder.position_at_start(cc); idx = self.builder.load(ip)
            self.builder.cbranch(self.builder.icmp_signed('<', idx, count), kb, ke)
            self.builder.position_at_start(kb); idx = self.builder.load(ip)
            np = self.builder.call(self._struct_get_name, [sptr, idx])
            self.builder.call(self._array_push, [ap, self.builder.ptrtoint(np, i64)])
            self.builder.store(self.builder.add(idx, ir.Constant(i64, 1)), ip); self.builder.branch(cc)
            self.builder.position_at_start(ke)
            return TypedValue(ap, ArrowType.ARRAY)

        elif name == "read_file":
            self._ensure_io_helpers()
            path_tv = self._compile_expr(args[0])
            path_str = path_tv.value if path_tv.type == ArrowType.STRING else self.builder.inttoptr(path_tv.value, i8_ptr)
            result = self.builder.call(self._read_file_fn, [path_str], name="file_data")
            return TypedValue(result, ArrowType.STRING)

        elif name == "write_file":
            self._ensure_io_helpers()
            path_tv = self._compile_expr(args[0])
            content_tv = self._compile_expr(args[1])
            path_str = path_tv.value if path_tv.type == ArrowType.STRING else self.builder.inttoptr(path_tv.value, i8_ptr)
            content_str = self._value_to_string(content_tv)
            result = self.builder.call(self._write_file_fn, [path_str, content_str], name="bytes_written")
            return TypedValue(result, ArrowType.INT)

        elif name == "input":
            self._ensure_io_helpers()
            if len(args) >= 1:
                prompt_tv = self._compile_expr(args[0])
                prompt_str = self._value_to_string(prompt_tv)
                self.builder.call(self.printf, [prompt_str])
                self.builder.call(self.fflush, [ir.Constant(i8_ptr, None)])
            result = self.builder.call(self._input_fn, [], name="user_input")
            return TypedValue(result, ArrowType.STRING)

        elif name == "char_code":
            tv = self._compile_expr(args[0])
            str_ptr = tv.value if tv.type == ArrowType.STRING else self.builder.inttoptr(tv.value, i8_ptr)
            # Load the first byte and zero-extend to i64
            ch = self.builder.load(str_ptr, name="ch")
            val = self.builder.zext(ch, i64, name="char_code")
            return TypedValue(val, ArrowType.INT)

        elif name == "from_char_code":
            tv = self._compile_expr(args[0])
            code_val = tv.value if tv.type == ArrowType.INT else self._to_int(tv)
            # Allocate a 2-byte string: [char, null]
            buf = self.builder.call(self.malloc, [ir.Constant(i64, 2)], name="ch_buf")
            ch = self.builder.trunc(code_val, i8, name="ch")
            self.builder.store(ch, buf)
            self.builder.store(ir.Constant(i8, 0), self.builder.gep(buf, [ir.Constant(i64, 1)]))
            return TypedValue(buf, ArrowType.STRING)

        elif name == "substring":
            s_tv = self._compile_expr(args[0])
            start_tv = self._compile_expr(args[1])
            end_tv = self._compile_expr(args[2])
            str_ptr = s_tv.value if s_tv.type == ArrowType.STRING else self.builder.inttoptr(s_tv.value, i8_ptr)
            start_val = start_tv.value if start_tv.type == ArrowType.INT else self._to_int(start_tv)
            end_val = end_tv.value if end_tv.type == ArrowType.INT else self._to_int(end_tv)
            # Length of substring
            sub_len = self.builder.sub(end_val, start_val, name="sub_len")
            buf_size = self.builder.add(sub_len, ir.Constant(i64, 1), name="buf_sz")
            buf = self.builder.call(self.malloc, [buf_size], name="sub_buf")
            # Source pointer: str_ptr + start
            src = self.builder.gep(str_ptr, [start_val], name="src")
            # Copy bytes using a loop
            idx_ptr = self._create_entry_alloca("_si", i64)
            self.builder.store(ir.Constant(i64, 0), idx_ptr)
            cond_bb = self.builder.append_basic_block("sub_c")
            body_bb = self.builder.append_basic_block("sub_b")
            end_bb = self.builder.append_basic_block("sub_e")
            self.builder.branch(cond_bb)
            self.builder.position_at_start(cond_bb)
            idx = self.builder.load(idx_ptr)
            self.builder.cbranch(self.builder.icmp_signed('<', idx, sub_len), body_bb, end_bb)
            self.builder.position_at_start(body_bb)
            idx = self.builder.load(idx_ptr)
            ch = self.builder.load(self.builder.gep(src, [idx]))
            self.builder.store(ch, self.builder.gep(buf, [idx]))
            self.builder.store(self.builder.add(idx, ir.Constant(i64, 1)), idx_ptr)
            self.builder.branch(cond_bb)
            self.builder.position_at_start(end_bb)
            # Null-terminate
            self.builder.store(ir.Constant(i8, 0), self.builder.gep(buf, [sub_len]))
            return TypedValue(buf, ArrowType.STRING)

        elif name == "char_at":
            s_tv = self._compile_expr(args[0])
            idx_tv = self._compile_expr(args[1])
            str_ptr = s_tv.value if s_tv.type == ArrowType.STRING else self.builder.inttoptr(s_tv.value, i8_ptr)
            idx_val = idx_tv.value if idx_tv.type == ArrowType.INT else self._to_int(idx_tv)
            buf = self.builder.call(self.malloc, [ir.Constant(i64, 2)], name="ca_buf")
            ch = self.builder.load(self.builder.gep(str_ptr, [idx_val]), name="ca_ch")
            self.builder.store(ch, buf)
            self.builder.store(ir.Constant(i8, 0), self.builder.gep(buf, [ir.Constant(i64, 1)]))
            return TypedValue(buf, ArrowType.STRING)

        raise CompileError(f"Unknown builtin: {name}")

    # ─── I/O helpers (lazy) ──────────────
    def _ensure_io_helpers(self):
        if self._io_helpers_built: return
        self._io_helpers_built = True
        # Declare C file I/O functions
        self._fopen = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i8_ptr]), name="fopen")
        self._fclose = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr]), name="fclose")
        self._fread = ir.Function(self.module, ir.FunctionType(i64, [i8_ptr, i64, i64, i8_ptr]), name="fread")
        self._fwrite = ir.Function(self.module, ir.FunctionType(i64, [i8_ptr, i64, i64, i8_ptr]), name="fwrite")
        self._fseek = ir.Function(self.module, ir.FunctionType(i32, [i8_ptr, i64, i32]), name="fseek")
        self._ftell = ir.Function(self.module, ir.FunctionType(i64, [i8_ptr]), name="ftell")
        self._fgets = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr, i32, i8_ptr]), name="fgets")
        self._free = ir.Function(self.module, ir.FunctionType(void, [i8_ptr]), name="free")
        # Get stdin — we'll use __acrt_iob_func on Windows, stdin global on Unix
        # For portability, build a helper that reads via fgets
        self._build_io_helpers()

    def _build_io_helpers(self):
        """Build read_file, write_file, and input as LLVM functions."""

        # ── arrow_read_file(path) -> i8* ──
        fn = ir.Function(self.module, ir.FunctionType(i8_ptr, [i8_ptr]), name="arrow_read_file")
        fn.args[0].name = "path"
        entry = fn.append_basic_block("entry")
        ok_bb = fn.append_basic_block("ok")
        fail_bb = fn.append_basic_block("fail")
        b = ir.IRBuilder(entry)
        rb = self._make_global_string_raw("rb")
        fp = b.call(self._fopen, [fn.args[0], rb], name="fp")
        is_null = b.icmp_signed('==', b.ptrtoint(fp, i64), ir.Constant(i64, 0))
        b.cbranch(is_null, fail_bb, ok_bb)

        b = ir.IRBuilder(fail_bb)
        empty = b.call(self.malloc, [ir.Constant(i64, 1)], name="empty")
        b.store(ir.Constant(i8, 0), empty)
        b.ret(empty)

        b = ir.IRBuilder(ok_bb)
        # Seek to end to get size
        b.call(self._fseek, [fp, ir.Constant(i64, 0), ir.Constant(i32, 2)])  # SEEK_END=2
        size = b.call(self._ftell, [fp], name="size")
        b.call(self._fseek, [fp, ir.Constant(i64, 0), ir.Constant(i32, 0)])  # SEEK_SET=0
        # Allocate buffer
        buf_size = b.add(size, ir.Constant(i64, 1))
        buf = b.call(self.malloc, [buf_size], name="buf")
        # Read file
        b.call(self._fread, [buf, ir.Constant(i64, 1), size, fp])
        # Null-terminate
        end_ptr = b.gep(buf, [size])
        b.store(ir.Constant(i8, 0), end_ptr)
        b.call(self._fclose, [fp])
        b.ret(buf)
        self._read_file_fn = fn

        # ── arrow_write_file(path, content) -> i64 bytes written ──
        fn = ir.Function(self.module, ir.FunctionType(i64, [i8_ptr, i8_ptr]), name="arrow_write_file")
        fn.args[0].name = "path"; fn.args[1].name = "content"
        entry = fn.append_basic_block("entry")
        ok_bb = fn.append_basic_block("ok")
        fail_bb = fn.append_basic_block("fail")
        b = ir.IRBuilder(entry)
        wb = self._make_global_string_raw("wb")
        fp = b.call(self._fopen, [fn.args[0], wb], name="fp")
        is_null = b.icmp_signed('==', b.ptrtoint(fp, i64), ir.Constant(i64, 0))
        b.cbranch(is_null, fail_bb, ok_bb)

        b = ir.IRBuilder(fail_bb)
        b.ret(ir.Constant(i64, 0))

        b = ir.IRBuilder(ok_bb)
        slen = b.call(self.strlen, [fn.args[1]], name="slen")
        written = b.call(self._fwrite, [fn.args[1], ir.Constant(i64, 1), slen, fp], name="written")
        b.call(self._fclose, [fp])
        b.ret(written)
        self._write_file_fn = fn

        # ── arrow_input() -> i8* (reads a line from stdin) ──
        fn = ir.Function(self.module, ir.FunctionType(i8_ptr, []), name="arrow_input")
        entry = fn.append_basic_block("entry")
        ok_bb = fn.append_basic_block("ok")
        b = ir.IRBuilder(entry)
        buf = b.call(self.malloc, [ir.Constant(i64, 4096)], name="buf")
        # Get stdin: use __acrt_iob_func(0) on Windows, or declare stdin
        if self._is_windows:
            iob_func = ir.Function(self.module, ir.FunctionType(i8_ptr, [i32]), name="__acrt_iob_func")
            stdin_fp = b.call(iob_func, [ir.Constant(i32, 0)], name="stdin")
        else:
            stdin_global = ir.GlobalVariable(self.module, i8_ptr, name="stdin")
            stdin_global.linkage = "external"
            stdin_fp = b.load(stdin_global, name="stdin")
        result = b.call(self._fgets, [buf, ir.Constant(i32, 4096), stdin_fp], name="result")
        # Strip trailing newline
        slen = b.call(self.strlen, [buf], name="slen")
        has_content = b.icmp_signed('>', slen, ir.Constant(i64, 0))
        b.cbranch(has_content, ok_bb, ok_bb)  # always go to ok
        b = ir.IRBuilder(ok_bb)
        slen = b.call(self.strlen, [buf])
        last_idx = b.sub(slen, ir.Constant(i64, 1))
        last_ptr = b.gep(buf, [last_idx])
        last_char = b.load(last_ptr)
        is_newline = b.icmp_signed('==', last_char, ir.Constant(i8, 10))  # '\n' = 10
        # If last char is newline, replace with null
        null_val = b.select(is_newline, ir.Constant(i8, 0), last_char)
        b.store(null_val, last_ptr)
        b.ret(buf)
        self._input_fn = fn

    def _make_global_string_raw(self, text):
        """Create a global string without using self.builder (for use in helper functions)."""
        encoded = (text + "\0").encode("utf-8")
        st = ir.ArrayType(i8, len(encoded))
        name = f".str.{self._str_counter}"; self._str_counter += 1
        gv = ir.GlobalVariable(self.module, st, name=name)
        gv.linkage = "private"; gv.global_constant = True
        gv.initializer = ir.Constant(st, bytearray(encoded))
        # Return a ConstantExpr GEP — no builder needed
        zero = ir.Constant(i64, 0)
        return gv.gep([zero, zero])

    def _compile_unary(self, op, operand):
        tv = self._compile_expr(operand)
        if op == '-':
            if tv.type == ArrowType.INT: return TypedValue(self.builder.neg(tv.value), ArrowType.INT)
            if tv.type == ArrowType.FLOAT: return TypedValue(self.builder.fneg(tv.value), ArrowType.FLOAT)
        if op == '!': return TypedValue(self.builder.not_(self._to_bool(tv)), ArrowType.BOOL)
        raise CompileError(f"Unknown unary: {op}")

    def _compile_binop(self, op, l, r):
        if op == '+':
            lt = self._compile_expr(l); rt = self._compile_expr(r)
            if lt.type == ArrowType.STRING or rt.type == ArrowType.STRING:
                return self._compile_string_concat(lt, rt)
            return self._compile_arith(op, lt, rt)
        lt = self._compile_expr(l); rt = self._compile_expr(r)
        if op in ('+','-','*','/','%'): return self._compile_arith(op, lt, rt)
        if op in ('<','>','<=','>=','=','!='): return self._compile_cmp(op, lt, rt)
        if op == '&&': return TypedValue(self.builder.and_(self._to_bool(lt), self._to_bool(rt)), ArrowType.BOOL)
        if op == '||': return TypedValue(self.builder.or_(self._to_bool(lt), self._to_bool(rt)), ArrowType.BOOL)
        raise CompileError(f"Unknown op: {op}")

    def _compile_arith(self, op, lt, rt):
        if lt.type == ArrowType.FLOAT or rt.type == ArrowType.FLOAT:
            lv, rv = self._to_float(lt), self._to_float(rt)
            ops = {'+':'fadd','-':'fsub','*':'fmul','/':'fdiv','%':'frem'}
            return TypedValue(getattr(self.builder, ops[op])(lv, rv), ArrowType.FLOAT)
        ops = {'+':'add','-':'sub','*':'mul','/':'sdiv','%':'srem'}
        return TypedValue(getattr(self.builder, ops[op])(lt.value, rt.value), ArrowType.INT)

    def _compile_cmp(self, op, lt, rt):
        cm = {'<':'<','>':'>','<=':'<=','>=':'>=','=':'==','!=':'!='}
        if lt.type == ArrowType.FLOAT or rt.type == ArrowType.FLOAT:
            return TypedValue(self.builder.fcmp_ordered(cm[op], self._to_float(lt), self._to_float(rt)), ArrowType.BOOL)
        if lt.type == ArrowType.STRING or rt.type == ArrowType.STRING:
            # String comparison via strcmp
            ls = lt.value if lt.type == ArrowType.STRING else self.builder.inttoptr(lt.value, i8_ptr)
            rs = rt.value if rt.type == ArrowType.STRING else self.builder.inttoptr(rt.value, i8_ptr)
            cmp_result = self.builder.call(self.strcmp, [ls, rs], name="scmp")
            return TypedValue(self.builder.icmp_signed(cm[op], cmp_result, ir.Constant(i32, 0)), ArrowType.BOOL)
        return TypedValue(self.builder.icmp_signed(cm[op], self._to_int(lt), self._to_int(rt)), ArrowType.BOOL)

    def _compile_string_concat(self, lt, rt):
        ls, rs = self._value_to_string(lt), self._value_to_string(rt)
        ll = self.builder.call(self.strlen, [ls]); rl = self.builder.call(self.strlen, [rs])
        buf = self.builder.call(self.malloc, [self.builder.add(self.builder.add(ll, rl), ir.Constant(i64, 1))])
        self.builder.call(self.strcpy, [buf, ls]); self.builder.call(self.strcat, [buf, rs])
        return TypedValue(buf, ArrowType.STRING)

    # ─── Type conversions ────────────────
    def _to_struct_ptr(self, tv):
        if tv.type == ArrowType.STRUCT: return tv.value
        if tv.type == ArrowType.INT: return self.builder.inttoptr(tv.value, i64_ptr, name="sptr")
        raise CompileError(f"Cannot convert {tv.type} to struct")

    def _to_array_ptr(self, tv):
        if tv.type == ArrowType.ARRAY: return tv.value
        if tv.type == ArrowType.INT: return self.builder.inttoptr(tv.value, array_ptr)
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
        if tv.type == ArrowType.STRUCT: return self.builder.ptrtoint(tv.value, i64)
        raise CompileError(f"Cannot convert {tv.type} to i64")

    def _arrow_type_to_llvm(self, at):
        m = {ArrowType.INT: i64, ArrowType.FLOAT: double, ArrowType.BOOL: i1,
             ArrowType.STRING: i8_ptr, ArrowType.FUNC: closure_ptr,
             ArrowType.ARRAY: array_ptr, ArrowType.STRUCT: i64_ptr}
        if at in m: return m[at]
        raise CompileError(f"Unknown type: {at}")

    def _create_entry_alloca(self, name, lt):
        return self.scope.alloca_builder.alloca(lt, name=name)

# ─── OPT / COMPILE / JIT ────────────────
def optimize_ir(llvm_ir):
    mod = binding.parse_assembly(llvm_ir); mod.verify()
    pm = binding.ModulePassManager(); pmb = binding.PassManagerBuilder()
    pmb.opt_level = 2; pmb.populate_module_pass_manager(pm); pm.run(mod)
    return str(mod)

def compile_to_executable(llvm_ir, output_path):
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir); ir_path = f.name
    try:
        subprocess.run(["clang", ir_path, "-o", output_path, "-O2"],
                       check=True, capture_output=True, text=True)
        print(f"Compiled to: {output_path}")
    except FileNotFoundError: print("Error: 'clang' not found."); sys.exit(1)
    except subprocess.CalledProcessError as e: print(f"Clang error:\n{e.stderr}"); sys.exit(1)
    finally: os.unlink(ir_path)

def jit_execute(llvm_ir):
    import ctypes, ctypes.util, platform
    try: binding.initialize_native_target()
    except Exception: pass
    try: binding.initialize_native_asmprinter()
    except Exception: pass
    system = platform.system()
    def _find(name, libs):
        for lib in libs:
            try: return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
            except (AttributeError, OSError): continue
        return None
    libs = []
    if system == "Windows":
        for dll in ["msvcrt","ucrtbase","api-ms-win-crt-stdio-l1-1-0",
                     "api-ms-win-crt-string-l1-1-0","api-ms-win-crt-heap-l1-1-0"]:
            try: libs.append(ctypes.CDLL(dll))
            except OSError: continue
    elif system == "Darwin": libs.append(ctypes.CDLL("libSystem.B.dylib"))
    else:
        ln = ctypes.util.find_library("c")
        if ln: libs.append(ctypes.CDLL(ln))
    all_c = ["printf","puts","fflush","malloc","realloc","sprintf","strlen","strcpy","strcat","strcmp",
             "fopen","fclose","fread","fwrite","fseek","ftell","fgets","free","__acrt_iob_func"]
    needed = [n for n in all_c if f'@"{n}"' in llvm_ir or f'@{n}(' in llvm_ir]
    c_funcs = {}
    for n in needed:
        a = _find(n, libs)
        if a: c_funcs[n] = a
    mod = binding.parse_assembly(llvm_ir); mod.verify()
    engine = binding.create_mcjit_compiler(mod, binding.Target.from_default_triple().create_target_machine())
    for n, a in c_funcs.items():
        try: engine.add_global_mapping(mod.get_function(n), a)
        except Exception: pass
    engine.finalize_object()
    return ctypes.CFUNCTYPE(ctypes.c_int)(engine.get_function_address("main"))()

def main():
    ap = argparse.ArgumentParser(description="Arrow Lang Compiler")
    ap.add_argument("file"); ap.add_argument("--emit-ir", action="store_true")
    ap.add_argument("--emit-ir-opt", action="store_true")
    ap.add_argument("-o", "--output", metavar="FILE")
    args = ap.parse_args()
    try:
        with open(args.file, encoding="utf-8") as f: source = f.read()
    except FileNotFoundError: print(f"File not found: {args.file}"); sys.exit(1)
    try: tokens = Lexer(source).tokenize(); program = Parser(tokens).parse()
    except (LexerError, ParseError) as e: print(f"Error: {e}"); sys.exit(1)
    try: compiler = Compiler(); llvm_ir = compiler.compile(program)
    except CompileError as e: print(f"Compile Error: {e}"); sys.exit(1)
    if args.emit_ir: print(llvm_ir)
    elif args.emit_ir_opt: print(optimize_ir(llvm_ir))
    elif args.output: compile_to_executable(llvm_ir, args.output)
    else: jit_execute(llvm_ir)

if __name__ == "__main__": main()