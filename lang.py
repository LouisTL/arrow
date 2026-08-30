#!/usr/bin/env python3
"""
Arrow Lang v0.4 — A small programming language.

Features:
    - Variables with <- assignment
    - Arithmetic, comparisons, logical operators
    - If/else, while loops
    - Named functions, arrow functions, closures, recursion
    - Arrays: [1, 2, 3], index access/assign, len/push/pop
    - Structs: {name: "Alice", age: 30}, dot access/assign
"""

import os
import sys
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────
#  TOKENS
# ─────────────────────────────────────────────
class TokenType(Enum):
    NUMBER    = auto()
    STRING    = auto()
    IDENT     = auto()
    BOOL      = auto()

    PLUS      = auto()  # +
    MINUS     = auto()  # -
    STAR      = auto()  # *
    SLASH     = auto()  # /
    PERCENT   = auto()  # %
    ARROW     = auto()  # <-
    FAT_ARROW = auto()  # =>
    MATCH     = auto()  # match keyword
    EQ        = auto()  # =
    NEQ       = auto()  # !=
    LT        = auto()  # <
    GT        = auto()  # >
    LTE       = auto()  # <=
    GTE       = auto()  # >=
    AND       = auto()  # &&
    OR        = auto()  # ||
    PIPE      = auto()  # |
    NOT       = auto()  # !
    DOT       = auto()  # .
    COLON     = auto()  # :

    LPAREN    = auto()  # (
    RPAREN    = auto()  # )
    LBRACE    = auto()  # {
    RBRACE    = auto()  # }
    LBRACKET  = auto()  # [
    RBRACKET  = auto()  # ]
    SEMI      = auto()  # ;
    COMMA     = auto()  # ,

    IF        = auto()
    ELSE      = auto()
    WHILE     = auto()
    FOR       = auto()
    IN        = auto()
    PRINT     = auto()
    FN        = auto()
    RETURN    = auto()
    IMPORT    = auto()
    VAR       = auto()

    EOF       = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


# ─────────────────────────────────────────────
#  LEXER
# ─────────────────────────────────────────────
KEYWORDS = {
    "match": TokenType.MATCH,
    "if": TokenType.IF, "else": TokenType.ELSE, "while": TokenType.WHILE,
    "for": TokenType.FOR, "in": TokenType.IN,
    "print": TokenType.PRINT, "fn": TokenType.FN, "return": TokenType.RETURN,
    "true": TokenType.BOOL, "false": TokenType.BOOL,
    "import": TokenType.IMPORT,
    "var": TokenType.VAR,
}


class LexerError(Exception):
    pass


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def _advance(self):
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _peek(self, offset=0):
        i = self.pos + offset
        return self.source[i] if i < len(self.source) else '\0'

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in ' \t\r\n':
                self._advance()
            elif ch == '/' and self._peek(1) == '/':
                while self.pos < len(self.source) and self._peek() != '\n':
                    self._advance()
            else:
                break

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.source):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                break

            ch = self._peek()
            line, col = self.line, self.col

            if ch.isdigit():
                tokens.append(self._read_number())
            elif ch == '"':
                tokens.append(self._read_string())
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_ident())

            # Two-character operators
            elif ch == '<' and self._peek(1) == '-':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.ARROW, '<-', line, col))
            elif ch == '=' and self._peek(1) == '>':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.FAT_ARROW, '=>', line, col))
            elif ch == '<' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.LTE, '<=', line, col))
            elif ch == '>' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.GTE, '>=', line, col))
            elif ch == '!' and self._peek(1) == '=':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.NEQ, '!=', line, col))
            elif ch == '&' and self._peek(1) == '&':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.AND, '&&', line, col))
            elif ch == '|' and self._peek(1) == '|':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.OR, '||', line, col))
            elif ch == '|':
                self._advance()
                tokens.append(Token(TokenType.PIPE, '|', line, col))

            # Single-character tokens
            elif ch == '+': self._advance(); tokens.append(Token(TokenType.PLUS,     '+', line, col))
            elif ch == '-': self._advance(); tokens.append(Token(TokenType.MINUS,    '-', line, col))
            elif ch == '*': self._advance(); tokens.append(Token(TokenType.STAR,     '*', line, col))
            elif ch == '/': self._advance(); tokens.append(Token(TokenType.SLASH,    '/', line, col))
            elif ch == '%': self._advance(); tokens.append(Token(TokenType.PERCENT,  '%', line, col))
            elif ch == '<': self._advance(); tokens.append(Token(TokenType.LT,       '<', line, col))
            elif ch == '>': self._advance(); tokens.append(Token(TokenType.GT,       '>', line, col))
            elif ch == '!': self._advance(); tokens.append(Token(TokenType.NOT,      '!', line, col))
            elif ch == '=': self._advance(); tokens.append(Token(TokenType.EQ,       '=', line, col))
            elif ch == '.': self._advance(); tokens.append(Token(TokenType.DOT,      '.', line, col))
            elif ch == ':': self._advance(); tokens.append(Token(TokenType.COLON,    ':', line, col))
            elif ch == '(': self._advance(); tokens.append(Token(TokenType.LPAREN,   '(', line, col))
            elif ch == ')': self._advance(); tokens.append(Token(TokenType.RPAREN,   ')', line, col))
            elif ch == '{': self._advance(); tokens.append(Token(TokenType.LBRACE,   '{', line, col))
            elif ch == '}': self._advance(); tokens.append(Token(TokenType.RBRACE,   '}', line, col))
            elif ch == '[': self._advance(); tokens.append(Token(TokenType.LBRACKET, '[', line, col))
            elif ch == ']': self._advance(); tokens.append(Token(TokenType.RBRACKET, ']', line, col))
            elif ch == ';': self._advance(); tokens.append(Token(TokenType.SEMI,     ';', line, col))
            elif ch == ',': self._advance(); tokens.append(Token(TokenType.COMMA,    ',', line, col))

            else:
                raise LexerError(f"Unexpected character '{ch}' at line {line}, col {col}")

        tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return tokens

    def _read_number(self) -> Token:
        line, col = self.line, self.col
        start = self.pos
        while self.pos < len(self.source) and (self._peek().isdigit() or self._peek() == '.'):
            self._advance()
        text = self.source[start:self.pos]
        value = float(text) if '.' in text else int(text)
        return Token(TokenType.NUMBER, value, line, col)

    def _read_string(self) -> Token:
        line, col = self.line, self.col
        self._advance()
        chars = []
        while self.pos < len(self.source) and self._peek() != '"':
            ch = self._advance()
            if ch == '\\':
                nxt = self._advance()
                escape = {'n': '\n', 't': '\t', '\\': '\\', '"': '"'}
                chars.append(escape.get(nxt, nxt))
            else:
                chars.append(ch)
        if self.pos >= len(self.source):
            raise LexerError(f"Unterminated string at line {line}, col {col}")
        self._advance()
        return Token(TokenType.STRING, ''.join(chars), line, col)

    def _read_ident(self) -> Token:
        line, col = self.line, self.col
        start = self.pos
        while self.pos < len(self.source) and (self._peek().isalnum() or self._peek() == '_'):
            self._advance()
        text = self.source[start:self.pos]
        if text in KEYWORDS:
            tt = KEYWORDS[text]
            value = (text == "true") if tt == TokenType.BOOL else text
            return Token(tt, value, line, col)
        return Token(TokenType.IDENT, text, line, col)


# ─────────────────────────────────────────────
#  AST NODES
# ─────────────────────────────────────────────
@dataclass
class NumberLit:
    value: float | int

@dataclass
class StringLit:
    value: str

@dataclass
class BoolLit:
    value: bool

@dataclass
class NoneLit:
    # The `none` literal expression. Evaluates to
    # Python None; classified as kind "none" everywhere.
    pass

@dataclass
class Identifier:
    name: str

@dataclass
class ArrayLit:
    elements: list

@dataclass
class StructLit:
    """Struct literal: {key: value, key: value, ...}"""
    fields: list  # list of (name_str, expr) tuples

@dataclass
class IndexExpr:
    obj: Any
    index: Any

@dataclass
class DotExpr:
    """Field access: expr.field_name"""
    obj: Any
    field: str

@dataclass
class IndexAssign:
    obj: Any
    index: Any
    value: Any

@dataclass
class DotAssign:
    """Field assignment: expr.field <- value;"""
    obj: Any
    field: str
    value: Any

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    op: str
    operand: Any

@dataclass
class Assignment:
    name: str
    expr: Any
    is_decl: bool = False  # True when introduced via `var`
    line: int = 0
    col: int = 0
    type_kind: str = ""  # annotation kind head ("" = unannotated)

@dataclass
class PrintStmt:
    expr: Any

@dataclass
class MatchArm:
    ptype_kind: str
    name: str | None
    body: list
    lit_kind: str | None = None
    lit_val: Any = None
    pfields: list | None = None

@dataclass
class MatchStmt:
    scrutinee: Any
    arms: list

@dataclass
class MatchExprArm:
    ptype_kind: str
    name: str | None
    value: Any
    lit_kind: str | None = None
    lit_val: Any = None
    pfields: list | None = None

@dataclass
class MatchExpr:
    scrutinee: Any
    arms: list

@dataclass
class IfStmt:
    condition: Any
    then_body: list
    else_body: list | None

@dataclass
class WhileStmt:
    condition: Any
    body: list

@dataclass
class ForInStmt:
    """For-in loop: for (var in iterable) { body }"""
    var_name: str
    iterable: Any
    body: list

@dataclass
class Block:
    statements: list

@dataclass
class Program:
    statements: list

@dataclass
class FnDecl:
    name: str
    params: list[str]
    body: list
    param_kinds: list = None   # per-param annotation kind heads
    ret_kind: str = ""     # return annotation kind head

@dataclass
class ArrowFn:
    params: list[str]
    body: Any
    param_kinds: list = None   # per-param annotation kind heads
    ret_kind: str = ""     # return annotation kind head

@dataclass
class ReturnStmt:
    expr: Any

@dataclass
class CallExpr:
    callee: Any
    args: list

@dataclass
class ImportStmt:
    """`import "path";` — namespace name defaults to basename of path.
    `items` lists destructured (name, bound_as) pairs from an optional
    `{a, b as c}` clause; empty when the clause is absent."""
    path: str
    name: str
    items: list = None
    line: int = 0
    col: int = 0


@dataclass
class TypeDecl:
    """`type Name <- Type;` — structural alias. rhs_kind is the parsed
    kind head (a builtin, or another type name pending resolution);
    rhs_pfields carries struct field names for match dispatch."""
    name: str
    rhs_kind: str
    rhs_pfields: Any
    line: int = 0
    col: int = 0


# ─────────────────────────────────────────────
#  PARSER
# ─────────────────────────────────────────────
class ParseError(Exception):
    pass


class Parser:
    # Sync points for statement-level error recovery — mirror compiler.arrow's
    # `sync_to_stmt_boundary`. SEMI ends a statement; RBRACE ends a block;
    # the top-level keywords start a fresh statement.
    _SYNC_STARTERS = frozenset({
        TokenType.FN, TokenType.IF, TokenType.WHILE, TokenType.FOR,
        TokenType.RETURN, TokenType.PRINT, TokenType.IMPORT,
    })

    def __init__(self, tokens: list[Token], src_file: str = "<unknown>"):
        self.tokens = tokens
        self.pos = 0
        # Accumulated parse errors for batch reporting. `parse()` returns
        # the program AND this list, so callers can decide whether to halt.
        self.errors: list[str] = []
        self.src_file = src_file
        # See compiler.arrow's panic_mode: once one error fires in a
        # statement we suppress follow-on errors until the recovery resyncs.
        self._panic = False

    def _record_error(self, msg: str, line: int, col: int):
        if self._panic:
            return
        self.errors.append(f"{self.src_file}:{line}:{col}: parse error: {msg}")
        self._panic = True

    def _sync(self):
        """Advance to a statement boundary so we can resume cleanly."""
        while self._current().type not in (
            TokenType.EOF, TokenType.SEMI, TokenType.RBRACE
        ) and self._current().type not in self._SYNC_STARTERS:
            self.pos += 1
        if self._current().type == TokenType.SEMI:
            self.pos += 1

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, tt: TokenType) -> Token:
        tok = self._current()
        if tok.type != tt:
            raise ParseError(
                f"expected {tt.name}, got {tok.type.name} ({tok.value!r}) "
                f"at line {tok.line}, col {tok.col}")
        self.pos += 1
        return tok

    def _match(self, *types: TokenType) -> Token | None:
        if self._current().type in types:
            tok = self._current()
            self.pos += 1
            return tok
        return None

    def _peek_type(self, offset=0) -> TokenType:
        i = self.pos + offset
        return self.tokens[i].type if i < len(self.tokens) else TokenType.EOF

    # ── Grammar ──────────────────────────────

    def parse(self) -> Program:
        stmts = []
        while self._current().type != TokenType.EOF:
            # A stray close-brace at top level is recovery residue: the
            # enclosing block was abandoned mid-parse because of an earlier
            # error. Skipping silently avoids cascading a confusing
            # "unexpected RBRACE" on top of the real diagnostic.
            if self._current().type == TokenType.RBRACE:
                self.pos += 1
                continue
            start_pos = self.pos
            self._panic = False
            try:
                cur = self._current()
                if (cur.type == TokenType.IDENT and cur.value == "type"
                        and self._peek_type(1) == TokenType.IDENT
                        and self._peek_type(2) == TokenType.ARROW):
                    stmts.append(self._type_decl())
                else:
                    stmts.append(self._statement())
            except ParseError as e:
                tok = self._current()
                msg = str(e)
                if " at line " in msg:
                    msg = msg.rsplit(" at line ", 1)[0]
                self._record_error(msg, tok.line, tok.col)
                if self.pos == start_pos:
                    self.pos += 1
                self._sync()
        return Program(stmts)

    def _statement(self):
        tok = self._current()

        if tok.type == TokenType.FN:
            return self._fn_decl()
        if tok.type == TokenType.RETURN:
            return self._return_stmt()
        if tok.type == TokenType.IF:
            return self._if_stmt()
        if tok.type == TokenType.WHILE:
            return self._while_stmt()
        if tok.type == TokenType.MATCH:
            return self._match_stmt()
        if tok.type == TokenType.FOR:
            return self._for_in_stmt()
        if tok.type == TokenType.PRINT:
            return self._print_stmt()
        if tok.type == TokenType.IMPORT:
            return self._import_stmt()
        if tok.type == TokenType.VAR:
            return self._var_decl()
        if tok.type == TokenType.LBRACE:
            # Distinguish block from struct literal used as expression stmt
            if self._is_struct_literal():
                expr = self._expression()
                self._eat(TokenType.SEMI)
                return expr
            return self._block()

        # Assignment: ident <- expr;
        if tok.type == TokenType.IDENT:
            if self._peek_type(1) == TokenType.ARROW:
                return self._assignment()
            # Typed assignment: ident : type <- expr;  (type parsed and ignored)
            # Type can start with IDENT (int, str, ...), LBRACKET ([int]), or LBRACE ({x:int})
            if (self._peek_type(1) == TokenType.COLON and
                    self._peek_type(2) in (TokenType.IDENT, TokenType.LBRACKET, TokenType.LBRACE)):
                return self._assignment(typed=True)
            # index/dot assignment: ident[...] <- ... OR ident.field <- ...
            if self._peek_type(1) in (TokenType.LBRACKET, TokenType.DOT):
                return self._try_postfix_assignment()

        # Expression statement
        expr = self._expression()
        self._eat(TokenType.SEMI)
        return expr

    def _is_struct_literal(self) -> bool:
        """Look ahead to distinguish { key: val } from { stmts }."""
        # { } is an empty struct
        if self._peek_type(1) == TokenType.RBRACE:
            return True
        # { IDENT : ... } is a struct
        if (self._peek_type(1) == TokenType.IDENT and
                self._peek_type(2) == TokenType.COLON):
            return True
        # { STRING : ... } is also a struct
        if (self._peek_type(1) == TokenType.STRING and
                self._peek_type(2) == TokenType.COLON):
            return True
        return False

    def _try_postfix_assignment(self):
        """Parse: ident.field <- val; or ident[idx] <- val; or expr stmt."""
        expr = self._expression()

        if self._current().type == TokenType.ARROW:
            self._eat(TokenType.ARROW)
            value = self._expression()
            self._eat(TokenType.SEMI)
            if isinstance(expr, IndexExpr):
                return IndexAssign(expr.obj, expr.index, value)
            elif isinstance(expr, DotExpr):
                return DotAssign(expr.obj, expr.field, value)
            else:
                raise ParseError("Invalid assignment target")

        self._eat(TokenType.SEMI)
        return expr

    def _block(self) -> Block:
        self._eat(TokenType.LBRACE)
        stmts = []
        while self._current().type != TokenType.RBRACE:
            stmts.append(self._statement())
        self._eat(TokenType.RBRACE)
        return Block(stmts)

    def _skip_type_ann(self):
        """Parse and discard a type annotation. Supports: int, str, [int], [[str]], {x: int, y: int}, ..."""
        if self._current().type == TokenType.LBRACKET:
            self._eat(TokenType.LBRACKET)
            self._skip_type_ann()
            self._eat(TokenType.RBRACKET)
        elif self._current().type == TokenType.LBRACE:
            # Struct type: {name: type, name: type, ...}
            self._eat(TokenType.LBRACE)
            if self._current().type != TokenType.RBRACE:
                self._eat(TokenType.IDENT)        # field name
                self._eat(TokenType.COLON)
                self._skip_type_ann()             # field type
                while self._match(TokenType.COMMA):
                    if self._current().type == TokenType.RBRACE:
                        break  # trailing comma
                    self._eat(TokenType.IDENT)
                    self._eat(TokenType.COLON)
                    self._skip_type_ann()
            self._eat(TokenType.RBRACE)
        else:
            self._eat(TokenType.IDENT)
            if self._match(TokenType.DOT):
                self._eat(TokenType.IDENT)
        while self._match(TokenType.PIPE):
            self._skip_type_ann()

    def _type_ann_kind(self) -> str:
        """Parse a type annotation (consuming exactly what _skip_type_ann
        would) and return its kind head for the runtime kind checks: int / float /
        bool / str / array / struct / any — or "" for anything the
        kind-level check does not cover. Type names (bare or one-dot
        qualified) pass through unresolved for the whole-program pass."""
        kind, _ = self._parse_type_kind()
        if self._current().type == TokenType.PIPE:
            members = [kind]
            while self._match(TokenType.PIPE):
                mk, _ = self._parse_type_kind()
                members.append(mk)
            return "union_none" if "none" in members else "union"
        if kind in ("int", "float", "bool", "str", "array", "struct", "any"):
            return kind
        if kind in ("none", "fn"):
            return ""
        return kind

    def _assignment(self, typed: bool = False) -> Assignment:
        ident_tok = self._eat(TokenType.IDENT)
        name = ident_tok.value
        tkind = ""
        if typed:
            # ': type' on a reassignment — kind head retained for the runtime check.
            self._eat(TokenType.COLON)
            tkind = self._type_ann_kind()
        self._eat(TokenType.ARROW)
        expr = self._expression()
        self._eat(TokenType.SEMI)
        return Assignment(name, expr, is_decl=False, line=ident_tok.line, col=ident_tok.col,
                          type_kind=tkind)

    def _var_decl(self) -> Assignment:
        var_tok = self._eat(TokenType.VAR)
        ident_tok = self._eat(TokenType.IDENT)
        name = ident_tok.value
        # Optional type annotation: var x: int <- expr; the kind head is
        # retained for the runtime kind check at this declaration edge.
        tkind = ""
        if self._current().type == TokenType.COLON:
            self._eat(TokenType.COLON)
            tkind = self._type_ann_kind()
        self._eat(TokenType.ARROW)
        expr = self._expression()
        self._eat(TokenType.SEMI)
        # Track at the `var` keyword position so error messages point there.
        return Assignment(name, expr, is_decl=True, line=var_tok.line, col=var_tok.col,
                          type_kind=tkind)

    def _print_stmt(self) -> PrintStmt:
        self._eat(TokenType.PRINT)
        self._eat(TokenType.LPAREN)
        expr = self._expression()
        self._eat(TokenType.RPAREN)
        self._eat(TokenType.SEMI)
        return PrintStmt(expr)

    def _import_stmt(self) -> ImportStmt:
        self._eat(TokenType.IMPORT)
        tok = self._current()
        if tok.type != TokenType.STRING:
            raise ParseError(f"import expects a string path at line {tok.line}, col {tok.col}, got {tok.type.name}")
        path = tok.value
        self._eat(TokenType.STRING)
        # Default namespace = basename of path, strip optional .arrow.
        base = path
        if base.endswith(".arrow"):
            base = base[:-len(".arrow")]
        for sep in ("/", "\\"):
            if sep in base:
                base = base.rsplit(sep, 1)[1]
        # Optional `as <ident>` override. `as` is not a reserved keyword;
        # we recognize it by peeking at the next IDENT token's value so it
        # remains usable as a regular variable name elsewhere.
        nxt = self._current()
        if nxt.type == TokenType.IDENT and nxt.value == "as":
            self._eat(TokenType.IDENT)
            alias_tok = self._current()
            if alias_tok.type != TokenType.IDENT:
                raise ParseError(f"`as` expects an identifier at line {alias_tok.line}, col {alias_tok.col}, got {alias_tok.type.name}")
            base = alias_tok.value
            self._eat(TokenType.IDENT)
        items = []
        if self._current().type == TokenType.LBRACE:
            self._eat(TokenType.LBRACE)
            items.append(self._import_item())
            while self._match(TokenType.COMMA):
                if self._current().type == TokenType.RBRACE:
                    break  # trailing comma
                items.append(self._import_item())
            self._eat(TokenType.RBRACE)
        self._eat(TokenType.SEMI)
        return ImportStmt(path=path, name=base, items=items,
                          line=tok.line, col=tok.col)

    def _import_item(self) -> tuple:
        """One `name` or `name as bound` entry of an import list."""
        itok = self._current()
        if itok.type != TokenType.IDENT:
            raise ParseError(f"import list expects an identifier at line {itok.line}, col {itok.col}, got {itok.type.name}")
        orig = self._eat(TokenType.IDENT).value
        bound = orig
        nxt = self._current()
        if nxt.type == TokenType.IDENT and nxt.value == "as":
            self._eat(TokenType.IDENT)
            btok = self._current()
            if btok.type != TokenType.IDENT:
                raise ParseError(f"`as` expects an identifier at line {btok.line}, col {btok.col}, got {btok.type.name}")
            bound = self._eat(TokenType.IDENT).value
        return (orig, bound)

    def _return_stmt(self) -> ReturnStmt:
        self._eat(TokenType.RETURN)
        if self._current().type == TokenType.SEMI:
            self._eat(TokenType.SEMI)
            return ReturnStmt(None)
        expr = self._expression()
        self._eat(TokenType.SEMI)
        return ReturnStmt(expr)

    def _fn_decl(self) -> FnDecl:
        self._eat(TokenType.FN)
        name = self._eat(TokenType.IDENT).value
        params, pkinds = self._param_list()
        # Optional return type annotation — kind head retained for the return check.
        rkind = ""
        if self._current().type == TokenType.COLON:
            self._eat(TokenType.COLON)
            rkind = self._type_ann_kind()
        body = self._block()
        return FnDecl(name, params, body.statements,
                      param_kinds=pkinds, ret_kind=rkind)

    def _param_list(self) -> tuple[list[str], list[str]]:
        self._eat(TokenType.LPAREN)
        params = []
        kinds = []
        if self._current().type != TokenType.RPAREN:
            params.append(self._eat(TokenType.IDENT).value)
            # Optional per-param type annotation: fn f(x: type, y: type) —
            # the kind head is retained for the param-bind checks.
            if self._current().type == TokenType.COLON:
                self._eat(TokenType.COLON)
                kinds.append(self._type_ann_kind())
            else:
                kinds.append("")
            while self._match(TokenType.COMMA):
                params.append(self._eat(TokenType.IDENT).value)
                if self._current().type == TokenType.COLON:
                    self._eat(TokenType.COLON)
                    kinds.append(self._type_ann_kind())
                else:
                    kinds.append("")
        self._eat(TokenType.RPAREN)
        return params, kinds

    def _type_decl(self) -> TypeDecl:
        """type Name <- Type; declare a structural alias. Runtime-typed, so
        only (kind, pfields) is needed, for match-arm dispatch; names on the
        right-hand side stay unresolved until the whole-program pass."""
        tok = self._current()
        self._eat(TokenType.IDENT)            # 'type'
        name = self._eat(TokenType.IDENT).value
        self._eat(TokenType.ARROW)
        kind, pfields = self._parse_type_kind()
        if self._current().type == TokenType.PIPE:
            while self._match(TokenType.PIPE):
                self._parse_type_kind()
            kind, pfields = "union", None
        self._eat(TokenType.SEMI)
        return TypeDecl(name=name, rhs_kind=kind, rhs_pfields=pfields,
                        line=tok.line, col=tok.col)

    def _parse_type_kind(self):
        """Parse a single (non-union) type for a match arm; return
        (kind, pfields). pfields is the list of field names for a struct
        pattern (used for structural arm dispatch), else None."""
        t = self._current().type
        if t == TokenType.LBRACKET:
            self._eat(TokenType.LBRACKET)
            self._skip_type_ann()
            self._eat(TokenType.RBRACKET)
            return "array", None
        if t == TokenType.LBRACE:
            self._eat(TokenType.LBRACE)
            fnames = []
            if self._current().type != TokenType.RBRACE:
                fnames.append(self._eat(TokenType.IDENT).value); self._eat(TokenType.COLON); self._skip_type_ann()
                while self._match(TokenType.COMMA):
                    if self._current().type == TokenType.RBRACE: break
                    fnames.append(self._eat(TokenType.IDENT).value); self._eat(TokenType.COLON); self._skip_type_ann()
            self._eat(TokenType.RBRACE)
            return "struct", fnames
        name = self._eat(TokenType.IDENT).value
        if self._current().type == TokenType.DOT:
            self._eat(TokenType.DOT)
            name = name + "." + self._eat(TokenType.IDENT).value
        return name, None

    def _parse_arm_pattern(self):
        """Parse one arm pattern (shared by statement and expression match).
        Returns (ptype_kind, name, lit_kind, lit_val, pfields); does not consume `=>`."""
        name = None
        lit_kind = None
        lit_val = None
        pfields = None
        cur = self._current()
        nxt_tok = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
        nxt = nxt_tok.type if nxt_tok else None
        if cur.type == TokenType.IDENT and cur.value == "_" and nxt == TokenType.FAT_ARROW:
            self._eat(TokenType.IDENT)  # consume the _
            ptype_kind = "_"
        elif cur.type == TokenType.NUMBER and isinstance(cur.value, int) and not isinstance(cur.value, bool):
            ptype_kind = "int"; lit_kind = "int"
            lit_val = self._eat(TokenType.NUMBER).value
        elif cur.type == TokenType.STRING:
            ptype_kind = "str"; lit_kind = "str"
            lit_val = self._eat(TokenType.STRING).value
        elif cur.type == TokenType.BOOL:
            ptype_kind = "bool"; lit_kind = "bool"
            lit_val = self._eat(TokenType.BOOL).value
        elif cur.type == TokenType.MINUS and nxt == TokenType.NUMBER and isinstance(nxt_tok.value, int) and not isinstance(nxt_tok.value, bool):
            self._eat(TokenType.MINUS)
            ptype_kind = "int"; lit_kind = "int"
            lit_val = -self._eat(TokenType.NUMBER).value
        else:
            if cur.type == TokenType.IDENT and nxt == TokenType.COLON:
                name = self._eat(TokenType.IDENT).value
                self._eat(TokenType.COLON)
            ptype_kind, pfields = self._parse_type_kind()
        return ptype_kind, name, lit_kind, lit_val, pfields

    def _match_stmt(self) -> MatchStmt:
        self._eat(TokenType.MATCH)
        self._eat(TokenType.LPAREN)
        scrutinee = self._expression()
        self._eat(TokenType.RPAREN)
        self._eat(TokenType.LBRACE)
        arms = []
        while self._current().type != TokenType.RBRACE:
            ptype_kind, name, lit_kind, lit_val, pfields = self._parse_arm_pattern()
            self._eat(TokenType.FAT_ARROW)
            body = self._block()
            arms.append(MatchArm(ptype_kind, name, body.statements, lit_kind, lit_val, pfields))
            self._match(TokenType.COMMA)
        self._eat(TokenType.RBRACE)
        return MatchStmt(scrutinee, arms)

    def _match_expr(self) -> MatchExpr:
        self._eat(TokenType.MATCH)
        self._eat(TokenType.LPAREN)
        scrutinee = self._expression()
        self._eat(TokenType.RPAREN)
        self._eat(TokenType.LBRACE)
        arms = []
        while self._current().type != TokenType.RBRACE:
            ptype_kind, name, lit_kind, lit_val, pfields = self._parse_arm_pattern()
            self._eat(TokenType.FAT_ARROW)
            value = self._expression()
            arms.append(MatchExprArm(ptype_kind, name, value, lit_kind, lit_val, pfields))
            if self._current().type != TokenType.RBRACE:
                self._eat(TokenType.COMMA)
        self._eat(TokenType.RBRACE)
        return MatchExpr(scrutinee, arms)

    def _if_stmt(self) -> IfStmt:
        self._eat(TokenType.IF)
        self._eat(TokenType.LPAREN)
        cond = self._expression()
        self._eat(TokenType.RPAREN)
        then_body = self._block()
        else_body = None
        if self._match(TokenType.ELSE):
            if self._current().type == TokenType.IF:
                else_body = Block([self._if_stmt()])
            else:
                else_body = self._block()
        return IfStmt(cond, then_body.statements, else_body.statements if else_body else None)

    def _while_stmt(self) -> WhileStmt:
        self._eat(TokenType.WHILE)
        self._eat(TokenType.LPAREN)
        cond = self._expression()
        self._eat(TokenType.RPAREN)
        body = self._block()
        return WhileStmt(cond, body.statements)

    def _for_in_stmt(self) -> ForInStmt:
        """Parse: for (x in expr) { body }"""
        self._eat(TokenType.FOR)
        self._eat(TokenType.LPAREN)
        var_name = self._eat(TokenType.IDENT).value
        self._eat(TokenType.IN)
        iterable = self._expression()
        self._eat(TokenType.RPAREN)
        body = self._block()
        return ForInStmt(var_name, iterable, body.statements)

    # ── Expressions ──────────────────────────

    def _expression(self):
        if self._is_arrow_fn():
            return self._arrow_fn()
        return self._or_expr()

    def _is_arrow_fn(self) -> bool:
        if self._current().type != TokenType.LPAREN:
            return False
        depth = 0
        i = self.pos
        while i < len(self.tokens):
            tt = self.tokens[i].type
            if tt == TokenType.LPAREN: depth += 1
            elif tt == TokenType.RPAREN:
                depth -= 1
                if depth == 0:
                    j = i + 1
                    # Skip optional return-type annotation: ): TYPE =>
                    if j < len(self.tokens) and self.tokens[j].type == TokenType.COLON:
                        j += 1
                        # Skip until we hit => or a block/statement terminator.
                        while (j < len(self.tokens)
                               and self.tokens[j].type not in (TokenType.FAT_ARROW,
                                                                TokenType.LBRACE,
                                                                TokenType.SEMI)):
                            j += 1
                    return (j < len(self.tokens)
                            and self.tokens[j].type == TokenType.FAT_ARROW)
            i += 1
        return False

    def _arrow_fn(self) -> ArrowFn:
        params, pkinds = self._param_list()
        # Optional return type annotation — kind head retained for the return check.
        rkind = ""
        if self._current().type == TokenType.COLON:
            self._eat(TokenType.COLON)
            rkind = self._type_ann_kind()
        self._eat(TokenType.FAT_ARROW)
        if self._current().type == TokenType.LBRACE and not self._is_struct_literal():
            body = self._block()
            return ArrowFn(params, body.statements,
                           param_kinds=pkinds, ret_kind=rkind)
        else:
            expr = self._expression()
            return ArrowFn(params, expr,
                           param_kinds=pkinds, ret_kind=rkind)

    def _or_expr(self):
        left = self._and_expr()
        while self._match(TokenType.OR):
            left = BinOp('||', left, self._and_expr())
        return left

    def _and_expr(self):
        left = self._equality()
        while self._match(TokenType.AND):
            left = BinOp('&&', left, self._equality())
        return left

    def _equality(self):
        left = self._comparison()
        while tok := self._match(TokenType.EQ, TokenType.NEQ):
            left = BinOp(tok.value, left, self._comparison())
        return left

    def _comparison(self):
        left = self._addition()
        while tok := self._match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            left = BinOp(tok.value, left, self._addition())
        return left

    def _addition(self):
        left = self._multiplication()
        while tok := self._match(TokenType.PLUS, TokenType.MINUS):
            left = BinOp(tok.value, left, self._multiplication())
        return left

    def _multiplication(self):
        left = self._unary()
        while tok := self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            left = BinOp(tok.value, left, self._unary())
        return left

    def _unary(self):
        if self._match(TokenType.MINUS):
            return UnaryOp('-', self._unary())
        if self._match(TokenType.NOT):
            return UnaryOp('!', self._unary())
        return self._postfix()

    def _postfix(self):
        """Parse postfix: calls f(args), indexing expr[i], dot access expr.field."""
        expr = self._primary()
        while True:
            if self._current().type == TokenType.LPAREN:
                self._eat(TokenType.LPAREN)
                args = []
                if self._current().type != TokenType.RPAREN:
                    args.append(self._expression())
                    while self._match(TokenType.COMMA):
                        args.append(self._expression())
                self._eat(TokenType.RPAREN)
                expr = CallExpr(expr, args)
            elif self._current().type == TokenType.LBRACKET:
                self._eat(TokenType.LBRACKET)
                index = self._expression()
                self._eat(TokenType.RBRACKET)
                expr = IndexExpr(expr, index)
            elif self._current().type == TokenType.DOT:
                self._eat(TokenType.DOT)
                field = self._eat(TokenType.IDENT).value
                expr = DotExpr(expr, field)
            else:
                break
        return expr

    def _primary(self):
        tok = self._current()

        if tok.type == TokenType.MATCH:
            return self._match_expr()
        if tok.type == TokenType.NUMBER:
            self.pos += 1
            return NumberLit(tok.value)
        if tok.type == TokenType.STRING:
            self.pos += 1
            return StringLit(tok.value)
        if tok.type == TokenType.BOOL:
            self.pos += 1
            return BoolLit(tok.value)
        if tok.type == TokenType.IDENT:
            if tok.value == "none":
                # `none` is the none-literal, not a name.
                self.pos += 1
                return NoneLit()
            self.pos += 1
            return Identifier(tok.value)

        if tok.type == TokenType.LBRACKET:
            return self._array_literal()

        if tok.type == TokenType.LBRACE:
            if self._is_struct_literal():
                return self._struct_literal()
            # Otherwise it's a block — but blocks aren't expressions in Arrow Lang,
            # so this would be a parse error in expression context
            raise ParseError(f"Unexpected '{{' in expression at line {tok.line}, col {tok.col}")

        if tok.type == TokenType.LPAREN:
            self._eat(TokenType.LPAREN)
            expr = self._expression()
            self._eat(TokenType.RPAREN)
            return expr

        raise ParseError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) "
            f"at line {tok.line}, col {tok.col}")

    def _array_literal(self) -> ArrayLit:
        self._eat(TokenType.LBRACKET)
        elements = []
        if self._current().type != TokenType.RBRACKET:
            elements.append(self._expression())
            while self._match(TokenType.COMMA):
                elements.append(self._expression())
        self._eat(TokenType.RBRACKET)
        return ArrayLit(elements)

    def _struct_literal(self) -> StructLit:
        """Parse: { key: expr, key: expr, ... }"""
        self._eat(TokenType.LBRACE)
        fields = []
        if self._current().type != TokenType.RBRACE:
            # Keys can be IDENT or STRING
            if self._current().type == TokenType.STRING:
                key = self._eat(TokenType.STRING).value
            else:
                key = self._eat(TokenType.IDENT).value
            self._eat(TokenType.COLON)
            val = self._expression()
            fields.append((key, val))
            while self._match(TokenType.COMMA):
                if self._current().type == TokenType.RBRACE:
                    break  # trailing comma
                if self._current().type == TokenType.STRING:
                    key = self._eat(TokenType.STRING).value
                else:
                    key = self._eat(TokenType.IDENT).value
                self._eat(TokenType.COLON)
                val = self._expression()
                fields.append((key, val))
        self._eat(TokenType.RBRACE)
        return StructLit(fields)


# ─────────────────────────────────────────────
#  INTERPRETER
# ─────────────────────────────────────────────
class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class RuntimeError_(Exception):
    pass


def _value_kind(v) -> str:
    """Map a runtime value to its universal kind name (native tag order;
    bool tested before int because Python bool subclasses int). Values
    with no mapping are 'opaque' and pass every runtime kind expectation."""
    if v is None: return "none"
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str): return "str"
    if isinstance(v, list): return "array"
    if isinstance(v, Struct): return "struct"
    if isinstance(v, Function): return "fn"
    return "opaque"


def _any_check(val, want: str):
    """Runtime kind check at an annotated edge. none/any expectations
    emit no check; opaque values pass everything; int/bool/float accept
    each other (no value conversion — the native float promote is a
    storage-representation detail the interpreter does not need). The
    message matches the native trap byte-for-byte once main() prefixes
    'Error: '."""
    if want == "" or want == "any" or want == "union" or want == "union_none":
        return
    got = _value_kind(val)
    if got == "opaque" or got == want:
        return
    num = ("int", "float", "bool")
    if want in num and got in num:
        return
    raise RuntimeError_(f"expected {want} in any, got {got}")


_BUILTIN_RET_KINDS = {
    "len": "int", "push": "int", "pop": "any", "keys": "array",
    "read_file": "str", "write_file": "int", "append_file": "int",
    "input": "str", "exec_cmd": "int", "args": "any",
    "char_code": "int", "from_char_code": "str", "substring": "str",
    "char_at": "str", "str_len": "int", "file_exists": "bool",
}


def _fn_is_unit(fn) -> bool:
    """True when no path returns a value. Cached."""
    c = getattr(fn, "_unit_cached", None)
    if c is not None:
        return c
    if not isinstance(fn.body, list):      # expression-bodied arrow fn
        fn._unit_cached = False
        return False
    found = [False]
    def walk(n):
        if found[0] or n is None or isinstance(n, (str, int, float, bool)):
            return
        if isinstance(n, list):
            for x in n:
                walk(x)
            return
        if isinstance(n, ReturnStmt):
            if n.expr is not None and n.expr is not False:
                found[0] = True
            return
        if isinstance(n, (FnDecl, ArrowFn)):
            return
        for v in getattr(n, "__dict__", {}).values():
            walk(v)
    walk(fn.body)
    fn._unit_cached = not found[0]
    return fn._unit_cached


def _classify_static(node, env) -> str:
    """Returns a kind head, "any" when
    unknown, or "unit" for calls of functions that return no value."""
    if isinstance(node, NumberLit):
        return "float" if isinstance(node.value, float) else "int"
    if isinstance(node, StringLit):
        return "str"
    if isinstance(node, BoolLit):
        return "bool"
    if isinstance(node, ArrayLit):
        return "array"
    if isinstance(node, StructLit):
        return "struct"
    if isinstance(node, ArrowFn):
        return "fn"
    if isinstance(node, Identifier):
        e = env
        while e is not None:
            if node.name in e.vars:
                k = e.kinds.get(node.name, "any")
                return k if k != "" else "any"
            e = e.parent
        return "any"
    if isinstance(node, UnaryOp):
        if node.op == "!":
            return "bool"
        return _classify_static(node.operand, env)
    if isinstance(node, BinOp):
        if node.op in ("=", "==", "!=", "<", ">", "<=", ">=", "&&", "||"):
            return "bool"
        lk = _classify_static(node.left, env)
        rk = _classify_static(node.right, env)
        if node.op == "+" and "str" in (lk, rk):
            return "str"
        if node.op in ("+", "-", "*", "/", "%"):
            if "float" in (lk, rk):
                return "float"
            if lk in ("int", "bool") and rk in ("int", "bool"):
                return "int"
        return "any"
    if isinstance(node, CallExpr) and isinstance(node.callee, Identifier):
        n = node.callee.name
        if n in _BUILTIN_RET_KINDS:
            return _BUILTIN_RET_KINDS[n]
        e, fnv = env, None
        while e is not None:
            if n in e.vars:
                fnv = e.vars[n]
                break
            e = e.parent
        if isinstance(fnv, Function):
            if fnv.ret_kind != "":
                return fnv.ret_kind
            return "unit" if _fn_is_unit(fnv) else "any"
        return "any"
    return "any"



class Environment:
    def __init__(self, parent: 'Environment | None' = None, is_fn_root: bool = False):
        self.vars: dict[str, Any] = {}
        # Annotation kind heads for bindings declared with a type.
        # Reassignments through assign() re-check against the owning
        # scope's recorded kind, mirroring the native declared-slot rule.
        self.kinds: dict[str, str] = {}
        self.parent = parent
        # Function-boundary marker: assignment walks up looking for an
        # existing binding to mutate, but stops at a function root so it
        # cannot write into a closure's captured scope. Reads (`get`) walk
        # freely across the boundary so closures can still see captured
        # values; assignments create or error instead, preserving Arrow's
        # by-snapshot closure convention.
        self.is_fn_root = is_fn_root

    def get(self, name: str):
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise RuntimeError_(f"Undefined variable '{name}'")

    def declare(self, name: str, value: Any, kind: str = ""):
        # Always create a fresh binding in the *current* scope. Error if a
        # name with the same identifier already exists in this same scope
        # (catches accidental redeclaration). Outer-scope bindings of the
        # same name are shadowed, not errored — that's intentional.
        if name in self.vars:
            raise RuntimeError_(f"redeclaration of '{name}' in the same scope")
        self.vars[name] = value
        if kind != "":
            self.kinds[name] = kind

    def assign(self, name: str, value: Any) -> bool:
        # Walk all the way up to find an existing binding. No function
        # boundary — a closure that writes a bare `name <- ...` to a
        # captured outer-fn local mutates the actual cell, not a snapshot.
        # Globals are the topmost env in the chain, so this also handles
        # function-writes-to-global without a separate fallback.
        env = self
        while env is not None:
            if name in env.vars:
                # A binding declared with an annotation keeps its kind
                # for life — later writes re-check, like the native slot.
                k = env.kinds.get(name, "")
                if k != "":
                    _any_check(value, k)
                env.vars[name] = value
                return True
            env = env.parent
        return False

    def set(self, name: str, value: Any):
        # Legacy auto-declare path kept around for any remaining call sites
        # during the var-keyword migration. New code goes through declare
        # or assign explicitly. Behaviour mirrors the original set(): walk
        # up within the function to find a binding, otherwise drop a fresh
        # one in the current scope.
        if not self.assign(name, value):
            self.vars[name] = value


class Function:
    def __init__(self, name: str, params: list[str], body: Any, closure: Environment,
                 param_kinds: list | None = None, ret_kind: str = ""):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
        # Annotation kind heads for the param-bind and return checks.
        self.param_kinds = param_kinds if param_kinds is not None else []
        self.ret_kind = ret_kind

    def __repr__(self):
        return f"<fn {self.name}({', '.join(self.params)})>"


class Struct:
    """Runtime representation of a struct/record."""
    def __init__(self, fields: dict[str, Any]):
        self._fields = fields

    def get(self, name: str):
        if name not in self._fields:
            raise RuntimeError_(f"Struct has no field '{name}'")
        return self._fields[name]

    def set(self, name: str, value: Any):
        if name not in self._fields:
            raise RuntimeError_(f"Struct has no field '{name}'")
        self._fields[name] = value

    def fields(self):
        return self._fields

    def __repr__(self):
        return "{" + ", ".join(f"{k}: ..." for k in self._fields) + "}"


BUILTINS = {"len", "push", "pop", "keys", "read_file", "write_file", "append_file", "input",
            "char_code", "from_char_code", "substring", "char_at", "str_len",
            "exec_cmd", "args", "file_exists"}


class Interpreter:
    def __init__(self):
        self.env = Environment()
        # Top-level scope kept separately so an assignment from inside a
        # function body can reach a global binding when there's no
        # matching local on the way up.
        self.globals = self.env
        self.output: list[str] = []

    def run(self, program: Program):
        for stmt in program.statements:
            self._exec(stmt)

    def _exec_block(self, stmts):
        """Run a sequence of statements in a fresh child scope. Bindings
        introduced inside the block die when this returns, which is the
        whole point of block scoping."""
        outer = self.env
        self.env = Environment(parent=outer)
        try:
            for s in stmts: self._exec(s)
        finally:
            self.env = outer

    def _exec(self, node):
        match node:
            case Assignment(name, expr, is_decl, line, col):
                val = self._eval(expr)
                if isinstance(val, Function) and val.name == "<arrow>":
                    val.name = name
                if is_decl:
                    # An annotated declaration checks the value's kind
                    # before binding. This runs OUTSIDE the try below — the
                    # trap message carries no position info, matching the
                    # native trap byte-for-byte.
                    if node.type_kind != "":
                        _any_check(val, node.type_kind)
                    # `var x <- expr;` — fresh binding in current scope.
                    # Redeclaration in the same scope is an error.
                    dk = node.type_kind
                    if dk == "":
                        # Unannotated decls lock to their
                        # classified static kind; unknown locks open (any).
                        dk = _classify_static(expr, self.env)
                        if dk == "unit":
                            # A no-value call yields
                            # none; binding it is legal and locks the
                            # name to kind "none".
                            dk = "none"
                    try:
                        self.env.declare(name, val, dk)
                    except RuntimeError_ as e:
                        # Re-raise with position info for better diagnostics.
                        raise RuntimeError_(f"{e} at line {line}, col {col}")
                else:
                    # `x <- expr;` — reassignment. assign() walks the env
                    # chain freely; that means closures can write to their
                    # captured outer locals (by-reference), and function
                    # bodies can write to globals via the topmost env. The
                    # only failure case is a name that doesn't exist
                    # anywhere — that's the typo case the user wanted
                    # caught.
                    # A typed reassignment (`x: str <- e;`) checks its
                    # own annotation; assign() then re-checks the kind
                    # recorded at declaration, whichever scope owns it.
                    if node.type_kind != "":
                        _any_check(val, node.type_kind)
                    if not self.env.assign(name, val):
                        raise RuntimeError_(
                            f"cannot reassign undeclared variable '{name}' "
                            f"at line {line}, col {col} — did you mean `var {name} <- ...`?"
                        )

            case IndexAssign(obj, index, value):
                target = self._eval(obj)
                idx = self._eval(index)
                val = self._eval(value)
                if not isinstance(target, list):
                    raise RuntimeError_(f"expected array in any, got {_value_kind(target)}")
                if not isinstance(idx, int):
                    raise RuntimeError_("Array index must be an integer")
                if idx < 0 or idx >= len(target):
                    raise RuntimeError_(f"Index {idx} out of bounds (length {len(target)})")
                target[idx] = val

            case DotAssign(obj, field, value):
                target = self._eval(obj)
                val = self._eval(value)
                if not isinstance(target, Struct):
                    raise RuntimeError_(f"expected struct in any, got {_value_kind(target)}")
                target.set(field, val)

            case PrintStmt(expr):
                val = self._eval(expr)
                text = self._format(val)
                print(text)
                self.output.append(text)

            case MatchStmt(scrutinee, arms):
                # Dynamically typed: a union value is just the underlying
                # value, so dispatch on its runtime type. bool is checked
                # before int (Python bool is an int subclass) — exactly the
                # native tag order. The matching arm runs in a fresh scope
                # with the binding (named, or the scrutinee variable shadowed)
                # bound to the value.
                val = self._eval(scrutinee)
                chosen = self._match_pick(arms, val)
                if chosen is not None:
                    outer = self.env
                    self.env = Environment(parent=outer)
                    try:
                        # A wildcard binds nothing — the scrutinee keeps its
                        # full union value, so no shadow is introduced.
                        if chosen.ptype_kind != "_":
                            bn = chosen.name
                            if bn is None and isinstance(scrutinee, Identifier):
                                bn = scrutinee.name
                            if bn is not None:
                                self.env.declare(bn, val)
                                self._bind_match_fields(chosen, val)
                        for s in chosen.body:
                            self._exec(s)
                    finally:
                        self.env = outer

            case IfStmt(cond, then_body, else_body):
                # The condition is evaluated in the enclosing scope. Each
                # branch body is its own block, so declarations inside it
                # die at the closing brace.
                if self._truthy(self._eval(cond)):
                    self._exec_block(then_body)
                elif else_body:
                    self._exec_block(else_body)

            case WhileStmt(cond, body):
                # Condition stays in the enclosing scope (it routinely
                # references the loop counter). Each iteration's body is a
                # fresh block — declarations inside die between iterations.
                iterations = 0
                while self._truthy(self._eval(cond)):
                    self._exec_block(body)
                    iterations += 1
                    if iterations > 10_000_000:
                        raise RuntimeError_("Infinite loop detected")

            case ForInStmt(var_name, iterable, body):
                # Iterable evaluated in enclosing scope. Each iteration gets
                # a fresh body scope, with the loop variable declared into
                # that scope (so it never accidentally mutates an outer
                # variable with the same name).
                collection = self._eval(iterable)
                if isinstance(collection, list):
                    items = list(collection)
                elif isinstance(collection, str):
                    items = list(collection)
                else:
                    raise RuntimeError_("for-in requires an array or string")
                outer = self.env
                for item in items:
                    self.env = Environment(parent=outer)
                    self.env.declare(var_name, item)
                    try:
                        for s in body: self._exec(s)
                    finally:
                        self.env = outer

            case Block(stmts):
                self._exec_block(stmts)

            case FnDecl(name, params, body):
                self.env.set(name, Function(name, params, body, self.env,
                                            param_kinds=node.param_kinds,
                                            ret_kind=node.ret_kind))

            case ReturnStmt(expr):
                raise ReturnSignal(self._eval(expr) if expr is not None else None)

            case _:
                self._eval(node)

    def _match_pick(self, arms, val):
        """Select the matching arm for `val` (shared by statement and
        expression match), or None. bool is checked before int (Python bool
        subclasses int), matching the native tag order."""
        if val is None: kind = "none"
        elif isinstance(val, bool): kind = "bool"
        elif isinstance(val, int): kind = "int"
        elif isinstance(val, str): kind = "str"
        elif isinstance(val, float): kind = "float"
        elif isinstance(val, list): kind = "array"
        else: kind = "struct"
        for arm in arms:
            if arm.ptype_kind == "_":
                return arm
            if arm.lit_kind is not None:
                if kind == arm.lit_kind and val == arm.lit_val:
                    return arm
                continue
            if arm.ptype_kind == kind:
                if kind == "struct":
                    # Distinct struct shapes are told apart by their exact
                    # field-name set — the runtime analogue of the native tag
                    # that encodes which struct member the value inhabits.
                    if isinstance(val, Struct) and arm.pfields is not None \
                            and set(arm.pfields) == set(val.fields().keys()):
                        return arm
                    continue
                return arm
        return None

    def _bind_match_fields(self, chosen, val):
        # Destructuring: a struct-pattern arm binds each of its pattern
        # fields as a local in the arm scope (statement and expression
        # arms alike). The native compiler reaches the same end via a
        # parser desugar (statement arms) and emit-time field binding
        # (expression arms).
        if chosen.ptype_kind == "struct" and chosen.pfields is not None \
                and isinstance(val, Struct):
            fv = val.fields()
            for fname in chosen.pfields:
                self.env.declare(fname, fv[fname])

    def _eval(self, node) -> Any:
        match node:
            case NumberLit(v): return v
            case StringLit(v): return v
            case BoolLit(v): return v
            case NoneLit(): return None
            case Identifier(name): return self.env.get(name)

            case ArrayLit(elements):
                return [self._eval(e) for e in elements]

            case StructLit(fields):
                return Struct({k: self._eval(v) for k, v in fields})

            case IndexExpr(obj, index):
                target = self._eval(obj)
                idx = self._eval(index)
                if isinstance(target, list):
                    if not isinstance(idx, int):
                        raise RuntimeError_("Array index must be an integer")
                    if idx < 0 or idx >= len(target):
                        raise RuntimeError_(f"Index {idx} out of bounds (length {len(target)})")
                    return target[idx]
                elif isinstance(target, str):
                    if not isinstance(idx, int):
                        raise RuntimeError_("String index must be an integer")
                    if idx < 0 or idx >= len(target):
                        raise RuntimeError_(f"Index {idx} out of bounds (length {len(target)})")
                    return target[idx]
                else:
                    raise RuntimeError_(f"expected array in any, got {_value_kind(target)}")

            case DotExpr(obj, field):
                target = self._eval(obj)
                if isinstance(target, Struct):
                    return target.get(field)
                raise RuntimeError_(f"expected struct in any, got {_value_kind(target)}")

            case UnaryOp('-', operand): return -self._eval(operand)
            case UnaryOp('!', operand): return not self._truthy(self._eval(operand))

            case BinOp(op, left, right):
                return self._eval_binop(op, left, right)

            case ArrowFn(params, body):
                return Function("<arrow>", params, body, self.env,
                                param_kinds=node.param_kinds,
                                ret_kind=node.ret_kind)

            case CallExpr(callee, args):
                return self._eval_call(callee, args)

            case MatchExpr(scrutinee, arms):
                val = self._eval(scrutinee)
                chosen = self._match_pick(arms, val)
                if chosen is None:
                    raise RuntimeError_("match expression had no matching arm")
                outer = self.env
                self.env = Environment(parent=outer)
                try:
                    if chosen.ptype_kind != "_":
                        bn = chosen.name
                        if bn is None and isinstance(scrutinee, Identifier):
                            bn = scrutinee.name
                        if bn is not None:
                            self.env.declare(bn, val)
                            self._bind_match_fields(chosen, val)
                    return self._eval(chosen.value)
                finally:
                    self.env = outer

            case _:
                if isinstance(node, TypeDecl):
                    return None

                raise RuntimeError_(f"Cannot evaluate node: {node}")

    def _eval_call(self, callee, args) -> Any:
        if isinstance(callee, Identifier) and callee.name in BUILTINS:
            return self._eval_builtin(callee.name, args)

        fn = self._eval(callee)
        if not isinstance(fn, Function):
            # A non-function callee can only arrive through an any/none
            # flow in checker-accepted programs — same trap as the native
            # declared-any callee check (expected tag 6).
            raise RuntimeError_(f"expected fn in any, got {_value_kind(fn)}")

        # Evaluate and kind-check arguments left-to-right, so a trap
        # on argument i fires before argument i+1 evaluates — matching the
        # native per-argument coercion order at the call site.
        pkinds = fn.param_kinds
        arg_vals = []
        for ai, a in enumerate(args):
            av = self._eval(a)
            _any_check(av, pkinds[ai] if ai < len(pkinds) else "")
            arg_vals.append(av)
        if len(arg_vals) != len(fn.params):
            raise RuntimeError_(
                f"Function {fn.name} expects {len(fn.params)} args, got {len(arg_vals)}")

        call_env = Environment(parent=fn.closure, is_fn_root=True)
        for pi, (param, val) in enumerate(zip(fn.params, arg_vals)):
            # `declare`, not `set` — the param shouldn't accidentally mutate
            # a same-named binding in the enclosing closure scope. The
            # annotation kind rides along so body reassignments re-check.
            call_env.declare(param, val,
                             pkinds[pi] if pi < len(pkinds) else "")

        prev_env = self.env
        self.env = call_env
        result = None
        try:
            if isinstance(fn.body, list):
                for stmt in fn.body: self._exec(stmt)
            else:
                result = self._eval(fn.body)
        except ReturnSignal as ret:
            result = ret.value
        finally:
            self.env = prev_env
        if fn.ret_kind != "":
            # Annotated return edge — same matrix as the native
            # checked unwrap at the return slot. A body that falls off
            # the end yields None, which maps to opaque and passes.
            _any_check(result, fn.ret_kind)
        return result

    def _eval_builtin(self, name: str, args: list) -> Any:
        if name == "len":
            val = self._eval(args[0])
            if isinstance(val, (list, str)):
                return len(val)
            raise RuntimeError_("len() requires an array or string")

        elif name == "push":
            arr = self._eval(args[0])
            val = self._eval(args[1])
            if not isinstance(arr, list):
                raise RuntimeError_("push() requires an array")
            arr.append(val)
            return len(arr)

        elif name == "pop":
            arr = self._eval(args[0])
            if not isinstance(arr, list):
                raise RuntimeError_("pop() requires an array")
            if len(arr) == 0:
                raise RuntimeError_("Cannot pop from empty array")
            return arr.pop()

        elif name == "keys":
            val = self._eval(args[0])
            if not isinstance(val, Struct):
                raise RuntimeError_("keys() requires a struct")
            return list(val.fields().keys())

        elif name == "read_file":
            if len(args) != 1:
                raise RuntimeError_("read_file() takes exactly 1 argument")
            path = self._eval(args[0])
            if not isinstance(path, str):
                raise RuntimeError_("read_file() requires a string path")
            # A missing (or unopenable) file is a runtime error — same
            # family as division by zero and out-of-bounds indexing, and
            # byte-identical to the native @arrow_read_file trap:
            # `Error: cannot read file: <path>` on stdout, exit 1.
            # Probing for optional files is file_exists()'s job (the
            # import resolver's search-path fallback uses it).
            try:
                # newline="": no CRLF translation on read — byte-faithful
                # to the native fopen("rb") helper on every platform.
                with open(path, encoding="utf-8", newline="") as f:
                    return f.read()
            except Exception:
                raise RuntimeError_(f"cannot read file: {path}")

        elif name == "file_exists":
            if len(args) != 1:
                raise RuntimeError_("file_exists() takes exactly 1 argument")
            path = self._eval(args[0])
            if not isinstance(path, str):
                raise RuntimeError_("file_exists() requires a string path")
            return os.path.isfile(path)

        elif name == "write_file":
            if len(args) != 2:
                raise RuntimeError_("write_file() takes exactly 2 arguments")
            path = self._eval(args[0])
            content = self._eval(args[1])
            if not isinstance(path, str):
                raise RuntimeError_("write_file() requires a string path")
            if not isinstance(content, str):
                content = self._format(content)
            try:
                # newline="": \n stays \n on Windows — matches the
                # native fopen("wb") helper byte-for-byte.
                with open(path, "w", encoding="utf-8", newline="") as f:
                    return f.write(content)
            except Exception as e:
                raise RuntimeError_(f"Error writing file: {e}")

        elif name == "append_file":
            if len(args) != 2:
                raise RuntimeError_("append_file() takes exactly 2 arguments")
            path = self._eval(args[0])
            content = self._eval(args[1])
            if not isinstance(path, str):
                raise RuntimeError_("append_file() requires a string path")
            if not isinstance(content, str):
                content = self._format(content)
            try:
                # newline="": \n stays \n on Windows — matches the
                # native fopen("ab") helper byte-for-byte.
                with open(path, "a", encoding="utf-8", newline="") as f:
                    return f.write(content)
            except Exception as e:
                raise RuntimeError_(f"Error appending to file: {e}")

        elif name == "input":
            if len(args) > 1:
                raise RuntimeError_("input() takes 0 or 1 arguments")
            if len(args) == 1:
                prompt = self._eval(args[0])
                return input(self._format(prompt))
            return input()

        elif name == "exec_cmd":
            if len(args) != 1:
                raise RuntimeError_("exec_cmd() takes exactly 1 argument")
            cmd = self._eval(args[0])
            if not isinstance(cmd, str):
                raise RuntimeError_("exec_cmd() requires a string command")
            import subprocess
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, end='', file=__import__('sys').stderr)
            return result.returncode

        elif name == "args":
            import sys
            if len(args) == 0:
                # Return number of script arguments (excluding interpreter and script file)
                # sys.argv = ['lang.py', 'compiler.arrow', arg1, arg2, ...]
                return len(sys.argv) - 2
            idx = self._eval(args[0])
            if not isinstance(idx, int):
                raise RuntimeError_("args() index must be an integer")
            # args(0) = first argument after the script file
            actual_idx = idx + 2  # skip 'lang.py' and the script filename
            if actual_idx < 0 or actual_idx >= len(sys.argv):
                return ""
            return sys.argv[actual_idx]

        elif name == "char_code":
            if len(args) != 1:
                raise RuntimeError_("char_code() takes exactly 1 argument")
            val = self._eval(args[0])
            if not isinstance(val, str) or len(val) == 0:
                raise RuntimeError_("char_code() requires a non-empty string")
            return ord(val[0])

        elif name == "from_char_code":
            if len(args) != 1:
                raise RuntimeError_("from_char_code() takes exactly 1 argument")
            val = self._eval(args[0])
            if not isinstance(val, int):
                raise RuntimeError_("from_char_code() requires an integer")
            return chr(val)

        elif name == "substring":
            if len(args) != 3:
                raise RuntimeError_("substring() takes exactly 3 arguments")
            s = self._eval(args[0])
            start = self._eval(args[1])
            end = self._eval(args[2])
            if not isinstance(s, str):
                raise RuntimeError_("substring() requires a string as first argument")
            if not isinstance(start, int) or not isinstance(end, int):
                raise RuntimeError_("substring() indices must be integers")
            return s[start:end]

        elif name == "char_at":
            if len(args) != 2:
                raise RuntimeError_("char_at() takes exactly 2 arguments")
            s = self._eval(args[0])
            idx = self._eval(args[1])
            if not isinstance(s, str):
                raise RuntimeError_("char_at() requires a string as first argument")
            if not isinstance(idx, int):
                raise RuntimeError_("char_at() index must be an integer")
            if idx < 0 or idx >= len(s):
                raise RuntimeError_(f"Index {idx} out of bounds (length {len(s)})")
            return s[idx]

        elif name == "str_len":
            if len(args) != 1:
                raise RuntimeError_("str_len() takes exactly 1 argument")
            val = self._eval(args[0])
            if not isinstance(val, str):
                raise RuntimeError_("str_len() requires a string")
            return len(val)

        raise RuntimeError_(f"Unknown builtin: {name}")

    def _eval_binop(self, op: str, left, right) -> Any:
        # Short-circuit && and || — must not evaluate right side eagerly
        if op == '&&':
            lv = self._eval(left)
            if not self._truthy(lv):
                return False
            return self._truthy(self._eval(right))
        if op == '||':
            lv = self._eval(left)
            if self._truthy(lv):
                return lv
            return self._eval(right)
        lv = self._eval(left)
        rv = self._eval(right)
        match op:
            case '+':
                if isinstance(lv, list) and isinstance(rv, list):
                    return lv + rv
                if isinstance(lv, str) or isinstance(rv, str):
                    return self._format(lv) + self._format(rv)
                return lv + rv
            case '-':  return lv - rv
            case '*':  return lv * rv
            case '/':
                if rv == 0: raise RuntimeError_("Division by zero")
                # Match Arrow's native semantics: int/int → int (truncate
                # toward zero), float promotion only when at least one
                # operand is float. Python's / always returns float, so we
                # explicitly route through int() when both are int.
                if isinstance(lv, int) and not isinstance(lv, bool) and isinstance(rv, int) and not isinstance(rv, bool):
                    # int(lv/rv) truncates toward zero for any sign combo.
                    return int(lv / rv)
                return lv / rv
            case '%':
                if rv == 0: raise RuntimeError_("Modulo by zero")
                return lv % rv
            case '<':  return lv < rv
            case '>':  return lv > rv
            case '<=': return lv <= rv
            case '>=': return lv >= rv
            case '=':  return lv == rv
            case '!=': return lv != rv
            case _: raise RuntimeError_(f"Unknown operator: {op}")

    def _truthy(self, val) -> bool:
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return val != 0
        if isinstance(val, str): return len(val) > 0
        if isinstance(val, list): return len(val) > 0
        return val is not None

    def _format(self, val, in_collection: bool = False) -> str:
        if val is None:
            return "none"
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, float):
            # Match Python's repr(float) — always show a decimal point so that
            # 2.0 prints as "2.0", not "2". This also matches the native
            # compiler's float formatter.
            return repr(val)
        if isinstance(val, list):
            return "[" + ", ".join(self._format(v, True) for v in val) + "]"
        if isinstance(val, Struct):
            parts = []
            for k, v in val.fields().items():
                parts.append(f"{k}: {self._format(v, True)}")
            return "{" + ", ".join(parts) + "}"
        if isinstance(val, Function):
            return repr(val)
        if isinstance(val, str) and in_collection:
            return '"' + val + '"'
        return str(val)


# ─────────────────────────────────────────────
#  MODULE RESOLVER
# ─────────────────────────────────────────────
# Mirrors compiler.arrow's resolver: walks the program, expands every
# `import "x";` into the imported file's top-level declarations with
# names mangled to `__mod_<modname>__<orig>`, and rewrites `mod.sym`
# DotExpr references in the importing file to the mangled identifiers.
def _validate_module(stmts: list, path: str) -> list[str]:
    errors = []
    for s in stmts:
        if not isinstance(s, (FnDecl, Assignment, ImportStmt, TypeDecl)):
            errors.append(f"{path}: module may only contain fn declarations, assignments, type declarations, or imports (got {type(s).__name__})")
    return errors


def _collect_top_names(stmts: list) -> set:
    names = set()
    for s in stmts:
        if isinstance(s, FnDecl):
            names.add(s.name)
        elif isinstance(s, Assignment):
            names.add(s.name)
    return names


def _mod_rewrite(node, top_names: set, scope: set, mod_name: str):
    """Mutate `node` in place — rewrite Identifier references to top-level
    names (not shadowed by `scope`) to their mangled form."""
    if isinstance(node, Identifier):
        if node.name in top_names and node.name not in scope:
            node.name = f"__mod_{mod_name}__{node.name}"
        return
    if isinstance(node, BinOp):
        _mod_rewrite(node.left, top_names, scope, mod_name)
        _mod_rewrite(node.right, top_names, scope, mod_name)
        return
    if isinstance(node, UnaryOp):
        _mod_rewrite(node.operand, top_names, scope, mod_name)
        return
    if isinstance(node, CallExpr):
        _mod_rewrite(node.callee, top_names, scope, mod_name)
        for a in node.args:
            _mod_rewrite(a, top_names, scope, mod_name)
        return
    if isinstance(node, IndexExpr):
        _mod_rewrite(node.obj, top_names, scope, mod_name)
        _mod_rewrite(node.index, top_names, scope, mod_name)
        return
    if isinstance(node, DotExpr):
        _mod_rewrite(node.obj, top_names, scope, mod_name)
        return
    if isinstance(node, ArrayLit):
        for e in node.elements:
            _mod_rewrite(e, top_names, scope, mod_name)
        return
    if isinstance(node, StructLit):
        for _, v in node.fields:
            _mod_rewrite(v, top_names, scope, mod_name)
        return
    if isinstance(node, ArrowFn):
        sub = scope | set(node.params)
        if isinstance(node.body, list):
            for s in node.body:
                _mod_rewrite_stmt(s, top_names, sub, mod_name)
        else:
            _mod_rewrite(node.body, top_names, sub, mod_name)
        return
    if isinstance(node, MatchExpr):
        _mod_rewrite(node.scrutinee, top_names, scope, mod_name)
        for arm in node.arms:
            _mod_rewrite(arm.value, top_names, scope, mod_name)
        return
    if isinstance(node, Assignment):
        _mod_rewrite(node.expr, top_names, scope, mod_name)
        return
    if isinstance(node, ReturnStmt):
        if node.expr is not None:
            _mod_rewrite(node.expr, top_names, scope, mod_name)
        return
    if isinstance(node, PrintStmt):
        _mod_rewrite(node.expr, top_names, scope, mod_name)
        return
    if isinstance(node, MatchStmt):
        _mod_rewrite(node.scrutinee, top_names, scope, mod_name)
        for arm in node.arms:
            for s in arm.body:
                _mod_rewrite_stmt(s, top_names, scope, mod_name)
        return
    if isinstance(node, IfStmt):
        _mod_rewrite(node.condition, top_names, scope, mod_name)
        for s in node.then_body:
            _mod_rewrite_stmt(s, top_names, scope, mod_name)
        if node.else_body:
            for s in node.else_body:
                _mod_rewrite_stmt(s, top_names, scope, mod_name)
        return
    if isinstance(node, WhileStmt):
        _mod_rewrite(node.condition, top_names, scope, mod_name)
        for s in node.body:
            _mod_rewrite_stmt(s, top_names, scope, mod_name)
        return
    if isinstance(node, ForInStmt):
        _mod_rewrite(node.iterable, top_names, scope, mod_name)
        for s in node.body:
            _mod_rewrite_stmt(s, top_names, scope | {node.var_name}, mod_name)
        return
    if isinstance(node, Block):
        for s in node.statements:
            _mod_rewrite_stmt(s, top_names, scope, mod_name)
        return
    if isinstance(node, IndexAssign):
        _mod_rewrite(node.obj, top_names, scope, mod_name)
        _mod_rewrite(node.index, top_names, scope, mod_name)
        _mod_rewrite(node.value, top_names, scope, mod_name)
        return
    if isinstance(node, DotAssign):
        _mod_rewrite(node.obj, top_names, scope, mod_name)
        _mod_rewrite(node.value, top_names, scope, mod_name)
        return
    # Literals — nothing to do


def _mod_rewrite_stmt(s, top_names: set, scope: set, mod_name: str):
    _mod_rewrite(s, top_names, scope, mod_name)


def _rename_module(stmts: list, mod_name: str) -> list:
    top_names = _collect_top_names(stmts)
    for s in stmts:
        if isinstance(s, FnDecl):
            s.name = f"__mod_{mod_name}__{s.name}"
            scope = set(s.params)
            for body_stmt in s.body:
                _mod_rewrite_stmt(body_stmt, top_names, scope, mod_name)
        elif isinstance(s, Assignment):
            s.name = f"__mod_{mod_name}__{s.name}"
            _mod_rewrite(s.expr, top_names, set(), mod_name)
    return stmts


def _main_rewrite(node, mod_names: set, canonical_for: dict):
    """In the importing module, rewrite `mod.sym` DotExpr to a mangled
    Identifier. `canonical_for` maps each alias to the canonical name under
    which the module's symbols were actually renamed (identity for the common
    one-import-per-file case). Returns the (possibly new) node."""
    if isinstance(node, DotExpr):
        if isinstance(node.obj, Identifier) and node.obj.name in mod_names:
            canon = canonical_for.get(node.obj.name, node.obj.name)
            return Identifier(name=f"__mod_{canon}__{node.field}")
        node.obj = _main_rewrite(node.obj, mod_names, canonical_for)
        return node
    if isinstance(node, BinOp):
        node.left = _main_rewrite(node.left, mod_names, canonical_for)
        node.right = _main_rewrite(node.right, mod_names, canonical_for)
        return node
    if isinstance(node, UnaryOp):
        node.operand = _main_rewrite(node.operand, mod_names, canonical_for)
        return node
    if isinstance(node, CallExpr):
        node.callee = _main_rewrite(node.callee, mod_names, canonical_for)
        node.args = [_main_rewrite(a, mod_names, canonical_for) for a in node.args]
        return node
    if isinstance(node, IndexExpr):
        node.obj = _main_rewrite(node.obj, mod_names, canonical_for)
        node.index = _main_rewrite(node.index, mod_names, canonical_for)
        return node
    if isinstance(node, ArrayLit):
        node.elements = [_main_rewrite(e, mod_names, canonical_for) for e in node.elements]
        return node
    if isinstance(node, StructLit):
        node.fields = [(k, _main_rewrite(v, mod_names, canonical_for)) for (k, v) in node.fields]
        return node
    if isinstance(node, ArrowFn):
        if isinstance(node.body, list):
            for s in node.body:
                _main_rewrite_stmt(s, mod_names, canonical_for)
        else:
            node.body = _main_rewrite(node.body, mod_names, canonical_for)
        return node
    if isinstance(node, MatchExpr):
        node.scrutinee = _main_rewrite(node.scrutinee, mod_names, canonical_for)
        for arm in node.arms:
            arm.value = _main_rewrite(arm.value, mod_names, canonical_for)
        return node
    return node


def _main_rewrite_stmt(s, mod_names: set, canonical_for: dict):
    """Walk a statement, rewriting `mod.sym` DotExpr to mangled Identifiers.
    Most cases mutate in place and return the same `s`. Bare-expression
    statements (the parser stores them as the raw expression node) need to
    return a possibly-new node, so callers must replace the slot in their
    list with the return value."""
    if isinstance(s, Assignment):
        s.expr = _main_rewrite(s.expr, mod_names, canonical_for)
        return s
    if isinstance(s, ReturnStmt):
        if s.expr is not None:
            s.expr = _main_rewrite(s.expr, mod_names, canonical_for)
        return s
    if isinstance(s, PrintStmt):
        s.expr = _main_rewrite(s.expr, mod_names, canonical_for)
        return s
    if isinstance(s, MatchStmt):
        s.scrutinee = _main_rewrite(s.scrutinee, mod_names, canonical_for)
        for arm in s.arms:
            for i, sub in enumerate(arm.body):
                arm.body[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        return s
    if isinstance(s, IfStmt):
        s.condition = _main_rewrite(s.condition, mod_names, canonical_for)
        for i, sub in enumerate(s.then_body):
            s.then_body[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        if s.else_body:
            for i, sub in enumerate(s.else_body):
                s.else_body[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        return s
    if isinstance(s, WhileStmt):
        s.condition = _main_rewrite(s.condition, mod_names, canonical_for)
        for i, sub in enumerate(s.body):
            s.body[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        return s
    if isinstance(s, ForInStmt):
        s.iterable = _main_rewrite(s.iterable, mod_names, canonical_for)
        for i, sub in enumerate(s.body):
            s.body[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        return s
    if isinstance(s, FnDecl):
        for i, sub in enumerate(s.body):
            s.body[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        return s
    if isinstance(s, Block):
        for i, sub in enumerate(s.statements):
            s.statements[i] = _main_rewrite_stmt(sub, mod_names, canonical_for)
        return s
    if isinstance(s, IndexAssign):
        s.obj = _main_rewrite(s.obj, mod_names, canonical_for)
        s.index = _main_rewrite(s.index, mod_names, canonical_for)
        s.value = _main_rewrite(s.value, mod_names, canonical_for)
        return s
    if isinstance(s, DotAssign):
        s.obj = _main_rewrite(s.obj, mod_names, canonical_for)
        s.value = _main_rewrite(s.value, mod_names, canonical_for)
        return s
    # Fallback: the parser returns bare expression statements (e.g. `foo();`,
    # `T.helper();`) as the raw expression node, not wrapped in a Stmt class.
    # Walk through _main_rewrite so DotExprs in them still get mangled.
    return _main_rewrite(s, mod_names, canonical_for)


def _dirname(path: str) -> str:
    # Handle both Unix '/' and Windows '\\' so the resolver works on either
    # platform regardless of which separator the user typed.
    fwd = path.rfind("/")
    bwd = path.rfind("\\")
    i = max(fwd, bwd)
    return path[: i + 1] if i >= 0 else ""



# ── whole-program type + import resolution ──
_CONCRETE_KINDS = ("int", "float", "bool", "str", "array", "struct", "any")
_RESERVED_KINDS = {"", "_", "none", "fn", "union", "union_none"} | set(_CONCRETE_KINDS)


def _ast_children(node):
    """Yield (owner, field, value) for every dataclass field of an AST node,
    flattening lists; owner+field let a caller reassign in place."""
    import dataclasses
    if not dataclasses.is_dataclass(node):
        return
    for f in dataclasses.fields(node):
        yield node, f.name, getattr(node, f.name)


class _TypeView:
    """Per-file resolver: own type decls + destructured type imports +
    namespace-qualified lookups. Resolution is transitive with cycle
    detection; results memoized."""

    def __init__(self, label, own, imported, ns_tables, terrors, views, memo):
        self.label = label            # file path for diagnostics; doubles as file id
        self.own = own                # name -> TypeDecl
        self.imported = imported      # bound name -> (owner_label, TypeDecl)
        self.ns_tables = ns_tables    # namespace -> (owner_label, {name -> TypeDecl})
        self.terrors = terrors
        self.views = views            # shared: label -> _TypeView, filled as files bind
        self.memo = memo              # shared: (label, name) -> (kind, pfields) or poison

    def _decl_for(self, name):
        """Return (owner_label, decl) — the file whose view governs the
        declaration's own right-hand side."""
        if "." in name:
            ns, base = name.split(".", 1)
            ent = self.ns_tables.get(ns)
            if ent is None:
                return None, None
            owner, table = ent
            d = table.get(base)
            return (owner, d) if d is not None else (owner, None)
        if name in self.own:
            return self.label, self.own[name]
        if name in self.imported:
            return self.imported[name]
        return None, None

    def resolve_name(self, name, stack=None):
        """name -> (kind, pfields) or None (unknown; caller reports)."""
        key = (self.label, name)
        if key in self.memo:
            return self.memo[key]
        if stack is None:
            stack = []
        if key in [e[0] for e in stack]:
            _, flabel, fline, fcol, fname = stack[0]
            self.terrors.append(f"{flabel}:{fline}:{fcol}: circular type alias '{fname}'")
            self.memo[key] = ("", None)   # error recorded; don't cascade
            return self.memo[key]
        olabel, decl = self._decl_for(name)
        if decl is None:
            return None  # caller reports unknown at its own site
        k, pf = decl.rhs_kind, decl.rhs_pfields
        if k not in _RESERVED_KINDS:
            owner = self.views.get(olabel, self)
            sub = owner.resolve_name(
                k, stack + [(key, olabel, decl.line, decl.col, name)])
            if sub is None:
                self.terrors.append(
                    f"{olabel}:{decl.line}:{decl.col}: unknown type '{k}'")
                self.memo[key] = ("", None)   # error recorded; don't cascade
                return self.memo[key]
            k, pf = sub
        self.memo[key] = (k, pf)
        return (k, pf)

    def ann_kind(self, k):
        """Annotation kind head: reserved values pass through; names resolve
        then filter to the concrete set exactly as annotation parsing did."""
        if k in _RESERVED_KINDS:
            return k, True
        r = self.resolve_name(k)
        if r is None:
            return "", False
        rk = r[0]
        return (rk if rk in _CONCRETE_KINDS else ""), True


def _walk_types(node, view, seen):
    """Resolve every stored annotation/arm kind under node, in place."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, list):
        for x in node:
            _walk_types(x, view, seen)
        return
    if isinstance(node, Assignment):
        if node.type_kind not in _RESERVED_KINDS:
            nk, ok = view.ann_kind(node.type_kind)
            if not ok:
                view.terrors.append(
                    f"{view.label}:{node.line}:{node.col}: unknown type '{node.type_kind}'")
            node.type_kind = nk
    if isinstance(node, (FnDecl, ArrowFn)):
        if node.param_kinds:
            for i, pk in enumerate(node.param_kinds):
                if pk not in _RESERVED_KINDS:
                    nk, ok = view.ann_kind(pk)
                    if not ok:
                        view.terrors.append(f"{view.label}: unknown type '{pk}'")
                    node.param_kinds[i] = nk
        if node.ret_kind not in _RESERVED_KINDS:
            nk, ok = view.ann_kind(node.ret_kind)
            if not ok:
                view.terrors.append(f"{view.label}: unknown type '{node.ret_kind}'")
            node.ret_kind = nk
    if isinstance(node, (MatchArm, MatchExprArm)):
        if node.ptype_kind not in _RESERVED_KINDS                 and node.lit_kind is None:
            r = view.resolve_name(node.ptype_kind)
            if r is None:
                view.terrors.append(
                    f"{view.label}: unknown type '{node.ptype_kind}'")
                node.ptype_kind = ""
            else:
                node.ptype_kind, node.pfields = r[0], r[1]
    for _, _, v in _ast_children(node):
        _walk_types(v, view, seen)


def _binder_names(node):
    """Names a node introduces into its own scope (for shadow tracking)."""
    if isinstance(node, Assignment) and node.is_decl:
        return [node.name]
    if isinstance(node, (FnDecl, ArrowFn)):
        return list(node.params or [])
    if isinstance(node, (MatchArm, MatchExprArm)) and node.name:
        return [node.name]
    if isinstance(node, ForInStmt):
        return [node.var_name]
    return []


def _bind_rewrite(node, bindmap, scope, errors, label):
    """Rewrite bare Identifier reads of destructured imports to their
    mangled targets, respecting local shadowing; reject writes."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, list):
        for x in node:
            _bind_rewrite(x, bindmap, scope, errors, label)
        return
    if isinstance(node, Assignment):
        if node.name in bindmap and node.name not in scope and not node.is_decl:
            errors.append(f"{label}:{node.line}:{node.col}: "
                          f"import error: cannot assign to imported '{node.name}'")
    if isinstance(node, DotExpr):
        # Namespace heads stay; only recurse below them.
        _bind_rewrite(node.obj, bindmap, scope, errors, label)
        return
    inner = scope | set(_binder_names(node))
    for owner, fname, v in _ast_children(node):
        if isinstance(v, Identifier) and v.name in bindmap and v.name not in inner:
            setattr(owner, fname, Identifier(name=bindmap[v.name]))
        elif isinstance(v, list):
            sub = list(inner)
            block = []
            for i, x in enumerate(v):
                if isinstance(x, Identifier) and x.name in bindmap                         and x.name not in set(sub):
                    v[i] = Identifier(name=bindmap[x.name])
                else:
                    _bind_rewrite(x, bindmap, set(sub), errors, label)
                for b in _binder_names(x):
                    sub.append(b)
                block.append(x)
        else:
            _bind_rewrite(v, bindmap, inner, errors, label)


def _collect_ns_refs(node, acc, seen):
    """Collect (namespace, field) for every DotExpr whose head is a bare
    Identifier — validated against module export sets before mangling."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, list):
        for x in node:
            _collect_ns_refs(x, acc, seen)
        return
    if isinstance(node, DotExpr) and isinstance(node.obj, Identifier):
        acc.append((node.obj.name, node.field))
    for _, _, v in _ast_children(node):
        _collect_ns_refs(v, acc, seen)

def resolve_imports(stmts: list, main_path: str):
    """Two phases: load every reachable module, then resolve types and
    destructured imports, then emit dependencies-first (post-order of the
    import graph, back-edges skipped). Returns (stmts, errors, type_errors)."""
    errors = []
    terrors = []
    all_paths = []          # dedup
    path_canonical = {}     # path → first-seen namespace name
    canonical_for = {}      # alias → canonical (identity entry per name)
    mod_names = set()
    records = {}            # resolved path → module record
    work = []               # list of (relative_path, raw_path, name, importer_edges)

    # Search-path fallback: lang.py's own directory contains the std/ tree,
    # so an import that doesn't resolve relative to the importing file can
    # still find a stdlib module under <lang.py's dir>/<raw_path>.arrow.
    # Matches the compiler-side fallback (dirname(args(-1))).
    _STDLIB_DIR = _dirname(os.path.abspath(__file__))

    # Seed from main file's imports.
    main_dir = _dirname(main_path)
    filtered_main = []
    main_types = {}
    main_edges = []   # (resolved-or-primary path, ns, items, line, col)
    for s in stmts:
        if isinstance(s, ImportStmt):
            slot = [None, s.name, s.items or [], s.line, s.col]
            main_edges.append(slot)
            work.append((main_dir + s.path + ".arrow", s.path, s.name, slot))
            mod_names.add(s.name)
        elif isinstance(s, TypeDecl):
            main_types[s.name] = s
        else:
            filtered_main.append(s)

    while work:
        primary_path, raw_path, name, slot = work.pop()
        # Resolve: relative-to-importer first, then exe-dir fallback.
        resolved_path = primary_path
        sub_src = None
        if os.path.exists(primary_path):
            with open(primary_path, encoding="utf-8") as f:
                sub_src = f.read()
        else:
            fb_path = _STDLIB_DIR + raw_path + ".arrow"
            if os.path.exists(fb_path):
                resolved_path = fb_path
                with open(fb_path, encoding="utf-8") as f:
                    sub_src = f.read()
        if sub_src is None:
            errors.append(f"import: file not found: {primary_path}")
            continue
        slot[0] = resolved_path
        if resolved_path in all_paths:
            # Already loaded; possibly a different alias.
            canon = path_canonical[resolved_path]
            if name != canon:
                canonical_for[name] = canon
            continue
        all_paths.append(resolved_path)
        path_canonical[resolved_path] = name
        canonical_for[name] = name
        try:
            sub_tokens = Lexer(sub_src).tokenize()
            sub_parser = Parser(sub_tokens, src_file=resolved_path)
            sub_program = sub_parser.parse()
            sub_stmts = sub_program.statements
            errors.extend(sub_parser.errors)
            if sub_parser.errors:
                continue
        except (LexerError, ParseError) as e:
            errors.append(f"{resolved_path}: {e}")
            continue

        errors.extend(_validate_module(sub_stmts, resolved_path))

        sub_dir = _dirname(resolved_path)
        new_filtered = []
        sub_types = {}
        sub_edges = []
        for s in sub_stmts:
            if isinstance(s, ImportStmt):
                sslot = [None, s.name, s.items or [], s.line, s.col]
                sub_edges.append(sslot)
                work.append((sub_dir + s.path + ".arrow", s.path, s.name, sslot))
                mod_names.add(s.name)
            elif isinstance(s, TypeDecl):
                sub_types[s.name] = s
            else:
                new_filtered.append(s)
        records[resolved_path] = {
            "label": resolved_path, "canon": name, "stmts": new_filtered,
            "types": sub_types, "edges": sub_edges,
            "values": _collect_top_names(new_filtered),
        }

    # ── per-file resolution: type views, destructured bindings, checks ──
    def file_pass(label, stmts_list, types, edges, own_values):
        own_decls = set()
        for s in stmts_list:
            if isinstance(s, FnDecl):
                own_decls.add(s.name)
            elif isinstance(s, Assignment) and s.is_decl:
                own_decls.add(s.name)
        ns_tables = {}
        imported_types = {}
        bindmap = {}
        bind_src = {}
        for (rp, ns, items, iline, icol) in edges:
            if rp is None:
                continue
            rec = records.get(rp)
            if rec is None:
                continue
            canon = path_canonical.get(rp, ns)
            ns_tables[ns] = (rec["label"], rec["types"])
            for (orig, bound) in items:
                is_val = orig in rec["values"]
                is_typ = orig in rec["types"]
                if not is_val and not is_typ:
                    errors.append(f"{label}:{iline}:{icol}: import error: "
                                  f"'{orig}' not found in module {rec['label']}")
                    continue
                key = (rp, orig)
                if bound in bind_src and bind_src[bound] != key:
                    errors.append(f"{label}:{iline}:{icol}: import error: "
                                  f"'{bound}' already imported from {records[bind_src[bound][0]]['label']}")
                    continue
                if bound in own_decls or bound in types:
                    errors.append(f"{label}:{iline}:{icol}: import error: "
                                  f"'{bound}' collides with a declaration in this file")
                    continue
                bind_src[bound] = key
                if is_typ:
                    imported_types[bound] = (rec["label"], rec["types"][orig])
                if is_val:
                    bindmap[bound] = f"__mod_{canon}__{orig}"
        view = _TypeView(label, types, imported_types, ns_tables, terrors,
                         views_by_label, shared_memo)
        views_by_label[label] = view
        pending_walks.append((stmts_list, view))
        if bindmap:
            _bind_rewrite(stmts_list, bindmap, set(), errors, label)
        # Namespace member validation (values or types both count as found).
        refs = []
        _collect_ns_refs(stmts_list, refs, set())
        for (ns, field) in refs:
            ent = ns_tables.get(ns)
            if ent is None:
                continue
            rp = None
            for (erp, ens, _i, _l, _c) in edges:
                if ens == ns and erp is not None:
                    rp = erp
                    break
            rec = records.get(rp)
            if rec and field not in rec["values"] and field not in rec["types"]:
                errors.append(f"{label}: import error: "
                              f"'{field}' not found in module {rec['label']}")

    views_by_label = {}
    shared_memo = {}
    pending_walks = []
    for rp, rec in records.items():
        file_pass(rec["label"], rec["stmts"], rec["types"], rec["edges"],
                  rec["values"])
    file_pass(main_path, filtered_main, main_types, main_edges,
              _collect_top_names(filtered_main))
    for _wstmts, _wview in pending_walks:
        _walk_types(_wstmts, _wview, set())

    # ── emission: dependencies first, each module once, back-edges skipped ──
    for rp, rec in records.items():
        _rename_module(rec["stmts"], rec["canon"])
    accumulated = []
    emitted = set()
    visiting = set()

    def emit(rp):
        if rp in emitted or rp in visiting:
            return
        visiting.add(rp)
        rec = records[rp]
        for (dep, _ns, _i, _l, _c) in rec["edges"]:
            if dep is not None and dep in records:
                emit(dep)
        visiting.discard(rp)
        emitted.add(rp)
        accumulated.extend(rec["stmts"])

    for (dep, _ns, _i, _l, _c) in main_edges:
        if dep is not None and dep in records:
            emit(dep)
    combined = accumulated + filtered_main
    for i, s in enumerate(combined):
        combined[i] = _main_rewrite_stmt(s, mod_names, canonical_for)
    return combined, errors, terrors


# ─────────────────────────────────────────────
#  RUN / REPL / MAIN
# ─────────────────────────────────────────────
def run_source(source: str, src_file: str = "<input>") -> Interpreter:
    tokens = Lexer(source).tokenize()
    parser = Parser(tokens, src_file=src_file)
    program = parser.parse()
    if parser.errors:
        for e in parser.errors:
            print(e)
        print("--")
        print(f"{len(parser.errors)} parse error(s). Compilation aborted.")
        sys.exit(1)
    interp = Interpreter()
    interp.run(program)
    return interp


def _paths_return(body) -> bool:
    """True when every path through this statement list returns a value.
    Conservative: while/for never guarantee; match guarantees only with
    a catch-all arm; if needs both branches."""
    for st in body:
        if isinstance(st, ReturnStmt):
            return st.expr is not None and st.expr is not False
        if isinstance(st, IfStmt):
            if st.else_body and _paths_return(st.then_body) \
                    and _paths_return(st.else_body):
                return True
        if isinstance(st, MatchStmt):
            arms = st.arms
            if arms and all(_paths_return(a.body) for a in arms) \
                    and any(getattr(a, "ptype_kind", "") == "_"
                            and not getattr(a, "lit_kind", None)
                            for a in arms):
                return True
    return False


def _check_totality(stmts) -> list:
    """Collect 1.10 violations over every FnDecl / block-bodied ArrowFn."""
    errors = []

    def scan_fn(name, body, ret_kind):
        has_val, has_bare = False, False
        def walk(n):
            nonlocal has_val, has_bare
            if n is None or isinstance(n, (str, int, float, bool)):
                return
            if isinstance(n, list):
                for x in n:
                    walk(x)
                return
            if isinstance(n, ReturnStmt):
                if n.expr is not None and n.expr is not False:
                    has_val = True
                else:
                    has_bare = True
                return
            if isinstance(n, (FnDecl, ArrowFn)):
                return
            for v in getattr(n, "__dict__", {}).values():
                walk(v)
        for s in body:
            walk(s)
        if ret_kind == "union_none":
            return    # declared nullable: mixed paths are legal
        if has_val and (has_bare or not _paths_return(body)):
            errors.append(
                f"'{name}' returns a value on some paths but "
                f"none on others; annotate ': T | none' or "
                f"return on every path")

    def find(n):
        if n is None or isinstance(n, (str, int, float, bool)):
            return
        if isinstance(n, list):
            for x in n:
                find(x)
            return
        if isinstance(n, FnDecl):
            scan_fn(n.name, n.body, n.ret_kind)
            find(n.body)
            return
        if isinstance(n, ArrowFn):
            if isinstance(n.body, list):
                scan_fn("<arrow fn>", n.body,
                        getattr(n, "ret_kind", ""))
                find(n.body)
            return
        for v in getattr(n, "__dict__", {}).values():
            find(v)

    find(stmts)
    return errors


def run_file(filepath: str) -> Interpreter:
    """Like run_source but with import resolution rooted at the given file."""
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    tokens = Lexer(source).tokenize()
    parser = Parser(tokens, src_file=filepath)
    program = parser.parse()
    # Parse errors take priority — type errors and import errors on a
    # malformed AST aren't useful, so report parse issues and stop.
    if parser.errors:
        for e in parser.errors:
            print(e)
        print("--")
        print(f"{len(parser.errors)} parse error(s). Compilation aborted.")
        sys.exit(1)
    resolved, errs, resolve_terrs = resolve_imports(program.statements, filepath)
    # `errs` may contain parser errors from imported files; classify them.
    parse_errs = [e for e in errs if ": parse error: " in e]
    other_errs = [e for e in errs if ": parse error: " not in e]
    if parse_errs:
        for e in parse_errs:
            print(e)
        print("--")
        print(f"{len(parse_errs)} parse error(s). Compilation aborted.")
        sys.exit(1)
    if other_errs:
        for e in other_errs:
            print(e)
        print("--")
        print(f"{len(other_errs)} import error(s). Compilation aborted.")
        sys.exit(1)
    type_errs = resolve_terrs + _check_totality(resolved)
    if type_errs:
        for e in type_errs:
            print(e)
        print("--")
        print(f"{len(type_errs)} type error(s). Compilation aborted.")
        sys.exit(1)
    interp = Interpreter()
    interp.run(Program(resolved))
    return interp


def repl():
    print("Arrow Lang v0.4 — Type 'exit' to quit")
    print("─" * 40)
    interp = Interpreter()
    while True:
        try:
            line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if line.strip() in ("exit", "quit"):
            print("Bye!"); break
        if not line.strip(): continue
        try:
            tokens = Lexer(line).tokenize()
            program = Parser(tokens).parse()
            interp.run(program)
        except (LexerError, ParseError, RuntimeError_) as e:
            print(f"Error: {e}")


def main():
    sys.setrecursionlimit(10000)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            run_file(filepath)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            sys.exit(1)
        except (LexerError, ParseError, RuntimeError_) as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        repl()


if __name__ == "__main__":
    main()