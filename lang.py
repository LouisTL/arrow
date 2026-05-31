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

@dataclass
class PrintStmt:
    expr: Any

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

@dataclass
class ArrowFn:
    params: list[str]
    body: Any

@dataclass
class ReturnStmt:
    expr: Any

@dataclass
class CallExpr:
    callee: Any
    args: list

@dataclass
class ImportStmt:
    """`import "path";` — namespace name defaults to basename of path."""
    path: str
    name: str


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
        while self._match(TokenType.PIPE):
            self._skip_type_ann()

    def _assignment(self, typed: bool = False) -> Assignment:
        ident_tok = self._eat(TokenType.IDENT)
        name = ident_tok.value
        if typed:
            # Skip ': type' — interpreter ignores type annotations
            self._eat(TokenType.COLON)
            self._skip_type_ann()
        self._eat(TokenType.ARROW)
        expr = self._expression()
        self._eat(TokenType.SEMI)
        return Assignment(name, expr, is_decl=False, line=ident_tok.line, col=ident_tok.col)

    def _var_decl(self) -> Assignment:
        var_tok = self._eat(TokenType.VAR)
        ident_tok = self._eat(TokenType.IDENT)
        name = ident_tok.value
        # Optional type annotation: var x: int <- expr;
        if self._current().type == TokenType.COLON:
            self._eat(TokenType.COLON)
            self._skip_type_ann()
        self._eat(TokenType.ARROW)
        expr = self._expression()
        self._eat(TokenType.SEMI)
        # Track at the `var` keyword position so error messages point there.
        return Assignment(name, expr, is_decl=True, line=var_tok.line, col=var_tok.col)

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
        self._eat(TokenType.SEMI)
        return ImportStmt(path=path, name=base)

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
        params = self._param_list()
        # Optional return type annotation: fn f(...) : type { ... }
        if self._current().type == TokenType.COLON:
            self._eat(TokenType.COLON)
            self._skip_type_ann()
        body = self._block()
        return FnDecl(name, params, body.statements)

    def _param_list(self) -> list[str]:
        self._eat(TokenType.LPAREN)
        params = []
        if self._current().type != TokenType.RPAREN:
            params.append(self._eat(TokenType.IDENT).value)
            # Optional per-param type annotation: fn f(x: type, y: type)
            if self._current().type == TokenType.COLON:
                self._eat(TokenType.COLON)
                self._skip_type_ann()
            while self._match(TokenType.COMMA):
                params.append(self._eat(TokenType.IDENT).value)
                if self._current().type == TokenType.COLON:
                    self._eat(TokenType.COLON)
                    self._skip_type_ann()
        self._eat(TokenType.RPAREN)
        return params

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
        params = self._param_list()
        # Optional return type annotation: skip it (interpreter ignores types)
        if self._current().type == TokenType.COLON:
            self._eat(TokenType.COLON)
            self._skip_type_ann()
        self._eat(TokenType.FAT_ARROW)
        if self._current().type == TokenType.LBRACE and not self._is_struct_literal():
            body = self._block()
            return ArrowFn(params, body.statements)
        else:
            expr = self._expression()
            return ArrowFn(params, expr)

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


class Environment:
    def __init__(self, parent: 'Environment | None' = None, is_fn_root: bool = False):
        self.vars: dict[str, Any] = {}
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

    def declare(self, name: str, value: Any):
        # Always create a fresh binding in the *current* scope. Error if a
        # name with the same identifier already exists in this same scope
        # (catches accidental redeclaration). Outer-scope bindings of the
        # same name are shadowed, not errored — that's intentional.
        if name in self.vars:
            raise RuntimeError_(f"redeclaration of '{name}' in the same scope")
        self.vars[name] = value

    def assign(self, name: str, value: Any) -> bool:
        # Walk all the way up to find an existing binding. No function
        # boundary — a closure that writes a bare `name <- ...` to a
        # captured outer-fn local mutates the actual cell, not a snapshot.
        # Globals are the topmost env in the chain, so this also handles
        # function-writes-to-global without a separate fallback.
        env = self
        while env is not None:
            if name in env.vars:
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
    def __init__(self, name: str, params: list[str], body: Any, closure: Environment):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure

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
            "exec_cmd", "args"}


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
                    # `var x <- expr;` — fresh binding in current scope.
                    # Redeclaration in the same scope is an error.
                    try:
                        self.env.declare(name, val)
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
                    raise RuntimeError_("Cannot index into non-array value")
                if not isinstance(idx, int):
                    raise RuntimeError_("Array index must be an integer")
                if idx < 0 or idx >= len(target):
                    raise RuntimeError_(f"Index {idx} out of bounds (length {len(target)})")
                target[idx] = val

            case DotAssign(obj, field, value):
                target = self._eval(obj)
                val = self._eval(value)
                if not isinstance(target, Struct):
                    raise RuntimeError_("Cannot set field on non-struct value")
                target.set(field, val)

            case PrintStmt(expr):
                val = self._eval(expr)
                text = self._format(val)
                print(text)
                self.output.append(text)

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
                self.env.set(name, Function(name, params, body, self.env))

            case ReturnStmt(expr):
                raise ReturnSignal(self._eval(expr) if expr is not None else None)

            case _:
                self._eval(node)

    def _eval(self, node) -> Any:
        match node:
            case NumberLit(v): return v
            case StringLit(v): return v
            case BoolLit(v): return v
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
                        raise RuntimeError_(f"Index {idx} out of bounds")
                    return target[idx]
                elif isinstance(target, str):
                    if not isinstance(idx, int):
                        raise RuntimeError_("String index must be an integer")
                    if idx < 0 or idx >= len(target):
                        raise RuntimeError_(f"Index {idx} out of bounds")
                    return target[idx]
                else:
                    raise RuntimeError_("Cannot index into this type")

            case DotExpr(obj, field):
                target = self._eval(obj)
                if isinstance(target, Struct):
                    return target.get(field)
                raise RuntimeError_(f"Cannot access field '{field}' on non-struct value")

            case UnaryOp('-', operand): return -self._eval(operand)
            case UnaryOp('!', operand): return not self._truthy(self._eval(operand))

            case BinOp(op, left, right):
                return self._eval_binop(op, left, right)

            case ArrowFn(params, body):
                return Function("<arrow>", params, body, self.env)

            case CallExpr(callee, args):
                return self._eval_call(callee, args)

            case _:
                raise RuntimeError_(f"Cannot evaluate node: {node}")

    def _eval_call(self, callee, args) -> Any:
        if isinstance(callee, Identifier) and callee.name in BUILTINS:
            return self._eval_builtin(callee.name, args)

        fn = self._eval(callee)
        if not isinstance(fn, Function):
            raise RuntimeError_(f"'{fn}' is not a function")

        arg_vals = [self._eval(a) for a in args]
        if len(arg_vals) != len(fn.params):
            raise RuntimeError_(
                f"Function {fn.name} expects {len(fn.params)} args, got {len(arg_vals)}")

        call_env = Environment(parent=fn.closure, is_fn_root=True)
        for param, val in zip(fn.params, arg_vals):
            # `declare`, not `set` — the param shouldn't accidentally mutate
            # a same-named binding in the enclosing closure scope.
            call_env.declare(param, val)

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
            # Return "" on missing file to match the native compiler's
            # behavior — Arrow programs that probe for optional files (the
            # resolver's search-path fallback, for one) rely on the empty
            # result as a signal rather than an exception.
            try:
                with open(path, encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return ""
            except Exception as e:
                raise RuntimeError_(f"Error reading file: {e}")

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
                with open(path, "w", encoding="utf-8") as f:
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
                with open(path, "a", encoding="utf-8") as f:
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
                raise RuntimeError_(f"char_at() index {idx} out of bounds")
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
        if not isinstance(s, (FnDecl, Assignment, ImportStmt)):
            errors.append(f"{path}: module may only contain fn declarations, assignments, or imports (got {type(s).__name__})")
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


def resolve_imports(stmts: list, main_path: str) -> tuple[list, list[str]]:
    """Iterative driver mirroring compiler.arrow's __main_logic loop.
    Returns (resolved_stmts, errors)."""
    errors = []
    all_paths = []          # dedup
    path_canonical = {}     # path → first-seen namespace name
    canonical_for = {}      # alias → canonical (identity entry per name)
    mod_names = set()
    work = []               # list of (relative_path, raw_path, name)

    # Search-path fallback: lang.py's own directory contains the std/ tree,
    # so an import that doesn't resolve relative to the importing file can
    # still find a stdlib module under <lang.py's dir>/<raw_path>.arrow.
    # Matches the compiler-side fallback (dirname(args(-1))).
    _STDLIB_DIR = _dirname(os.path.abspath(__file__))

    # Seed from main file's imports.
    main_dir = _dirname(main_path)
    filtered_main = []
    for s in stmts:
        if isinstance(s, ImportStmt):
            work.append((main_dir + s.path + ".arrow", s.path, s.name))
            mod_names.add(s.name)
        else:
            filtered_main.append(s)

    accumulated = []
    while work:
        primary_path, raw_path, name = work.pop()
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
        for s in sub_stmts:
            if isinstance(s, ImportStmt):
                work.append((sub_dir + s.path + ".arrow", s.path, s.name))
                mod_names.add(s.name)
            else:
                new_filtered.append(s)
        _rename_module(new_filtered, name)
        accumulated.extend(new_filtered)

    combined = accumulated + filtered_main
    for i, s in enumerate(combined):
        combined[i] = _main_rewrite_stmt(s, mod_names, canonical_for)
    return combined, errors


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
    resolved, errs = resolve_imports(program.statements, filepath)
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