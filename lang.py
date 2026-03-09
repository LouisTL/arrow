#!/usr/bin/env python3
"""
Arrow Lang — A small programming language with C-like syntax and <- assignment.

Example:
    x <- 10;
    y <- x + 5;
    print(y);
"""

import sys
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────
#  TOKENS
# ─────────────────────────────────────────────
class TokenType(Enum):
    # Literals & identifiers
    NUMBER   = auto()
    STRING   = auto()
    IDENT    = auto()
    BOOL     = auto()

    # Operators
    PLUS     = auto()  # +
    MINUS    = auto()  # -
    STAR     = auto()  # *
    SLASH    = auto()  # /
    PERCENT  = auto()  # %
    ARROW    = auto()  # <-
    EQ       = auto()  # =
    NEQ      = auto()  # !=
    LT       = auto()  # <
    GT       = auto()  # >
    LTE      = auto()  # <=
    GTE      = auto()  # >=
    AND      = auto()  # &&
    OR       = auto()  # ||
    NOT      = auto()  # !

    # Delimiters
    LPAREN   = auto()  # (
    RPAREN   = auto()  # )
    LBRACE   = auto()  # {
    RBRACE   = auto()  # }
    SEMI     = auto()  # ;
    COMMA    = auto()  # ,

    # Keywords
    IF       = auto()
    ELSE     = auto()
    WHILE    = auto()
    PRINT    = auto()

    EOF      = auto()


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
    "if":    TokenType.IF,
    "else":  TokenType.ELSE,
    "while": TokenType.WHILE,
    "print": TokenType.PRINT,
    "true":  TokenType.BOOL,
    "false": TokenType.BOOL,
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
                # Line comment
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

            # Numbers
            if ch.isdigit():
                tokens.append(self._read_number())

            # Strings
            elif ch == '"':
                tokens.append(self._read_string())

            # Identifiers / keywords
            elif ch.isalpha() or ch == '_':
                tokens.append(self._read_ident())

            # Two-character operators
            elif ch == '<' and self._peek(1) == '-':
                self._advance(); self._advance()
                tokens.append(Token(TokenType.ARROW, '<-', line, col))
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

            # Single-character tokens
            elif ch == '+': self._advance(); tokens.append(Token(TokenType.PLUS,   '+', line, col))
            elif ch == '-': self._advance(); tokens.append(Token(TokenType.MINUS,  '-', line, col))
            elif ch == '*': self._advance(); tokens.append(Token(TokenType.STAR,   '*', line, col))
            elif ch == '/': self._advance(); tokens.append(Token(TokenType.SLASH,  '/', line, col))
            elif ch == '%': self._advance(); tokens.append(Token(TokenType.PERCENT,'%', line, col))
            elif ch == '<': self._advance(); tokens.append(Token(TokenType.LT,     '<', line, col))
            elif ch == '>': self._advance(); tokens.append(Token(TokenType.GT,     '>', line, col))
            elif ch == '!': self._advance(); tokens.append(Token(TokenType.NOT,    '!', line, col))
            elif ch == '=': self._advance(); tokens.append(Token(TokenType.EQ,     '=', line, col))
            elif ch == '(': self._advance(); tokens.append(Token(TokenType.LPAREN, '(', line, col))
            elif ch == ')': self._advance(); tokens.append(Token(TokenType.RPAREN, ')', line, col))
            elif ch == '{': self._advance(); tokens.append(Token(TokenType.LBRACE, '{', line, col))
            elif ch == '}': self._advance(); tokens.append(Token(TokenType.RBRACE, '}', line, col))
            elif ch == ';': self._advance(); tokens.append(Token(TokenType.SEMI,   ';', line, col))
            elif ch == ',': self._advance(); tokens.append(Token(TokenType.COMMA,  ',', line, col))

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
        self._advance()  # skip opening "
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
        self._advance()  # skip closing "
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
class Block:
    statements: list

@dataclass
class Program:
    statements: list


# ─────────────────────────────────────────────
#  PARSER
# ─────────────────────────────────────────────
class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, tt: TokenType) -> Token:
        tok = self._current()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r}) "
                f"at line {tok.line}, col {tok.col}"
            )
        self.pos += 1
        return tok

    def _match(self, *types: TokenType) -> Token | None:
        if self._current().type in types:
            tok = self._current()
            self.pos += 1
            return tok
        return None

    # ── Grammar ──────────────────────────────

    def parse(self) -> Program:
        stmts = []
        while self._current().type != TokenType.EOF:
            stmts.append(self._statement())
        return Program(stmts)

    def _statement(self):
        tok = self._current()

        if tok.type == TokenType.IF:
            return self._if_stmt()
        if tok.type == TokenType.WHILE:
            return self._while_stmt()
        if tok.type == TokenType.PRINT:
            return self._print_stmt()
        if tok.type == TokenType.LBRACE:
            return self._block()
        if tok.type == TokenType.IDENT:
            # Look ahead for <-
            if self.tokens[self.pos + 1].type == TokenType.ARROW:
                return self._assignment()

        # Expression statement
        expr = self._expression()
        self._eat(TokenType.SEMI)
        return expr

    def _block(self) -> Block:
        self._eat(TokenType.LBRACE)
        stmts = []
        while self._current().type != TokenType.RBRACE:
            stmts.append(self._statement())
        self._eat(TokenType.RBRACE)
        return Block(stmts)

    def _assignment(self) -> Assignment:
        name = self._eat(TokenType.IDENT).value
        self._eat(TokenType.ARROW)
        expr = self._expression()
        self._eat(TokenType.SEMI)
        return Assignment(name, expr)

    def _print_stmt(self) -> PrintStmt:
        self._eat(TokenType.PRINT)
        self._eat(TokenType.LPAREN)
        expr = self._expression()
        self._eat(TokenType.RPAREN)
        self._eat(TokenType.SEMI)
        return PrintStmt(expr)

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

    # ── Expressions (precedence climbing) ────

    def _expression(self):
        return self._or_expr()

    def _or_expr(self):
        left = self._and_expr()
        while self._match(TokenType.OR):
            right = self._and_expr()
            left = BinOp('||', left, right)
        return left

    def _and_expr(self):
        left = self._equality()
        while self._match(TokenType.AND):
            right = self._equality()
            left = BinOp('&&', left, right)
        return left

    def _equality(self):
        left = self._comparison()
        while tok := self._match(TokenType.EQ, TokenType.NEQ):
            right = self._comparison()
            left = BinOp(tok.value, left, right)
        return left

    def _comparison(self):
        left = self._addition()
        while tok := self._match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            right = self._addition()
            left = BinOp(tok.value, left, right)
        return left

    def _addition(self):
        left = self._multiplication()
        while tok := self._match(TokenType.PLUS, TokenType.MINUS):
            right = self._multiplication()
            left = BinOp(tok.value, left, right)
        return left

    def _multiplication(self):
        left = self._unary()
        while tok := self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            right = self._unary()
            left = BinOp(tok.value, left, right)
        return left

    def _unary(self):
        if tok := self._match(TokenType.MINUS):
            return UnaryOp('-', self._unary())
        if tok := self._match(TokenType.NOT):
            return UnaryOp('!', self._unary())
        return self._primary()

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

        if tok.type == TokenType.LPAREN:
            self._advance_paren()
            expr = self._expression()
            self._eat(TokenType.RPAREN)
            return expr

        raise ParseError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) "
            f"at line {tok.line}, col {tok.col}"
        )

    def _advance_paren(self):
        self._eat(TokenType.LPAREN)


# ─────────────────────────────────────────────
#  INTERPRETER
# ─────────────────────────────────────────────
class RuntimeError_(Exception):
    pass


class Environment:
    def __init__(self):
        self.vars: dict[str, Any] = {}

    def get(self, name: str):
        if name not in self.vars:
            raise RuntimeError_(f"Undefined variable '{name}'")
        return self.vars[name]

    def set(self, name: str, value: Any):
        self.vars[name] = value


class Interpreter:
    def __init__(self):
        self.env = Environment()
        self.output: list[str] = []  # captured output for testing

    def run(self, program: Program):
        for stmt in program.statements:
            self._exec(stmt)

    def _exec(self, node):
        match node:
            case Assignment(name, expr):
                self.env.set(name, self._eval(expr))

            case PrintStmt(expr):
                val = self._eval(expr)
                text = self._format(val)
                print(text)
                self.output.append(text)

            case IfStmt(cond, then_body, else_body):
                if self._truthy(self._eval(cond)):
                    for s in then_body:
                        self._exec(s)
                elif else_body:
                    for s in else_body:
                        self._exec(s)

            case WhileStmt(cond, body):
                iterations = 0
                while self._truthy(self._eval(cond)):
                    for s in body:
                        self._exec(s)
                    iterations += 1
                    if iterations > 100_000:
                        raise RuntimeError_("Infinite loop detected (>100,000 iterations)")

            case Block(stmts):
                for s in stmts:
                    self._exec(s)

            case _:
                # Expression statement — evaluate and discard
                self._eval(node)

    def _eval(self, node) -> Any:
        match node:
            case NumberLit(v):
                return v
            case StringLit(v):
                return v
            case BoolLit(v):
                return v
            case Identifier(name):
                return self.env.get(name)

            case UnaryOp('-', operand):
                return -self._eval(operand)
            case UnaryOp('!', operand):
                return not self._truthy(self._eval(operand))

            case BinOp(op, left, right):
                return self._eval_binop(op, left, right)

            case _:
                raise RuntimeError_(f"Cannot evaluate node: {node}")

    def _eval_binop(self, op: str, left, right) -> Any:
        lv = self._eval(left)
        rv = self._eval(right)

        match op:
            case '+':
                if isinstance(lv, str) or isinstance(rv, str):
                    return self._format(lv) + self._format(rv)
                return lv + rv
            case '-':  return lv - rv
            case '*':  return lv * rv
            case '/':
                if rv == 0:
                    raise RuntimeError_("Division by zero")
                return lv / rv
            case '%':
                if rv == 0:
                    raise RuntimeError_("Modulo by zero")
                return lv % rv
            case '<':  return lv < rv
            case '>':  return lv > rv
            case '<=': return lv <= rv
            case '>=': return lv >= rv
            case '=':  return lv == rv
            case '!=': return lv != rv
            case '&&': return self._truthy(lv) and self._truthy(rv)
            case '||': return self._truthy(lv) or self._truthy(rv)
            case _:
                raise RuntimeError_(f"Unknown operator: {op}")

    def _truthy(self, val) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        if isinstance(val, str):
            return len(val) > 0
        return val is not None

    def _format(self, val) -> str:
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)


# ─────────────────────────────────────────────
#  RUN HELPER
# ─────────────────────────────────────────────
def run_source(source: str) -> Interpreter:
    """Lex, parse, and interpret a source string. Returns the interpreter."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    interp = Interpreter()
    interp.run(program)
    return interp


# ─────────────────────────────────────────────
#  REPL & FILE RUNNER
# ─────────────────────────────────────────────
def repl():
    print("Arrow Lang v0.1 — Type 'exit' to quit")
    print("─" * 40)
    interp = Interpreter()
    while True:
        try:
            line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if line.strip() in ("exit", "quit"):
            print("Bye!")
            break
        if not line.strip():
            continue
        try:
            tokens = Lexer(line).tokenize()
            program = Parser(tokens).parse()
            interp.run(program)
        except (LexerError, ParseError, RuntimeError_) as e:
            print(f"Error: {e}")


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            with open(filepath) as f:
                source = f.read()
            run_source(source)
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