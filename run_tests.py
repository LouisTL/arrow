#!/usr/bin/env python3
"""
run_tests.py — Runs every .arrow file in examples/ and verifies behaviour
against per-file annotations.

Annotation format (anywhere in the file as `// ...` comments):

    // EXPECT: ok | type_fail | scope_fail | parse_fail
    //OUT: <line of expected stdout>     (zero or more, in source order)
    //ERR: <substring required in compiler diagnostic>  (zero or more)

Rules per EXPECT category:

  ok          (or no annotation, back-compat)
              interpreter must succeed; native compile + run must succeed.
              If any //OUT: lines exist, both interp and native stdout must
              equal that expected text. With no //OUT:, only the interp-vs-
              native check fires (flagged "weak" in the report — that means
              "they agree but we never checked they're right").

  type_fail   native compile must report "type error" and abort.
  scope_fail  native compile must report "scope error" and abort.
  parse_fail  native compile must report "parse error" and abort.

  For all *_fail categories: every //ERR: line must appear as a substring
  of the compiler's diagnostic output (its stdout). The interpreter side
  is not consulted — error categories are about the compile-time checker.

Usage:
    python run_tests.py                 # runs all examples
    python run_tests.py pattern         # runs examples matching pattern
    python run_tests.py -v              # show full output on every test

Exit code: 0 if all pass, 1 if any fail.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent.resolve()
EXAMPLES = HERE / "examples"
LANG = HERE / "lang.py"
COMPILER = HERE / "compiler.arrow"

INTERACTIVE_KEYWORDS = ["input("]


def parse_header(src: str) -> dict:
    """Pull EXPECT / OUT / ERR annotations out of the source. Annotations
    can live anywhere in the file as `//` comments; we just scan every
    line. Returns a dict with `expect`, `output`, `contains`."""
    expect = None
    out_lines = []
    contains = []
    for line in src.splitlines():
        s = line.lstrip()
        if not s.startswith("//"):
            continue
        m = re.match(r"//\s*EXPECT:\s*(\S+)", s)
        if m:
            expect = m.group(1)
            continue
        if s.startswith("//OUT:"):
            content = s[len("//OUT:"):]
            if content.startswith(" "):
                content = content[1:]
            out_lines.append(content)
            continue
        if s.startswith("//ERR:"):
            contains.append(s[len("//ERR:"):].strip())
            continue
    output = ("\n".join(out_lines) + "\n") if out_lines else None
    return {"expect": expect, "output": output, "contains": contains}


def run_interp(example: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(LANG), str(example)],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_compile_only(example: Path) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "a.exe"
        proc = subprocess.run(
            [sys.executable, str(LANG), str(COMPILER), str(example),
             "-o", str(exe)],
            capture_output=True, text=True, timeout=180,
        )
        return proc.returncode, proc.stdout, proc.stderr


def run_compile_and_native(example: Path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / "a.exe"
        compile_proc = subprocess.run(
            [sys.executable, str(LANG), str(COMPILER), str(example),
             "-o", str(exe)],
            capture_output=True, text=True, timeout=180,
        )
        if compile_proc.returncode != 0:
            return (compile_proc.returncode, compile_proc.stdout,
                    compile_proc.stderr or "(compile failed)",
                    compile_proc.stdout)
        real_exe = exe if exe.exists() else tmp / "a"
        if not real_exe.exists():
            exes = list(tmp.glob("*.exe")) or list(tmp.glob("a*"))
            real_exe = exes[0] if exes else exe
        if not real_exe.exists():
            return (99, compile_proc.stdout,
                    f"(no output binary; compile output: {compile_proc.stdout})",
                    compile_proc.stdout)
        run_proc = subprocess.run(
            [str(real_exe)], capture_output=True, text=True, timeout=60,
        )
        return (run_proc.returncode, run_proc.stdout, run_proc.stderr,
                compile_proc.stdout)


def is_interactive(example: Path) -> bool:
    src = example.read_text(errors="replace")
    return any(kw in src for kw in INTERACTIVE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────
#  Dispatchers per EXPECT category
# ─────────────────────────────────────────────────────────────────────

def check_ok(example: Path, header: dict, verbose: bool):
    try:
        ic, iout, ierr = run_interp(example)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT (interp)", "")
    if ic != 0:
        last = (ierr or iout).strip().splitlines()
        return ("interp ERROR", last[-1] if last else "")

    if is_interactive(example):
        return ("skipped (interactive)", "")

    try:
        nc, nout, nerr, _ = run_compile_and_native(example)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT (native)", "")
    if nc != 0:
        last = (nerr or nout).strip().splitlines()
        return ("native ERROR", last[-1] if last else "")

    expected = header["output"]
    if expected is not None:
        if iout != expected:
            return ("MISMATCH (interp != OUT)", brief_diff(expected, iout))
        if nout != expected:
            return ("MISMATCH (native != OUT)", brief_diff(expected, nout))
        return ("ok", "")
    if iout != nout:
        return ("MISMATCH (interp != native)", f"interp:{iout!r} vs native:{nout!r}")
    return ("weak (no //OUT:)", "")


def check_fail(example: Path, header: dict, error_kind: str, verbose: bool):
    try:
        rc, stdout, stderr = run_compile_only(example)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT (compile)", "")
    marker = f"{error_kind} error"
    if marker not in stdout:
        return (f"UNEXPECTED ({error_kind})",
                f"compiler output didn't mention {marker!r}: "
                f"{stdout.strip().splitlines()[-3:]}")
    if "Compilation aborted" not in stdout:
        return (f"UNEXPECTED ({error_kind})",
                f"compiler didn't abort: {stdout.strip().splitlines()[-3:]}")
    missing = [c for c in header["contains"] if c not in stdout]
    if missing:
        return ("CONTAINS missing", f"required substrings absent: {missing}")
    return (f"{error_kind} fail", "")


def brief_diff(expected: str, actual: str, max_lines: int = 4) -> str:
    el, al = expected.splitlines(), actual.splitlines()
    n = max(len(el), len(al))
    diffs = []
    for i in range(n):
        e = el[i] if i < len(el) else "<EOF>"
        a = al[i] if i < len(al) else "<EOF>"
        if e != a:
            diffs.append(f"line {i+1}: want {e!r} got {a!r}")
            if len(diffs) >= max_lines:
                break
    return "; ".join(diffs) if diffs else f"length differs ({len(el)} vs {len(al)})"


CATEGORY_DISPATCH = {
    "ok":          lambda ex, h, v: check_ok(ex, h, v),
    "type_fail":   lambda ex, h, v: check_fail(ex, h, "type", v),
    "scope_fail":  lambda ex, h, v: check_fail(ex, h, "scope", v),
    "parse_fail":  lambda ex, h, v: check_fail(ex, h, "parse", v),
}

PASS_STATUSES = {
    "ok", "type fail", "scope fail", "parse fail",
    "skipped (interactive)", "weak (no //OUT:)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default="",
                    help="substring filter for example filenames")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show full output on every test")
    args = ap.parse_args()

    examples = sorted([p for p in EXAMPLES.glob("*.arrow")
                       if args.pattern in p.name])
    if not examples:
        print(f"no examples matching {args.pattern!r}")
        return 1

    results = []
    for ex in examples:
        src = ex.read_text(errors="replace")
        header = parse_header(src)
        expect = header["expect"] or "ok"
        dispatch = CATEGORY_DISPATCH.get(expect)
        if dispatch is None:
            results.append((ex.name, f"UNKNOWN EXPECT: {expect}", ""))
            continue
        status, note = dispatch(ex, header, args.verbose)
        results.append((ex.name, status, note))

    passed = failed = weak = 0
    for name, status, note in results:
        ok = status in PASS_STATUSES
        mark = "OK " if ok else "!! "
        if ok:
            passed += 1
            if status == "weak (no //OUT:)":
                weak += 1
        else:
            failed += 1
        line = f"{mark}{name:32s}  {status}"
        if note:
            line += f"   {note}"
        print(line)

    summary = f"\n{passed} passed, {failed} failed"
    if weak:
        summary += f" ({weak} weak — no //OUT: block, only cross-checked interp vs native)"
    print(summary)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())