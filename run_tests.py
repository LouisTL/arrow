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

  runtime_fail
              native compile must succeed; then BOTH the interpreter and
              the native binary must exit 1 with byte-identical output.
              //OUT: lines are the expected pre-trap stdout (matched as a
              prefix); //ERR: lines are substrings required in the output
              (the trap message).

  For all *_fail categories: every //ERR: line must appear as a substring
  of the compiler's diagnostic output (its stdout). The interpreter side
  is not consulted — error categories are about the compile-time checker.

Usage:
    python run_tests.py                 # runs all examples
    python run_tests.py pattern         # runs examples matching pattern
    python run_tests.py -v              # show full output on every test
    python run_tests.py --native-compiler ./arrow2
                                        # additive: every test ALSO goes
                                        # through the given native compiler
                                        # binary, cross-checked against the
                                        # interp-hosted oracle

Additive --native-compiler mode: the oracle path (lang.py hosting
compiler.arrow) runs exactly as described above; per test the runner
additionally compiles with the native binary and requires (a) the emitted
.ll byte-identical to the oracle's on compile-clean categories, (b)
compiler diagnostics byte-identical on *_fail categories, and (c) run
output / exit codes identical to the interp and oracle-native legs on
ok / runtime_fail. Interactive tests still skip every native leg. The
suite expects to run from the repo root (the std-import exe-dir fallback
and the file_exists_ok pin both assume it).

Exit code: 0 if all pass, 1 if any fail.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent.resolve()
EXAMPLES = HERE / "examples"
LANG = HERE / "lang.py"
COMPILER = HERE / "compiler.arrow"

# Set by --native-compiler: absolute Path to a native arrow compiler
# binary. None = oracle-only (the default, byte-identical to before).
NATIVE_COMPILER = None

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


def oracle_cmd(example: Path, exe: Path, keep_ll: bool) -> list:
    """Compile command for the interp-hosted oracle (lang.py running
    compiler.arrow) — the default host, unchanged from before."""
    cmd = [sys.executable, str(LANG), str(COMPILER), str(example),
           "-o", str(exe)]
    if keep_ll:
        cmd.append("--keep-ll")
    return cmd


def native_cmd(example: Path, exe: Path, keep_ll: bool) -> list:
    """Compile command for the --native-compiler binary. Same driver argv
    contract as the oracle: <compiler> <input> -o <exe> [--keep-ll]."""
    cmd = [str(NATIVE_COMPILER), str(example), "-o", str(exe)]
    if keep_ll:
        cmd.append("--keep-ll")
    return cmd


def run_compile_only(example: Path, cmd_builder=oracle_cmd) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "a.exe"
        proc = subprocess.run(
            cmd_builder(example, exe, False),
            capture_output=True, text=True, timeout=180,
        )
        return proc.returncode, proc.stdout, proc.stderr


def run_compile_and_native(example: Path, cmd_builder=oracle_cmd,
                           keep_ll: bool = False):
    """Compile with the given host command, then run the produced binary.
    Returns (rc, stdout, stderr, compile_stdout, ll_bytes); ll_bytes is
    the kept intermediate IR when keep_ll (driver: -o a.exe -> a.ll),
    else None."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / "a.exe"
        compile_proc = subprocess.run(
            cmd_builder(example, exe, keep_ll),
            capture_output=True, text=True, timeout=180,
        )
        ll = tmp / "a.ll"
        ll_bytes = ll.read_bytes() if (keep_ll and ll.exists()) else None
        if compile_proc.returncode != 0:
            return (compile_proc.returncode, compile_proc.stdout,
                    compile_proc.stderr or "(compile failed)",
                    compile_proc.stdout, ll_bytes)
        real_exe = exe if exe.exists() else tmp / "a"
        if not real_exe.exists():
            exes = (list(tmp.glob("*.exe"))
                    or [p for p in tmp.glob("a*") if p.suffix != ".ll"])
            real_exe = exes[0] if exes else exe
        if not real_exe.exists():
            return (99, compile_proc.stdout,
                    f"(no output binary; compile output: {compile_proc.stdout})",
                    compile_proc.stdout, ll_bytes)
        run_proc = subprocess.run(
            [str(real_exe)], capture_output=True, text=True, timeout=60,
        )
        return (run_proc.returncode, run_proc.stdout, run_proc.stderr,
                compile_proc.stdout, ll_bytes)


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

    keep = NATIVE_COMPILER is not None
    try:
        nc, nout, nerr, _, oll = run_compile_and_native(
            example, oracle_cmd, keep)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT (native)", "")
    if nc != 0:
        last = (nerr or nout).strip().splitlines()
        return ("native ERROR", last[-1] if last else "")

    if NATIVE_COMPILER is not None:
        try:
            c2, out2, err2, _, nll = run_compile_and_native(
                example, native_cmd, True)
        except subprocess.TimeoutExpired:
            return ("TIMEOUT (nc-native)", "")
        if c2 != 0:
            last = (err2 or out2).strip().splitlines()
            return ("nc-native ERROR", last[-1] if last else "")
        if oll != nll:
            return ("MISMATCH (.ll oracle != nc)", ll_diff_note(oll, nll))
        if out2 != nout:
            return ("MISMATCH (nc-native != native)",
                    f"native:{nout!r} vs nc:{out2!r}")

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
    if NATIVE_COMPILER is not None:
        try:
            rc2, stdout2, stderr2 = run_compile_only(example, native_cmd)
        except subprocess.TimeoutExpired:
            return ("TIMEOUT (nc-compile)", "")
        if stdout2 != stdout:
            return ("MISMATCH (nc diag != oracle)", brief_diff(stdout, stdout2))
    return (f"{error_kind} fail", "")


def check_runtime_fail(example: Path, header: dict, verbose: bool):
    """Compiles cleanly, then traps at runtime: both implementations must
    exit 1 with byte-identical output (pre-trap prints + the error line)."""
    try:
        ic, iout, ierr = run_interp(example)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT (interp)", "")
    if ic == 0:
        return ("UNEXPECTED (interp ran clean)", "")
    keep = NATIVE_COMPILER is not None
    try:
        nc, nout, nerr, compile_out, oll = run_compile_and_native(
            example, oracle_cmd, keep)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT (native)", "")
    if "Compilation aborted" in compile_out:
        last = compile_out.strip().splitlines()
        return ("UNEXPECTED (compile failed)", last[-1] if last else "")
    if nc == 0:
        return ("UNEXPECTED (native ran clean)", "")
    if ic != 1 or nc != 1:
        return ("BAD EXIT CODE", f"interp rc={ic}, native rc={nc} (want 1)")
    if iout != nout:
        return ("MISMATCH (interp != native)",
                f"interp:{iout!r} vs native:{nout!r}")
    expected = header["output"]
    if expected is not None and not iout.startswith(expected):
        return ("MISMATCH (pre-trap OUT)", brief_diff(expected, iout))
    missing = [c for c in header["contains"] if c not in iout]
    if missing:
        return ("CONTAINS missing", f"required substrings absent: {missing}")
    if NATIVE_COMPILER is not None:
        try:
            c2, out2, err2, compile_out2, nll = run_compile_and_native(
                example, native_cmd, True)
        except subprocess.TimeoutExpired:
            return ("TIMEOUT (nc-native)", "")
        if "Compilation aborted" in compile_out2:
            last = compile_out2.strip().splitlines()
            return ("UNEXPECTED (nc compile failed)", last[-1] if last else "")
        if oll != nll:
            return ("MISMATCH (.ll oracle != nc)", ll_diff_note(oll, nll))
        if c2 != 1:
            return ("BAD EXIT CODE (nc)", f"nc rc={c2} (want 1)")
        if out2 != iout:
            return ("MISMATCH (nc-native != interp)",
                    f"interp:{iout!r} vs nc:{out2!r}")
    return ("runtime fail", "")


def ll_diff_note(a, b) -> str:
    """Describe a .ll byte mismatch between the oracle-emitted IR and
    the native-compiler-emitted IR."""
    if a is None or b is None:
        return (f"missing IR (oracle={'present' if a is not None else 'absent'}, "
                f"nc={'present' if b is not None else 'absent'})")
    n = min(len(a), len(b))
    off = next((i for i in range(n) if a[i] != b[i]), n)
    return f"first diff at byte {off} (sizes {len(a)} vs {len(b)})"


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
    "runtime_fail": lambda ex, h, v: check_runtime_fail(ex, h, v),
}

PASS_STATUSES = {
    "ok", "type fail", "scope fail", "parse fail", "runtime fail",
    "skipped (interactive)", "weak (no //OUT:)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default="",
                    help="substring filter for example filenames")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show full output on every test")
    ap.add_argument("--native-compiler", metavar="PATH", default=None,
                    help="additive mode: also compile every test with this "
                         "native arrow compiler binary and cross-check .ll "
                         "bytes, diagnostics, and run output against the "
                         "interp-hosted oracle")
    args = ap.parse_args()

    global NATIVE_COMPILER
    if args.native_compiler:
        p = Path(args.native_compiler).resolve()
        if not p.is_file() or not os.access(p, os.X_OK):
            print(f"--native-compiler: not an executable file: {p}")
            return 1
        NATIVE_COMPILER = p
        print(f"additive native-compiler mode: {p}")

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
