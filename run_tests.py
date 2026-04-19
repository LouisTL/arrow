#!/usr/bin/env python3
"""
run_tests.py — Runs every .arrow file in examples/ through both the interpreter
and the compiler, flagging any that behave differently.

Usage:
    python run_tests.py                 # runs all examples
    python run_tests.py pattern         # runs examples matching pattern

Exit code: 0 if all pass, 1 if any differ or crash.

A test has three possible outcomes:
  1. "type-check fail"  — compiler rejects at check-time (OK, expected for *_fail.arrow)
  2. "match"            — interpreter and native binary produce identical stdout
  3. "mismatch"         — they differ (or one crashed and the other didn't)

The runner skips clang/native for examples that need interactive input
(detected by the presence of input() in source).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent.resolve()
EXAMPLES = HERE / "examples"
LANG = HERE / "lang.py"
COMPILER = HERE / "compiler.arrow"

# Files we expect to produce type errors on purpose — don't run natively, just check the type checker fires.
EXPECT_FAIL = {"test_fail.arrow", "test_arr_fail.arrow", "test_struct_fail.arrow"}

# Files that need interactive input — skip native run for these.
INTERACTIVE_KEYWORDS = ["input("]


def run_interp(example: Path) -> tuple[int, str, str]:
    """Run example.arrow through lang.py interpreter directly."""
    proc = subprocess.run(
        [sys.executable, str(LANG), str(example)],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_compile_and_native(example: Path) -> tuple[int, str, str]:
    """Compile example through compiler.arrow (via lang.py), then run the exe."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Compile: python lang.py compiler.arrow <example> -o <tmp>/a.exe
        exe = tmp / "a.exe"
        compile_proc = subprocess.run(
            [sys.executable, str(LANG), str(COMPILER), str(example),
             "-o", str(exe)],
            capture_output=True, text=True, timeout=180,
        )
        if compile_proc.returncode != 0:
            return compile_proc.returncode, compile_proc.stdout, compile_proc.stderr or "(compile failed)"
        # Find the resulting exe (on Linux it may just be `a`)
        real_exe = exe if exe.exists() else tmp / "a"
        if not real_exe.exists():
            # try finding *.exe in tmp
            exes = list(tmp.glob("*.exe")) or list(tmp.glob("a*"))
            real_exe = exes[0] if exes else exe
        if not real_exe.exists():
            return 99, compile_proc.stdout, f"(no output binary found; compile output: {compile_proc.stdout})"
        run_proc = subprocess.run(
            [str(real_exe)], capture_output=True, text=True, timeout=60,
        )
        return run_proc.returncode, run_proc.stdout, run_proc.stderr


def check_type_fail(example: Path) -> bool:
    """For EXPECT_FAIL files: ensure compiler.arrow reports type errors and aborts."""
    proc = subprocess.run(
        [sys.executable, str(LANG), str(COMPILER), str(example)],
        capture_output=True, text=True, timeout=120,
    )
    # Expect stdout to contain "type error" and "Compilation aborted"
    return "type error" in proc.stdout and "aborted" in proc.stdout.lower(), proc.stdout


def is_interactive(example: Path) -> bool:
    src = example.read_text(errors="replace")
    return any(kw in src for kw in INTERACTIVE_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default="", help="substring filter for example filenames")
    args = ap.parse_args()

    examples = sorted([p for p in EXAMPLES.glob("*.arrow")
                       if args.pattern in p.name])
    if not examples:
        print(f"no examples matching {args.pattern!r}")
        return 1

    results = []  # list of (example, status, note)

    for ex in examples:
        name = ex.name
        # Case 1: expected-fail type-check tests
        if name in EXPECT_FAIL:
            ok, output = check_type_fail(ex)
            results.append((name, "type-check fail" if ok else "UNEXPECTED",
                            "" if ok else output.strip().splitlines()[-1:]))
            continue

        # Case 2: interpreter run
        try:
            ic, iout, ierr = run_interp(ex)
        except subprocess.TimeoutExpired:
            results.append((name, "TIMEOUT (interp)", ""))
            continue

        if is_interactive(ex):
            results.append((name, "skipped (interactive)", f"interp output: {iout.strip()[:40]}"))
            continue

        # Case 3: compile-and-run
        try:
            nc, nout, nerr = run_compile_and_native(ex)
        except subprocess.TimeoutExpired:
            results.append((name, "TIMEOUT (native)", ""))
            continue

        if ic != 0 and "error" in ierr.lower():
            results.append((name, "interp ERROR", ierr.strip().splitlines()[-1]))
            continue

        # Compare stdouts
        if iout == nout:
            results.append((name, "match", ""))
        else:
            results.append((name, "MISMATCH", f"interp:{iout!r} vs native:{nout!r}"))

    # Report
    passed = failed = 0
    for name, status, note in results:
        mark = "OK " if status in ("match", "type-check fail", "skipped (interactive)") else "!! "
        if mark == "OK ":
            passed += 1
        else:
            failed += 1
        line = f"{mark}{name:30s}  {status}"
        if note:
            line += f"   {note}"
        print(line)

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
