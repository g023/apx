"""Language registry & dispatch for polyglot.

Supports Python, PHP, and Node.js. Stdlib only. All external interpreter
invocations go through ``apex.tools.run_command``.

Language tags
-------------
- ``"python"`` — ``.py`` files, ``pytest -q``.
- ``"php"``    — ``.php`` files, ``phpunit`` if available else ``php tests/*.php``.
- ``"node"``   — ``.js`` / ``.mjs`` / ``.cjs`` / ``.ts`` files, ``npm test`` if a
  ``package.json`` defines a ``test`` script, else ``node --test tests/``.
- ``"mixed"`` / ``"unknown"`` — reported by ``detect_language`` only.
"""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

from . import tools as _tools

# --------------------------------------------------------------------------- #
# Extension table
# --------------------------------------------------------------------------- #

LANG_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "php": (".php",),
    "node": (".js", ".mjs", ".cjs", ".ts"),
}

ALL_CODE_EXTENSIONS: tuple[str, ...] = tuple(
    ext for exts in LANG_EXTENSIONS.values() for ext in exts
)

_SKIP_DIRS = {
    ".git", ".apex", "__pycache__", ".venv", ".mypy_cache",
    ".pytest_cache", "node_modules", "vendor",
}


def ext_for(lang: str) -> tuple[str, ...]:
    return LANG_EXTENSIONS.get(lang, ())


def lang_for_file(path: str) -> str | None:
    suf = Path(path).suffix.lower()
    for lang, exts in LANG_EXTENSIONS.items():
        if suf in exts:
            return lang
    return None


# --------------------------------------------------------------------------- #
# Project-level detection
# --------------------------------------------------------------------------- #


def _iter_code_files(root: str) -> Iterable[Path]:
    rp = Path(root)
    for p in rp.rglob("*"):
        if not p.is_file():
            continue
        if set(p.parts) & _SKIP_DIRS:
            continue
        if p.suffix.lower() in ALL_CODE_EXTENSIONS:
            yield p


def detect_language(root: str = ".") -> str:
    """Return the dominant language tag for a workspace.

    Priority signals (strong → weak):
    1. ``composer.json`` → ``php``
    2. ``package.json``  → ``node``
    3. ``pyproject.toml`` / ``setup.py`` / ``setup.cfg`` → ``python``
    4. File-count majority over indexed code files.

    Returns ``"unknown"`` if no signals are found, or ``"mixed"`` when two
    languages each hold ≥30 % share with neither at ≥60 %.
    """
    rp = Path(root)
    if (rp / "composer.json").is_file():
        return "php"
    if (rp / "package.json").is_file():
        return "node"
    if (
        (rp / "pyproject.toml").is_file()
        or (rp / "setup.py").is_file()
        or (rp / "setup.cfg").is_file()
    ):
        return "python"

    counts: dict[str, int] = {"python": 0, "php": 0, "node": 0}
    total = 0
    for fp in _iter_code_files(str(rp)):
        lang = lang_for_file(str(fp))
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
            total += 1
    if total == 0:
        return "unknown"
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top, top_n = ranked[0]
    if top_n / total >= 0.6:
        return top
    second_n = ranked[1][1] if len(ranked) > 1 else 0
    if top_n / total >= 0.3 and second_n / total >= 0.3:
        return "mixed"
    return top


# --------------------------------------------------------------------------- #
# Interpreter availability
# --------------------------------------------------------------------------- #


def interpreter_available(lang: str) -> bool:
    if lang == "python":
        return True  # we're running in python
    if lang == "php":
        return shutil.which("php") is not None
    if lang == "node":
        return shutil.which("node") is not None
    return False


def interpreter_version(lang: str) -> str | None:
    flag = {"python": "--version", "php": "--version", "node": "--version"}.get(lang)
    if not flag:
        return None
    binary = {"python": sys.executable, "php": "php", "node": "node"}[lang]
    rc, out, err = _tools.run_command([binary, flag], timeout=5)
    if rc != 0:
        return None
    text = (out or err or "").strip().splitlines()
    return text[0] if text else None


# --------------------------------------------------------------------------- #
# Syntax checks (used by SCM apply_simulation)
# --------------------------------------------------------------------------- #


def syntax_check(path: str, source: str) -> list[str]:
    """Return a list of human-readable warnings ([] = clean).

    Uses ``php -l`` for PHP and ``node --check`` for JS/TS. Python is
    handled by the SCM's own AST parse — callers shouldn't pass ``.py``.
    """
    lang = lang_for_file(path)
    if lang not in ("php", "node"):
        return []
    if not interpreter_available(lang):
        return []  # silently skip if interpreter missing
    import tempfile
    suf = Path(path).suffix or ({"php": ".php", "node": ".js"}[lang])
    with tempfile.NamedTemporaryFile(
        "w", suffix=suf, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(source)
        tmp = fh.name
    try:
        if lang == "php":
            rc, out, err = _tools.run_command(["php", "-l", tmp], timeout=10)
        else:
            rc, out, err = _tools.run_command(["node", "--check", tmp], timeout=10)
        if rc == 0:
            return []
        msg = (err or out or "").strip().splitlines()
        return [f"syntax error: {msg[0] if msg else 'unknown'}"]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Test dispatch
# --------------------------------------------------------------------------- #


def _python_run_tests(cwd: str, timeout: int) -> tuple[float, str]:
    import re
    tests_dir = Path(cwd) / "tests"
    if not tests_dir.exists():
        return 1.0, "no-tests"
    rc, out, err = _tools.run_command(
        [sys.executable, "-m", "pytest", "-q", "--maxfail=5"],
        cwd=cwd, timeout=timeout,
    )
    raw = (out or "") + (err or "")
    passed = failed = 0
    for line in reversed(raw.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        pm = re.search(r"(\d+)\s+passed", line)
        fm = re.search(r"(\d+)\s+failed", line)
        if pm or fm:
            if pm:
                passed = int(pm.group(1))
            if fm:
                failed = int(fm.group(1))
            break
    total = passed + failed
    if total == 0:
        if rc == 5:
            return 1.0, raw
        return 0.0 if rc != 0 else 1.0, raw
    return passed / total, raw


def _php_run_tests(cwd: str, timeout: int) -> tuple[float, str]:
    import re
    if not interpreter_available("php"):
        return 1.0, "no-php"
    tests_dir = Path(cwd) / "tests"
    # Prefer phpunit if available (project-local or global).
    phpunit_candidates: list[list[str]] = []
    local_phpunit = Path(cwd) / "vendor" / "bin" / "phpunit"
    if local_phpunit.exists():
        phpunit_candidates.append([str(local_phpunit)])
    if shutil.which("phpunit"):
        phpunit_candidates.append(["phpunit"])

    for cmd in phpunit_candidates:
        rc, out, err = _tools.run_command(
            cmd + ["--colors=never"], cwd=cwd, timeout=timeout
        )
        raw = (out or "") + (err or "")
        # PHPUnit summary: "OK (N tests, M assertions)" or "Tests: N, ..."
        m_ok = re.search(r"OK\s*\((\d+)\s+tests?", raw)
        if m_ok and rc == 0:
            return 1.0, raw
        m_summary = re.search(
            r"Tests:\s*(\d+).*?(?:Failures:\s*(\d+))?.*?(?:Errors:\s*(\d+))?",
            raw, re.DOTALL,
        )
        if m_summary:
            total = int(m_summary.group(1) or 0)
            failures = int(m_summary.group(2) or 0)
            errors = int(m_summary.group(3) or 0)
            if total > 0:
                bad = failures + errors
                return max(0.0, (total - bad) / total), raw
        if rc == 0:
            return 1.0, raw
        # phpunit not found despite which() lying — try next candidate
        if rc == 127:
            continue
        return 0.0, raw

    # Fallback: run each tests/*.php with `php`; non-zero = fail.
    if not tests_dir.exists():
        return 1.0, "no-tests"
    files = sorted(tests_dir.glob("*.php"))
    if not files:
        return 1.0, "no-tests"
    passed = 0
    logs: list[str] = []
    for tf in files:
        rc, out, err = _tools.run_command(
            ["php", str(tf)], cwd=cwd, timeout=timeout
        )
        logs.append(f"--- {tf.name} (rc={rc}) ---\n{out}{err}")
        if rc == 0:
            passed += 1
    return passed / len(files), "\n".join(logs)


def _node_run_tests(cwd: str, timeout: int) -> tuple[float, str]:
    import re
    if not interpreter_available("node"):
        return 1.0, "no-node"
    pkg = Path(cwd) / "package.json"
    has_npm_test = False
    if pkg.is_file():
        try:
            data = json.loads(pkg.read_text(encoding="utf-8"))
            has_npm_test = bool(data.get("scripts", {}).get("test"))
        except Exception:
            has_npm_test = False
    if has_npm_test and shutil.which("npm"):
        rc, out, err = _tools.run_command(
            ["npm", "test", "--silent"], cwd=cwd, timeout=timeout
        )
        raw = (out or "") + (err or "")
        # node --test summary lines: "# pass N" / "# fail N"
        pm = re.search(r"#\s*pass\s+(\d+)", raw)
        fm = re.search(r"#\s*fail\s+(\d+)", raw)
        if pm or fm:
            p = int(pm.group(1)) if pm else 0
            f = int(fm.group(1)) if fm else 0
            tot = p + f
            return (p / tot if tot else (1.0 if rc == 0 else 0.0)), raw
        return (1.0 if rc == 0 else 0.0), raw

    tests_dir = Path(cwd) / "tests"
    if not tests_dir.exists():
        return 1.0, "no-tests"
    files = [
        p for p in sorted(tests_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in (".js", ".mjs", ".cjs")
    ]
    if not files:
        return 1.0, "no-tests"
    rc, out, err = _tools.run_command(
        ["node", "--test"] + [str(p) for p in files],
        cwd=cwd, timeout=timeout,
    )
    raw = (out or "") + (err or "")
    pm = re.search(r"#\s*pass\s+(\d+)", raw)
    fm = re.search(r"#\s*fail\s+(\d+)", raw)
    if pm or fm:
        p = int(pm.group(1)) if pm else 0
        f = int(fm.group(1)) if fm else 0
        tot = p + f
        return (p / tot if tot else (1.0 if rc == 0 else 0.0)), raw
    return (1.0 if rc == 0 else 0.0), raw


_DISPATCH = {
    "python": _python_run_tests,
    "php": _php_run_tests,
    "node": _node_run_tests,
}


def run_tests_for(lang: str, cwd: str, timeout: int = 60) -> tuple[float, str]:
    """Run the appropriate test command for ``lang`` in ``cwd``.

    Falls back to python behaviour for ``"mixed"`` / ``"unknown"`` so callers
    that don't pre-detect get the previous default.
    """
    fn = _DISPATCH.get(lang, _python_run_tests)
    return fn(cwd, timeout)
