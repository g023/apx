"""
debug_trace.py — Live debug tracing for the pipeline stages.

Injects timestamped, stage-labeled debug prints into every phase of the
orchestrator pipeline so the user can see exactly what's happening,
where failures occur, and how long each stage takes.

Usage:
    from .debug_trace import T, stage_timer

    with stage_timer("Spec Elicitation"):
        ...
"""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


# Global flag — set False to silence all debug output.
_ENABLED = True


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:12]


def T(label: str, *args: Any, **kw: Any) -> None:
    """Print a timestamped debug trace line to stderr."""
    if not _ENABLED:
        return
    extra = " ".join(str(a) for a in args) if args else ""
    if kw:
        extra += " " + " ".join(f"{k}={v!r}" for k, v in kw.items())
    line = f"[{_ts()}] [TRACE] {label}"
    if extra:
        line += f" — {extra}"
    print(line, file=sys.stderr, flush=True)


@contextmanager
def stage_timer(label: str, *args: Any, **kw: Any) -> Iterator[None]:
    """Context manager that traces start/end + elapsed time for a stage."""
    T(f"▶ {label}", *args, **kw)
    t0 = time.monotonic()
    try:
        yield
    except Exception as e:
        elapsed = time.monotonic() - t0
        T(f"✗ {label}", f"FAIL after {elapsed:.2f}s:", str(e))
        raise
    else:
        elapsed = time.monotonic() - t0
        T(f"✓ {label}", f"OK ({elapsed:.2f}s)")


def trace_llm_call(method: str, model: str, msg_count: int, result_preview: str = "") -> None:
    """Trace an LLM API call."""
    preview = (result_preview[:120] + "...") if result_preview else "(no preview)"
    T("LLM", f"{method} model={model!r} msgs={msg_count}", f"→ {preview}")


def trace_git_op(op: str, *args: str, result: str = "") -> None:
    """Trace a git/filesystem operation."""
    args_str = " ".join(str(a) for a in args) if args else ""
    res_str = f" → {result}" if result else ""
    T("GIT", f"{op} {args_str}{res_str}")


def trace_mcts_iter(i: int, node_id: str, reward: float, branch: str, detail: str = "") -> None:
    """Trace a single MCTS iteration."""
    extra = f" — {detail}" if detail else ""
    T("MCTS", f"iter={i} node={node_id[:10]} reward={reward:.4f} branch={branch}{extra}")


def trace_expand(parent: str, child: str, branch: str, file: str, status: str) -> None:
    """Trace an MCTS expansion."""
    T("EXPAND", f"parent={parent[:10]} child={child[:10]} branch={branch} file={file} → {status}")


def trace_simulate(node: str, test_ratio: float, spec_score: float, critic: float, errors: int) -> None:
    """Trace simulation reward components."""
    T("SIMULATE", f"node={node[:10]} tests={test_ratio:.2f} spec={spec_score:.2f} critic={critic:.2f} errors={errors}")


def trace_audit(all_met: bool, reports: int, tests_ok: bool, files_clean: bool) -> None:
    """Trace final audit results."""
    T("AUDIT", f"all_met={all_met} reports={reports} tests_ok={tests_ok} files_clean={files_clean}")
