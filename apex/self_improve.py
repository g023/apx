"""self-improvement smoke: propose a single _ds4 enhancement."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

from pathlib import Path
from typing import Any

from .logging_util import get_logger
from .scm import SCM

_log = get_logger(__name__)


_SI_PROMPT = """You are improving the _ds4 DeepSeek client.
Below is a structural summary of the module (NOT the full source).
Propose exactly ONE small, safe improvement (e.g., better error message,
extra docstring, a small helper). Return your suggestion as a unified
diff snippet wrapped in a fenced block, plus a one-line rationale.
Do NOT propose breaking changes.

SCM summary:
{summary}
"""


def propose_ds4_improvement(llm: Any | None = None) -> str:
    """Ask the LLM for one small improvement to _ds4.py.

    Reads only the SCM summary (NOT the full file). Returns the LLM's
    suggestion text. Does NOT write any file.
    """
    if llm is None:
        from .llm import default_llm
        llm = default_llm()

    root = Path(__file__).resolve().parent.parent
    scm = SCM(root=str(root))
    try:
        scm.scan(paths=[str(root / "_ds4.py")])
        summary = scm.summary(max_chars=1500)
    except Exception as e:
        _log.warning("self_improve summary failed: %s", e)
        summary = f"(summary unavailable: {e})"

    prompt = _SI_PROMPT.format(summary=summary)
    try:
        text = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_turns=1,
        )
    except Exception as e:
        _log.warning("self_improve llm failed: %s", e)
        return f"(no suggestion: {e})"
    return text or "(no suggestion)"
