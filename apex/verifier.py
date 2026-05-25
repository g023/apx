"""Final spec verifier + repo audit."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import tools as _tools
from .debug_trace import T
from .logging_util import get_logger
from .mcts import run_tests
from .memory import trajectory

_log = get_logger(__name__)


_VERIFY_PROMPT = """You verify whether a single requirement has been
implemented in a codebase. You will see a codebase summary and the
diff against the project's main branch.

Requirement:
{requirement}

Codebase summary:
{summary}

Diff (truncated):
{diff}

Respond ONLY with JSON:
{{"met": <bool>, "evidence": "<short reason>", "score": <0..1 float>}}
"""


@dataclass
class VerificationReport:
    requirement_id: str
    met: bool
    evidence: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _find_main_ref(repo_root: str) -> str:
    """Return 'main' if it exists, else 'master', else HEAD~1, else HEAD.

    In non-git mode (filesystem fallback), returns the root branch name.
    """
    if not _tools.is_git_repo(repo_root):
        # In filesystem mode, use the root branch.
        try:
            return _tools.fs_current_branch(repo_root)
        except Exception:
            return "__fs__"
    for name in ("main", "master"):
        rc, _, _ = _tools.run_command(
            ["git", "rev-parse", "--verify", f"refs/heads/{name}"], cwd=repo_root
        )
        if rc == 0:
            return name
    rc, _, _ = _tools.run_command(["git", "rev-parse", "HEAD~1"], cwd=repo_root)
    if rc == 0:
        return "HEAD~1"
    return "HEAD"


class Verifier:
    """Verify Spec requirements against the working repo + audit cleanliness."""

    def __init__(self, scm: Any, llm: Any | None = None) -> None:
        self.scm = scm
        self._llm = llm

    def _get_llm(self) -> Any:
        from .llm import lazy_llm
        self._llm = lazy_llm(self._llm)
        return self._llm

    def _diff(self, repo_root: str) -> str:
        ref = _find_main_ref(repo_root)
        try:
            diff = _tools.git_diff(ref, "HEAD", cwd=repo_root)
            if diff:
                return diff
        except Exception as e:
            _log.warning("verifier diff failed: %s", e)
        # Fallback: if no git diff available (empty repo / fs mode),
        # show the current file listing as a pseudo-diff.
        try:
            from . import lang as _lang
            lines = []
            for ext in _lang.ALL_CODE_EXTENSIONS:
                for fp in _tools.list_files(root=repo_root, pattern=f"**/*{ext}"):
                    try:
                        content = Path(fp).read_text(encoding="utf-8", errors="replace")
                        rel = Path(fp).relative_to(repo_root)
                        lines.append(f"--- a/{rel}")
                        lines.append(f"+++ b/{rel}")
                        for i, line in enumerate(content.splitlines(), 1):
                            lines.append(f" {line}")
                    except Exception:
                        pass
            return "\n".join(lines[:4000])
        except Exception as e2:
            _log.warning("verifier fallback diff also failed: %s", e2)
            return ""

    def verify(self, spec: Any, repo_root: str) -> list[VerificationReport]:
        T("VERIFY", f"Verifying {len(getattr(spec, 'requirements', []))} requirements")
        try:
            summary = self.scm.summary(max_chars=1500)
            T("VERIFY", f"SCM summary: {len(summary)} chars")
        except Exception as e:
            summary = ""
            T("VERIFY", f"SCM summary failed: {e}")
        diff = self._diff(repo_root)
        T("VERIFY", f"Diff length: {len(diff)} chars")
        llm = self._get_llm()
        reports: list[VerificationReport] = []
        for req in getattr(spec, "requirements", []) or []:
            req_text = f"{req.id}: {req.description} [accept: {req.acceptance}]"
            prompt = _VERIFY_PROMPT.format(
                requirement=req_text,
                summary=summary[:1500],
                diff=(diff or "")[:4000],
            )
            T("VERIFY", f"Checking {req.id}...")
            try:
                obj = llm.chat_json(
                    [{"role": "user", "content": prompt}],
                    schema_hint='{"met": bool, "evidence": str, "score": float}',
                )
                met = bool(obj.get("met", False))
                evidence = str(obj.get("evidence", ""))
                try:
                    score = float(obj.get("score", 1.0 if met else 0.0))
                except Exception:
                    score = 1.0 if met else 0.0
                T("VERIFY", f"  {req.id}: met={met} score={score:.2f} evidence={evidence[:60]}")
            except Exception as e:
                _log.warning("verify %s failed: %s", req.id, e)
                met, evidence, score = False, f"llm-error: {e}", 0.0
                T("VERIFY", f"  {req.id}: LLM ERROR: {e}")
            reports.append(
                VerificationReport(
                    requirement_id=req.id,
                    met=met,
                    evidence=evidence,
                    score=score,
                )
            )
        trajectory().log(
            "verify",
            total=len(reports),
            met=sum(1 for r in reports if r.met),
        )
        T("VERIFY", f"Done: {sum(1 for r in reports if r.met)}/{len(reports)} met")
        return reports

    # ---------------- audit ----------------

    def _audit_files(self, repo_root: str) -> dict:
        """Scan repo files for leftover markers across all supported languages."""
        from . import lang as _lang
        markers = ("TODO", "FIXME", "apex-stub", "NotImplementedError")
        flagged: list[dict] = []
        seen: set[str] = set()
        for ext in _lang.ALL_CODE_EXTENSIONS:
            pattern = f"**/*{ext}"
            for fp in _tools.list_files(root=repo_root, pattern=pattern):
                if fp in seen:
                    continue
                seen.add(fp)
                try:
                    txt = Path(fp).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                hits = [m for m in markers if m in txt]
                if hits:
                    flagged.append({"file": fp, "markers": hits})
        return {"marker_hits": flagged, "clean": not flagged}

    def _audit_imports(self) -> dict:
        """Detect Python imports that reference unresolvable modules via importlib.

        Non-Python files (php/js/ts) are skipped — their import resolution is
        package-manager-specific and not the verifier's concern.
        """
        from . import lang as _lang
        dead: list[dict] = []
        try:
            files = getattr(self.scm, "files", {}) or {}
        except Exception:
            files = {}
        symbol_names: set[str] = set()
        for fi in files.values():
            for s in getattr(fi, "symbols", []) or []:
                symbol_names.add(s.name)
            if _lang.lang_for_file(fi.path) != "python":
                continue
            for imp in getattr(fi, "imports", []) or []:
                head = imp.split(".", 1)[0]
                try:
                    import importlib.util
                    spec = importlib.util.find_spec(head)
                    if spec is None:
                        dead.append({
                            "file": fi.path,
                            "import": imp,
                            "reason": f"module '{head}' not found",
                        })
                except (ModuleNotFoundError, ValueError, ImportError):
                    dead.append({
                        "file": fi.path,
                        "import": imp,
                        "reason": f"module '{head}' not importable",
                    })
        return {"symbols": len(symbol_names), "dead_imports": dead}

    def final_audit(self, spec: Any, repo_root: str) -> dict:
        T("AUDIT", "Starting final audit")
        reports = self.verify(spec, repo_root)
        files_audit = self._audit_files(repo_root)
        T("AUDIT", f"Files audit: clean={files_audit.get('clean')}, "
          f"marker_hits={len(files_audit.get('marker_hits', []))}")
        imports_audit = self._audit_imports()
        T("AUDIT", f"Imports audit: {len(imports_audit.get('dead_imports', []))} dead imports")
        try:
            pass_ratio, raw = run_tests(repo_root, timeout=60)
            T("AUDIT", f"Tests: pass_ratio={pass_ratio:.2f}")
        except Exception as e:
            pass_ratio, raw = 0.0, f"run_tests error: {e}"
            T("AUDIT", f"Tests error: {e}")
        tests_ok = pass_ratio >= 0.999
        all_met = (
            all(r.met for r in reports)
            and files_audit.get("clean", False)
            and tests_ok
        )
        result = {
            "all_met": all_met,
            "reports": [r.to_dict() for r in reports],
            "audit": {
                "files": files_audit,
                "imports": imports_audit,
                "tests": {"pass_ratio": pass_ratio, "ok": tests_ok},
            },
        }
        trajectory().log("final_audit", all_met=all_met, tests_ok=tests_ok)
        T("AUDIT", f"Final: all_met={all_met} (reports={sum(1 for r in reports if r.met)}/{len(reports)}, "
          f"files_clean={files_audit.get('clean')}, tests_ok={tests_ok})")
        return result
