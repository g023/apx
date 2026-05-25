"""Top-level orchestrator wiring spec → MCTS → merge → audit."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import tools as _tools
from .config import APEX_DIR, MAX_MCTS_ITERATIONS, STATE_DIR
from .critics import CriticEnsemble
from .debug_trace import T, stage_timer, trace_audit
from .discovery import DiscoveryResult, run_discovery
from .logging_util import get_logger
from .mcts import MCTSCode
from .memory import trajectory
from .meta import MetaController
from .scm import SCM
from .spec_engine import Spec, SpecEngine
from .verifier import Verifier

_log = get_logger(__name__)


class Orchestrator:
    """State-machine wiring everything together."""

    def __init__(
        self,
        root_dir: str = ".",
        llm: Any | None = None,
        fast_llm: Any | None = None,
    ) -> None:
        self.root_dir = str(Path(root_dir).resolve())
        # Lazy LLM resolution.
        if llm is None:
            from .llm import default_llm
            llm = default_llm()
        if fast_llm is None:
            from .llm import fast_llm as _fast
            fast_llm = _fast()
        self.llm = llm
        self.fast_llm = fast_llm

        # Ensure state dir exists.
        STATE_DIR()
        self.scm = SCM(root=self.root_dir)
        self.meta = MetaController(llm=llm)
        self.critics = CriticEnsemble(llm=fast_llm)
        self.verifier = Verifier(scm=self.scm, llm=llm)

    def _spec_path(self) -> str:
        return str(Path(self.root_dir) / APEX_DIR / "spec.json")

    def run(
        self,
        user_goal: str,
        ask: Callable[[str], str] | None = None,
        mcts_iterations: int | None = None,
        discovery: bool = True,
    ) -> dict:
        result: dict = {
            "spec": None,
            "best_node": None,
            "audit": None,
            "merged_sha": None,
            "discovery": None,
            "errors": [],
        }
        trajectory().log("orch_start", goal=user_goal)
        T("ORCH", "Starting pipeline", goal=user_goal)

        # 0. Pre-spec discovery (skipped if user opted out).
        disc_ctx = ""
        if discovery:
            with stage_timer("Pre-Spec Discovery"):
                try:
                    disc = run_discovery(self.root_dir, user_goal)
                    result["discovery"] = disc.to_dict()
                    trajectory().log("orch_discovery", mode=disc.mode,
                                     empty=disc.empty, llm_used=disc.llm_used)
                    T("ORCH", f"Discovery: mode={disc.mode} empty={disc.empty} "
                             f"anchors={len(disc.anchors)} llm={disc.llm_used}")
                    if not disc.empty:
                        disc_ctx = (
                            f"Mode: {disc.mode}\n"
                            f"Summary: {disc.summary}\n"
                            f"Languages: {', '.join(disc.languages) or 'n/a'}\n"
                            f"Anchor files: {', '.join(disc.anchors) or 'n/a'}"
                        )
                except Exception as e:
                    _log.warning("discovery failed: %s", e)
                    result["errors"].append(f"discovery: {e}")
                    T("ORCH", f"Discovery FAILED: {e}")

        # 1. Spec.
        with stage_timer("Spec Elicitation"):
            try:
                spec = SpecEngine().run(user_goal, ask=ask, llm=self.llm,
                                        context=disc_ctx)
                spec.save(self._spec_path())
                result["spec"] = spec.to_dict()
                trajectory().log("orch_spec", reqs=len(spec.requirements))
                T("ORCH", f"Spec done: {len(spec.requirements)} requirements, confidence={spec.confidence:.2f}")
            except Exception as e:
                _log.warning("spec failed: %s", e)
                result["errors"].append(f"spec: {e}")
                T("ORCH", f"Spec FAILED: {e}")
                return result

        # 2. SCM.
        with stage_timer("SCM Scan"):
            try:
                self.scm.scan()
                self.scm.save()
                T("ORCH", f"SCM scanned: {len(getattr(self.scm, 'files', {}))} files")
            except Exception as e:
                _log.warning("scm scan failed: %s", e)
                result["errors"].append(f"scm: {e}")
                T("ORCH", f"SCM scan FAILED: {e}")

        # 3. Meta decompose (reserved; just log).
        with stage_timer("Meta Decompose"):
            try:
                specialists = self.meta.decompose(spec)
                trajectory().log(
                    "orch_specialists",
                    names=[s.name for s in specialists],
                )
                T("ORCH", f"Meta decompose: {[s.name for s in specialists]}")
            except Exception as e:
                _log.warning("meta decompose failed: %s", e)
                T("ORCH", f"Meta decompose FAILED: {e}")

        # 4. MCTS.
        best_node = None
        with stage_timer("MCTS Search"):
            try:
                mcts = MCTSCode(
                    root_dir=self.root_dir,
                    spec=spec,
                    scm=self.scm,
                    llm=self.llm,
                    fast_llm=self.fast_llm,
                    critics=self.critics,
                    max_iterations=mcts_iterations or MAX_MCTS_ITERATIONS,
                )
                mcts.init_root()
                T("ORCH", f"MCTS root initialized: {mcts.root_id[:10] if mcts.root_id else 'N/A'}")
                best_node = mcts.run(iterations=mcts_iterations)
                if best_node is not None:
                    result["best_node"] = best_node.to_dict()
                    T("ORCH", f"MCTS done: best node {best_node.id[:10]} q={best_node.q:.4f} visits={best_node.visits}")
                else:
                    T("ORCH", "MCTS returned None (no best node)")
                # 5. Merge.
                try:
                    if best_node and best_node.id != mcts.root_id:
                        merged = mcts.merge_best(message="apex: merge best leaf")
                        result["merged_sha"] = merged
                        T("ORCH", f"Merged best leaf: {merged[:12]}")
                    else:
                        T("ORCH", "No merge needed (best is root or None)")
                except Exception as e:
                    _log.warning("merge failed: %s", e)
                    result["errors"].append(f"merge: {e}")
                    T("ORCH", f"Merge FAILED: {e}")
            except Exception as e:
                _log.warning("mcts failed (safe-mode): %s", e)
                trajectory().log("orch_mcts_safe_mode", error=str(e))
                result["errors"].append(f"mcts: {e}")
                T("ORCH", f"MCTS FAILED (safe-mode): {e}")

        # 6. Final audit.
        with stage_timer("Final Audit"):
            try:
                # Re-scan for latest state.
                try:
                    self.scm.scan()
                except Exception:
                    pass
                result["audit"] = self.verifier.final_audit(spec, self.root_dir)
                audit = result["audit"] or {}
                trace_audit(
                    all_met=audit.get("all_met", False),
                    reports=len(audit.get("reports", [])),
                    tests_ok=audit.get("audit", {}).get("tests", {}).get("ok", False),
                    files_clean=audit.get("audit", {}).get("files", {}).get("clean", False),
                )
            except Exception as e:
                _log.warning("audit failed: %s", e)
                result["errors"].append(f"audit: {e}")
                T("ORCH", f"Audit FAILED: {e}")

        trajectory().log("orch_done", merged=bool(result.get("merged_sha")))
        T("ORCH", "Pipeline complete", errors=len(result.get("errors", [])),
          merged=bool(result.get("merged_sha")),
          all_met=(result.get("audit") or {}).get("all_met"))
        return result
