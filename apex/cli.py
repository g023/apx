"""CLI entry point."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import Orchestrator
from .spec_engine import SpecEngine
from .verifier import Verifier
from .scm import SCM


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="g023_apx_agent",
        description="Adaptive Programming via eXploratory edit search.",
    )
    p.add_argument("goal", nargs="?", default=None, help="natural-language goal")
    p.add_argument("--no-questions", action="store_true",
                   help="skip clarifying questions")
    p.add_argument("--iterations", type=int, default=None,
                   help="MCTS iterations (default from config)")
    p.add_argument("--root", default=".", help="repo root (default: cwd)")
    p.add_argument("--verify-only", action="store_true",
                   help="only run final audit against existing .apex/spec.json")
    p.add_argument("--spec-only", action="store_true",
                   help="elicit spec and exit (no MCTS, no merge)")
    p.add_argument("--no-discovery", action="store_true",
                   help="skip the pre-spec discovery phase")
    p.add_argument("--discovery-only", action="store_true",
                   help="run only the pre-spec discovery phase and print the result")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = str(Path(args.root).resolve())

    if args.verify_only:
        from .spec_engine import Spec
        spec_path = Path(root) / ".apex" / "spec.json"
        if not spec_path.exists():
            print(json.dumps({"error": f"no spec at {spec_path}"}, indent=2))
            return 2
        spec = Spec.load(str(spec_path))
        scm = SCM(root=root)
        scm.scan()
        verifier = Verifier(scm=scm)
        audit = verifier.final_audit(spec, root)
        print(json.dumps(audit, indent=2))
        return 0 if audit.get("all_met") else 1

    if args.discovery_only:
        from .discovery import run_discovery
        goal = args.goal or ""
        res = run_discovery(root, goal)
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if not args.goal:
        parser.print_help()
        return 2

    if args.spec_only:
        spec = SpecEngine().run(args.goal, ask=None)
        spec.save(str(Path(root) / ".apex" / "spec.json"))
        print(json.dumps(spec.to_dict(), indent=2))
        return 0

    ask = None if args.no_questions else input
    orch = Orchestrator(root_dir=root)
    result = orch.run(args.goal, ask=ask, mcts_iterations=args.iterations,
                      discovery=not args.no_discovery)
    # Concise summary.
    summary = {
        "goal": result.get("spec", {}).get("goal") if result.get("spec") else None,
        "requirements": len((result.get("spec") or {}).get("requirements") or []),
        "merged_sha": result.get("merged_sha"),
        "all_met": (result.get("audit") or {}).get("all_met"),
        "errors": result.get("errors") or [],
    }
    print(json.dumps(summary, indent=2))
    ok = bool(summary["all_met"]) and not summary["errors"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
