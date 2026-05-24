# g023's - Adaptive Programming via eXploratory edit search Agent

This is an autonomous coding agent that turns a natural-language goal into
working code by combining four ideas: a formal **Intent Specification**
elicited from the user, a **Symbolic Code Model** (zero-dep static analysis),
**Monte Carlo Tree Search over code edits** that forks the workspace via git
branches, and an **LLM-Critic ensemble** that scores every candidate patch
along correctness, maintainability, and performance axes. All LLM calls go
through **DeepSeek v4 Flash** via the local `_ds4.py` client; everything else is
Python 3.11+ stdlib.

Author: g023 (github.com/g023)

License: MIT

## Quickstart

```bash
# API key file expected at ../K.dat (relative to this repo).
python3 -m apex "Add a CLI flag to do X"
```

Useful flags:

```bash
python3 -m apex "..." --no-questions      # skip clarifying questions
python3 -m apex "..." --iterations 16     # MCTS budget
python3 -m apex "..." --root /path/repo   # operate on another repo
python3 -m apex "..." --spec-only         # elicit spec then exit
python3 -m apex --verify-only             # re-audit using .apex/spec.json

# example run from another dir (assuming /tmp/project1 exists)
cd /tmp/project1 && PYTHONPATH=/path/to/g023_apex:$PYTHONPATH python3 -m apex "Add a power(a, b) function that raises a to the power b, and a modulus(a, b) function that returns a % b. Add corresponding tests." --no-questions
```

## Architecture

```
            ┌─────────────────────────────┐
            │         orchestrator        │
            │  spec → meta → MCTS → audit │
            └─────┬────────────┬──────────┘
                  │            │
        ┌─────────▼──┐    ┌────▼──────────┐
        │ spec_engine│    │     mcts      │
        │   (Spec)   │    │   (MCTSCode)  │
        └─────┬──────┘    └────┬──────────┘
              │                │
              │      ┌─────────▼────────┐
              │      │     critics      │
              │      │  (3× ensemble)   │
              │      └─────────┬────────┘
              │                │
        ┌─────▼─────────┐  ┌───▼──────┐  ┌──────────┐
        │     meta      │  │   scm    │  │  tools   │
        │ (specialists) │  │  (AST)   │  │ (git/io) │
        └───────┬───────┘  └──────────┘  └────┬─────┘
                │                              │
                └───────────► llm ◄────────────┘
                              (_ds4)
```

State lives under `.apex/` (`spec.json`, `tree.json`, `scm.pickle`,
`memory/`). The `verifier` module performs the final audit (per-requirement
LLM check + leftover-marker scan + pytest pass-ratio).

## Running tests

```bash
python3 -m pytest tests/ -q
```

All tests use stub LLMs — no network calls.

## License

MIT
