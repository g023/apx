# g023's APX Agent — Adaptive Programming via eXploratory edit search

An autonomous coding agent that turns a natural-language goal into working code.
Combines **Intent Specification** elicitation, a **Symbolic Code Model** (zero-dep
static analysis), **Monte Carlo Tree Search over code edits** (forking the workspace
via git branches), and an **LLM-Critic ensemble** scoring every candidate patch.
All LLM calls go through **DeepSeek v4 Flash** (`_ds4.py`); everything else is
**Python 3.11+ stdlib only**.

**Polyglot** — works with Python, PHP, and Node.js projects.

Author: **g023** ([github.com/g023](https://github.com/g023))  
License: **MIT**  
Version: **0.1.0a**  

---

## Quickstart

```bash
# API key file expected at ../K.dat (relative to this repo).
python3 -m apex "Add a CLI flag to do X"
```

### Useful flags

```bash
python3 -m apex "..." --no-questions       # skip clarifying questions
python3 -m apex "..." --iterations 16      # MCTS budget
python3 -m apex "..." --root /path/repo    # operate on another repo
python3 -m apex "..." --spec-only          # elicit spec then exit
python3 -m apex --verify-only              # re-audit using .apex/spec.json
python3 -m apex "..." --discovery-only     # explore workspace then exit
python3 -m apex "..." --no-discovery       # skip pre-spec discovery
```

### Run on another repo

```bash
cd /tmp/project1
PYTHONPATH=/path/to/g023_apx:$PYTHONPATH python3 -m apex \
  "Add power(a,b) and modulus(a,b) functions with tests." --no-questions
```

---

## Architecture

```
            ┌──────────────────────────────────┐
            │       apex.orchestrator          │
            │  discovery → spec → meta → MCTS  │
            │  → merge → audit                 │
            └─────┬────────────┬───────────────┘
                  │            │
        ┌─────────▼──┐    ┌───▼───────────┐
        │ spec_engine│    │     mcts      │
        │   (Spec)   │    │   (MCTSCode)  │
        └─────┬──────┘    └────┬──────────┘
              │                │
              │      ┌─────────▼────────┐
              │      │     critics      │
              │      │  (3× ensemble)   │
              │      └─────────┬────────┘
              │                │
        ┌─────▼─────────┐  ┌──▼───────┐  ┌──────────┐
        │     meta      │  │   scm    │  │  tools   │
        │ (specialists) │  │ (AST/re) │  │ (git/io) │
        └───────┬───────┘  └──────────┘  └────┬─────┘
                │                             │
                └───────────► llm ◄───────────┘
                           (_ds4.py)
```

### Pipeline stages

| Stage | Module | What it does |
|-------|--------|-------------|
| **0. Discovery** | `discovery.py` | Understand existing workspace before planning — algorithmic summary + optional LLM exploration. Persists to `.apex/discovery.json`. |
| **1. Spec** | `spec_engine.py` | Elicit a formal `Spec` (requirements, constraints, schemas) from the user goal. Asks clarifying questions until confidence ≥ 0.9. |
| **2. Decompose** | `meta.py` | Meta-Controller decomposes the spec into specialist sub-agents (e.g. `php_coder`, `node_coder`, `polyglot_reviewer`). |
| **3. MCTS** | `mcts.py` | Monte Carlo Tree Search over code patches. Each node = a git branch with a candidate patch. Select → Propose → Expand → Simulate (test + critic + spec compliance) → Backpropagate. |
| **4. Merge** | `mcts.py` | Merge the best path back to the working branch. |
| **5. Audit** | `verifier.py` | Per-requirement LLM verification + leftover-marker scan + test pass ratio. |

### State directory

Everything lives under `.apex/` (auto-gitignored):

```
.apex/
├── spec.json          # elicited specification
├── tree.json          # MCTS tree snapshot
├── scm.pickle         # serialised Symbolic Code Model
├── discovery.json     # pre-spec workspace analysis
├── memory/
│   ├── trajectory.jsonl   # JSONL log of every action
│   └── anti_patterns.json # learned anti-pattern penalties
├── fcache/            # content-hash-keyed file summaries
└── apex.log           # structured log output
```

---

## Package layout

```
apex/
├── __init__.py
├── __main__.py        # CLI entry point
├── cli.py             # argparse + dispatch
├── config.py          # central knobs (model names, MCTS params, weights)
├── llm.py             # thin wrapper over _ds4 with structured output helpers
├── tools.py           # file ops, run_command, git branching (stdlib subprocess)
├── memory.py          # blackboard, trajectory log, anti-pattern memory
├── scm.py             # Symbolic Code Model (ast + regex for PHP/JS)
├── spec_engine.py     # Intent Spec elicitation + validation
├── critics.py         # 3 critics (correctness, maintainability, performance) + ensemble
├── mcts.py            # MCTS-Code tree, UCB1, expansion, rollout, backprop
├── meta.py            # Meta-Controller + dynamic agent brewing
├── verifier.py        # final audit (per-req LLM check + marker scan + tests)
├── orchestrator.py    # main state machine wiring everything
├── discovery.py       # pre-spec workspace exploration
├── lang.py            # polyglot registry (Python/PHP/Node detection + test runners)
├── self_improve.py    # self-improvement smoke: propose _ds4 enhancements
├── debug_trace.py     # live stage-timer debug tracing
├── logging_util.py    # structured logging (file + stderr)
└── agents/            # YAML subagent templates
    ├── php_coder.yaml
    ├── node_coder.yaml
    └── polyglot_reviewer.yaml

tools/                 # stdlib helpers (importable as tools.<fn>)
├── INDEX.md
├── lex.py             # lexical/regex search with file globs
├── struct.py          # lightweight AST structural scan
├── diffs.py           # unified-diff utilities (difflib)
├── patch.py           # safe whole-file write with backup
└── subagent_run.py    # YAML template loader + CLI
```

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| **No OpenAI** — all LLM calls via `_ds4.py` | Single dependency; **DeepSeek v4 Flash** only model available |
| **Stdlib only** (Python 3.11+) | Zero pip installs beyond test framework |
| **Git branches for MCTS nodes** | Full `git diff` for free; easy rollback; no in-memory patching |
| **Polyglot via regex + AST** | Python uses `ast`; PHP/JS use regex extractors; `php -l` / `node --check` for syntax validation |
| **Pre-spec discovery** | Understands existing codebase before planning — reduces hallucinated integrations |
| **Content-hash file cache** | `.apex/fcache/` — avoids re-parsing unchanged files across runs |
| **Parallel critic evaluation** | 3 critics run concurrently via `ThreadPoolExecutor` (~3× speedup) |
| **Anti-pattern memory** | Accumulates penalties for patterns that failed in past MCTS rollouts |

---

## License

MIT
