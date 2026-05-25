"""Central configuration constants for g023's apx agent."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import os
from pathlib import Path

# Model knobs
DEFAULT_MODEL = "deepseek-v4-flash"
FAST_MODEL = "deepseek-v4-flash"

# LLM call defaults
MAX_LLM_TOKENS = 64000
TEMPERATURE = 0.2

# MCTS knobs
MAX_MCTS_ITERATIONS = 50
MCTS_EXPLORATION_CONSTANT = 1.4
PARALLEL_BRANCHES = 3

# Reward weights
W_TEST = 1.0
W_SPEC = 1.0
W_CRITIC = 0.5
W_ERRORS = 1.0
W_TOKENS = 0.001

# State dir
APEX_DIR = ".apex"
MAX_CLARIFY_QUESTIONS = 3


def STATE_DIR() -> str:
    """Return absolute path to `.apex/` under cwd, ensuring it exists."""
    p = Path(os.getcwd()) / APEX_DIR
    p.mkdir(parents=True, exist_ok=True)
    (p / "memory").mkdir(parents=True, exist_ok=True)
    return str(p.resolve())
