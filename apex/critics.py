"""Critic ensemble — three specialized critics scoring candidate patches."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .debug_trace import T
from .llm import fast_llm, lazy_llm
from .logging_util import get_logger

_log = get_logger(__name__)


CORRECTNESS_PROMPT = """You are the Correctness Critic.
Given a requirement and a proposed code diff, rate (0..1) how well the
diff fulfills the requirement. List any missing aspects.

Requirement:
{requirement}

Context:
{context}

Diff:
{diff}

Respond ONLY with a JSON object: {{"score": <float 0..1>, "notes": [..], "suggestions": [..]}}.
"""

MAINTAINABILITY_PROMPT = """You are the Maintainability Critic.
Examine the diff for: duplication, poor naming, missing docstrings,
style/convention violations, dead code. Rate maintainability (0..1).

Requirement:
{requirement}

Context:
{context}

Diff:
{diff}

Respond ONLY with a JSON object: {{"score": <float 0..1>, "notes": [..], "suggestions": [..]}}.
"""

PERFORMANCE_PROMPT = """You are the Performance Critic.
Identify obvious inefficiencies (nested loops on large data, repeated I/O,
missing indices, O(n^2) where O(n) would do). Rate performance (0..1).

Requirement:
{requirement}

Context:
{context}

Diff:
{diff}

Respond ONLY with a JSON object: {{"score": <float 0..1>, "notes": [..], "suggestions": [..]}}.
"""


@dataclass
class CriticScore:
    name: str
    score: float
    notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class EnsembleResult:
    scores: list[CriticScore]
    aggregate: float
    weights: dict[str, float]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


class Critic:
    """Base critic — subclasses set `name` and `PROMPT`."""

    name: str = "base"
    PROMPT: str = ""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm  # resolved lazily

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = lazy_llm(fast=True)
        return self._llm

    def prompt(self, requirement: str, diff: str, context: str) -> str:
        return self.PROMPT.format(
            requirement=requirement or "(none)",
            context=context or "(none)",
            diff=diff or "(empty)",
        )

    def evaluate(self, requirement: str, diff: str, context: str = "") -> CriticScore:
        T("CRITIC", f"Evaluating {self.name} (req={requirement[:60]}, diff={len(diff)} chars)")
        text = self.prompt(requirement, diff, context)
        messages = [{"role": "user", "content": text}]
        try:
            llm = self._get_llm()
            obj = llm.chat_json(
                messages,
                schema_hint='{"score": float, "notes": [str], "suggestions": [str]}',
            )
        except Exception as e:
            _log.warning("critic %s failed: %s", self.name, e)
            T("CRITIC", f"{self.name} FAILED: {e}")
            return CriticScore(name=self.name, score=0.5, notes=[f"llm-error: {e}"])

        try:
            raw_score = obj.get("score", 0.5)
            score = _clamp01(float(raw_score))
        except Exception:
            score = 0.5
        notes = _as_str_list(obj.get("notes"))
        suggestions = _as_str_list(obj.get("suggestions"))
        if "score" not in obj:
            notes.append(f"raw: {obj}")
        return CriticScore(
            name=self.name, score=score, notes=notes, suggestions=suggestions
        )


class CorrectnessCritic(Critic):
    name = "correctness"
    PROMPT = CORRECTNESS_PROMPT


class MaintainabilityCritic(Critic):
    name = "maintainability"
    PROMPT = MAINTAINABILITY_PROMPT


class PerformanceCritic(Critic):
    name = "performance"
    PROMPT = PERFORMANCE_PROMPT


DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.5,
    "maintainability": 0.25,
    "performance": 0.25,
}


class CriticEnsemble:
    """Run three critics and aggregate weighted scores."""

    def __init__(
        self,
        llm: Any | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.llm = llm
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.critics: list[Critic] = [
            CorrectnessCritic(llm=llm),
            MaintainabilityCritic(llm=llm),
            PerformanceCritic(llm=llm),
        ]
        self._cache: dict[str, EnsembleResult] = {}

    @staticmethod
    def _cache_key(requirement: str, diff: str) -> str:
        h = hashlib.sha256()
        h.update((requirement or "").encode("utf-8"))
        h.update(b"\0")
        h.update((diff or "").encode("utf-8"))
        return h.hexdigest()

    def evaluate(
        self,
        requirement: str,
        diff: str,
        context: str = "",
        anti_pattern_penalty: float = 0.0,
    ) -> EnsembleResult:
        key = self._cache_key(requirement, diff)
        if key in self._cache:
            return self._cache[key]

        # Parallelize the three critic LLM calls — each is I/O-bound; the
        # _ds4 client + rate limiter are thread-safe enough for this small fan-out.
        with ThreadPoolExecutor(max_workers=len(self.critics)) as ex:
            futures = [
                ex.submit(c.evaluate, requirement, diff, context)
                for c in self.critics
            ]
            scores: list[CriticScore] = [f.result() for f in futures]

        total_w = sum(self.weights.get(s.name, 0.0) for s in scores) or 1.0
        weighted = sum(self.weights.get(s.name, 0.0) * s.score for s in scores)
        agg = weighted / total_w
        agg = _clamp01(agg - float(anti_pattern_penalty or 0.0))

        result = EnsembleResult(scores=scores, aggregate=agg, weights=dict(self.weights))
        self._cache[key] = result
        T("CRITIC", f"Ensemble: agg={agg:.3f} "
          f"correctness={scores[0].score:.2f} maint={scores[1].score:.2f} perf={scores[2].score:.2f}")
        return result
