"""Intent Specification engine — LLM-elicited structured spec."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .config import MAX_CLARIFY_QUESTIONS
from .debug_trace import T


ELICIT_PROMPT = """You are the Intent Specification Engine.
Given a user goal, output ONLY a single JSON object with these keys:

{
  "goal": "<unambiguous restatement>",
  "language": "python|php|node",
  "requirements": [
    {"id": "R1", "description": "...", "acceptance": "concrete observable criterion"}
  ],
  "constraints": ["non-functional constraints, style, performance, security"],
  "schemas": {"<name>": {<json schema or contract sketch>}},
  "unknowns": ["explicit list of ambiguities"],
  "questions": ["up to 3 prioritized clarifying questions"],
  "confidence": 0.0
}

Rules:
- requirements MUST have unique ids R1..Rn.
- acceptance must be testable.
- language MUST be one of "python", "php", "node" — infer from the goal
  (e.g. mentions of PHP, Laravel, composer => "php"; Node, npm, Express,
  JavaScript, TypeScript => "node"; otherwise "python").
- confidence is in [0,1]; lower if many unknowns.
- Output JSON only. No prose, no fences.
"""

_VALID_LANGUAGES = {"python", "php", "node"}


@dataclass
class Requirement:
    id: str
    description: str
    acceptance: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Requirement":
        return cls(
            id=str(d.get("id", "")),
            description=str(d.get("description", "")),
            acceptance=str(d.get("acceptance", "")),
        )


@dataclass
class Spec:
    goal: str
    requirements: list[Requirement] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    schemas: dict = field(default_factory=dict)
    unknowns: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    language: str = "python"

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "language": self.language,
            "requirements": [r.to_dict() for r in self.requirements],
            "constraints": list(self.constraints),
            "schemas": dict(self.schemas),
            "unknowns": list(self.unknowns),
            "questions": list(self.questions),
            "confidence": float(self.confidence),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        reqs_raw = d.get("requirements") or []
        reqs: list[Requirement] = []
        for i, r in enumerate(reqs_raw, 1):
            if isinstance(r, dict):
                if not r.get("id"):
                    r = dict(r)
                    r["id"] = f"R{i}"
                reqs.append(Requirement.from_dict(r))
            elif isinstance(r, str):
                reqs.append(Requirement(id=f"R{i}", description=r, acceptance=""))
        try:
            conf = float(d.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        lang = str(d.get("language", "python")).lower().strip()
        if lang not in _VALID_LANGUAGES:
            lang = "python"
        return cls(
            goal=str(d.get("goal", "")),
            requirements=reqs,
            constraints=[str(c) for c in (d.get("constraints") or [])],
            schemas=dict(d.get("schemas") or {}),
            unknowns=[str(u) for u in (d.get("unknowns") or [])],
            questions=[str(q) for q in (d.get("questions") or [])],
            confidence=conf,
            language=lang,
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Spec":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class SpecEngine:
    """Elicit + refine an Intent Specification via an LLM."""

    def elicit(self, user_goal: str, llm=None, context: str = "") -> Spec:
        T("SPEC", f"Eliciting spec for goal: {user_goal[:100]}")
        from .llm import lazy_llm
        llm = lazy_llm(llm)
        user_content = f"User goal: {user_goal}"
        if context:
            user_content = (
                f"Workspace context (from Pre-Spec Discovery):\n{context}\n\n"
                + user_content
                + "\n\nFactor the workspace context into requirements: prefer "
                  "extension/integration over rebuilding, and reference existing "
                  "anchor files where appropriate."
            )
        messages = [
            {"role": "system", "content": ELICIT_PROMPT},
            {"role": "user", "content": user_content},
        ]
        T("SPEC", "Calling LLM for spec elicitation...")
        data = llm.chat_json(messages, schema_hint="APEX Spec v1")
        spec = Spec.from_dict(data)
        T("SPEC", f"Elicited: {len(spec.requirements)} reqs, confidence={spec.confidence:.2f}, "
          f"questions={len(spec.questions)}")
        return spec

    def clarify(self, spec: Spec, answers: dict[str, str], llm=None) -> Spec:
        T("SPEC", f"Clarifying spec with {len(answers)} answers")
        from .llm import lazy_llm
        llm = lazy_llm(llm)
        prior = spec.to_dict()
        msg = (
            "Refine the spec given the following clarifications. "
            "Resolve referenced unknowns/questions. Keep all prior IDs stable.\n"
            f"Prior spec:\n{json.dumps(prior)}\n"
            f"Answers (question -> answer):\n{json.dumps(answers)}\n"
        )
        messages = [
            {"role": "system", "content": ELICIT_PROMPT},
            {"role": "user", "content": msg},
        ]
        data = llm.chat_json(messages, schema_hint="APEX Spec v1 refined")
        spec = Spec.from_dict(data)
        T("SPEC", f"Clarified: {len(spec.requirements)} reqs, confidence={spec.confidence:.2f}")
        return spec

    def run(
        self,
        user_goal: str,
        ask: Callable[[str], str] | None = None,
        llm=None,
        context: str = "",
    ) -> Spec:
        T("SPEC", "SpecEngine.run starting")
        spec = self.elicit(user_goal, llm=llm, context=context)
        if ask is None:
            T("SPEC", "No ask callback, returning initial spec")
            return spec
        rounds = 0
        while rounds < MAX_CLARIFY_QUESTIONS and spec.questions and spec.confidence < 0.9:
            answers: dict[str, str] = {}
            # Ask top question this round.
            q = spec.questions[0]
            T("SPEC", f"Clarify round {rounds+1}: asking '{q[:80]}'")
            try:
                ans = ask(q)
            except Exception:
                T("SPEC", "ask callback raised, breaking")
                break
            if not ans:
                T("SPEC", "empty answer, breaking")
                break
            answers[q] = ans
            spec = self.clarify(spec, answers, llm=llm)
            rounds += 1
        T("SPEC", f"SpecEngine.run done: {len(spec.requirements)} reqs, confidence={spec.confidence:.2f}, "
          f"rounds={rounds}")
        return spec
