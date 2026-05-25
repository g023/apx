"""Meta-Controller + dynamic agent brewing.

The Meta-Controller decomposes a Spec into a list of `SpecialistDef`s
(role + system prompt + tool whitelist + success criteria). Each
specialist can be "brewed" into a `DynamicAgent` — a thin ReAct-style
loop driven by the LLM that emits JSON actions limited to a
whitelisted tool set.
"""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import tools as _tools
from .debug_trace import T
from .logging_util import get_logger
from .memory import trajectory

_log = get_logger(__name__)


# Tools the dynamic agent layer is allowed to expose. Everything outside
# this set is rejected even if a SpecialistDef whitelists it.
ALLOWED_TOOLS: set[str] = {
    "read_file",
    "write_patch",
    "list_files",
    "run_command",
    "git_diff",
    "scm_query",
    "scm_summary",
}


META_PROMPT = """You are the APEX Meta-Controller.
Given an Intent Specification, decompose the work into a small list
of specialist agents. For each specialist provide:
  - name: short snake_case identifier
  - system_prompt: focused instructions for that role
  - tool_whitelist: subset of {tools}
  - success_criteria: observable, requirement-linked

Respond ONLY with JSON of the form:
{{
  "specialists": [
    {{"name": "...", "system_prompt": "...",
      "tool_whitelist": ["..."], "success_criteria": "..."}}
  ]
}}
""".replace("{tools}", json.dumps(sorted(ALLOWED_TOOLS)))


@dataclass
class SpecialistDef:
    name: str
    system_prompt: str
    tool_whitelist: list[str] = field(default_factory=list)
    success_criteria: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpecialistDef":
        return cls(
            name=str(d.get("name", "specialist")),
            system_prompt=str(d.get("system_prompt", "")),
            tool_whitelist=[str(t) for t in (d.get("tool_whitelist") or [])],
            success_criteria=str(d.get("success_criteria", "")),
        )


class MetaController:
    """Decompose a Spec into a list of SpecialistDefs via the LLM."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> Any:
        from .llm import lazy_llm
        self._llm = lazy_llm(self._llm)
        return self._llm

    def decompose(self, spec: Any) -> list[SpecialistDef]:
        T("META", "Decomposing spec into specialists")
        spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        messages = [
            {"role": "system", "content": META_PROMPT},
            {
                "role": "user",
                "content": "Intent Specification:\n" + json.dumps(spec_dict, indent=2),
            },
        ]
        try:
            llm = self._get_llm()
            obj = llm.chat_json(messages, schema_hint="{specialists:[...]}")
            raw = obj.get("specialists") or []
            out: list[SpecialistDef] = []
            for entry in raw:
                if isinstance(entry, dict) and entry.get("name"):
                    out.append(SpecialistDef.from_dict(entry))
            if out:
                trajectory().log(
                    "meta_decompose",
                    count=len(out),
                    names=[s.name for s in out],
                )
                T("META", f"Decomposed into {len(out)} specialists: {[s.name for s in out]}")
                return out
        except Exception as e:
            _log.warning("MetaController.decompose failed: %s", e)
            T("META", f"Decompose FAILED: {e}")
        # Fallback: a single general coder.
        default = SpecialistDef(
            name="general_coder",
            system_prompt=(
                "You are a focused coding assistant. Implement the task "
                "by reading and writing files via the allowed tools."
            ),
            tool_whitelist=sorted(ALLOWED_TOOLS),
            success_criteria="Task completed; all tests pass.",
        )
        trajectory().log("meta_decompose_fallback", names=[default.name])
        T("META", "Falling back to general_coder")
        return [default]


# ---------------- Dynamic agent ----------------


def _build_tool_registry(
    whitelist: list[str],
    scm: Any | None,
) -> dict[str, Callable[..., Any]]:
    """Return name -> callable for tools in (whitelist ∩ ALLOWED_TOOLS)."""
    allowed = [t for t in whitelist if t in ALLOWED_TOOLS]
    reg: dict[str, Callable[..., Any]] = {}

    def _read_file(path: str, start: int = 1, end: int | None = None) -> str:
        return _tools.read_file(path, start=start, end=end)

    def _write_patch(path: str, content: str) -> str:
        _tools.write_patch(path, content)
        return f"wrote {path} ({len(content)} bytes)"

    def _list_files(root: str = ".", pattern: str = "**/*.py") -> list[str]:
        return _tools.list_files(root=root, pattern=pattern)

    def _run_command(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> dict:
        rc, out, err = _tools.run_command(cmd, cwd=cwd, timeout=timeout)
        return {"rc": rc, "stdout": out, "stderr": err}

    def _git_diff(ref_a: str = "HEAD", ref_b: str | None = None, cwd: str | None = None) -> str:
        return _tools.git_diff(ref_a=ref_a, ref_b=ref_b, cwd=cwd)

    def _scm_query(name: str) -> list[dict]:
        if scm is None:
            return []
        syms = scm.find_symbol(name) or []
        return [
            {
                "name": s.name,
                "kind": s.kind,
                "file": s.file,
                "lineno": s.lineno,
                "qualname": s.qualname,
            }
            for s in syms
        ]

    def _scm_summary(max_chars: int = 1500) -> str:
        if scm is None:
            return ""
        return scm.summary(max_chars=max_chars)

    candidates = {
        "read_file": _read_file,
        "write_patch": _write_patch,
        "list_files": _list_files,
        "run_command": _run_command,
        "git_diff": _git_diff,
        "scm_query": _scm_query,
        "scm_summary": _scm_summary,
    }
    for name in allowed:
        if name in candidates:
            reg[name] = candidates[name]
    return reg


_AGENT_SYS_SUFFIX = """

You have access to these tools (by name): {tool_names}

On each turn, respond with EXACTLY one JSON object, either:
  {{"action": "<tool_name>", "args": {{...}}}}
to call a tool, or
  {{"final": "<answer>"}}
to finish. Output JSON ONLY — no prose, no fences.
"""


class DynamicAgent:
    """A small ReAct-style agent restricted to a tool whitelist."""

    def __init__(
        self,
        definition: SpecialistDef,
        llm: Any | None = None,
        scm: Any | None = None,
    ) -> None:
        self.definition = definition
        self._llm = llm
        self.scm = scm
        self.tools = _build_tool_registry(definition.tool_whitelist, scm)

    def _get_llm(self) -> Any:
        from .llm import lazy_llm
        self._llm = lazy_llm(self._llm)
        return self._llm

    def _system_prompt(self) -> str:
        names = sorted(self.tools.keys())
        return self.definition.system_prompt + _AGENT_SYS_SUFFIX.format(
            tool_names=json.dumps(names)
        )

    def _dispatch(self, action: str, args: dict) -> str:
        if action not in self.tools:
            return f"error: tool '{action}' not whitelisted"
        fn = self.tools[action]
        try:
            result = fn(**(args or {}))
        except TypeError as e:
            return f"error: bad args for {action}: {e}"
        except Exception as e:
            return f"error: tool {action} raised: {e}"
        if isinstance(result, (dict, list)):
            return json.dumps(result, default=str)[:4000]
        return str(result)[:4000]

    def run(self, task: str, max_turns: int = 6) -> str:
        llm = self._get_llm()
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Task: {task}\nSuccess: {self.definition.success_criteria}"},
        ]
        final_content = ""
        turns_taken = 0
        for turn in range(max_turns):
            turns_taken = turn + 1
            try:
                obj = llm.chat_json(
                    messages,
                    schema_hint='{"action":str,"args":obj} or {"final":str}',
                )
            except Exception as e:
                _log.warning("dynamic_agent[%s] llm error: %s", self.definition.name, e)
                final_content = f"error: {e}"
                break
            if "final" in obj:
                final_content = str(obj.get("final", ""))
                break
            action = str(obj.get("action", "")).strip()
            args = obj.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            observation = self._dispatch(action, args)
            messages.append({"role": "assistant", "content": json.dumps(obj)})
            messages.append({"role": "user", "content": f"observation: {observation}"})
        trajectory().log(
            "dynamic_agent",
            name=self.definition.name,
            turns=turns_taken,
            final=final_content[:200],
        )
        return final_content


def brew_agent(
    definition: SpecialistDef,
    llm: Any | None = None,
    scm: Any | None = None,
) -> DynamicAgent:
    """Factory for a dynamic agent from a SpecialistDef."""
    return DynamicAgent(definition=definition, llm=llm, scm=scm)
