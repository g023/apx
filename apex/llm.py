"""Thin LLM wrapper over `_ds4.DeepSeekV4`."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL, FAST_MODEL, MAX_LLM_TOKENS, TEMPERATURE
from .debug_trace import T, trace_llm_call

# Allow importing _ds4 from project root (sibling of `apex/`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_CACHE_MAX = 128
# Maximum number of messages kept per chat_json call (drop oldest user/assistant
# turns above the cap; system message at index 0 is always preserved).
_MAX_MESSAGES = 32


def lazy_llm(existing: Any | None = None, *, fast: bool = False) -> Any:
    """Return `existing` if non-None, otherwise the cached default/fast singleton.

    Centralizes the `if llm is None: from .llm import default_llm; llm = default_llm()`
    pattern repeated across spec_engine / critics / meta / verifier.
    """
    if existing is not None:
        return existing
    return fast_llm() if fast else default_llm()


def _cap_messages(messages: list[dict]) -> list[dict]:
    """Keep the system message (if first) plus the most recent _MAX_MESSAGES-1 turns."""
    if len(messages) <= _MAX_MESSAGES:
        return messages
    head: list[dict] = []
    if messages and messages[0].get("role") == "system":
        head = [messages[0]]
        rest = messages[1:]
    else:
        rest = messages
    keep = _MAX_MESSAGES - len(head)
    return head + rest[-keep:]


def _cache_key(model: str, messages: list[dict], temperature: float) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(f"{temperature:.4f}".encode("ascii"))
    h.update(b"\0")
    for m in messages:
        h.update((m.get("role") or "").encode("utf-8"))
        h.update(b"\1")
        h.update((m.get("content") or "").encode("utf-8"))
        h.update(b"\2")
    return h.hexdigest()


class LLM:
    """Lazy wrapper around `_ds4.DeepSeekV4`."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client: Any | None = None
        # Bounded LRU for chat_json: identical (model, messages, temp) → cached dict.
        self._json_cache: "OrderedDict[str, dict]" = OrderedDict()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import _ds4  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Cannot import _ds4 — the DeepSeek client module. "
                    "Make sure _ds4.py is in the project root (sibling of apex/). "
                    f"sys.path includes: {_ROOT}"
                ) from e
            self._client = _ds4.DeepSeekV4(model=self.model)
        return self._client

    def chat(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
        max_turns: int = 1,
    ) -> str:
        T("LLM", f"chat: model={self.model} msgs={len(messages)} max_turns={max_turns}")
        client = self._get_client()
        _msgs, choice = client.chat_with_tools(
            messages,
            max_turns=max_turns,
            temperature=temperature,
            include_reasoning=False,
        )
        try:
            content = choice["message"]["content"] or ""
            trace_llm_call("chat", self.model, len(messages), content[:120])
            return content
        except Exception:
            T("LLM", "chat: failed to extract content from response")
            return ""

    def chat_json(
        self,
        messages: list[dict],
        schema_hint: str = "",
        temperature: float = 0.1,
    ) -> dict:
        """Force JSON object output. Retries once on parse failure."""
        T("LLM", f"chat_json: model={self.model} msgs={len(messages)} hint={schema_hint!r}")
        # Skip the system-reminder if the caller already supplied one at index 0.
        sys_reminder_text = (
            "Respond with ONLY a single valid JSON object. "
            "Do not include code fences, prose, or anything else."
            + (f" Schema hint: {schema_hint}" if schema_hint else "")
        )
        has_sys = bool(messages) and messages[0].get("role") == "system"
        if has_sys:
            msgs = list(messages)
            # Augment the existing system message instead of stacking another.
            msgs[0] = {
                "role": "system",
                "content": (msgs[0].get("content") or "") + "\n\n" + sys_reminder_text,
            }
        else:
            msgs = [{"role": "system", "content": sys_reminder_text}] + list(messages)
        msgs = _cap_messages(msgs)

        # Cache lookup.
        ck = _cache_key(self.model, msgs, temperature)
        cached = self._json_cache.get(ck)
        if cached is not None:
            self._json_cache.move_to_end(ck)
            T("LLM", "chat_json: cache hit")
            return dict(cached)

        client = self._get_client()

        # First attempt: prefix completion forcing a JSON object.
        try:
            T("LLM", "chat_json: trying prefix_complete")
            raw = client.chat_prefix_complete(
                messages=msgs,
                prefix_content="{\n",
                stop=None,
                max_tokens=MAX_LLM_TOKENS,
                temperature=temperature,
            )
            text = "{\n" + (raw or "")
            result = _parse_json_object(text)
            T("LLM", f"chat_json: prefix_complete OK, keys={list(result.keys())}")
            self._json_cache_put(ck, result)
            return result
        except Exception as e:
            T("LLM", f"chat_json: prefix_complete failed: {e}, trying fallback")

        # Retry: plain chat then regex-extract.
        try:
            raw = self.chat(msgs, temperature=temperature, max_turns=1)
            result = _parse_json_object(raw)
            T("LLM", f"chat_json: fallback OK, keys={list(result.keys())}")
            self._json_cache_put(ck, result)
            return result
        except Exception as e:
            T("LLM", f"chat_json: fallback also failed: {e}")
            raise ValueError(f"chat_json failed to parse: {e}")

    def _json_cache_put(self, key: str, value: dict) -> None:
        self._json_cache[key] = dict(value)
        self._json_cache.move_to_end(key)
        while len(self._json_cache) > _JSON_CACHE_MAX:
            self._json_cache.popitem(last=False)

    def fim(self, prefix: str, suffix: str, max_tokens: int = 128) -> str:
        client = self._get_client()
        return client.fim_complete(prompt=prefix, suffix=suffix, max_tokens=max_tokens)


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object from arbitrary text."""
    if not text:
        raise ValueError("empty response")
    # Try direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try extracting first {...} block.
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


@lru_cache(maxsize=1)
def default_llm() -> LLM:
    return LLM(model=DEFAULT_MODEL)


@lru_cache(maxsize=1)
def fast_llm() -> LLM:
    return LLM(model=FAST_MODEL)
