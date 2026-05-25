#!/usr/bin/env python3
"""
_ds4.py – Minimal DeepSeek client with conversation truncation.
No external dependencies besides standard library.

Author: g023
License: MIT
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
MAX_OUTPUT_TOKENS = 128000
HTTP_TIMEOUT = 600
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_SLEEP = 1.0
RETRY_MAX_SLEEP = 60.0
MAX_TOOL_TURNS = 12
RATE_LIMIT_REQUESTS_PER_SECOND = 5
RATE_LIMIT_BURST = 10
TOOL_EXECUTION_TIMEOUT = 300  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Token Bucket Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, rate: float = RATE_LIMIT_REQUESTS_PER_SECOND,
                 burst: int = RATE_LIMIT_BURST):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            deficit = tokens - self.tokens
            wait = deficit / self.rate
            self.tokens = 0.0
            self.last_refill += wait
        if wait > 0:
            time.sleep(wait)
        return wait

    def __call__(self, tokens: float = 1.0) -> float:
        return self.acquire(tokens)


_rate_limiter = _RateLimiter()


# ─────────────────────────────────────────────────────────────────────────────
# API Key resolution (relative to script location)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_api_key() -> str:
    """Look for K.dat in the script's parent directory (../K.dat)."""
    script_dir = Path(__file__).resolve().parent
    key_file = script_dir.parent / "K.dat"
    try:
        with open(key_file, "r") as f:
            key = f.read().strip()
            if key:
                return key
    except Exception:
        pass
    # fallback to env var
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    raise RuntimeError(
        "DeepSeek API key not found. Place K.dat one directory above the script "
        "or set DEEPSEEK_API_KEY environment variable."
    )


def _retry_with_backoff(req_fn, max_attempts=MAX_RETRY_ATTEMPTS):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        _rate_limiter.acquire(1.0)
        try:
            return req_fn()
        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read().decode() if e.fp else ""
            last_exc = RuntimeError(f"DeepSeek HTTP {status}: {body}")
            if status == 429:
                sleep_time = min(RETRY_BASE_SLEEP * (2 ** (attempt - 1)), RETRY_MAX_SLEEP)
            elif status >= 500:
                sleep_time = min(RETRY_BASE_SLEEP * (2 ** (attempt - 1)), RETRY_MAX_SLEEP)
            else:
                raise last_exc
            if attempt < max_attempts:
                time.sleep(sleep_time)
            else:
                raise last_exc
        except urllib.error.URLError as e:
            last_exc = RuntimeError(f"Connection error: {e.reason}")
            if attempt < max_attempts:
                time.sleep(min(RETRY_BASE_SLEEP * (2 ** (attempt - 1)), RETRY_MAX_SLEEP))
            else:
                raise last_exc
    raise last_exc


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    handler: Callable | None = None
    max_result_chars: int = 8000

    def to_openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class StreamBuffer:
    """Accumulates streaming response chunks into complete message."""
    def __init__(self):
        self.reasoning = ""
        self.content = ""
        self.tool_calls: Dict[int, dict] = {}
        self.finish_reason = None
        self.usage = None
        self.chunk_count = 0

    def process_chunk(self, chunk: dict) -> None:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if "reasoning_content" in delta:
                rc = delta["reasoning_content"]
                self.reasoning = "" if rc is None else self.reasoning + rc
            if "content" in delta:
                ct = delta["content"]
                self.content = "" if ct is None else self.content + ct
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index")
                if idx is None:
                    continue
                if idx not in self.tool_calls:
                    self.tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                cur = self.tool_calls[idx]
                if tc.get("id"):
                    cur["id"] = tc["id"]
                if tc.get("type"):
                    cur["type"] = tc["type"]
                func = tc.get("function", {})
                if func.get("name"):
                    cur["function"]["name"] = func["name"]
                if func.get("arguments"):
                    cur["function"]["arguments"] += func["arguments"]
            if "message" in choice:
                msg = choice["message"]
                if msg.get("reasoning_content") is not None:
                    self.reasoning = msg["reasoning_content"]
                if msg.get("content") is not None:
                    self.content = msg["content"]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if "usage" in chunk:
            self.usage = chunk["usage"]

    def build_assistant_message(self) -> dict:
        msg = {"role": "assistant"}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.content:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[k] for k in sorted(self.tool_calls)]
        return msg


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek Client (with context truncation)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSeekV4:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        http_timeout: int = HTTP_TIMEOUT,
        thinking_enabled: bool = False,
    ):
        self.api_key = api_key or _resolve_api_key()
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.http_timeout = http_timeout
        self.thinking_enabled = thinking_enabled
        self.tools: List[ToolDef] = []
        self.total_tokens_used = 0
        self.api_calls = 0

    def add_tool(self, tool: ToolDef) -> None:
        self.tools.append(tool)

    def set_thinking_mode(self, enabled: bool) -> None:
        self.thinking_enabled = enabled

    def get_stats(self) -> dict:
        return {
            "total_tokens_used": self.total_tokens_used,
            "api_calls": self.api_calls,
            "avg_tokens_per_call": self.total_tokens_used // max(1, self.api_calls),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Truncation helper – keep only last N messages and truncate long content
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _truncate_conversation(messages: List[dict], max_messages: int = 20) -> List[dict]:
        """Keep only the last `max_messages` messages, ensuring tool message integrity.
        Aggressively truncate older tool results to save context."""
        # First trim to max_messages if needed
        if len(messages) > max_messages:
            result = []
            if messages and messages[0].get("role") == "system":
                result.append(messages[0])
                messages = messages[1:]

            keep_from = len(messages) - (max_messages - len(result))
            if keep_from < 0:
                keep_from = 0
            result.extend(messages[keep_from:])
        else:
            result = messages

        # Remove orphaned tool messages (tool message without matching assistant with tool_calls)
        final = []
        for i, msg in enumerate(result):
            if msg.get("role") == "tool":
                found_match = False
                for j in range(i - 1, -1, -1):
                    if result[j].get("role") == "assistant":
                        tool_calls = result[j].get("tool_calls", [])
                        tc_ids = [tc.get("id") for tc in tool_calls]
                        if msg.get("tool_call_id") in tc_ids:
                            found_match = True
                        break
                if not found_match:
                    continue
            final.append(msg)

        # Aggressive truncation: keep last 3 tool results at full length, truncate older ones
        for i, msg in enumerate(final):
            if msg.get("role") == "tool":
                # Count how many tool results come after this one
                remaining_tools = sum(1 for m in final[i+1:] if m.get("role") == "tool")

                content = msg.get("content", "")
                if isinstance(content, str):
                    # Keep last 3 tool results at full 2000 chars, truncate older ones to 500 chars
                    if remaining_tools < 3:
                        if len(content) > 2000:
                            msg["content"] = content[:2000] + "\n[... truncated ...]"
                    else:
                        if len(content) > 500:
                            msg["content"] = content[:500] + "\n[... truncated ...]"
            elif msg.get("role") in ("user", "assistant"):
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 3000:
                    msg["content"] = content[:3000] + "\n[... truncated ...]"

        return final

    # ─────────────────────────────────────────────────────────────────────────
    # Core API call
    # ─────────────────────────────────────────────────────────────────────────

    def _make_request(self, payload: dict, stream: bool) -> urllib.request.Request:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        return urllib.request.Request(
            f"{DEEPSEEK_BASE}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )

    def _parse_stream(self, response) -> Iterator[dict]:
        """Parse Server-Sent Events from streaming response."""
        for line in response:
            line = line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    def chat_with_tools(
        self,
        messages: List[dict],
        max_turns: int = MAX_TOOL_TURNS,
        temperature: float = 0.2,
        include_reasoning: bool = True,
    ) -> Tuple[List[dict], dict]:
        """
        Execute a tool‑augmented conversation with reasoning support.
        Returns (updated_messages, final_response_choice_dict).

        Args:
            messages: Conversation history
            max_turns: Max turns for tool execution loop
            temperature: Model temperature
            include_reasoning: Include reasoning in tool call instructions
        """
        import inspect
        import concurrent.futures

        working = [m.copy() for m in messages]
        final_choice = None
        turn = 0
        tool_turns = 0

        while turn < max_turns:
            turn += 1
            working = self._truncate_conversation(working, max_messages=20)

            thinking_config = (
                {"type": "enabled", "budget_tokens": 8000}
                if self.thinking_enabled
                else {"type": "disabled"}
            )

            payload = {
                "model": self.model,
                "messages": working,
                "stream": False,
                "max_tokens": self.max_output_tokens,
                "temperature": temperature,
                "thinking": thinking_config,
            }
            if self.tools:
                payload["tools"] = [t.to_openai_spec() for t in self.tools]

            req = self._make_request(payload, stream=False)
            def _do():
                with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                    return json.loads(resp.read().decode())
            body = _retry_with_backoff(_do)

            self.api_calls += 1
            if "usage" in body:
                self.total_tokens_used += body["usage"].get("total_tokens", 0)

            choice = body["choices"][0]
            final_choice = choice
            message = choice["message"].copy()

            working.append(message)

            if not message.get("tool_calls"):
                break

            tool_turns += 1

            for tc in message["tool_calls"]:
                tool_name = tc["function"]["name"]
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool or not tool.handler:
                    result = {"error": f"Tool '{tool_name}' not found"}
                else:
                    try:
                        sig = inspect.signature(tool.handler)
                        params = list(sig.parameters.keys())
                        args = ()
                        kwargs = {}
                        if len(params) == 1 and params[0] == "params":
                            args = (json.loads(tc["function"]["arguments"]),)
                        else:
                            kwargs = json.loads(tc["function"]["arguments"])
                    except Exception as e:
                        result = {"error": f"Argument parsing failed: {e}"}
                    else:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(tool.handler, *args, **kwargs)
                            try:
                                res = future.result(timeout=TOOL_EXECUTION_TIMEOUT)
                                result = {"output": res} if isinstance(res, str) else res
                            except concurrent.futures.TimeoutError:
                                result = {"error": f"Timeout after {TOOL_EXECUTION_TIMEOUT}s"}

                result_str = json.dumps(result, default=str)
                if tool and tool.max_result_chars and len(result_str) > tool.max_result_chars:
                    result_str = result_str[:tool.max_result_chars] + "\n[... truncated ...]"

                working.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        return working, final_choice

    # ─────────────────────────────────────────────────────────────────────────
    # FIM (Fill-in-the-Middle) — Beta endpoint
    # ─────────────────────────────────────────────────────────────────────────

    def fim_complete(
        self,
        prompt: str,
        suffix: str = "",
        max_tokens: int = 128,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        """Fill-in-the-Middle completion via Beta endpoint."""
        import urllib.request as _urllib_request
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            f"{DEEPSEEK_BASE}/beta/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        def _do():
            with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                return json.loads(resp.read().decode())
        body = _retry_with_backoff(_do)
        self.api_calls += 1
        try:
            return body["choices"][0]["text"]
        except (KeyError, IndexError):
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # Chat Prefix Completion — Beta endpoint
    # ─────────────────────────────────────────────────────────────────────────

    def chat_prefix_complete(
        self,
        messages: list[dict],
        prefix_content: str = "",
        stop: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        """Force the model to continue from a specific assistant prefix."""
        import urllib.request as _urllib_request
        working = [m.copy() for m in messages]
        if prefix_content:
            working.append({
                "role": "assistant",
                "content": prefix_content,
                "prefix": True,
            })
        payload = {
            "model": model or self.model,
            "messages": working,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            f"{DEEPSEEK_BASE}/beta/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        def _do():
            with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                return json.loads(resp.read().decode())
        body = _retry_with_backoff(_do)
        self.api_calls += 1
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""