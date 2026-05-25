"""Thread-safe blackboard + persistence layer."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import STATE_DIR


class Blackboard:
    """Thread-safe key/value store with append semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def append(self, key: str, value: Any) -> None:
        with self._lock:
            cur = self._data.get(key)
            if cur is None:
                self._data[key] = [value]
            elif isinstance(cur, list):
                cur.append(value)
            else:
                self._data[key] = [cur, value]

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


class TrajectoryLog:
    """Append-only JSONL writer."""

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = str(Path(STATE_DIR()) / "memory" / "trajectory.jsonl")
        self.path = path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event_type: str, **fields: Any) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
        }
        rec.update(fields)
        line = json.dumps(rec, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class AntiPatternMemory:
    """Persistent JSON map of substring patterns to penalty weights."""

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = str(Path(STATE_DIR()) / "memory" / "anti_patterns.json")
        self.path = path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, float]:
        p = Path(self.path)
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save(self, data: dict[str, float]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def add(self, pattern: str, penalty: float = 0.1) -> None:
        with self._lock:
            data = self._load()
            data[pattern] = float(penalty)
            self._save(data)

    def all(self) -> dict[str, float]:
        with self._lock:
            return self._load()

    def penalty_for(self, text: str) -> float:
        with self._lock:
            data = self._load()
        if not text:
            return 0.0
        total = 0.0
        for pat, pen in data.items():
            if pat and pat in text:
                total += pen
        return total


@lru_cache(maxsize=1)
def blackboard() -> Blackboard:
    return Blackboard()


@lru_cache(maxsize=1)
def trajectory() -> TrajectoryLog:
    return TrajectoryLog()


@lru_cache(maxsize=1)
def anti_patterns() -> AntiPatternMemory:
    return AntiPatternMemory()
