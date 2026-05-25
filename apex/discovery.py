"""Pre-Spec Discovery — understand an existing workspace before planning.

Pipeline:
    is_empty_workspace()  → fast yes/no (zero LLM)
    FileCache             → content-hash-keyed compact summaries at .apex/fcache/
    local_summary()       → algorithmic (AST/regex), zero LLM
    directory_tree()      → importance-ranked, ignore-aware
    DiscoveryAgent.run()  → DeepSeek loop (thinking=False) with tools:
                              peek_file, read_file, grep, list_dir
                            → DiscoveryResult { mode, summary, anchors }

Persists to .apex/discovery.json; caller passes summary into SpecEngine so
the elicited spec reflects integration realities.

Stdlib only.
"""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import APEX_DIR
from .debug_trace import T

# ---- ignore rules ---------------------------------------------------------

IGNORE_DIRS = {
    ".git", ".apex", "__pycache__", ".venv", "venv", "env",
    "node_modules", "vendor", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode", "dist", "build", ".tox", ".cache",
    "target", "coverage", ".next", ".nuxt",
}
IGNORE_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".class", ".jar",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac",
    ".sqlite", ".sqlite3", ".db", ".bin", ".dat",
}
MAX_FILE_BYTES = 1_000_000  # skip files > 1MB from inline/summary
TEXT_PEEK_BYTES = 32_000     # cap text read for summary

# Language → extensions
LANG_EXT: dict[str, set[str]] = {
    "python": {".py"},
    "php": {".php"},
    "node": {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"},
}


def _is_ignored(path: Path) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return True
    if path.is_file() and path.suffix.lower() in IGNORE_SUFFIXES:
        return True
    return False


def _lang_of(path: Path) -> str:
    ext = path.suffix.lower()
    for lang, exts in LANG_EXT.items():
        if ext in exts:
            return lang
    return "other"


# ---- emptiness check ------------------------------------------------------

def is_empty_workspace(root: str) -> bool:
    """True if the directory has no non-ignored, non-hidden content.

    `.git/` and `.apex/` do NOT count as content. Hidden dotfiles DO count
    (they may indicate a real project) except for the agent's own state.
    """
    base = Path(root)
    if not base.exists():
        return True
    for entry in base.iterdir():
        name = entry.name
        if name in (".git", APEX_DIR):
            continue
        if name in IGNORE_DIRS:
            continue
        return False
    return True


# ---- importance ranking ---------------------------------------------------

_HIGH_VALUE_NAMES = {
    "readme.md", "readme.rst", "readme.txt", "readme",
    "package.json", "composer.json", "pyproject.toml", "setup.py",
    "setup.cfg", "cargo.toml", "go.mod", "gemfile", "requirements.txt",
    "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "claude.md", "agents.md",
}
_ENTRYPOINT_HINTS = {"main", "__main__", "index", "app", "server", "cli"}


def importance_score(path: Path, root: Path) -> float:
    """Higher = more orientation value. Pure heuristic, cheap."""
    name = path.name.lower()
    rel = path.relative_to(root) if path.is_absolute() else path
    depth = len(rel.parts) - 1
    score = 0.0
    if name in _HIGH_VALUE_NAMES:
        score += 10.0
    stem = path.stem.lower()
    if stem in _ENTRYPOINT_HINTS:
        score += 4.0
    if path.suffix.lower() in {".md", ".rst", ".txt"}:
        score += 1.5
    if path.suffix.lower() in {ext for exts in LANG_EXT.values() for ext in exts}:
        score += 1.0
    score -= 0.5 * depth  # shallower files matter more
    try:
        size = path.stat().st_size
        if size and size < 4_000:
            score += 0.5
        if size > 200_000:
            score -= 2.0
    except OSError:
        pass
    return score


# ---- algorithmic local summary (no LLM) -----------------------------------

_PY_DOC_LIMIT = 240
_PHP_FUNC_RE = re.compile(r"^\s*(?:public|private|protected|static|final|abstract|\s)*function\s+(\w+)\s*\(", re.MULTILINE)
_PHP_CLASS_RE = re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)", re.MULTILINE)
_JS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE)
_JS_ARROW_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(?[^=]*\)?\s*=>", re.MULTILINE)
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+(\w+)", re.MULTILINE)


def _summarize_python(text: str) -> dict:
    out: dict[str, Any] = {"defs": [], "classes": [], "imports": [], "doc": ""}
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        out["error"] = f"SyntaxError: {e.msg} (line {e.lineno})"
        return out
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        out["doc"] = tree.body[0].value.value.strip().splitlines()[0][:_PY_DOC_LIMIT]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            out["defs"].append(f"{node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            out["classes"].append({"name": node.name, "methods": methods[:12]})
        elif isinstance(node, ast.Import):
            for n in node.names:
                out["imports"].append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                out["imports"].append(f"{mod}.{n.name}" if mod else n.name)
    return out


def _summarize_regex(text: str, fn_re: re.Pattern, cls_re: re.Pattern,
                     arrow_re: re.Pattern | None = None) -> dict:
    defs = fn_re.findall(text)
    if arrow_re is not None:
        defs += arrow_re.findall(text)
    return {
        "defs": [f"{d}()" for d in defs[:40]],
        "classes": [{"name": c, "methods": []} for c in cls_re.findall(text)[:20]],
        "imports": [],
        "doc": "",
    }


def _summarize_text(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = " ".join(lines[:3])[:_PY_DOC_LIMIT]
    return {"defs": [], "classes": [], "imports": [], "doc": head}


def local_summary(path: str | Path) -> dict:
    """Return a compact, JSON-friendly summary of a file. Zero LLM."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        return {"path": str(p), "error": f"stat: {e}"}
    if st.st_size > MAX_FILE_BYTES:
        return {"path": str(p), "size": st.st_size, "skip": "too-large"}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")[:TEXT_PEEK_BYTES]
    except OSError as e:
        return {"path": str(p), "error": f"read: {e}"}
    lang = _lang_of(p)
    if lang == "python":
        body = _summarize_python(text)
    elif lang == "php":
        body = _summarize_regex(text, _PHP_FUNC_RE, _PHP_CLASS_RE)
    elif lang == "node":
        body = _summarize_regex(text, _JS_FUNC_RE, _JS_CLASS_RE, _JS_ARROW_RE)
    else:
        body = _summarize_text(text)
    body["path"] = str(p)
    body["lang"] = lang
    body["size"] = st.st_size
    body["loc"] = text.count("\n") + 1
    return body


# ---- file cache (content-hash addressed) ---------------------------------

class FileCache:
    """Two-tier cache for per-file summaries.

    Layout under ``<root>/.apex/fcache/``:

    - ``<sha256>.json`` — content-addressed summary blob
    - ``index.json``    — ``{ "<abs path>": {"sha":..., "mtime":..., "size":...} }``

    A cache hit on path uses the (mtime,size) shortcut to avoid re-hashing;
    a content-hash hit avoids regenerating the summary even after renames.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.dir = self.root / APEX_DIR / "fcache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self._index: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists():
            try:
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except Exception:
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._index), encoding="utf-8")
        os.replace(tmp, self.index_path)

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def summary_for(self, path: str | Path) -> dict:
        p = Path(path).resolve()
        key = str(p)
        try:
            st = p.stat()
        except OSError as e:
            return {"path": key, "error": f"stat: {e}"}
        entry = self._index.get(key)
        sha: str | None = None
        if entry and entry.get("mtime") == st.st_mtime and entry.get("size") == st.st_size:
            sha = entry.get("sha")
        if sha is None:
            sha = self._hash_file(p)
        blob = self.dir / f"{sha}.json"
        if blob.exists():
            try:
                data = json.loads(blob.read_text(encoding="utf-8"))
                self.hits += 1
                self._index[key] = {"sha": sha, "mtime": st.st_mtime, "size": st.st_size}
                self._save_index()
                return data
            except Exception:
                pass
        # Miss: compute fresh.
        self.misses += 1
        data = local_summary(p)
        data["sha"] = sha
        try:
            blob.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass
        self._index[key] = {"sha": sha, "mtime": st.st_mtime, "size": st.st_size}
        self._save_index()
        return data

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "ratio": (self.hits / total) if total else 0.0,
            "entries": len(self._index),
            "blobs": sum(1 for _ in self.dir.glob("*.json")
                         if _.name != "index.json"),
        }


# ---- directory tree (ignore-aware, ranked) -------------------------------

def _walk(root: Path, max_depth: int, max_entries: int) -> list[Path]:
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(out) < max_entries:
        cur, depth = stack.pop(0)
        if _is_ignored(cur):
            continue
        try:
            kids = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            continue
        for k in kids:
            if _is_ignored(k):
                continue
            out.append(k)
            if len(out) >= max_entries:
                break
            if k.is_dir() and depth + 1 < max_depth:
                stack.append((k, depth + 1))
    return out


def directory_tree(root: str, max_depth: int = 4, max_entries: int = 400) -> str:
    base = Path(root).resolve()
    if not base.exists():
        return f"(missing) {base}"
    items = _walk(base, max_depth, max_entries)
    # Build a tree string sorted by (path).
    lines = [f"{base.name}/"]
    for p in sorted(items, key=lambda x: x.relative_to(base).parts):
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        depth = len(rel.parts) - 1
        prefix = "  " * depth + ("└─ " if depth >= 0 else "")
        suffix = "/" if p.is_dir() else ""
        size = ""
        if p.is_file():
            try:
                sz = p.stat().st_size
                size = f"  ({sz}B)" if sz < 4096 else f"  ({sz//1024}K)"
            except OSError:
                pass
        lines.append(f"{prefix}{p.name}{suffix}{size}")
    if len(items) >= max_entries:
        lines.append(f"... (truncated at {max_entries} entries)")
    return "\n".join(lines)


def ranked_files(root: str, top: int = 12) -> list[Path]:
    base = Path(root).resolve()
    items = [p for p in _walk(base, max_depth=6, max_entries=2000) if p.is_file()]
    items.sort(key=lambda p: importance_score(p, base), reverse=True)
    return items[:top]


# ---- DiscoveryResult ------------------------------------------------------

@dataclass
class DiscoveryResult:
    mode: str = "new"            # new | integration | extend | unknown
    summary: str = ""            # short prose understanding
    anchors: list[str] = field(default_factory=list)  # important file paths
    languages: list[str] = field(default_factory=list)
    empty: bool = False
    cache_stats: dict = field(default_factory=dict)
    elapsed: float = 0.0
    llm_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiscoveryResult":
        return cls(
            mode=str(d.get("mode", "new")),
            summary=str(d.get("summary", "")),
            anchors=[str(a) for a in (d.get("anchors") or [])],
            languages=[str(l) for l in (d.get("languages") or [])],
            empty=bool(d.get("empty", False)),
            cache_stats=dict(d.get("cache_stats") or {}),
            elapsed=float(d.get("elapsed", 0.0)),
            llm_used=bool(d.get("llm_used", False)),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "DiscoveryResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---- DiscoveryAgent -------------------------------------------------------

_DISCOVERY_SYSTEM = """You are the APEX Pre-Spec Discovery agent.

You are given:
- A directory tree of an existing workspace.
- Compact summaries of the most-important files.
- The user's goal.
- Tools to peek/read/grep additional files on demand.

Reason briefly and call tools only when needed. Your job: decide whether
the user's goal is:
  - "new"          — the workspace is irrelevant / empty / template.
  - "integration"  — goal must be merged into an existing project.
  - "extend"       — goal is a natural extension of what's here.
Identify 1-5 important anchor files the implementer must read first.

When done, output ONLY one JSON object:
{
  "mode": "new|integration|extend",
  "summary": "<<=400 chars: what this codebase is and how the goal fits>>",
  "anchors": ["path/one", "path/two"],
  "languages": ["python","php","node"]
}
No prose outside the JSON.
"""


class DiscoveryAgent:
    """Drive a frugal, tool-augmented discovery loop.

    Uses ``subagent.SubAgent.tool_loop`` so reasoning is OFF and DeepSeek
    answers fast/cheap. Falls back to a purely-algorithmic result if the
    LLM is unavailable or errors.
    """

    def __init__(
        self,
        root: str,
        subagent: Any | None = None,
        max_turns: int = 6,
        inline_top: int = 6,
        max_inline_chars: int = 8000,
    ) -> None:
        self.root = str(Path(root).resolve())
        self.cache = FileCache(self.root)
        self._sa = subagent
        self.max_turns = max_turns
        self.inline_top = inline_top
        self.max_inline_chars = max_inline_chars

    # ---- public ----

    def run(self, user_goal: str) -> DiscoveryResult:
        t0 = time.time()
        if is_empty_workspace(self.root):
            T("DISCOVERY", "empty workspace → mode=new (no LLM)")
            return DiscoveryResult(
                mode="new", summary="Workspace is empty.", empty=True,
                cache_stats=self.cache.stats(), elapsed=time.time() - t0,
                llm_used=False,
            )

        anchors_local = [str(p.relative_to(self.root))
                         for p in ranked_files(self.root, top=self.inline_top)]
        languages = self._detect_languages()
        T("DISCOVERY", f"non-empty: top={anchors_local[:5]} langs={languages}")

        # Build inline context (free, cached summaries).
        tree = directory_tree(self.root, max_depth=4, max_entries=300)
        inline = self._build_inline_context(anchors_local)

        # Try LLM tool-loop; if unavailable, return algorithmic fallback.
        try:
            sa = self._subagent()
            tools = self._build_tools()
            user_msg = (
                f"User goal: {user_goal}\n\n"
                f"Workspace tree (depth ≤ 4):\n```\n{tree}\n```\n\n"
                f"Top-ranked file summaries (algorithmic, free):\n{inline}\n\n"
                "Decide mode, summarize, list anchors. Use tools sparingly."
            )
            raw = sa.tool_loop(
                system=_DISCOVERY_SYSTEM,
                user=user_msg,
                tools=tools,
                max_turns=self.max_turns,
                temperature=0.1,
                thinking=False,
            )
            data = _parse_json_lenient(raw)
            res = DiscoveryResult(
                mode=str(data.get("mode") or "unknown").lower(),
                summary=str(data.get("summary") or "")[:600],
                anchors=[str(a) for a in (data.get("anchors") or [])][:6],
                languages=[str(l) for l in (data.get("languages") or languages)],
                empty=False,
                cache_stats=self.cache.stats(),
                elapsed=time.time() - t0,
                llm_used=True,
            )
            if res.mode not in {"new", "integration", "extend", "unknown"}:
                res.mode = "unknown"
            return res
        except Exception as e:
            T("DISCOVERY", f"LLM loop failed → algorithmic fallback: {e}")
            return DiscoveryResult(
                mode="integration" if anchors_local else "new",
                summary=f"Algorithmic fallback (LLM unavailable: {e}). "
                        f"Top files: {', '.join(anchors_local[:5])}.",
                anchors=anchors_local[:5],
                languages=languages,
                empty=False,
                cache_stats=self.cache.stats(),
                elapsed=time.time() - t0,
                llm_used=False,
            )

    # ---- internals ----

    def _subagent(self) -> Any:
        if self._sa is not None:
            return self._sa
        # Lazy import to avoid mandatory dependency in tests.
        import sys as _sys
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from subagent import SubAgent  # type: ignore
        self._sa = SubAgent()
        return self._sa

    def _detect_languages(self) -> list[str]:
        counts: dict[str, int] = {}
        for p in _walk(Path(self.root), max_depth=6, max_entries=2000):
            if not p.is_file():
                continue
            lang = _lang_of(p)
            if lang != "other":
                counts[lang] = counts.get(lang, 0) + 1
        return [l for l, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    def _build_inline_context(self, rel_paths: list[str]) -> str:
        parts: list[str] = []
        budget = self.max_inline_chars
        for rel in rel_paths:
            abs_p = Path(self.root) / rel
            if not abs_p.exists() or not abs_p.is_file():
                continue
            summ = self.cache.summary_for(abs_p)
            chunk = f"- {rel} [{summ.get('lang','?')}, {summ.get('loc',0)} loc]: "
            if summ.get("doc"):
                chunk += summ["doc"][:160]
            sig_bits: list[str] = []
            for d in (summ.get("defs") or [])[:6]:
                sig_bits.append(d)
            for c in (summ.get("classes") or [])[:4]:
                sig_bits.append(f"class {c.get('name')}")
            if sig_bits:
                chunk += "  | " + ", ".join(sig_bits)
            chunk += "\n"
            if len(chunk) > budget:
                break
            parts.append(chunk)
            budget -= len(chunk)
        return "".join(parts) or "(no inline summaries available)"

    def _build_tools(self) -> list[dict]:
        """Return ToolDef-compatible dicts for the tool_loop."""
        root = self.root
        cache = self.cache

        def peek_file(params: dict) -> dict:
            rel = str(params.get("path", "")).strip()
            target = (Path(root) / rel).resolve()
            if not _safe_under(root, target):
                return {"error": "path escapes workspace"}
            if not target.exists() or not target.is_file():
                return {"error": f"not a file: {rel}"}
            return cache.summary_for(target)

        def read_file(params: dict) -> dict:
            rel = str(params.get("path", "")).strip()
            n = int(params.get("max_bytes", 4000))
            n = max(256, min(n, 16000))
            target = (Path(root) / rel).resolve()
            if not _safe_under(root, target):
                return {"error": "path escapes workspace"}
            if not target.exists() or not target.is_file():
                return {"error": f"not a file: {rel}"}
            try:
                text = target.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return {"error": str(e)}
            return {"path": rel, "bytes": len(text), "content": text[:n],
                    "truncated": len(text) > n}

        def grep(params: dict) -> dict:
            pat = str(params.get("pattern", ""))
            glob = str(params.get("glob", "**/*"))
            try:
                rx = re.compile(pat)
            except re.error as e:
                return {"error": f"bad regex: {e}"}
            hits: list[dict] = []
            base = Path(root)
            for p in base.rglob(glob):
                if not p.is_file() or _is_ignored(p):
                    continue
                try:
                    for i, line in enumerate(
                        p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                    ):
                        if rx.search(line):
                            hits.append({"path": str(p.relative_to(base)),
                                         "line": i, "text": line[:200]})
                            if len(hits) >= 60:
                                return {"hits": hits, "truncated": True}
                except OSError:
                    continue
            return {"hits": hits, "truncated": False}

        def list_dir(params: dict) -> dict:
            rel = str(params.get("path", ".")).strip() or "."
            target = (Path(root) / rel).resolve()
            if not _safe_under(root, target) or not target.exists():
                return {"error": f"not a dir: {rel}"}
            try:
                kids = []
                for k in sorted(target.iterdir(), key=lambda p: p.name):
                    if _is_ignored(k):
                        continue
                    kids.append({"name": k.name, "dir": k.is_dir(),
                                 "size": (k.stat().st_size if k.is_file() else 0)})
                return {"path": rel, "entries": kids[:200]}
            except OSError as e:
                return {"error": str(e)}

        return [
            {
                "name": "peek_file",
                "description": "Compact cached summary (defs/classes/imports) of one file. Free, cached.",
                "parameters": {"type": "object", "required": ["path"],
                               "properties": {"path": {"type": "string"}}},
                "handler": peek_file,
            },
            {
                "name": "read_file",
                "description": "Read up to max_bytes (default 4000) of a workspace file.",
                "parameters": {"type": "object", "required": ["path"],
                               "properties": {"path": {"type": "string"},
                                              "max_bytes": {"type": "integer"}}},
                "handler": read_file,
            },
            {
                "name": "grep",
                "description": "Regex search across files matching glob (default '**/*').",
                "parameters": {"type": "object", "required": ["pattern"],
                               "properties": {"pattern": {"type": "string"},
                                              "glob": {"type": "string"}}},
                "handler": grep,
            },
            {
                "name": "list_dir",
                "description": "List a workspace directory (non-recursive).",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}}},
                "handler": list_dir,
            },
        ]


# ---- helpers --------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_lenient(text: str) -> dict:
    if not text:
        return {}
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text or "")
    if not m:
        return {}
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _safe_under(root: str, target: Path) -> bool:
    try:
        target.relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def run_discovery(root: str, user_goal: str,
                  subagent: Any | None = None) -> DiscoveryResult:
    """Convenience: run discovery, persist to .apex/discovery.json, return result."""
    agent = DiscoveryAgent(root=root, subagent=subagent)
    res = agent.run(user_goal)
    try:
        out = Path(root) / APEX_DIR / "discovery.json"
        res.save(str(out))
    except OSError:
        pass
    return res
