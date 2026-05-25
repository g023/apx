"""Symbolic Code Model — AST-driven map of symbols, imports, and calls."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import ast
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import lang as _lang
from .config import STATE_DIR
from .debug_trace import T
from . import tools as _tools

_SKIP_DIRS = {
    ".git", ".apex", "__pycache__", ".venv", ".mypy_cache",
    ".pytest_cache", "node_modules", "vendor",
}

# Lightweight regex extractors for non-Python files. Not perfect — meant to
# expose top-level function / class names to the rest of the agent.
_PHP_FUNC_RE = re.compile(
    r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
)
_PHP_CLASS_RE = re.compile(
    r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
)
_PHP_USE_RE = re.compile(r"(?:^|\n)\s*use\s+([\\A-Za-z0-9_]+)\s*;")

_JS_FUNC_RE = re.compile(
    r"\b(?:async\s+)?function\s*\*?\s+([A-Za-z_$][\w$]*)\s*\(",
)
_JS_ARROW_RE = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
)
_JS_CLASS_RE = re.compile(
    r"\bclass\s+([A-Za-z_$][\w$]*)",
)
_JS_IMPORT_RE = re.compile(
    r"""(?:^|\n)\s*(?:import[^'"]*['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\))""",
)


@dataclass
class Symbol:
    name: str
    kind: str  # "function" | "class" | "variable" | "import"
    file: str
    lineno: int
    end_lineno: int | None = None
    signature: str | None = None
    docstring: str | None = None
    parent: str | None = None  # qualified parent name (e.g. ClassName)

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass
class FileInfo:
    path: str
    mtime: float
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    # (caller_qualname, callee_name)


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args})"


def _is_skipped(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & _SKIP_DIRS)


class SCM:
    """Symbolic Code Model — fast static index of a Python codebase."""

    def __init__(self, root: str) -> None:
        self.root = str(Path(root).resolve())
        self.files: dict[str, FileInfo] = {}
        self.symbol_index: dict[str, list[Symbol]] = {}

    # -------------- Public API --------------

    def scan(self, paths: list[str] | None = None) -> None:
        """Full or partial reparse across all supported languages."""
        if paths is None:
            target = list(self._iter_code_files())
            self.files.clear()
            T("SCM", f"Full scan: {len(target)} code files")
        else:
            target = []
            for raw in paths:
                p = Path(raw)
                if not p.is_absolute():
                    p = Path(self.root) / p
                if (
                    p.exists()
                    and p.suffix.lower() in _lang.ALL_CODE_EXTENSIONS
                    and not _is_skipped(p)
                ):
                    target.append(p)
                # Drop file regardless to ensure stale removal on missing.
                key = str(p.resolve())
                self.files.pop(key, None)
            T("SCM", f"Partial scan: {len(target)} files")

        parsed = 0
        for p in target:
            try:
                lang = _lang.lang_for_file(str(p))
                if lang == "python":
                    info = self._parse_file(str(p))
                elif lang == "php":
                    info = self._parse_php_file(str(p))
                elif lang == "node":
                    info = self._parse_js_file(str(p))
                else:
                    continue
                self.files[info.path] = info
                parsed += 1
            except Exception:
                # Skip unparseable; user can re-scan after fix.
                continue
        self._rebuild_index()
        T("SCM", f"Scan done: {parsed} parsed, {len(self.files)} files in index, "
          f"{len(self.symbol_index)} symbols")

    def find_symbol(self, name: str) -> list[Symbol]:
        return list(self.symbol_index.get(name, []))

    def callers_of(self, qualname: str) -> list[Symbol]:
        """Return symbols (functions/methods) whose body calls qualname or its short name."""
        short = qualname.rsplit(".", 1)[-1]
        out: list[Symbol] = []
        seen: set[tuple[str, str, int]] = set()
        for info in self.files.values():
            for caller, callee in info.calls:
                if callee == short or callee == qualname:
                    # find caller symbol
                    for sym in info.symbols:
                        if sym.qualname == caller and sym.kind in ("function",):
                            key = (sym.file, sym.qualname, sym.lineno)
                            if key not in seen:
                                seen.add(key)
                                out.append(sym)
                            break
        return out

    def get_function_source(self, qualname: str) -> str | None:
        for info in self.files.values():
            for sym in info.symbols:
                if sym.qualname == qualname and sym.kind == "function":
                    return self._slice_source(info.path, sym.lineno, sym.end_lineno)
            for sym in info.symbols:
                if sym.kind == "function" and sym.name == qualname and sym.parent is None:
                    return self._slice_source(info.path, sym.lineno, sym.end_lineno)
        return None

    def summary(self, max_chars: int = 1500) -> str:
        parts: list[str] = []
        for path in sorted(self.files):
            info = self.files[path]
            classes = sorted({s.name for s in info.symbols if s.kind == "class"})
            funcs = sorted({s.qualname for s in info.symbols if s.kind == "function"})
            try:
                rel = os.path.relpath(path, self.root)
            except Exception:
                rel = path
            parts.append(f"{rel}: classes={classes}, funcs={funcs}")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def save(self, dest: str | None = None) -> str:
        if dest is None:
            dest = str(Path(STATE_DIR()) / "scm.pickle")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            pickle.dump({"root": self.root, "files": self.files}, f)
        return dest

    @classmethod
    def load(cls, src: str) -> "SCM":
        with open(src, "rb") as f:
            blob = pickle.load(f)
        scm = cls(root=blob["root"])
        scm.files = blob["files"]
        scm._rebuild_index()
        return scm

    def apply_simulation(self, path: str, new_source: str) -> list[str]:
        """Parse new_source, return warnings without writing to disk."""
        warnings: list[str] = []
        lang = _lang.lang_for_file(path)
        if lang and lang != "python":
            # Non-Python: defer to interpreter for syntax; skip cross-ref checks.
            return _lang.syntax_check(path, new_source)
        try:
            tree = ast.parse(new_source)
        except SyntaxError as e:
            return [f"syntax error: {e.msg} at line {e.lineno}"]

        # Find symbols defined in new source.
        new_top_names: set[str] = set()
        new_qualnames: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                new_top_names.add(node.name)
                new_qualnames.add(node.name)
                if isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            new_qualnames.add(f"{node.name}.{sub.name}")

        # Find what the OLD version of this file defined.
        abs_path = str(Path(path).resolve())
        old_info = self.files.get(abs_path)
        old_top: set[str] = set()
        old_qualnames: set[str] = set()
        if old_info is not None:
            for sym in old_info.symbols:
                if sym.kind in ("function", "class") and sym.parent is None:
                    old_top.add(sym.name)
                if sym.kind in ("function", "class"):
                    old_qualnames.add(sym.qualname)

        removed = old_qualnames - new_qualnames
        for sym_name in sorted(removed):
            short = sym_name.rsplit(".", 1)[-1]
            for info in self.files.values():
                if info.path == abs_path:
                    continue
                for caller, callee in info.calls:
                    if callee == short or callee == sym_name:
                        warnings.append(
                            f"removed symbol {sym_name} still referenced by {caller} in {info.path}"
                        )

        # Unresolved imports — check if "from X import Y" where Y is not findable
        # in either the new file or known SCM modules. We use a simple heuristic.
        known_modules = {
            self._module_for(p) for p in self.files if self._module_for(p)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod and mod not in known_modules and not _is_stdlib_or_external(mod):
                    warnings.append(f"unresolved import {mod}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if (
                        "." not in mod
                        and mod not in known_modules
                        and not _is_stdlib_or_external(mod)
                    ):
                        # ambiguous; skip
                        pass
        return warnings

    def update_from_dirty(self) -> None:
        dirty = _tools.consume_dirty()
        if dirty:
            self.scan(paths=dirty)

    # -------------- Internals --------------

    def _iter_py_files(self) -> Iterable[Path]:
        """Backwards-compat: Python-only iteration."""
        root = Path(self.root)
        for p in root.rglob("*.py"):
            if _is_skipped(p):
                continue
            if p.is_file():
                yield p

    def _iter_code_files(self) -> Iterable[Path]:
        root = Path(self.root)
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if _is_skipped(p):
                continue
            if p.suffix.lower() in _lang.ALL_CODE_EXTENSIONS:
                yield p

    # ----- non-Python parsers (regex-based; advisory only) -----

    def _parse_php_file(self, path: str) -> FileInfo:
        abs_path = str(Path(path).resolve())
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        info = FileInfo(path=abs_path, mtime=mtime)
        for m in _PHP_CLASS_RE.finditer(source):
            lineno = source.count("\n", 0, m.start()) + 1
            info.symbols.append(Symbol(
                name=m.group(1), kind="class", file=abs_path,
                lineno=lineno, signature=f"class {m.group(1)}",
            ))
        for m in _PHP_FUNC_RE.finditer(source):
            lineno = source.count("\n", 0, m.start()) + 1
            info.symbols.append(Symbol(
                name=m.group(1), kind="function", file=abs_path,
                lineno=lineno, signature=f"function {m.group(1)}(...)",
            ))
        for m in _PHP_USE_RE.finditer(source):
            info.imports.append(m.group(1).lstrip("\\"))
        return info

    def _parse_js_file(self, path: str) -> FileInfo:
        abs_path = str(Path(path).resolve())
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        info = FileInfo(path=abs_path, mtime=mtime)
        for m in _JS_CLASS_RE.finditer(source):
            lineno = source.count("\n", 0, m.start()) + 1
            info.symbols.append(Symbol(
                name=m.group(1), kind="class", file=abs_path,
                lineno=lineno, signature=f"class {m.group(1)}",
            ))
        for regex in (_JS_FUNC_RE, _JS_ARROW_RE):
            for m in regex.finditer(source):
                lineno = source.count("\n", 0, m.start()) + 1
                info.symbols.append(Symbol(
                    name=m.group(1), kind="function", file=abs_path,
                    lineno=lineno, signature=f"function {m.group(1)}(...)",
                ))
        for m in _JS_IMPORT_RE.finditer(source):
            mod = m.group(1) or m.group(2)
            if mod:
                info.imports.append(mod)
        return info

    def _parse_file(self, path: str) -> FileInfo:
        abs_path = str(Path(path).resolve())
        with open(abs_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=abs_path)
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        info = FileInfo(path=abs_path, mtime=mtime)

        # Module-level walk.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._collect_function(node, info, parent=None)
            elif isinstance(node, ast.ClassDef):
                self._collect_class(node, info)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._collect_module_assign(node, info)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
                    info.symbols.append(
                        Symbol(
                            name=alias.asname or alias.name,
                            kind="import",
                            file=abs_path,
                            lineno=node.lineno,
                            end_lineno=getattr(node, "end_lineno", node.lineno),
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    full = f"{mod}.{alias.name}" if mod else alias.name
                    info.imports.append(full)
                    info.symbols.append(
                        Symbol(
                            name=alias.asname or alias.name,
                            kind="import",
                            file=abs_path,
                            lineno=node.lineno,
                            end_lineno=getattr(node, "end_lineno", node.lineno),
                        )
                    )

        # Collect call edges via traversal of each function body.
        for sym in list(info.symbols):
            if sym.kind != "function":
                continue
            src_slice_node = self._find_funcdef_node(tree, sym)
            if src_slice_node is None:
                continue
            for sub in ast.walk(src_slice_node):
                if isinstance(sub, ast.Call):
                    callee = _call_name(sub.func)
                    if callee:
                        info.calls.append((sym.qualname, callee))

        return info

    def _find_funcdef_node(self, tree: ast.AST, sym: Symbol) -> ast.AST | None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == sym.name and node.lineno == sym.lineno:
                    return node
        return None

    def _collect_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        info: FileInfo,
        parent: str | None,
    ) -> None:
        info.symbols.append(
            Symbol(
                name=node.name,
                kind="function",
                file=info.path,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                signature=_format_signature(node),
                docstring=ast.get_docstring(node),
                parent=parent,
            )
        )

    def _collect_class(self, node: ast.ClassDef, info: FileInfo) -> None:
        info.symbols.append(
            Symbol(
                name=node.name,
                kind="class",
                file=info.path,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                signature=f"class {node.name}",
                docstring=ast.get_docstring(node),
                parent=None,
            )
        )
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._collect_function(sub, info, parent=node.name)
            elif isinstance(sub, ast.ClassDef):
                # Nested classes — register, skip deep recursion.
                info.symbols.append(
                    Symbol(
                        name=sub.name,
                        kind="class",
                        file=info.path,
                        lineno=sub.lineno,
                        end_lineno=getattr(sub, "end_lineno", None),
                        signature=f"class {sub.name}",
                        docstring=ast.get_docstring(sub),
                        parent=node.name,
                    )
                )

    def _collect_module_assign(
        self, node: ast.Assign | ast.AnnAssign, info: FileInfo
    ) -> None:
        targets: list[str] = []
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                targets.append(node.target.id)
        else:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        for name in targets:
            info.symbols.append(
                Symbol(
                    name=name,
                    kind="variable",
                    file=info.path,
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

    def _rebuild_index(self) -> None:
        self.symbol_index.clear()
        for info in self.files.values():
            for sym in info.symbols:
                self.symbol_index.setdefault(sym.name, []).append(sym)
                if sym.parent:
                    self.symbol_index.setdefault(sym.qualname, []).append(sym)

    def _slice_source(self, path: str, start: int, end: int | None) -> str | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return None
        s = max(0, start - 1)
        e = end if end is not None else len(lines)
        return "".join(lines[s:e])

    def _module_for(self, path: str) -> str | None:
        try:
            rel = os.path.relpath(path, self.root)
        except Exception:
            return None
        if rel.endswith(".py"):
            rel = rel[:-3]
        parts = [p for p in rel.split(os.sep) if p and p != "__init__"]
        if not parts:
            return None
        return ".".join(parts)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_stdlib_or_external(mod: str) -> bool:
    """Check if a module is stdlib or externally installed using importlib."""
    head = mod.split(".", 1)[0]
    # Fast-path: known common modules.
    known = {
        "os", "sys", "io", "re", "json", "math", "time", "datetime", "subprocess",
        "pathlib", "typing", "collections", "itertools", "functools", "threading",
        "asyncio", "ast", "symtable", "tokenize", "logging", "pickle", "dataclasses",
        "urllib", "http", "email", "argparse", "shutil", "tempfile", "unittest",
        "pytest", "contextlib", "enum", "abc", "inspect", "warnings", "traceback",
        "operator", "random", "string", "hashlib", "base64", "uuid", "copy",
    }
    if head in known:
        return True
    # Dynamic check via importlib.
    try:
        import importlib.util
        spec = importlib.util.find_spec(head)
        return spec is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False
