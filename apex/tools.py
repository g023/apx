"""File ops, command execution, and git helpers (stdlib only)."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

_SKIP_DIRS = {".git", ".apex", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"}

# Tracking patched files so SCM can incrementally re-scan.
_dirty_files: set[str] = set()

# Filesystem-based fallback (used when no git repo is present).
_FS_STAGING_DIR = ".apex/staging"
_FS_MANIFEST_FILE = "manifest.json"
_FS_HEAD_REF = "HEAD"  # ref file storing current branch name
_FS_ROOT_BRANCH = "__fs__"


# ---------------- File operations ----------------

def read_file(path: str, start: int = 1, end: int | None = None) -> str:
    """Read file lines [start, end] (1-indexed, inclusive)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if start < 1:
        start = 1
    s = start - 1
    e = end if end is not None else len(lines)
    return "".join(lines[s:e])


def write_file(path: str, content: str) -> None:
    """Write content to path, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)


def write_patch(path: str, content: str) -> None:
    """Write content and mark path dirty for SCM."""
    write_file(path, content)
    try:
        abs_p = str(Path(path).resolve())
    except Exception:
        abs_p = path
    _dirty_files.add(abs_p)


def consume_dirty() -> list[str]:
    """Return and clear the dirty file set."""
    out = sorted(_dirty_files)
    _dirty_files.clear()
    return out


def list_files(root: str = ".", pattern: str = "**/*.py") -> list[str]:
    """List files under root matching glob pattern; skip noise dirs."""
    rp = Path(root)
    out: list[str] = []
    for p in rp.rglob(pattern):
        if not p.is_file():
            continue
        parts = set(p.parts)
        if parts & _SKIP_DIRS:
            continue
        out.append(str(p))
    return sorted(out)


def run_command(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int = 60,
) -> tuple[int, str, str]:
    """Run a subprocess command. Returns (rc, stdout, stderr). Never shell=True."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + f"\n[timeout after {timeout}s]"
    except FileNotFoundError as e:
        return 127, "", str(e)


# ---------------- Filesystem fallback (non-git repos) ----------------


def is_git_repo(path: str | None = None) -> bool:
    """Check whether *path* (or cwd) is inside a git repository."""
    p = Path(path or os.getcwd())
    # Walk up looking for .git.
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return True
    return False


def _fs_staging_root(cwd: str | None = None) -> Path:
    return Path(cwd or os.getcwd()) / _FS_STAGING_DIR


def _fs_manifest_path(cwd: str | None = None) -> Path:
    return _fs_staging_root(cwd) / _FS_MANIFEST_FILE


def _fs_snapshot_dir(name: str, cwd: str | None = None) -> Path:
    return _fs_staging_root(cwd) / name


def _fs_load_manifest(cwd: str | None = None) -> dict:
    """Load the staging manifest; return {name: {file: sha256, ...}, ...}."""
    mp = _fs_manifest_path(cwd)
    if mp.exists():
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            # Migrate old format (plain {file: hash}) to new format.
            if isinstance(data, dict):
                needs_migration = False
                for key, val in list(data.items()):
                    if isinstance(val, dict) and "type" not in val:
                        # Check if this looks like old format (all values are 16-char hex strings)
                        if val and all(isinstance(v, str) and len(v) == 16 for v in val.values()):
                            data[key] = {"type": "snapshot", "files": val}
                            needs_migration = True
                if needs_migration:
                    mp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return data
        except Exception:
            return {}
    return {}


def _fs_save_manifest(manifest: dict, cwd: str | None = None) -> None:
    mp = _fs_manifest_path(cwd)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _fs_read_head(cwd: str | None = None) -> str:
    """Return the current branch name from the HEAD ref file."""
    hp = _fs_staging_root(cwd) / _FS_HEAD_REF
    if hp.exists():
        try:
            return hp.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return _FS_ROOT_BRANCH


def _fs_write_head(branch: str, cwd: str | None = None) -> None:
    """Write the current branch name to the HEAD ref file."""
    hp = _fs_staging_root(cwd) / _FS_HEAD_REF
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text(branch.strip(), encoding="utf-8")


def _fs_file_hash(path: str) -> str:
    """SHA-256 of file content."""
    h = hashlib.sha256()
    try:
        h.update(Path(path).read_bytes())
    except FileNotFoundError:
        h.update(b"")
    return h.hexdigest()[:16]


def _fs_snapshot_tree(cwd: str | None = None) -> dict[str, str]:
    """Return {relative_path: sha256} for every tracked file under cwd."""
    root = Path(cwd or os.getcwd())
    tree: dict[str, str] = {}
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        parts = set(fp.relative_to(root).parts)
        if parts & _SKIP_DIRS:
            continue
        rel = str(fp.relative_to(root))
        tree[rel] = _fs_file_hash(str(fp))
    return tree


def _fs_copy_tree_to_snapshot(name: str, cwd: str | None = None) -> dict[str, str]:
    """Copy all project files into a snapshot directory. Returns file-hash map."""
    root = Path(cwd or os.getcwd())
    snap_dir = _fs_snapshot_dir(name, cwd)
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    tree: dict[str, str] = {}
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        parts = set(fp.relative_to(root).parts)
        if parts & _SKIP_DIRS:
            continue
        rel = str(fp.relative_to(root))
        dest = snap_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(fp), str(dest))
        tree[rel] = _fs_file_hash(str(fp))
    return tree


def _fs_restore_snapshot(name: str, cwd: str | None = None) -> None:
    """Restore all files from a snapshot into the working tree."""
    root = Path(cwd or os.getcwd())
    snap_dir = _fs_snapshot_dir(name, cwd)
    if not snap_dir.exists():
        raise RuntimeError(f"snapshot '{name}' does not exist")
    for fp in snap_dir.rglob("*"):
        if not fp.is_file():
            continue
        rel = fp.relative_to(snap_dir)
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(fp), str(dest))


def fs_current_branch(cwd: str | None = None) -> str:
    """Return the sentinel branch name for filesystem mode."""
    return _FS_ROOT_BRANCH


def fs_head_sha(cwd: str | None = None) -> str:
    """Return the SHA of the latest snapshot, or a content-based hash if none."""
    manifest = _fs_load_manifest(cwd)
    if manifest:
        # Get the current branch's commit.
        cur_branch = _fs_read_head(cwd)
        if cur_branch in manifest:
            entry = manifest[cur_branch]
            if isinstance(entry, dict) and entry.get("type") == "branch":
                commit = entry.get("commit", "")
                if commit:
                    return commit
        # Fallback: find the latest snapshot entry.
        for key in reversed(list(manifest.keys())):
            entry = manifest[key]
            if isinstance(entry, dict) and entry.get("type") == "snapshot":
                return key
        # Last resort: return the last key.
        return list(manifest.keys())[-1]
    # Return a hash of the current working tree so it's a valid key.
    tree = _fs_snapshot_tree(cwd)
    h = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]
    return h


def _fs_resolve_ref(ref: str, manifest: dict, cwd: str | None = None) -> str:
    """Resolve a symbolic ref to an actual snapshot key.

    In the manifest, branch entries (like ``__fs__``) store ``{"type": "branch",
    "commit": "<snapshot_hash>"}``, while snapshot entries store
    ``{"type": "snapshot", "files": {file: hash}}``.

    This function resolves a branch name to its commit hash, or returns
    the ref unchanged if it's already a snapshot key.
    """
    if ref == "HEAD":
        ref = _fs_read_head(cwd)
    if ref in manifest:
        entry = manifest[ref]
        if isinstance(entry, dict) and entry.get("type") == "branch":
            commit = entry.get("commit")
            if commit and commit in manifest:
                return commit
            # Branch points to a non-existent commit — return branch itself.
            return ref
        # It's a snapshot entry directly.
        return ref
    # Not found — return as-is (caller will handle).
    return ref


def fs_create_branch(name: str, from_ref: str = "HEAD", cwd: str | None = None, force_new: bool = False) -> None:
    """Create a new branch pointing to the same commit as *from_ref*.

    Always creates a manifest entry for *name* and switches HEAD to it.
    If *force_new* is True and the branch already exists, it is deleted first.
    """
    manifest = _fs_load_manifest(cwd)
    if force_new and name in manifest:
        manifest.pop(name, None)
    # Resolve from_ref to a commit hash.
    resolved = _fs_resolve_ref(from_ref, manifest, cwd)
    commit_hash = resolved if (resolved in manifest and isinstance(manifest[resolved], dict) and manifest[resolved].get("type") == "snapshot") else None
    if commit_hash is None:
        # from_ref doesn't point to a snapshot — create one from working tree.
        tree = _fs_snapshot_tree(cwd)
        commit_hash = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]
        _fs_copy_tree_to_snapshot(commit_hash, cwd)
        manifest[commit_hash] = {"type": "snapshot", "files": tree}
    # Create branch entry pointing to the commit.
    manifest[name] = {"type": "branch", "commit": commit_hash}
    _fs_save_manifest(manifest, cwd)
    _fs_write_head(name, cwd)


def fs_checkout(ref: str, cwd: str | None = None) -> None:
    """Restore working tree from a snapshot."""
    manifest = _fs_load_manifest(cwd)
    resolved = _fs_resolve_ref(ref, manifest, cwd)
    if resolved not in manifest:
        # If manifest is empty, there's nothing to restore — just record the branch.
        if not manifest:
            _fs_write_head(ref, cwd)
            return
        # Try to create a snapshot from the current working tree.
        tree = _fs_snapshot_tree(cwd)
        h = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]
        _fs_copy_tree_to_snapshot(h, cwd)
        manifest[h] = {"type": "snapshot", "files": tree}
        manifest[ref] = {"type": "branch", "commit": h}
        _fs_save_manifest(manifest, cwd)
        _fs_write_head(ref, cwd)
        return
    _fs_restore_snapshot(resolved, cwd)
    # Record the branch name so fs_commit_all can update the right entry.
    _fs_write_head(ref, cwd)


def fs_commit_all(message: str, cwd: str | None = None) -> str:
    """Snapshot the current working tree and return a content-hash key.

    Also updates the current branch's manifest entry so the branch
    pointer moves with the commit (mirroring git's behaviour).
    """
    manifest = _fs_load_manifest(cwd)
    tree = _fs_snapshot_tree(cwd)
    h = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]
    # Skip if this exact snapshot already exists.
    if h in manifest:
        return h
    # Copy files into snapshot directory named after the hash.
    _fs_copy_tree_to_snapshot(h, cwd)
    manifest[h] = {"type": "snapshot", "files": tree}
    # Update the current branch pointer so it points to this new commit.
    cur_branch = _fs_read_head(cwd)
    manifest[cur_branch] = {"type": "branch", "commit": h}
    _fs_save_manifest(manifest, cwd)
    return h


def fs_merge(branch: str, message: str | None = None, cwd: str | None = None) -> None:
    """Merge (copy) all files from *branch* snapshot into working tree."""
    manifest = _fs_load_manifest(cwd)
    resolved = _fs_resolve_ref(branch, manifest, cwd)
    if resolved not in manifest:
        # If the branch doesn't exist as a snapshot, just snapshot the current tree.
        tree = _fs_snapshot_tree(cwd)
        h = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]
        _fs_copy_tree_to_snapshot(h, cwd)
        manifest[h] = {"type": "snapshot", "files": tree}
        cur_branch = _fs_read_head(cwd)
        manifest[cur_branch] = {"type": "branch", "commit": h}
        _fs_save_manifest(manifest, cwd)
        return
    _fs_restore_snapshot(resolved, cwd)
    # Record the merge as a new snapshot.
    tree = _fs_snapshot_tree(cwd)
    h = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]
    _fs_copy_tree_to_snapshot(h, cwd)
    manifest[h] = {"type": "snapshot", "files": tree}
    # Update current branch pointer.
    cur_branch = _fs_read_head(cwd)
    manifest[cur_branch] = {"type": "branch", "commit": h}
    _fs_save_manifest(manifest, cwd)


def _fs_get_files(entry: dict | list | str) -> dict[str, str]:
    """Extract the file-hash map from a manifest entry."""
    if isinstance(entry, dict):
        if "files" in entry:
            return entry["files"]
        # Old-style: entry is itself the file map.
        # Check if values look like hashes (16-char hex strings).
        if entry.get("type") == "snapshot" and "files" in entry:
            return entry["files"]
        # Could be old format where entry IS the file map.
        return entry
    return {}


def fs_diff(ref_a: str = "HEAD", ref_b: str | None = None, cwd: str | None = None) -> str:
    """Compute unified diff between two snapshots (or snapshot vs working tree)."""
    manifest = _fs_load_manifest(cwd)

    if not manifest:
        # No manifest yet — show current working tree as a pseudo-diff.
        tree = _fs_snapshot_tree(cwd)
        if not tree:
            return ""
        lines: list[str] = []
        for fname in sorted(tree):
            content = _fs_read_working_file(fname, cwd)
            if content:
                lines.append(f"--- a/{fname}")
                lines.append(f"+++ b/{fname}")
                for line in content.splitlines():
                    lines.append(f" {line}")
        return "\n".join(lines)

    # Resolve ref_a.
    ref_a = _fs_resolve_ref(ref_a, manifest, cwd)
    if ref_a not in manifest:
        return ""

    entry_a = manifest[ref_a]
    tree_a = _fs_get_files(entry_a)

    if ref_b is None or ref_b == "HEAD":
        # Diff against working tree.
        tree_b = _fs_snapshot_tree(cwd)
    else:
        ref_b = _fs_resolve_ref(ref_b, manifest, cwd)
        if ref_b not in manifest:
            return ""
        entry_b = manifest[ref_b]
        tree_b = _fs_get_files(entry_b)

    # Compute diffs for all files in either tree.
    all_files = set(tree_a.keys()) | set(tree_b.keys())
    lines = []
    for fname in sorted(all_files):
        content_a = _fs_read_snapshot_file(ref_a, fname, cwd)
        content_b = _fs_read_snapshot_file(ref_b, fname, cwd) if ref_b and ref_b != "HEAD" else _fs_read_working_file(fname, cwd)
        if content_a != content_b:
            diff = difflib.unified_diff(
                content_a.splitlines(keepends=True),
                content_b.splitlines(keepends=True),
                fromfile=f"a/{fname}",
                tofile=f"b/{fname}",
            )
            lines.extend(diff)
    return "".join(lines)


def _fs_read_snapshot_file(snapshot: str, rel_path: str, cwd: str | None = None) -> str:
    """Read a file from a snapshot directory."""
    fp = _fs_snapshot_dir(snapshot, cwd) / rel_path
    try:
        return fp.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _fs_read_working_file(rel_path: str, cwd: str | None = None) -> str:
    """Read a file from the working tree."""
    fp = Path(cwd or os.getcwd()) / rel_path
    try:
        return fp.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


# ---------------- Git helpers ----------------

def _git(args: list[str], cwd: str | None = None, timeout: int = 60) -> tuple[int, str, str]:
    return run_command(["git", *args], cwd=cwd, timeout=timeout)


def _git_check(args: list[str], cwd: str | None = None) -> str:
    rc, out, err = _git(args, cwd=cwd)
    if rc != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {err.strip() or out.strip()}")
    return out


def git_root(cwd: str | None = None) -> str:
    if not is_git_repo(cwd):
        return str(Path(cwd or os.getcwd()).resolve())
    return _git_check(["rev-parse", "--show-toplevel"], cwd=cwd).strip()


def git_current_branch(cwd: str | None = None) -> str:
    if not is_git_repo(cwd):
        return fs_current_branch(cwd)
    rc, out, err = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if rc != 0:
        # Detached HEAD or no commits — return a sentinel.
        return "main"
    branch = out.strip()
    if branch == "HEAD":
        return "main"
    return branch


def git_create_branch(name: str, from_ref: str = "HEAD", cwd: str | None = None, force_new: bool = False) -> None:
    if not is_git_repo(cwd):
        fs_create_branch(name, from_ref, cwd, force_new=force_new)
        return
    # If already on target, no-op.
    try:
        cur = git_current_branch(cwd=cwd)
        if cur == name:
            return
    except RuntimeError:
        pass
    # Check if branch exists.
    rc, out, _ = _git(["rev-parse", "--verify", f"refs/heads/{name}"], cwd=cwd)
    if rc == 0:
        if force_new:
            git_branch_delete(name, force=True, cwd=cwd)
            _git_check(["checkout", "-b", name, from_ref], cwd=cwd)
        else:
            _git_check(["checkout", name], cwd=cwd)
    else:
        # Check if from_ref exists
        rc_ref, _, _ = _git(["rev-parse", "--verify", from_ref], cwd=cwd)
        if rc_ref != 0:
            # from_ref doesn't exist — create orphan branch
            _git_check(["checkout", "--orphan", name], cwd=cwd)
            _git_check(["reset", "--hard"], cwd=cwd)
        else:
            _git_check(["checkout", "-b", name, from_ref], cwd=cwd)


def git_checkout(ref: str, cwd: str | None = None) -> None:
    if not is_git_repo(cwd):
        fs_checkout(ref, cwd)
        return
    rc, out, err = _git(["checkout", ref], cwd=cwd)
    if rc != 0:
        # If pathspec didn't match, try creating an orphan branch
        if "pathspec" in err and "did not match" in err:
            _git_check(["checkout", "--orphan", ref], cwd=cwd)
            _git_check(["reset", "--hard"], cwd=cwd)
            return
        raise RuntimeError(f"git checkout failed: {err.strip() or out.strip()}")


def git_commit_all(message: str, cwd: str | None = None) -> str:
    """Stage everything and commit. Tolerates nothing-to-commit by returning HEAD."""
    if not is_git_repo(cwd):
        return fs_commit_all(message, cwd)
    _git_check(["add", "-A"], cwd=cwd)
    # Check if there's anything to commit
    rc_st, out_st, _ = _git(["status", "--porcelain"], cwd=cwd)
    if rc_st == 0 and not out_st.strip():
        return git_head_sha(cwd=cwd)
    # For initial commit, set up user config if missing
    rc, out, err = _git(["commit", "-m", message], cwd=cwd)
    if rc != 0:
        combined = (out + err).lower()
        if "nothing to commit" in combined or "no changes added" in combined or "nothing added" in combined:
            return git_head_sha(cwd=cwd)
        if "please tell me who you are" in combined or "author identity unknown" in combined:
            # Auto-configure for headless operation
            _git(["config", "user.email", "local@localhost"], cwd=cwd)
            _git(["config", "user.name", "local_agent"], cwd=cwd)
            rc, out, err = _git(["commit", "-m", message], cwd=cwd)
            if rc != 0:
                raise RuntimeError(f"git commit failed after config: {err.strip() or out.strip()}")
        else:
            raise RuntimeError(f"git commit failed: {err.strip() or out.strip()}")
    return git_head_sha(cwd=cwd)


def git_branch_delete(name: str, force: bool = True, cwd: str | None = None) -> None:
    if not is_git_repo(cwd):
        # In fs mode, just remove the branch entry from manifest.
        manifest = _fs_load_manifest(cwd)
        manifest.pop(name, None)
        _fs_save_manifest(manifest, cwd)
        return
    flag = "-D" if force else "-d"
    rc, out, err = _git(["branch", flag, name], cwd=cwd)
    if rc != 0:
        # Branch might not exist — that's fine
        if "not found" in err or "not a valid branch" in err or "not found" in out:
            return
        raise RuntimeError(f"git branch delete failed: {err.strip() or out.strip()}")


def git_merge(branch: str, message: str | None = None, cwd: str | None = None) -> None:
    if not is_git_repo(cwd):
        fs_merge(branch, message, cwd)
        return
    args = ["merge", "--no-ff", branch]
    if message:
        args += ["-m", message]
    _git_check(args, cwd=cwd)


def git_diff(ref_a: str = "HEAD", ref_b: str | None = None, cwd: str | None = None) -> str:
    if not is_git_repo(cwd):
        return fs_diff(ref_a, ref_b, cwd)
    # Check if ref_a exists
    rc_a, _, _ = _git(["rev-parse", "--verify", ref_a], cwd=cwd)
    if rc_a != 0:
        # ref_a doesn't exist — try showing the working tree as a pseudo-diff.
        rc_show, out_show, _ = _git(["show", "HEAD", "--stat"], cwd=cwd)
        if rc_show != 0:
            return ""
        return out_show
    args = ["diff", ref_a]
    if ref_b:
        rc_b, _, _ = _git(["rev-parse", "--verify", ref_b], cwd=cwd)
        if rc_b != 0:
            return ""
        args.append(ref_b)
    rc, out, err = _git(args, cwd=cwd)
    if rc != 0:
        return ""
    return out


def git_head_sha(cwd: str | None = None) -> str:
    if not is_git_repo(cwd):
        return fs_head_sha(cwd)
    rc, out, err = _git(["rev-parse", "HEAD"], cwd=cwd)
    if rc != 0:
        # No commits yet — return a synthetic hash.
        return hashlib.sha256(b"empty").hexdigest()[:16]
    return out.strip()
