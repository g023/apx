"""MCTS-Code — Monte Carlo Tree Search over code edits with git branches."""
# g023's APX Agent — Adaptive Programming via eXploratory edit search - MIT License
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import tools as _tools
from .config import (
    APEX_DIR,
    MAX_MCTS_ITERATIONS,
    MCTS_EXPLORATION_CONSTANT,
    W_CRITIC,
    W_ERRORS,
    W_SPEC,
    W_TEST,
    W_TOKENS,
)
from .critics import CriticEnsemble
from .debug_trace import T, trace_expand, trace_mcts_iter, trace_simulate
from .logging_util import get_logger
from .memory import anti_patterns, trajectory

_log = get_logger(__name__)

# Hard cap on branches created per run.
_MAX_BRANCHES_PER_RUN = 64


# ---------------- Data types ----------------


@dataclass
class PatchProposal:
    file: str
    new_content: str
    rationale: str = ""
    requirement_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PatchProposal":
        return cls(
            file=str(d.get("file", "")),
            new_content=str(d.get("new_content", "")),
            rationale=str(d.get("rationale", "")),
            requirement_id=str(d.get("requirement_id", "")),
        )


@dataclass
class RewardComponents:
    test_pass: float = 0.0
    spec_compliance: float = 0.0
    critic_score: float = 0.0
    new_errors: float = 0.0
    token_cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_reward(rc: RewardComponents) -> float:
    """Weighted aggregate using config weights."""
    r = (
        W_TEST * rc.test_pass
        + W_SPEC * rc.spec_compliance
        + W_CRITIC * rc.critic_score
        - W_ERRORS * rc.new_errors
        - W_TOKENS * rc.token_cost
    )
    return r


@dataclass
class Node:
    id: str
    parent_id: str | None
    branch_name: str
    commit_sha: str
    children_ids: list[str] = field(default_factory=list)
    patch: PatchProposal | None = None
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[PatchProposal] = field(default_factory=list)
    terminal: bool = False
    reward_breakdown: dict = field(default_factory=dict)
    expansion_failures: int = 0

    @property
    def q(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb1(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        if parent_visits <= 0:
            return self.q
        return self.q + c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "branch_name": self.branch_name,
            "commit_sha": self.commit_sha,
            "children_ids": list(self.children_ids),
            "patch": self.patch.to_dict() if self.patch else None,
            "visits": int(self.visits),
            "total_reward": float(self.total_reward),
            "untried_actions": [p.to_dict() for p in self.untried_actions],
            "terminal": bool(self.terminal),
            "reward_breakdown": dict(self.reward_breakdown),
            "expansion_failures": int(self.expansion_failures),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        patch = PatchProposal.from_dict(d["patch"]) if d.get("patch") else None
        untried = [PatchProposal.from_dict(p) for p in d.get("untried_actions", [])]
        return cls(
            id=d["id"],
            parent_id=d.get("parent_id"),
            branch_name=d.get("branch_name", ""),
            commit_sha=d.get("commit_sha", ""),
            children_ids=list(d.get("children_ids", [])),
            patch=patch,
            visits=int(d.get("visits", 0)),
            total_reward=float(d.get("total_reward", 0.0)),
            untried_actions=untried,
            terminal=bool(d.get("terminal", False)),
            reward_breakdown=dict(d.get("reward_breakdown", {})),
            expansion_failures=int(d.get("expansion_failures", 0)),
        )


# ---------------- Reward helpers ----------------


def run_tests(cwd: str, timeout: int = 60) -> tuple[float, str]:
    """Detect project language and dispatch to the appropriate test runner.

    Pure delegation to :func:`apex.lang.run_tests_for`; kept here for backwards
    compatibility (verifier + tests import this symbol directly).
    """
    from . import lang as _lang
    project_lang = _lang.detect_language(cwd)
    return _lang.run_tests_for(project_lang, cwd, timeout=timeout)


def spec_compliance(spec: Any, diff: str, llm: Any) -> float:
    """Ask the LLM what fraction of requirements the diff moves us toward."""
    if not diff:
        return 0.0
    try:
        reqs = []
        for r in getattr(spec, "requirements", []) or []:
            try:
                reqs.append({"id": r.id, "description": r.description, "acceptance": r.acceptance})
            except Exception:
                continue
        prompt = (
            "You evaluate code progress against a specification.\n"
            "Requirements:\n" + json.dumps(reqs, indent=2) + "\n\n"
            "Diff:\n" + diff[:4000] + "\n\n"
            'Respond ONLY with JSON: {"compliance": <float 0..1>}.'
        )
        obj = llm.chat_json(
            [{"role": "user", "content": prompt}],
            schema_hint='{"compliance": float}',
        )
        val = obj.get("compliance", obj.get("score", 0.0))
        try:
            v = float(val)
        except Exception:
            return 0.0
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v
    except Exception as e:
        _log.warning("spec_compliance failed: %s", e)
        return 0.0


def static_errors(scm: Any, patch: PatchProposal) -> int:
    """Return count of warnings from scm.apply_simulation for this patch."""
    try:
        warnings = scm.apply_simulation(patch.file, patch.new_content)
        return len(warnings or [])
    except Exception as e:
        _log.warning("static_errors failed: %s", e)
        return 0


# ---------------- MCTSCode ----------------


class MCTSCode:
    """Monte Carlo Tree Search over code edits."""

    def __init__(
        self,
        root_dir: str,
        spec: Any,
        scm: Any,
        llm: Any,
        fast_llm: Any,
        critics: CriticEnsemble,
        exploration_c: float = MCTS_EXPLORATION_CONSTANT,
        max_iterations: int = MAX_MCTS_ITERATIONS,
        patches_per_expansion: int = 3,
        rollout_depth: int = 2,
        branch_prefix: str = "apex/node-",
    ) -> None:
        self.root_dir = str(Path(root_dir).resolve())
        self.spec = spec
        self.scm = scm
        self.llm = llm
        self.fast_llm = fast_llm
        self.critics = critics
        self.exploration_c = exploration_c
        self.max_iterations = max_iterations
        self.patches_per_expansion = patches_per_expansion
        self.rollout_depth = rollout_depth
        self.branch_prefix = branch_prefix

        self.nodes: dict[str, Node] = {}
        self.root_id: str | None = None
        self._branches_created = 0

    # -------- init / persistence --------

    def init_root(self) -> Node:
        T("MCTS", "init_root starting")
        # Ensure .apex/ is gitignored so persistence doesn't pollute commits.
        gi = Path(self.root_dir) / ".gitignore"
        ignore_line = f"{APEX_DIR}/"
        existing = gi.read_text() if gi.exists() else ""
        if ignore_line not in existing.splitlines():
            with open(gi, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(ignore_line + "\n")
            try:
                _tools.git_commit_all("apex: ignore .apex/", cwd=self.root_dir)
            except Exception:
                pass
        cur_branch = _tools.git_current_branch(cwd=self.root_dir)
        is_git = _tools.is_git_repo(self.root_dir)
        T("MCTS", f"init_root: is_git={is_git} branch={cur_branch!r}")
        # In non-git mode, create an initial snapshot so the root branch exists.
        if not is_git:
            T("MCTS", "init_root: non-git mode, creating fs snapshot")
            _tools.fs_commit_all("apex: root snapshot", cwd=self.root_dir)
        head = _tools.git_head_sha(cwd=self.root_dir)
        T("MCTS", f"init_root: head={head[:16]}")
        root = Node(
            id=uuid.uuid4().hex,
            parent_id=None,
            branch_name=cur_branch,
            commit_sha=head,
            patch=None,
        )
        self.nodes = {root.id: root}
        self.root_id = root.id
        self.save()
        T("MCTS", f"init_root done: root_id={root.id[:10]}")
        return root

    def _tree_path(self) -> str:
        return str(Path(self.root_dir) / APEX_DIR / "tree.json")

    def save(self) -> str:
        path = self._tree_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        return path

    def load(self) -> None:
        path = self._tree_path()
        if not Path(path).exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        self.root_id = blob.get("root_id")
        self.nodes = {
            nid: Node.from_dict(d) for nid, d in (blob.get("nodes") or {}).items()
        }

    # -------- selection --------

    def select(self, node: Node) -> Node:
        cur = node
        depth = 0
        while True:
            if cur.terminal:
                T("MCTS", f"select: terminal node {cur.id[:10]} at depth {depth}")
                return cur
            if cur.untried_actions:
                T("MCTS", f"select: untried actions available at {cur.id[:10]} depth={depth}")
                return cur
            if not cur.children_ids:
                # Lazy-populate untried actions.
                T("MCTS", f"select: leaf {cur.id[:10]} depth={depth}, proposing patches")
                actions = self.propose_patches(cur, self.patches_per_expansion)
                if actions:
                    cur.untried_actions = actions
                    return cur
                cur.expansion_failures += 1
                T("MCTS", f"select: no proposals, expansion_failures={cur.expansion_failures}")
                if cur.expansion_failures >= 3:
                    cur.terminal = True
                    T("MCTS", f"select: marking {cur.id[:10]} terminal after 3 failures")
                return cur
            # Descend to best child by UCB1.
            best: Node | None = None
            best_score = -float("inf")
            for cid in cur.children_ids:
                child = self.nodes.get(cid)
                if child is None:
                    continue
                s = child.ucb1(cur.visits, self.exploration_c)
                if s > best_score:
                    best_score = s
                    best = child
            if best is None:
                T("MCTS", f"select: no best child at {cur.id[:10]} depth={depth}")
                return cur
            depth += 1
            cur = best

    # -------- expansion --------

    def propose_patches(self, node: Node, k: int) -> list[PatchProposal]:
        """Ask fast_llm for k candidate patches."""
        T("MCTS", f"propose_patches: k={k} node={node.id[:10]}")
        try:
            summary = self.scm.summary(max_chars=1200)
            T("MCTS", f"propose_patches: scm summary length={len(summary)}")
        except Exception as e:
            summary = ""
            T("MCTS", f"propose_patches: scm summary failed: {e}")
        reqs = []
        for r in getattr(self.spec, "requirements", []) or []:
            try:
                reqs.append(f"{r.id}: {r.description}")
            except Exception:
                continue
        # Get the target language from the spec so the LLM generates correct file types.
        target_lang = getattr(self.spec, "language", "python") or "python"
        lang_extensions = {"python": ".py", "php": ".php", "node": ".js"}
        target_ext = lang_extensions.get(target_lang, ".py")
        prompt = (
            "Propose " + str(k) + " concrete file edits that move the codebase toward "
            "satisfying the spec. Each patch fully replaces a file's content.\n\n"
            f"IMPORTANT: The target language is **{target_lang}**. "
            f"All source files MUST use the `{target_ext}` extension. "
            f"Do NOT create files in any other language.\n\n"
            "Codebase summary:\n" + summary + "\n\n"
            "Requirements:\n" + "\n".join(reqs) + "\n\n"
            "Respond ONLY with JSON: "
            '{"patches": [{"file": "path", "new_content": "...", '
            '"rationale": "...", "requirement_id": "R1"}, ...]}'
        )
        try:
            T("MCTS", "propose_patches: calling LLM...")
            obj = self.fast_llm.chat_json(
                [{"role": "user", "content": prompt}],
                schema_hint='{"patches": [{file, new_content, rationale, requirement_id}]}',
            )
            T("MCTS", f"propose_patches: LLM returned keys={list(obj.keys())}")
        except Exception as e:
            _log.warning("propose_patches failed: %s", e)
            T("MCTS", f"propose_patches: LLM call FAILED: {e}")
            return []
        raw = obj.get("patches") or []
        out: list[PatchProposal] = []
        if isinstance(raw, list):
            for p in raw[:k]:
                if isinstance(p, dict) and p.get("file"):
                    out.append(PatchProposal.from_dict(p))
        T("MCTS", f"propose_patches: got {len(out)} valid patches")
        if out:
            for p in out:
                T("MCTS", f"  patch: file={p.file} req={p.requirement_id} rationale={p.rationale[:60]}")
        return out

    def expand(self, node: Node) -> Node | None:
        if not node.untried_actions:
            T("MCTS", "expand: no untried actions")
            return None
        if self._branches_created >= _MAX_BRANCHES_PER_RUN:
            _log.warning("hit branch cap (%d)", _MAX_BRANCHES_PER_RUN)
            T("MCTS", f"expand: hit branch cap {_MAX_BRANCHES_PER_RUN}")
            return None

        action = node.untried_actions.pop(0)
        orig_branch = None
        try:
            orig_branch = _tools.git_current_branch(cwd=self.root_dir)
        except Exception:
            pass

        new_branch = self.branch_prefix + uuid.uuid4().hex[:10]
        T("MCTS", f"expand: parent={node.id[:10]} file={action.file} branch={new_branch}")
        try:
            # Make sure we start from the parent's branch.
            T("MCTS", f"expand: checkout {node.branch_name}")
            try:
                _tools.git_checkout(node.branch_name, cwd=self.root_dir)
            except Exception as e:
                T("MCTS", f"expand: checkout of parent branch failed: {e}")
                # In non-git mode, the branch might not exist as a snapshot yet.
                # Try to create it from HEAD first.
                if not _tools.is_git_repo(self.root_dir):
                    _tools.fs_create_branch(node.branch_name, from_ref="HEAD", cwd=self.root_dir)
                    _tools.fs_checkout(node.branch_name, cwd=self.root_dir)
                else:
                    raise
            T("MCTS", f"expand: create branch {new_branch} from {node.branch_name}")
            _tools.git_create_branch(new_branch, from_ref=node.branch_name, cwd=self.root_dir)
            self._branches_created += 1

            # Apply patch.
            target = action.file
            target_path = Path(target)
            if not target_path.is_absolute():
                target_path = Path(self.root_dir) / target_path
            T("MCTS", f"expand: writing {target_path} ({len(action.new_content)} bytes)")
            _tools.write_patch(str(target_path), action.new_content)
            try:
                self.scm.update_from_dirty()
            except Exception:
                pass
            sha = _tools.git_commit_all(
                f"apex(node): {action.rationale or action.file}", cwd=self.root_dir
            )
            T("MCTS", f"expand: committed {sha[:16]}")

            child = Node(
                id=uuid.uuid4().hex,
                parent_id=node.id,
                branch_name=new_branch,
                commit_sha=sha,
                patch=action,
            )
            self.nodes[child.id] = child
            node.children_ids.append(child.id)
            self.save()
            trajectory().log("mcts_expand", node=node.id, child=child.id, branch=new_branch)
            trace_expand(node.id, child.id, new_branch, action.file, "OK")
            return child
        except Exception as e:
            _log.warning("expand failed: %s", e)
            trajectory().log("mcts_expand_fail", node=node.id, error=str(e))
            trace_expand(node.id, "?", new_branch, action.file, f"FAIL: {e}")
            return None
        finally:
            if orig_branch:
                try:
                    _tools.git_checkout(orig_branch, cwd=self.root_dir)
                except Exception:
                    pass

    # -------- simulation --------

    def simulate(self, node: Node) -> RewardComponents:
        """Compute reward components for the node on its branch."""
        T("MCTS", f"simulate: node={node.id[:10]} branch={node.branch_name}")
        orig_branch = None
        try:
            orig_branch = _tools.git_current_branch(cwd=self.root_dir)
        except Exception:
            pass

        rc = RewardComponents()
        try:
            try:
                T("MCTS", f"simulate: checkout {node.branch_name}")
                _tools.git_checkout(node.branch_name, cwd=self.root_dir)
            except Exception as e:
                _log.warning("simulate checkout %s: %s", node.branch_name, e)
                T("MCTS", f"simulate checkout FAILED: {e}")
                # In non-git mode, try to create the branch snapshot first
                if not _tools.is_git_repo(self.root_dir):
                    try:
                        _tools.fs_create_branch(node.branch_name, from_ref="HEAD", cwd=self.root_dir)
                        _tools.fs_checkout(node.branch_name, cwd=self.root_dir)
                        T("MCTS", "simulate: recovered by creating fs branch")
                    except Exception as e2:
                        T("MCTS", f"simulate: fs recovery also failed: {e2}")

            # Optional rollout: cheap follow-up patches.
            rollout_count = 0
            for _ in range(max(0, self.rollout_depth)):
                actions = self.propose_patches(node, 1)
                if not actions:
                    break
                a = actions[0]
                try:
                    tp = Path(a.file)
                    if not tp.is_absolute():
                        tp = Path(self.root_dir) / tp
                    _tools.write_patch(str(tp), a.new_content)
                    try:
                        self.scm.update_from_dirty()
                    except Exception:
                        pass
                    _tools.git_commit_all(
                        f"apex(rollout): {a.rationale or a.file}", cwd=self.root_dir
                    )
                    rollout_count += 1
                except Exception as e:
                    _log.warning("rollout patch failed: %s", e)
                    break
            if rollout_count:
                T("MCTS", f"simulate: {rollout_count} rollout patches applied")

            # Tests.
            rc.test_pass, raw_test = run_tests(self.root_dir, timeout=60)
            T("MCTS", f"simulate: tests pass_ratio={rc.test_pass:.2f}")

            # Diff vs root.
            diff = ""
            try:
                root_node = self.nodes.get(self.root_id) if self.root_id else None
                if root_node:
                    diff = _tools.git_diff(
                        root_node.commit_sha, "HEAD", cwd=self.root_dir
                    )
                    T("MCTS", f"simulate: diff length={len(diff)}")
            except Exception as e:
                _log.warning("diff failed: %s", e)

            # Spec compliance.
            rc.spec_compliance = spec_compliance(self.spec, diff, self.fast_llm)
            T("MCTS", f"simulate: spec_compliance={rc.spec_compliance:.2f}")

            # Critic ensemble on primary requirement.
            primary_req = ""
            try:
                if getattr(self.spec, "requirements", None):
                    r0 = self.spec.requirements[0]
                    primary_req = f"{r0.id}: {r0.description} [accept: {r0.acceptance}]"
            except Exception:
                pass
            penalty = 0.0
            try:
                penalty = anti_patterns().penalty_for(diff)
            except Exception:
                pass
            try:
                ens = self.critics.evaluate(
                    primary_req, diff, context="", anti_pattern_penalty=penalty
                )
                rc.critic_score = ens.aggregate
                T("MCTS", f"simulate: critic_score={rc.critic_score:.2f}")
            except Exception as e:
                _log.warning("critics failed: %s", e)
                rc.critic_score = 0.5

            # Static errors for this node's patch.
            if node.patch is not None:
                rc.new_errors = float(static_errors(self.scm, node.patch))
                T("MCTS", f"simulate: static_errors={rc.new_errors}")

            # Token cost stub.
            rc.token_cost = 0.0

            node.reward_breakdown = rc.to_dict()
            trace_simulate(node.id, rc.test_pass, rc.spec_compliance, rc.critic_score, int(rc.new_errors))
            return rc
        finally:
            if orig_branch:
                try:
                    _tools.git_checkout(orig_branch, cwd=self.root_dir)
                except Exception:
                    pass

    # -------- backprop --------

    def backpropagate(self, node: Node, reward: float) -> None:
        cur: Node | None = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += reward
            if cur.parent_id is None:
                break
            cur = self.nodes.get(cur.parent_id)

    # -------- main loop --------

    def run(self, iterations: int | None = None) -> Node:
        T("MCTS", f"run: starting with {iterations or self.max_iterations} iterations")
        if self.root_id is None or self.root_id not in self.nodes:
            self.init_root()
        assert self.root_id is not None
        root = self.nodes[self.root_id]

        budget = iterations if iterations is not None else self.max_iterations
        trajectory().log("mcts_run_start", iterations=budget)

        for i in range(budget):
            if self._branches_created >= _MAX_BRANCHES_PER_RUN:
                _log.warning("branch cap reached, stopping run")
                T("MCTS", f"run: branch cap {_MAX_BRANCHES_PER_RUN} reached at iter {i}")
                break
            sel = self.select(root)
            target = sel
            if sel.untried_actions and not sel.terminal:
                child = self.expand(sel)
                if child is not None:
                    target = child
            try:
                rc = self.simulate(target)
                reward = aggregate_reward(rc)
            except Exception as e:
                _log.warning("simulate failed: %s", e)
                reward = 0.0
            self.backpropagate(target, reward)
            self.save()
            trace_mcts_iter(i, target.id, reward, target.branch_name,
                           detail=f"visits={target.visits} q={target.q:.4f}")
            trajectory().log(
                "mcts_iter", i=i, node=target.id, reward=reward
            )

        # Best leaf by q with visits > 0.
        # Prefer non-root leaves; if all leaves have same q, pick the deepest.
        best = root
        best_q = root.q
        best_depth = 0

        def _node_depth(nid: str) -> int:
            d = 0
            cur = nid
            while cur:
                n = self.nodes.get(cur)
                if n is None or n.parent_id is None:
                    break
                d += 1
                cur = n.parent_id
            return d

        for n in self.nodes.values():
            if n.visits > 0 and n.id != self.root_id:
                depth = _node_depth(n.id)
                # Prefer higher q; if equal, prefer deeper (more work done).
                if n.q > best_q or (n.q == best_q and depth > best_depth):
                    best = n
                    best_q = n.q
                    best_depth = depth
        if best is root:
            # Fallback: any visited node with highest q.
            for n in self.nodes.values():
                if n.visits > 0 and n.q > best_q:
                    best = n
                    best_q = n.q
        T("MCTS", f"run: done. best node={best.id[:10]} q={best.q:.4f} visits={best.visits} "
          f"branches_created={self._branches_created} total_nodes={len(self.nodes)}")
        return best

    def best_path(self) -> list[Node]:
        """Root → best leaf following highest-q children."""
        if self.root_id is None:
            return []
        path: list[Node] = []
        cur = self.nodes.get(self.root_id)
        while cur is not None:
            path.append(cur)
            if not cur.children_ids:
                break
            best_child: Node | None = None
            best_q = -float("inf")
            for cid in cur.children_ids:
                ch = self.nodes.get(cid)
                if ch is None:
                    continue
                if ch.q > best_q:
                    best_q = ch.q
                    best_child = ch
            if best_child is None:
                break
            cur = best_child
        return path

    def merge_best(self, message: str | None = None) -> str:
        path = self.best_path()
        if not path or len(path) < 2:
            T("MCTS", "merge_best: no best leaf (path too short), trying direct branch merge")
            # Fallback: find the best non-root node and merge its branch directly.
            best = None
            best_q = -float("inf")
            for n in self.nodes.values():
                if n.id != self.root_id and n.visits > 0 and n.q > best_q:
                    best = n
                    best_q = n.q
            if best is None:
                raise RuntimeError("no best leaf to merge")
            # Apply the best node's patch directly to the working tree.
            if best.patch is not None:
                target = best.patch.file
                target_path = Path(target)
                if not target_path.is_absolute():
                    target_path = Path(self.root_dir) / target_path
                _tools.write_patch(str(target_path), best.patch.new_content)
                try:
                    self.scm.update_from_dirty()
                except Exception:
                    pass
                sha = _tools.git_commit_all(
                    message or f"apex: merge best leaf {best.id[:8]}",
                    cwd=self.root_dir,
                )
                T("MCTS", f"merge_best: applied best patch directly, sha={sha[:16]}")
                return sha
            raise RuntimeError("no best leaf to merge (no patch)")
        root = path[0]
        leaf = path[-1]
        T("MCTS", f"merge_best: root={root.id[:10]} leaf={leaf.id[:10]} leaf_branch={leaf.branch_name}")
        orig_branch = _tools.git_current_branch(cwd=self.root_dir)
        try:
            T("MCTS", f"merge_best: checkout {root.branch_name}")
            try:
                _tools.git_checkout(root.branch_name, cwd=self.root_dir)
            except Exception as e:
                T("MCTS", f"merge_best: checkout root failed: {e}")
                if not _tools.is_git_repo(self.root_dir):
                    _tools.fs_create_branch(root.branch_name, from_ref="HEAD", cwd=self.root_dir)
                    _tools.fs_checkout(root.branch_name, cwd=self.root_dir)
                else:
                    raise
            T("MCTS", f"merge_best: merging {leaf.branch_name} into {root.branch_name}")
            _tools.git_merge(
                leaf.branch_name,
                message=message or f"apex: merge best leaf {leaf.id[:8]}",
                cwd=self.root_dir,
            )
            sha = _tools.git_head_sha(cwd=self.root_dir)
            T("MCTS", f"merge_best: merged at {sha[:16]}")
            return sha
        finally:
            try:
                if orig_branch and _tools.git_current_branch(cwd=self.root_dir) != orig_branch:
                    _tools.git_checkout(orig_branch, cwd=self.root_dir)
            except Exception:
                pass
