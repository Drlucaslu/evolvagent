"""
Git operation Skills: GitInfoSkill (read-only) and GitActionSkill (write).

Split into two Skills so the trust model is clean:
  - GitInfoSkill (AUTO): status, diff, log, branch — no side effects
  - GitActionSkill (SUGGEST): commit, stash — requires user confirmation
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from ..core.llm import LLMClient
from ..core.skill import BaseSkill, SkillMetadata, SkillOrigin, SkillResult, TrustLevel

MAX_OUTPUT_LENGTH = 8000

_COMMIT_MSG_SYSTEM = """\
You are a git commit message generator. Given a git diff and optional user description,
output ONLY the commit message. No explanation, no markdown.

Rules:
- First line: concise summary under 72 chars (imperative mood)
- If the diff is large, add a blank line followed by bullet points for details
- Do not wrap in quotes
"""

# --- Operation detection ---

_READ_OPS = {
    "diff": (["git", "diff"], ["git diff", "diff", "what changed", "show changes"]),
    "diff_staged": (["git", "diff", "--staged"], ["staged", "cached"]),
    "status": (["git", "status", "--short"], ["git status", "status", "changes"]),
    "log": (["git", "log", "--oneline", "-15"], ["git log", "log", "history", "recent commits"]),
    "branch": (["git", "branch"], ["git branch", "branches", "current branch"]),
}

_WRITE_TRIGGERS = {
    "commit": ["commit", "save changes"],
    "stash": ["stash"],
    "stash_pop": ["stash pop", "unstash", "pop stash"],
}


def _detect_read_op(intent: str) -> str | None:
    """Return the matching read operation name, or None."""
    lower = intent.lower()
    # Check specific ops in priority order
    if "diff" in lower and "staged" in lower:
        return "diff_staged"
    for op_name, (_, triggers) in _READ_OPS.items():
        for trigger in triggers:
            if trigger in lower:
                return op_name
    return None


def _detect_write_op(intent: str) -> str | None:
    """Return the matching write operation name, or None."""
    lower = intent.lower()
    if "stash pop" in lower or "unstash" in lower or "pop stash" in lower:
        return "stash_pop"
    for op_name, triggers in _WRITE_TRIGGERS.items():
        for trigger in triggers:
            if trigger in lower:
                return op_name
    return None


def _run_git(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_LENGTH:
        lines = text.split("\n")
        return "\n".join(lines[:80]) + f"\n\n... ({len(lines)} total lines, showing first 80)"
    return text


# ---------------------------------------------------------------------------
# GitInfoSkill — read-only operations (AUTO trust)
# ---------------------------------------------------------------------------

class GitInfoSkill(BaseSkill):
    """Read-only Git operations: status, diff, log, branch."""

    def __init__(self):
        super().__init__(metadata=SkillMetadata(
            name="git_info",
            category="git",
            description="Git read-only operations: status, diff, log, branch",
            tags=["git", "status", "diff", "log", "branch", "history"],
            trigger_conditions=["git status", "git diff", "git log", "git branch",
                                "changes", "diff", "history", "commits", "branches"],
            trust_level=TrustLevel.AUTO,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="active",
        ))

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        op = _detect_read_op(intent)
        if op:
            # Higher score for exact git commands
            lower = intent.lower()
            if lower.startswith("git ") or "git " in lower:
                return 0.95
            return 0.85
        return 0.0

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        user_input = context.get("user_input", "")
        if not user_input.strip():
            return SkillResult(success=False, error="Empty input")

        op = _detect_read_op(user_input)
        if not op:
            return SkillResult(success=False, error="Could not determine git operation")

        cmd = _READ_OPS[op][0]
        start = time.time()
        try:
            proc = _run_git(cmd)
            elapsed_ms = (time.time() - start) * 1000
            output = proc.stdout.strip()
            if proc.stderr.strip():
                output += ("\n" + proc.stderr.strip()) if output else proc.stderr.strip()

            if proc.returncode != 0:
                return SkillResult(
                    success=False, output=output,
                    error=f"git exited with code {proc.returncode}",
                    data={"command": " ".join(cmd)},
                    execution_time_ms=elapsed_ms,
                )

            return SkillResult(
                success=True,
                output=_truncate(output) if output else "(no output)",
                data={"command": " ".join(cmd), "operation": op},
                execution_time_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False, error="Git command timed out",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(success=False, error=str(e), execution_time_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# GitActionSkill — write operations (SUGGEST trust)
# ---------------------------------------------------------------------------

class GitActionSkill(BaseSkill):
    """Git write operations: commit, stash. Requires user confirmation."""

    def __init__(self, llm_client: LLMClient | None = None):
        super().__init__(metadata=SkillMetadata(
            name="git_action",
            category="git",
            description="Git write operations: commit, stash (requires confirmation)",
            tags=["git", "commit", "stash", "save"],
            trigger_conditions=["commit", "stash", "save changes"],
            trust_level=TrustLevel.SUGGEST,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="active",
        ))
        self.llm = llm_client

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        op = _detect_write_op(intent)
        if op:
            lower = intent.lower()
            if lower.startswith("git ") or "git " in lower:
                return 0.95
            return 0.85
        return 0.0

    async def preview(self, context: dict[str, Any]) -> str:
        user_input = context.get("user_input", "")
        op = _detect_write_op(user_input)

        if op == "commit":
            # Show what would be committed
            proc = _run_git(["git", "status", "--short"])
            status = proc.stdout.strip() or "(no changes)"
            msg = await self._generate_commit_message(context) if self.llm else "..."
            return f"Will commit with message: {msg}\n\nFiles:\n{status}"

        elif op == "stash":
            return "Will run: git stash"

        elif op == "stash_pop":
            return "Will run: git stash pop"

        return f"Git action: {op}"

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        user_input = context.get("user_input", "")
        if not user_input.strip():
            return SkillResult(success=False, error="Empty input")

        op = _detect_write_op(user_input)
        if not op:
            return SkillResult(success=False, error="Could not determine git action")

        start = time.time()
        try:
            if op == "commit":
                return await self._do_commit(context, start)
            elif op == "stash":
                return self._do_simple(["git", "stash"], start)
            elif op == "stash_pop":
                return self._do_simple(["git", "stash", "pop"], start)
            else:
                return SkillResult(success=False, error=f"Unknown git action: {op}")
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(success=False, error=str(e), execution_time_ms=elapsed_ms)

    async def _generate_commit_message(self, context: dict[str, Any]) -> str:
        """Use LLM to generate a commit message from the current diff."""
        if not self.llm:
            return "Update files"
        diff_proc = _run_git(["git", "diff", "--staged"])
        diff = diff_proc.stdout.strip()
        if not diff:
            diff_proc = _run_git(["git", "diff"])
            diff = diff_proc.stdout.strip()
        if not diff:
            return "Update files"
        # Truncate diff for LLM
        if len(diff) > 4000:
            diff = diff[:4000] + "\n... (truncated)"
        user_input = context.get("user_input", "")
        prompt = f"User request: {user_input}\n\nGit diff:\n{diff}"
        response = await self.llm.complete(prompt=prompt, system=_COMMIT_MSG_SYSTEM, temperature=0.2)
        return response.content.strip().strip('"').strip("'")

    async def _do_commit(self, context: dict[str, Any], start: float) -> SkillResult:
        msg = await self._generate_commit_message(context)
        # Stage all changes
        _run_git(["git", "add", "-A"])
        proc = _run_git(["git", "commit", "-m", msg])
        elapsed_ms = (time.time() - start) * 1000
        output = proc.stdout.strip()
        if proc.stderr.strip():
            output += ("\n" + proc.stderr.strip()) if output else proc.stderr.strip()
        if proc.returncode != 0:
            return SkillResult(
                success=False, output=output,
                error=f"Commit failed (exit {proc.returncode})",
                data={"command": f'git commit -m "{msg}"'},
                execution_time_ms=elapsed_ms,
            )
        return SkillResult(
            success=True,
            output=output or f"Committed: {msg}",
            data={"command": f'git commit -m "{msg}"', "message": msg},
            execution_time_ms=elapsed_ms,
        )

    def _do_simple(self, cmd: list[str], start: float) -> SkillResult:
        proc = _run_git(cmd)
        elapsed_ms = (time.time() - start) * 1000
        output = proc.stdout.strip()
        if proc.stderr.strip():
            output += ("\n" + proc.stderr.strip()) if output else proc.stderr.strip()
        if proc.returncode != 0:
            return SkillResult(
                success=False, output=output,
                error=f"Command failed (exit {proc.returncode})",
                data={"command": " ".join(cmd)},
                execution_time_ms=elapsed_ms,
            )
        return SkillResult(
            success=True,
            output=output or "(done)",
            data={"command": " ".join(cmd)},
            execution_time_ms=elapsed_ms,
        )
