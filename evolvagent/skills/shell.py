"""
ShellCommandSkill: Execute shell commands parsed from natural language.

Default trust level is SUGGEST — the agent shows the command and waits
for user approval before executing. This is the poster child for the
progressive trust model.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any

from ..core.llm import LLMClient
from ..core.skill import BaseSkill, SkillMetadata, SkillOrigin, SkillResult, TrustLevel

_SYSTEM_PROMPT = """\
You are a shell command generator. Given a user's request in natural language,
output ONLY the shell command to execute. No explanation, no markdown, no quotes
around the command. Just the raw command.

Rules:
- Use common Unix/macOS commands
- For dangerous operations (rm -rf, etc.), add safety flags when possible
- If the request is ambiguous, pick the most common interpretation
- Output exactly one command (use && to chain if needed)

Examples:
  User: list files in current directory
  Output: ls -la

  User: find all python files
  Output: find . -name "*.py"

  User: show disk usage
  Output: df -h
"""

MAX_OUTPUT_LENGTH = 5000


class ShellCommandSkill(BaseSkill):
    """
    Translates natural language into shell commands and executes them.

    Trust model:
      OBSERVE  — shows the parsed command, does not execute
      SUGGEST  — shows command, asks for confirmation, then executes
      AUTO     — parses and executes immediately
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(metadata=SkillMetadata(
            name="shell_command",
            category="system",
            description="Execute shell commands from natural language instructions",
            tags=["shell", "command", "terminal", "run", "execute"],
            trigger_conditions=["run", "execute", "shell", "command", "terminal", "list files",
                                "find", "show", "delete", "create", "move", "copy"],
            trust_level=TrustLevel.SUGGEST,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="active",
        ))
        self.llm = llm_client
        self._cached_command: dict[str, str] = {}

    async def _parse_command(self, user_input: str) -> str:
        """Use LLM to convert natural language to a shell command."""
        if user_input in self._cached_command:
            return self._cached_command[user_input]

        response = await self.llm.complete(
            prompt=user_input,
            system=_SYSTEM_PROMPT,
            temperature=0.1,
        )
        command = response.content.strip()
        # Strip markdown code fences if the LLM wraps the output
        if command.startswith("```") and command.endswith("```"):
            command = command[3:-3].strip()
        if command.startswith("`") and command.endswith("`"):
            command = command[1:-1].strip()
        # Remove "bash\n" or "sh\n" prefix from code blocks
        for prefix in ("bash\n", "sh\n", "zsh\n"):
            if command.startswith(prefix):
                command = command[len(prefix):]

        self._cached_command[user_input] = command
        return command

    async def preview(self, context: dict[str, Any]) -> str:
        """Show the command that would be executed."""
        user_input = context.get("user_input", "")
        command = await self._parse_command(user_input)
        return f"Will execute shell command:\n  $ {command}"

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        """Match on shell/command-related keywords."""
        intent_lower = intent.lower()
        strong_triggers = ["run ", "execute ", "shell ", "command "]
        for trigger in strong_triggers:
            if trigger in intent_lower:
                return 0.9
        # Weaker match on general trigger_conditions
        for trigger in self.metadata.trigger_conditions:
            if trigger.lower() in intent_lower:
                return 0.7
        return 0.0

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        """Parse the command via LLM and execute it."""
        user_input = context.get("user_input", "")
        if not user_input.strip():
            return SkillResult(success=False, error="Empty input")

        start = time.time()
        try:
            command = await self._parse_command(user_input)
            # Clear cache after use
            self._cached_command.pop(user_input, None)

            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            elapsed_ms = (time.time() - start) * 1000
            output = proc.stdout
            if proc.stderr:
                output += ("\n" + proc.stderr) if output else proc.stderr
            if len(output) > MAX_OUTPUT_LENGTH:
                output = output[:MAX_OUTPUT_LENGTH] + "\n... (truncated)"

            if proc.returncode == 0:
                return SkillResult(
                    success=True,
                    output=output or "(command completed with no output)",
                    data={"command": command, "return_code": proc.returncode},
                    execution_time_ms=elapsed_ms,
                )
            else:
                return SkillResult(
                    success=False,
                    output=output,
                    error=f"Command exited with code {proc.returncode}",
                    data={"command": command, "return_code": proc.returncode},
                    execution_time_ms=elapsed_ms,
                )

        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error="Command timed out after 30 seconds",
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
