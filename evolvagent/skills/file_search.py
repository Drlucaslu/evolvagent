"""
FileSearchSkill: Search files by name or content using find/grep.

Trust level AUTO — pure read-only, no side effects.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ..core.skill import BaseSkill, SkillMetadata, SkillOrigin, SkillResult, TrustLevel

MAX_OUTPUT_LENGTH = 8000

# Patterns that indicate content search (grep) vs filename search (find)
_CONTENT_KEYWORDS = {"grep", "search for", "search in", "containing", "content", "look for"}
_FILENAME_KEYWORDS = {"find file", "find all", "find *."}

_EXCLUDED_DIRS = [".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"]


def _detect_search_type(user_input: str) -> str:
    """Detect whether the user wants a filename search or content search."""
    lower = user_input.lower()
    for kw in _FILENAME_KEYWORDS:
        if kw in lower:
            return "filename"
    for kw in _CONTENT_KEYWORDS:
        if kw in lower:
            return "content"
    # Default: if there's a file extension pattern like "*.py", treat as filename
    if re.search(r"\*\.\w+", user_input):
        return "filename"
    # Default to content search
    return "content"


def _extract_pattern(user_input: str) -> str:
    """Extract the search pattern from user input."""
    # Check for quoted strings first
    quoted = re.findall(r'"([^"]+)"', user_input)
    if quoted:
        return quoted[0]
    quoted = re.findall(r"'([^']+)'", user_input)
    if quoted:
        return quoted[0]

    # Remove common filler words and extract the likely search term
    lower = user_input.lower()
    # Remove command-like prefixes
    for prefix in [
        "search for", "search in", "find file", "find all",
        "grep for", "grep", "find", "search", "look for", "where is",
    ]:
        if lower.startswith(prefix):
            remainder = user_input[len(prefix):].strip()
            if remainder:
                # Take the first meaningful token(s)
                return remainder.split(" in ")[0].strip()

    # Fallback: last word(s) that look like a pattern
    words = user_input.split()
    # Filter out common filler
    fillers = {"for", "in", "the", "all", "files", "file", "please", "can", "you"}
    meaningful = [w for w in words if w.lower() not in fillers]
    return meaningful[-1] if meaningful else user_input.strip()


def _extract_file_types(user_input: str) -> list[str]:
    """Extract file type filters from user input."""
    lower = user_input.lower()
    types = []
    if "python" in lower or ".py" in lower:
        types.append("*.py")
    if "javascript" in lower or ".js" in lower:
        types.append("*.js")
    if "typescript" in lower or ".ts" in lower:
        types.append("*.ts")
    if "rust" in lower or ".rs" in lower:
        types.append("*.rs")
    if "go " in lower or ".go" in lower:
        types.append("*.go")
    if "java " in lower or ".java" in lower:
        types.append("*.java")
    # Glob patterns like *.py
    globs = re.findall(r"\*\.\w+", user_input)
    for g in globs:
        if g not in types:
            types.append(g)
    return types


class FileSearchSkill(BaseSkill):
    """
    Search files by name or content using find/grep.

    Supports:
      - Filename search: find files matching a pattern
      - Content search: grep for patterns inside files
    """

    def __init__(self):
        super().__init__(metadata=SkillMetadata(
            name="file_search",
            category="search",
            description="Search files by name or content (grep/find)",
            tags=["search", "find", "grep", "file", "code search"],
            trigger_conditions=["search for", "find file", "grep", "search in",
                                "find all", "where is", "look for"],
            trust_level=TrustLevel.AUTO,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="active",
        ))

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        intent_lower = intent.lower()
        strong = ["search for", "find file", "grep ", "search in", "find all"]
        for trigger in strong:
            if trigger in intent_lower:
                return 0.9
        weak = ["find", "search", "where is", "look for"]
        for trigger in weak:
            if trigger in intent_lower:
                return 0.7
        return 0.0

    async def preview(self, context: dict[str, Any]) -> str:
        user_input = context.get("user_input", "")
        search_type = _detect_search_type(user_input)
        pattern = _extract_pattern(user_input)
        workspace = context.get("workspace", ".")
        if search_type == "filename":
            return f"Will search for files matching '{pattern}' in {workspace}"
        else:
            file_types = _extract_file_types(user_input)
            type_desc = ", ".join(file_types) if file_types else "all files"
            return f"Will grep for '{pattern}' in {type_desc} under {workspace}"

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        user_input = context.get("user_input", "")
        if not user_input.strip():
            return SkillResult(success=False, error="Empty input")

        workspace = context.get("workspace", str(Path.cwd()))
        search_type = _detect_search_type(user_input)
        pattern = _extract_pattern(user_input)

        start = time.time()
        try:
            if search_type == "filename":
                output = self._search_filename(pattern, workspace)
            else:
                file_types = _extract_file_types(user_input)
                output = self._search_content(pattern, workspace, file_types)

            elapsed_ms = (time.time() - start) * 1000

            if not output.strip():
                return SkillResult(
                    success=True,
                    output=f"No results found for '{pattern}'.",
                    data={"search_type": search_type, "pattern": pattern},
                    execution_time_ms=elapsed_ms,
                )

            if len(output) > MAX_OUTPUT_LENGTH:
                lines = output.split("\n")
                truncated = "\n".join(lines[:50])
                output = truncated + f"\n\n... ({len(lines)} total matches, showing first 50)"

            return SkillResult(
                success=True,
                output=output,
                data={"search_type": search_type, "pattern": pattern},
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(success=False, error=str(e), execution_time_ms=elapsed_ms)

    def _search_filename(self, pattern: str, workspace: str) -> str:
        excludes = []
        for d in _EXCLUDED_DIRS:
            excludes.extend(["-not", "-path", f"*/{d}/*"])
        cmd = ["find", workspace, "-name", f"*{pattern}*"] + excludes
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return proc.stdout.strip()

    def _search_content(
        self, pattern: str, workspace: str, file_types: list[str] | None = None
    ) -> str:
        cmd = ["grep", "-rn"]
        if file_types:
            for ft in file_types:
                cmd.extend(["--include", ft])
        for d in _EXCLUDED_DIRS:
            cmd.extend(["--exclude-dir", d])
        cmd.extend([pattern, workspace])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return proc.stdout.strip()
