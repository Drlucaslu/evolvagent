"""
WorkspaceContextSkill: Gather workspace intelligence for Claude Code integration.

Analyzes git history, directory structure, project metadata, and recent
EvolvAgent activity to produce a structured workspace summary suitable
for CLAUDE.md — the file Claude Code reads on startup.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from ..core.llm import LLMClient
from ..core.skill import BaseSkill, SkillMetadata, SkillOrigin, SkillResult, TrustLevel

_NOISE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
    ".tox", "dist", "build", ".mypy_cache", ".ruff_cache", ".eggs",
    "egg-info", ".benchmarks", ".DS_Store",
}

_KEY_FILES = [
    "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".github", ".gitlab-ci.yml", "Jenkinsfile",
    "tox.ini", "setup.cfg", "setup.py", ".editorconfig",
    "requirements.txt", "Pipfile", "Cargo.toml", "go.mod",
]

_ENHANCE_SYSTEM_PROMPT = """\
You are a project analyst. Given raw workspace data (git log, directory structure,
project metadata), produce a concise "Current Focus" section (2-3 sentences)
describing what the developer is currently working on, based on recent commits
and the active branch name.

Also produce a "Conventions" section listing detected code style and tooling
conventions as bullet points.

Output ONLY these two sections in markdown, nothing else:

## Current Focus
<your analysis>

## Conventions
<bullet points>
"""


class WorkspaceContextSkill(BaseSkill):
    """
    Gathers workspace intelligence and renders a CLAUDE.md template.

    Trust level is AUTO — this skill only reads and analyzes, never writes files.
    File writing is handled by the CLI command with user confirmation.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(metadata=SkillMetadata(
            name="workspace_context",
            category="workspace",
            description="Analyze workspace and generate project context for Claude Code",
            tags=["context", "workspace", "claude", "project", "analyze"],
            trigger_conditions=["context", "workspace", "claude.md", "project context",
                                "analyze project", "generate context"],
            trust_level=TrustLevel.AUTO,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="passive",
        ))
        self.llm = llm_client

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        """Match on context/workspace-related keywords."""
        intent_lower = intent.lower()
        strong = ["context", "claude.md", "workspace context", "project context"]
        for kw in strong:
            if kw in intent_lower:
                return 0.9
        weak = ["analyze", "workspace", "project structure", "document"]
        for kw in weak:
            if kw in intent_lower:
                return 0.6
        return 0.0

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        """Gather workspace data and render context markdown."""
        workspace_raw = context.get("workspace", ".")
        workspace = Path(workspace_raw).resolve()
        store = context.get("store")

        start = time.time()
        try:
            # Query activity with both raw and resolved paths (macOS /var vs /private/var)
            activity = self._recent_activity(store, str(workspace))
            if not activity and str(workspace) != workspace_raw:
                activity = self._recent_activity(store, workspace_raw)

            data = {
                "workspace": str(workspace),
                "git": self._git_info(workspace),
                "project": self._project_metadata(workspace),
                "structure": self._directory_structure(workspace),
                "conventions": self._detect_conventions(workspace),
                "activity": activity,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            markdown = self._render_template(data)

            if self.llm and context.get("use_llm", True):
                try:
                    markdown = await self._enhance_with_llm(data, markdown)
                except Exception:
                    pass  # Fallback to template

            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(
                success=True,
                output=markdown,
                data=data,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(success=False, error=str(e), execution_time_ms=elapsed_ms)

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    @staticmethod
    def _run_git(workspace: Path, args: list[str]) -> str:
        """Run a git command and return stdout, or empty string on failure."""
        try:
            proc = subprocess.run(
                ["git", "-C", str(workspace)] + args,
                capture_output=True, text=True, timeout=10,
            )
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    def _git_info(self, workspace: Path) -> dict[str, Any]:
        """Gather git repository information."""
        # Check if it's a git repo
        if not self._run_git(workspace, ["rev-parse", "--git-dir"]):
            return {}

        recent_commits = self._run_git(workspace, [
            "log", "--oneline", "--no-decorate", "-15",
        ])
        active_branch = self._run_git(workspace, [
            "rev-parse", "--abbrev-ref", "HEAD",
        ])
        branches = self._run_git(workspace, [
            "branch", "--format=%(refname:short)",
        ])
        diff_stat = self._run_git(workspace, [
            "diff", "--stat", "HEAD~5..HEAD", "--",
        ])

        return {
            "recent_commits": recent_commits.split("\n") if recent_commits else [],
            "active_branch": active_branch or "unknown",
            "branches": branches.split("\n") if branches else [],
            "recent_changes": diff_stat,
        }

    def _project_metadata(self, workspace: Path) -> dict[str, Any]:
        """Read project metadata from pyproject.toml or package.json."""
        meta: dict[str, Any] = {"name": workspace.name}

        # Try pyproject.toml
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[no-redef]
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                project = data.get("project", {})
                meta["name"] = project.get("name", meta["name"])
                meta["version"] = project.get("version", "")
                meta["description"] = project.get("description", "")
                meta["python_requires"] = project.get("requires-python", "")
                meta["dependencies"] = project.get("dependencies", [])
                meta["source"] = "pyproject.toml"

                # Extract scripts/commands
                scripts = project.get("scripts", {})
                if scripts:
                    meta["scripts"] = scripts

                # Check for tool configs
                tools = data.get("tool", {})
                if tools:
                    meta["tools"] = list(tools.keys())
            except Exception:
                pass

        # Try package.json
        pkg_json = workspace / "package.json"
        if pkg_json.exists() and "source" not in meta:
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8"))
                meta["name"] = data.get("name", meta["name"])
                meta["version"] = data.get("version", "")
                meta["description"] = data.get("description", "")
                meta["source"] = "package.json"
                scripts = data.get("scripts", {})
                if scripts:
                    meta["scripts"] = scripts
                deps = list(data.get("dependencies", {}).keys())
                if deps:
                    meta["dependencies"] = deps
            except Exception:
                pass

        return meta

    def _directory_structure(self, workspace: Path) -> dict[str, Any]:
        """Scan top-level directory structure and file type statistics."""
        dirs = []
        files = []
        file_types: dict[str, int] = {}
        key_files: list[str] = []

        if not workspace.is_dir():
            return {"dirs": dirs, "top_level_files": files, "key_files": key_files,
                    "file_type_counts": {}, "total_files": 0}

        try:
            for item in sorted(workspace.iterdir()):
                name = item.name
                if name.startswith(".") and name not in (".github", ".gitlab-ci.yml"):
                    continue
                if name in _NOISE_DIRS:
                    continue

                if item.is_dir():
                    # Count files in subdirectory (shallow)
                    try:
                        count = sum(1 for _ in item.iterdir())
                    except PermissionError:
                        count = 0
                    dirs.append({"name": name, "items": count})
                elif item.is_file():
                    files.append(name)
                    ext = item.suffix.lower() or "(no ext)"
                    file_types[ext] = file_types.get(ext, 0) + 1

            # Check for key files
            for kf in _KEY_FILES:
                if (workspace / kf).exists():
                    key_files.append(kf)

        except PermissionError:
            pass

        # Count files by type across the project (bounded walk)
        total_by_ext: dict[str, int] = {}
        file_count = 0
        try:
            for p in workspace.rglob("*"):
                if file_count > 5000:
                    break
                rel = str(p.relative_to(workspace))
                if any(noise in rel.split("/") for noise in _NOISE_DIRS):
                    continue
                if rel.startswith("."):
                    continue
                if p.is_file():
                    file_count += 1
                    ext = p.suffix.lower() or "(no ext)"
                    total_by_ext[ext] = total_by_ext.get(ext, 0) + 1
        except (PermissionError, OSError):
            pass

        return {
            "dirs": dirs,
            "top_level_files": files,
            "key_files": key_files,
            "file_type_counts": dict(sorted(
                total_by_ext.items(), key=lambda x: x[1], reverse=True
            )[:15]),
            "total_files": file_count,
        }

    def _detect_conventions(self, workspace: Path) -> dict[str, Any]:
        """Detect code conventions from config files."""
        conventions: dict[str, Any] = {}

        # Python tools
        pyproject = workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[no-redef]
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                tools = data.get("tool", {})

                if "ruff" in tools:
                    ruff = tools["ruff"]
                    conventions["linter"] = "ruff"
                    if "line-length" in ruff:
                        conventions["line_length"] = ruff["line-length"]

                if "pytest" in tools or "pytest.ini_options" in tools:
                    conventions["test_framework"] = "pytest"

                if "black" in tools:
                    conventions["formatter"] = "black"

                if "mypy" in tools:
                    conventions["type_checker"] = "mypy"
            except Exception:
                pass

        # Other config files
        if (workspace / "pytest.ini").exists() or (workspace / "conftest.py").exists():
            conventions.setdefault("test_framework", "pytest")
        if (workspace / ".eslintrc.json").exists() or (workspace / ".eslintrc.js").exists():
            conventions["linter"] = "eslint"
        if (workspace / ".prettierrc").exists() or (workspace / ".prettierrc.json").exists():
            conventions["formatter"] = "prettier"
        if (workspace / ".editorconfig").exists():
            conventions["editorconfig"] = True
        if (workspace / "Makefile").exists():
            conventions["build_tool"] = "make"

        # Detect test directory
        for test_dir in ["tests", "test", "spec", "__tests__"]:
            if (workspace / test_dir).is_dir():
                conventions["test_dir"] = test_dir
                break

        return conventions

    @staticmethod
    def _recent_activity(store: Any, workspace: str) -> list[dict[str, Any]]:
        """Load recent EvolvAgent activity from the store."""
        if store is None:
            return []
        try:
            return store.recent_activity(workspace=workspace, limit=10)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_template(self, data: dict[str, Any]) -> str:
        """Render workspace data into a CLAUDE.md markdown template."""
        lines: list[str] = []

        lines.append("# Project Context")
        lines.append("")
        lines.append(f"> Auto-generated by EvolvAgent on {data['generated_at']}.")
        lines.append("> Re-generate with: `evolvagent context`")
        lines.append("")

        # Overview
        proj = data["project"]
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Name**: {proj.get('name', 'unknown')}")
        if proj.get("description"):
            lines.append(f"- **Description**: {proj['description']}")
        if proj.get("version"):
            lines.append(f"- **Version**: {proj['version']}")

        # Tech stack from file types
        structure = data["structure"]
        ftc = structure.get("file_type_counts", {})
        languages = []
        lang_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
            ".c": "C", ".cpp": "C++", ".swift": "Swift", ".kt": "Kotlin",
        }
        for ext, lang in lang_map.items():
            if ext in ftc:
                languages.append(f"{lang} ({ftc[ext]})")
        if languages:
            lines.append(f"- **Tech Stack**: {', '.join(languages[:5])}")
        if proj.get("python_requires"):
            lines.append(f"- **Python**: {proj['python_requires']}")
        lines.append("")

        # Directory Structure
        lines.append("## Directory Structure")
        lines.append("")
        lines.append("```")
        for d in structure.get("dirs", []):
            lines.append(f"  {d['name']}/  ({d['items']} items)")
        for f in structure.get("top_level_files", [])[:10]:
            lines.append(f"  {f}")
        lines.append("```")
        lines.append("")

        if structure.get("total_files"):
            lines.append(f"Total: {structure['total_files']} files")
            lines.append("")

        # Build & Run Commands
        scripts = proj.get("scripts", {})
        if scripts or data.get("conventions", {}).get("test_framework"):
            lines.append("## Build & Run Commands")
            lines.append("")
            lines.append("```bash")
            if scripts:
                for name, cmd in scripts.items():
                    if isinstance(cmd, str):
                        lines.append(f"# {name}")
                        lines.append(cmd)
                    else:
                        lines.append(f"{name}")
            test_fw = data.get("conventions", {}).get("test_framework")
            test_dir = data.get("conventions", {}).get("test_dir", "tests")
            if test_fw == "pytest":
                lines.append(f"# Run tests")
                lines.append(f"pytest {test_dir}/ -v")
            linter = data.get("conventions", {}).get("linter")
            if linter == "ruff":
                lines.append("# Lint")
                lines.append("ruff check .")
            lines.append("```")
            lines.append("")

        # Recent Git Activity
        git = data.get("git", {})
        commits = git.get("recent_commits", [])
        if commits:
            lines.append("## Recent Git Activity")
            lines.append("")
            branch = git.get("active_branch", "unknown")
            lines.append(f"Active branch: `{branch}`")
            lines.append("")
            for commit in commits[:10]:
                lines.append(f"- {commit}")
            lines.append("")

        # Recent EvolvAgent Activity
        activity = data.get("activity", [])
        if activity:
            lines.append("## Recent EvolvAgent Activity")
            lines.append("")
            for entry in activity[:5]:
                ts = time.strftime("%m-%d %H:%M", time.localtime(entry.get("timestamp", 0)))
                query = entry.get("query", "")[:80]
                skill = entry.get("skill_used", "")
                status = "ok" if entry.get("success") else "fail"
                lines.append(f"- [{ts}] {query} → {skill} ({status})")
            lines.append("")

        # Conventions
        conv = data.get("conventions", {})
        if conv:
            lines.append("## Conventions")
            lines.append("")
            if conv.get("test_framework"):
                lines.append(f"- Test framework: {conv['test_framework']}")
            if conv.get("test_dir"):
                lines.append(f"- Test directory: `{conv['test_dir']}/`")
            if conv.get("linter"):
                linter_info = conv["linter"]
                if conv.get("line_length"):
                    linter_info += f" (line-length={conv['line_length']})"
                lines.append(f"- Linter: {linter_info}")
            if conv.get("formatter"):
                lines.append(f"- Formatter: {conv['formatter']}")
            if conv.get("type_checker"):
                lines.append(f"- Type checker: {conv['type_checker']}")
            if conv.get("build_tool"):
                lines.append(f"- Build tool: {conv['build_tool']}")
            lines.append("")

        # Current Focus (template fallback — LLM will enhance this)
        if commits:
            lines.append("## Current Focus")
            lines.append("")
            branch = git.get("active_branch", "")
            lines.append(f"Based on recent activity on branch `{branch}`:")
            lines.append("")
            for commit in commits[:3]:
                lines.append(f"- {commit}")
            lines.append("")

        return "\n".join(lines)

    async def _enhance_with_llm(self, data: dict[str, Any], template: str) -> str:
        """Use LLM to add Current Focus and refined Conventions sections."""
        git = data.get("git", {})
        summary_parts = []
        if git.get("active_branch"):
            summary_parts.append(f"Active branch: {git['active_branch']}")
        commits = git.get("recent_commits", [])
        if commits:
            summary_parts.append("Recent commits:\n" + "\n".join(commits[:10]))
        conv = data.get("conventions", {})
        if conv:
            summary_parts.append(f"Detected tools: {json.dumps(conv)}")

        if not summary_parts:
            return template

        prompt = "\n\n".join(summary_parts)
        response = await self.llm.complete(
            prompt=prompt,
            system=_ENHANCE_SYSTEM_PROMPT,
            temperature=0.3,
        )

        enhanced = response.content.strip()
        if not enhanced:
            return template

        # Replace the template's Current Focus and Conventions with LLM output
        result_lines = []
        skip_section = False
        for line in template.split("\n"):
            if line.startswith("## Current Focus") or line.startswith("## Conventions"):
                skip_section = True
                continue
            if skip_section and line.startswith("## "):
                skip_section = False
            if not skip_section:
                result_lines.append(line)

        # Append LLM-generated sections
        result_lines.append(enhanced)
        result_lines.append("")

        return "\n".join(result_lines)
