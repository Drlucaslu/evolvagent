"""Tests for WorkspaceContextSkill, activity logging, and context CLI."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.config import Settings, AgentConfig
from evolvagent.core.llm import LLMResponse
from evolvagent.core.storage import SkillStore
from evolvagent.skills.workspace_context import WorkspaceContextSkill


def run(coro):
    return asyncio.run(coro)


def _tmp_settings():
    return Settings(agent=AgentConfig(data_dir=tempfile.mkdtemp()))


def _mock_llm(content="## Current Focus\nWorking on tests.\n\n## Conventions\n- pytest\n"):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=content, model="mock", tokens_in=50, tokens_out=100,
    ))
    return llm


def _make_git_repo(path: Path) -> None:
    """Create a minimal git repo with a few commits."""
    subprocess.run(["git", "init", str(path)], capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        capture_output=True,
    )
    # Create initial commit
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "Initial commit"],
        capture_output=True,
    )
    # Second commit
    (path / "main.py").write_text("print('hello')\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "Add main script"],
        capture_output=True,
    )


# ===================================================================
# WorkspaceContextSkill Tests
# ===================================================================

class TestWorkspaceContextSkill(unittest.TestCase):

    def test_can_handle_strong_match(self):
        skill = WorkspaceContextSkill()
        self.assertGreaterEqual(skill.can_handle("generate context"), 0.9)
        self.assertGreaterEqual(skill.can_handle("update claude.md"), 0.9)
        self.assertGreaterEqual(skill.can_handle("workspace context"), 0.9)

    def test_can_handle_weak_match(self):
        skill = WorkspaceContextSkill()
        self.assertGreater(skill.can_handle("analyze this project"), 0)

    def test_can_handle_no_match(self):
        skill = WorkspaceContextSkill()
        self.assertEqual(skill.can_handle("what is python?"), 0.0)

    def test_execute_non_git_dir(self):
        """Skill works in a non-git directory."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "hello.py").write_text("print('hi')")
            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("Project Context", result.output)
            # No git section since not a repo
            self.assertNotIn("Recent Git Activity", result.output)

    def test_execute_git_dir(self):
        """Skill gathers git info from a real repo."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _make_git_repo(tmppath)

            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("Recent Git Activity", result.output)
            self.assertIn("Initial commit", result.output)
            self.assertIn("Add main script", result.output)

    def test_template_has_expected_sections(self):
        """Output markdown contains all expected section headers."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _make_git_repo(tmppath)
            # Add a pyproject.toml
            (tmppath / "pyproject.toml").write_text(
                '[project]\nname = "testproj"\nversion = "1.0"\n'
                'description = "A test project"\n'
            )
            # Add some dirs
            (tmppath / "src").mkdir()
            (tmppath / "tests").mkdir()
            (tmppath / "src" / "app.py").write_text("pass")

            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("## Overview", result.output)
            self.assertIn("## Directory Structure", result.output)
            self.assertIn("testproj", result.output)
            self.assertIn("A test project", result.output)

    def test_no_llm_works(self):
        """Skill with no LLM client produces template-only output."""
        skill = WorkspaceContextSkill(llm_client=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("Project Context", result.output)

    def test_with_mock_llm(self):
        """Skill calls LLM and integrates enhanced output."""
        llm = _mock_llm()
        skill = WorkspaceContextSkill(llm_client=llm)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _make_git_repo(tmppath)

            result = run(skill.execute({
                "workspace": tmpdir,
                "use_llm": True,
            }))
            self.assertTrue(result.success)
            llm.complete.assert_called_once()
            # LLM enhancement should appear
            self.assertIn("Working on tests", result.output)

    def test_llm_failure_falls_back_to_template(self):
        """If LLM raises, skill still returns template output."""
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        skill = WorkspaceContextSkill(llm_client=llm)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            _make_git_repo(tmppath)

            result = run(skill.execute({
                "workspace": tmpdir,
                "use_llm": True,
            }))
            self.assertTrue(result.success)
            self.assertIn("Project Context", result.output)

    def test_project_metadata_pyproject(self):
        """Detects project metadata from pyproject.toml."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "pyproject.toml").write_text(
                '[project]\nname = "myapp"\nversion = "2.0"\n'
                'description = "My awesome app"\nrequires-python = ">=3.9"\n'
            )
            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("myapp", result.output)
            self.assertIn("My awesome app", result.output)
            self.assertIn("2.0", result.output)

    def test_project_metadata_package_json(self):
        """Detects project metadata from package.json."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "package.json").write_text(json.dumps({
                "name": "my-js-app",
                "version": "3.0.0",
                "description": "A JS project",
                "scripts": {"test": "jest"},
            }))
            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("my-js-app", result.output)
            self.assertIn("A JS project", result.output)

    def test_directory_structure_skips_noise(self):
        """Noise directories like __pycache__ are excluded."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "src").mkdir()
            (tmppath / "src" / "main.py").write_text("pass")
            (tmppath / "__pycache__").mkdir()
            (tmppath / "__pycache__" / "cache.pyc").write_bytes(b"")
            (tmppath / "node_modules").mkdir()

            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("src/", result.output)
            self.assertNotIn("__pycache__", result.output)
            self.assertNotIn("node_modules", result.output)

    def test_detect_conventions(self):
        """Detects code conventions from config files."""
        skill = WorkspaceContextSkill()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "pyproject.toml").write_text(
                '[project]\nname = "test"\n'
                '[tool.ruff]\nline-length = 100\n'
                '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
            )
            (tmppath / "tests").mkdir()
            (tmppath / "Makefile").write_text("all:\n\techo hello\n")

            result = run(skill.execute({"workspace": tmpdir}))
            self.assertTrue(result.success)
            self.assertIn("ruff", result.output)
            self.assertIn("100", result.output)
            self.assertIn("pytest", result.output)
            self.assertIn("make", result.output)

    def test_metadata(self):
        skill = WorkspaceContextSkill()
        self.assertEqual(skill.metadata.name, "workspace_context")
        self.assertEqual(skill.metadata.category, "workspace")

    def test_use_llm_false_skips_llm(self):
        """use_llm=False in context skips the LLM call."""
        llm = _mock_llm()
        skill = WorkspaceContextSkill(llm_client=llm)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(skill.execute({
                "workspace": tmpdir,
                "use_llm": False,
            }))
            self.assertTrue(result.success)
            llm.complete.assert_not_called()


# ===================================================================
# Activity Log Tests
# ===================================================================

class TestActivityLog(unittest.TestCase):

    def _make_store(self) -> SkillStore:
        tmpdir = tempfile.mkdtemp()
        return SkillStore(Path(tmpdir) / "test.db")

    def test_log_and_retrieve(self):
        store = self._make_store()
        store.log_activity(
            workspace="/tmp/test",
            action="request",
            query="hello world",
            skill_used="general_assistant",
            success=True,
        )
        entries = store.recent_activity(workspace="/tmp/test")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "request")
        self.assertEqual(entries[0]["query"], "hello world")
        self.assertEqual(entries[0]["skill_used"], "general_assistant")
        self.assertEqual(entries[0]["success"], 1)
        store.close()

    def test_filter_by_workspace(self):
        store = self._make_store()
        store.log_activity(workspace="/proj/a", action="request", query="q1")
        store.log_activity(workspace="/proj/b", action="request", query="q2")
        store.log_activity(workspace="/proj/a", action="request", query="q3")

        a_entries = store.recent_activity(workspace="/proj/a")
        self.assertEqual(len(a_entries), 2)

        b_entries = store.recent_activity(workspace="/proj/b")
        self.assertEqual(len(b_entries), 1)

        all_entries = store.recent_activity()
        self.assertEqual(len(all_entries), 3)
        store.close()

    def test_limit(self):
        store = self._make_store()
        for i in range(30):
            store.log_activity(workspace="/test", action="request", query=f"query {i}")

        entries = store.recent_activity(workspace="/test", limit=10)
        self.assertEqual(len(entries), 10)
        # Most recent first
        self.assertIn("query 29", entries[0]["query"])
        store.close()

    def test_query_truncation(self):
        store = self._make_store()
        long_query = "x" * 1000
        store.log_activity(workspace="/test", action="request", query=long_query)
        entries = store.recent_activity(workspace="/test")
        self.assertLessEqual(len(entries[0]["query"]), 500)
        store.close()

    def test_activity_in_context_skill(self):
        """WorkspaceContextSkill includes activity log in output."""
        store = self._make_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            store.log_activity(
                workspace=tmpdir,
                action="request",
                query="test query for context",
                skill_used="general",
                success=True,
            )

            skill = WorkspaceContextSkill()
            result = run(skill.execute({
                "workspace": tmpdir,
                "store": store,
            }))
            self.assertTrue(result.success)
            self.assertIn("EvolvAgent Activity", result.output)
            self.assertIn("test query for context", result.output)
        store.close()


# ===================================================================
# Integration: Agent activity logging
# ===================================================================

class TestAgentActivityLogging(unittest.TestCase):

    def test_handle_request_logs_activity(self):
        """Agent.handle_request() logs activity to the store."""
        from evolvagent.core.agent import Agent
        from evolvagent.core.skill import BaseSkill, SkillMetadata, SkillResult, TrustLevel

        class EchoSkill(BaseSkill):
            async def execute(self, context):
                return SkillResult(success=True, output="echoed")

        settings = _tmp_settings()
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            agent.workspace = "/test/workspace"
            skill = EchoSkill(SkillMetadata(
                name="echo", trigger_conditions=["test"],
                trust_level=TrustLevel.AUTO))
            agent.register_skill(skill)
            await agent.handle_request("test this")
            # Check activity was logged
            entries = agent._store.recent_activity(workspace="/test/workspace")
            await agent.shutdown()
            return entries

        entries = run(_test())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "request")
        self.assertEqual(entries[0]["skill_used"], "echo")
        self.assertEqual(entries[0]["success"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
