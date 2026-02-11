"""Tests for GitInfoSkill and GitActionSkill."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.llm import LLMResponse
from evolvagent.skills.git_ops import (
    GitActionSkill,
    GitInfoSkill,
    _detect_read_op,
    _detect_write_op,
)


def run(coro):
    return asyncio.run(coro)


def _make_mock_llm(content="Update files", cost=0.001):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        model="mock-model",
        tokens_in=10,
        tokens_out=20,
        cost_usd=cost,
    ))
    return llm


class TestDetectReadOp(unittest.TestCase):

    def test_status(self):
        self.assertEqual(_detect_read_op("git status"), "status")
        self.assertEqual(_detect_read_op("show me the status"), "status")

    def test_diff(self):
        self.assertEqual(_detect_read_op("git diff"), "diff")
        self.assertEqual(_detect_read_op("what changed"), "diff")

    def test_diff_staged(self):
        self.assertEqual(_detect_read_op("show staged diff"), "diff_staged")

    def test_log(self):
        self.assertEqual(_detect_read_op("git log"), "log")
        self.assertEqual(_detect_read_op("show recent commits"), "log")
        self.assertEqual(_detect_read_op("history"), "log")

    def test_branch(self):
        self.assertEqual(_detect_read_op("git branch"), "branch")
        self.assertEqual(_detect_read_op("list branches"), "branch")

    def test_no_match(self):
        self.assertIsNone(_detect_read_op("what is python?"))


class TestDetectWriteOp(unittest.TestCase):

    def test_commit(self):
        self.assertEqual(_detect_write_op("commit changes"), "commit")
        self.assertEqual(_detect_write_op("git commit"), "commit")

    def test_stash(self):
        self.assertEqual(_detect_write_op("git stash"), "stash")

    def test_stash_pop(self):
        self.assertEqual(_detect_write_op("stash pop"), "stash_pop")
        self.assertEqual(_detect_write_op("unstash"), "stash_pop")

    def test_no_match(self):
        self.assertIsNone(_detect_write_op("what is python?"))


class TestGitInfoSkillCanHandle(unittest.TestCase):

    def setUp(self):
        self.skill = GitInfoSkill()

    def test_strong_match(self):
        self.assertGreaterEqual(self.skill.can_handle("git status"), 0.9)
        self.assertGreaterEqual(self.skill.can_handle("git diff"), 0.9)
        self.assertGreaterEqual(self.skill.can_handle("git log"), 0.9)

    def test_weaker_match(self):
        score = self.skill.can_handle("show recent commits")
        self.assertGreaterEqual(score, 0.8)

    def test_no_match(self):
        self.assertEqual(self.skill.can_handle("what is python?"), 0.0)

    def test_metadata(self):
        self.assertEqual(self.skill.metadata.name, "git_info")
        self.assertEqual(self.skill.metadata.trust_level.value, "auto")


class TestGitInfoSkillExecute(unittest.TestCase):
    """Execute tests — require an actual git repo."""

    def setUp(self):
        self.skill = GitInfoSkill()
        # Create a temporary git repo
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"],
                       cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=self.tmpdir, capture_output=True)
        # Create a file and commit
        (Path(self.tmpdir) / "test.txt").write_text("hello\n")
        subprocess.run(["git", "add", "."], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.tmpdir, capture_output=True)
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_status(self):
        result = run(self.skill.execute({"user_input": "git status"}))
        self.assertTrue(result.success)
        # Clean repo — might be empty output
        self.assertIn("command", result.data)

    def test_log(self):
        result = run(self.skill.execute({"user_input": "git log"}))
        self.assertTrue(result.success)
        self.assertIn("init", result.output)

    def test_branch(self):
        result = run(self.skill.execute({"user_input": "git branch"}))
        self.assertTrue(result.success)

    def test_empty_input(self):
        result = run(self.skill.execute({"user_input": ""}))
        self.assertFalse(result.success)

    def test_unknown_op(self):
        result = run(self.skill.execute({"user_input": "something unrelated"}))
        self.assertFalse(result.success)


class TestGitActionSkillCanHandle(unittest.TestCase):

    def setUp(self):
        self.skill = GitActionSkill()

    def test_commit_match(self):
        self.assertGreaterEqual(self.skill.can_handle("commit changes"), 0.8)
        self.assertGreaterEqual(self.skill.can_handle("git commit"), 0.9)

    def test_stash_match(self):
        self.assertGreaterEqual(self.skill.can_handle("git stash"), 0.8)

    def test_no_match(self):
        self.assertEqual(self.skill.can_handle("what is python?"), 0.0)

    def test_metadata(self):
        self.assertEqual(self.skill.metadata.name, "git_action")
        self.assertEqual(self.skill.metadata.trust_level.value, "suggest")


class TestGitActionSkillPreview(unittest.TestCase):

    def setUp(self):
        self.skill = GitActionSkill(_make_mock_llm("fix: update config"))
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"],
                       cwd=self.tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"],
                       cwd=self.tmpdir, capture_output=True)
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)

    def test_commit_preview(self):
        (Path(self.tmpdir) / "test.txt").write_text("hello\n")
        preview = run(self.skill.preview({"user_input": "commit changes"}))
        self.assertIn("commit", preview.lower())

    def test_stash_preview(self):
        preview = run(self.skill.preview({"user_input": "git stash"}))
        self.assertIn("stash", preview.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
