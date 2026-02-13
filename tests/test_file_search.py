"""Tests for FileSearchSkill."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.skills.file_search import (
    FileSearchSkill,
    _detect_search_type,
    _extract_file_types,
    _extract_pattern,
)


def run(coro):
    return asyncio.run(coro)


class TestSearchTypeDetection(unittest.TestCase):

    def test_content_search_keywords(self):
        self.assertEqual(_detect_search_type("grep for config"), "content")
        self.assertEqual(_detect_search_type("search for TODO in files"), "content")
        self.assertEqual(_detect_search_type("search in python files"), "content")

    def test_filename_search_keywords(self):
        self.assertEqual(_detect_search_type("find file config.py"), "filename")
        self.assertEqual(_detect_search_type("find all test files"), "filename")

    def test_glob_pattern_is_filename(self):
        self.assertEqual(_detect_search_type("find *.py files"), "filename")

    def test_default_is_content(self):
        self.assertEqual(_detect_search_type("where is the database"), "content")


class TestPatternExtraction(unittest.TestCase):

    def test_quoted_pattern(self):
        self.assertEqual(_extract_pattern('search for "hello world"'), "hello world")
        self.assertEqual(_extract_pattern("grep for 'TODO'"), "TODO")

    def test_prefix_stripping(self):
        self.assertEqual(_extract_pattern("search for config"), "config")
        self.assertEqual(_extract_pattern("grep TODO"), "TODO")
        self.assertEqual(_extract_pattern("find all test"), "test")

    def test_in_clause(self):
        self.assertEqual(_extract_pattern("search for config in python files"), "config")


class TestFileTypeExtraction(unittest.TestCase):

    def test_python_detection(self):
        types = _extract_file_types("search in python files")
        self.assertIn("*.py", types)

    def test_javascript_detection(self):
        types = _extract_file_types("grep in javascript files")
        self.assertIn("*.js", types)

    def test_glob_extraction(self):
        types = _extract_file_types("search *.rs files")
        self.assertIn("*.rs", types)

    def test_no_types(self):
        types = _extract_file_types("search for config")
        self.assertEqual(types, [])


class TestFileSearchSkillCanHandle(unittest.TestCase):

    def setUp(self):
        self.skill = FileSearchSkill()

    def test_strong_match(self):
        self.assertGreaterEqual(self.skill.can_handle("search for config"), 0.9)
        self.assertGreaterEqual(self.skill.can_handle("grep TODO"), 0.9)

    def test_weak_match(self):
        score = self.skill.can_handle("find something")
        self.assertGreaterEqual(score, 0.7)
        self.assertLess(score, 0.9)

    def test_no_match(self):
        self.assertEqual(self.skill.can_handle("what is python?"), 0.0)

    def test_metadata(self):
        self.assertEqual(self.skill.metadata.name, "file_search")
        self.assertEqual(self.skill.metadata.trust_level.value, "auto")


class TestFileSearchSkillExecute(unittest.TestCase):

    def setUp(self):
        self.skill = FileSearchSkill()
        # Create a temp directory with test files
        self.tmpdir = tempfile.mkdtemp()
        (Path(self.tmpdir) / "foo.py").write_text("# config value\nprint('hello')\n")
        (Path(self.tmpdir) / "bar.py").write_text("# nothing here\n")
        subdir = Path(self.tmpdir) / "sub"
        subdir.mkdir()
        (subdir / "baz.txt").write_text("config = true\n")

    def test_content_search(self):
        result = run(self.skill.execute({
            "user_input": "search for config in python files",
            "workspace": self.tmpdir,
        }))
        self.assertTrue(result.success)
        self.assertIn("config", result.output)
        self.assertIn("foo.py", result.output)

    def test_filename_search(self):
        result = run(self.skill.execute({
            "user_input": "find file baz",
            "workspace": self.tmpdir,
        }))
        self.assertTrue(result.success)
        self.assertIn("baz", result.output)

    def test_no_results(self):
        result = run(self.skill.execute({
            "user_input": "search for zzz_nonexistent_pattern_zzz",
            "workspace": self.tmpdir,
        }))
        self.assertTrue(result.success)
        self.assertIn("No results", result.output)

    def test_empty_input(self):
        result = run(self.skill.execute({"user_input": ""}))
        self.assertFalse(result.success)

    def test_preview(self):
        preview = run(self.skill.preview({
            "user_input": "search for config in python files",
            "workspace": self.tmpdir,
        }))
        self.assertIn("config", preview)


if __name__ == "__main__":
    unittest.main(verbosity=2)
