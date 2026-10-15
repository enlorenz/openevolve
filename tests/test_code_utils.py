"""
Tests for code utilities in openevolve.utils.code_utils
"""

import unittest

from openevolve.utils.code_utils import (
    _format_block_lines,
    apply_diff,
    code_without_comments_and_docstrings,
    extract_diffs,
    format_diff_summary,
    select_distinct_code_programs,
)


class TestCodeWithoutCommentsAndDocstrings(unittest.TestCase):
    """Tests for the commentary-insensitive exact-source comparison key."""

    def test_inline_comment_content_and_presence_are_ignored(self):
        first = "x = 1  # explanation A\n"
        second = "x = 1\t# completely different explanation\n"
        absent = "x = 1\n"

        self.assertEqual(code_without_comments_and_docstrings(first), absent)
        self.assertEqual(code_without_comments_and_docstrings(second), absent)

    def test_comment_only_line_content_and_presence_are_ignored(self):
        first = "# explanation A\nx = 1\n"
        second = "  # completely different explanation\nx = 1\n"
        absent = "x = 1\n"

        self.assertEqual(code_without_comments_and_docstrings(first), absent)
        self.assertEqual(code_without_comments_and_docstrings(second), absent)

    def test_hash_characters_inside_strings_are_preserved(self):
        first = 'x = "# foo"\n'
        second = 'x = "# bar"\n'

        self.assertEqual(code_without_comments_and_docstrings(first), first)
        self.assertEqual(code_without_comments_and_docstrings(second), second)
        self.assertNotEqual(
            code_without_comments_and_docstrings(first),
            code_without_comments_and_docstrings(second),
        )

    def test_function_docstring_content_and_presence_are_ignored(self):
        short_docstring = (
            "def policy(state):\n"
            '    """Strategy A."""\n'
            "    return (1, 0, 0)\n"
        )
        multiline_docstring = (
            "def policy(state):\n"
            '    """A completely different\n'
            "    multi-line strategy description.\n"
            '    """\n'
            "    return (1, 0, 0)\n"
        )
        absent = "def policy(state):\n    return (1, 0, 0)\n"

        self.assertEqual(
            code_without_comments_and_docstrings(short_docstring), absent
        )
        self.assertEqual(
            code_without_comments_and_docstrings(multiline_docstring), absent
        )

    def test_module_class_and_async_function_docstrings_are_removed(self):
        source = (
            '"""Module documentation."""\n'
            "class Policy:\n"
            '    """Class documentation."""\n'
            "    async def choose(self):\n"
            '        """Method documentation."""\n'
            "        return 1\n"
        )
        expected = (
            "class Policy:\n"
            "    async def choose(self):\n"
            "        return 1\n"
        )

        self.assertEqual(code_without_comments_and_docstrings(source), expected)

    def test_nonleading_string_expression_is_preserved(self):
        source = (
            "def policy():\n"
            "    marker = 1\n"
            '    "not a docstring"\n'
            "    return marker\n"
        )

        self.assertEqual(code_without_comments_and_docstrings(source), source)

    def test_executable_string_values_remain_distinct(self):
        cheap = 'def policy():\n    mode = "cheap"\n    return mode\n'
        expensive = 'def policy():\n    mode = "expensive"\n    return mode\n'

        self.assertNotEqual(
            code_without_comments_and_docstrings(cheap),
            code_without_comments_and_docstrings(expensive),
        )

    def test_assigned_triple_quoted_strings_remain_distinct(self):
        cheap = 'def policy():\n    mode = """cheap"""\n    return mode\n'
        expensive = 'def policy():\n    mode = """expensive"""\n    return mode\n'

        self.assertEqual(code_without_comments_and_docstrings(cheap), cheap)
        self.assertEqual(code_without_comments_and_docstrings(expensive), expensive)
        self.assertNotEqual(
            code_without_comments_and_docstrings(cheap),
            code_without_comments_and_docstrings(expensive),
        )

    def test_executable_changes_remain_distinct(self):
        first = "def policy():\n    return (1, 0, 0)\n"
        second = "def policy():\n    return (2, 0, 0)\n"

        self.assertNotEqual(
            code_without_comments_and_docstrings(first),
            code_without_comments_and_docstrings(second),
        )

    def test_surviving_formatting_is_not_normalized(self):
        compact = "x=1\n"
        spaced = "x = 1\n"

        self.assertEqual(code_without_comments_and_docstrings(compact), compact)
        self.assertEqual(code_without_comments_and_docstrings(spaced), spaced)
        self.assertNotEqual(
            code_without_comments_and_docstrings(compact),
            code_without_comments_and_docstrings(spaced),
        )

    def test_same_line_docstring_removes_following_statement_separator(self):
        with_docstring = 'def policy(): "strategy"; return 1\n'
        absent = "def policy(): return 1\n"

        self.assertEqual(
            code_without_comments_and_docstrings(with_docstring), absent
        )

    def test_commentary_removal_preserves_crlf_in_surviving_source(self):
        with_commentary = (
            "def policy():\r\n"
            '    """Strategy."""\r\n'
            "    value = 1  # explanation\r\n"
            "    return value\r\n"
        )
        absent = (
            "def policy():\r\n"
            "    value = 1\r\n"
            "    return value\r\n"
        )

        self.assertEqual(
            code_without_comments_and_docstrings(with_commentary), absent
        )

    def test_invalid_python_is_returned_raw(self):
        first = "int main() { // comment A\n  return 0;\n}\n"
        second = "int main() { // comment B\n  return 0;\n}\n"

        self.assertEqual(code_without_comments_and_docstrings(first), first)
        self.assertEqual(code_without_comments_and_docstrings(second), second)
        self.assertNotEqual(
            code_without_comments_and_docstrings(first),
            code_without_comments_and_docstrings(second),
        )

    def test_unicode_before_same_line_docstring_uses_correct_ast_offsets(self):
        with_docstring = 'def café(): "dóc"; return "é"\n'
        absent = 'def café(): return "é"\n'

        self.assertEqual(
            code_without_comments_and_docstrings(with_docstring), absent
        )

    def test_distinct_program_selection_uses_commentary_stripped_key(self):
        programs = [
            ("rank-1", "x = 1  # first\n"),
            ("rank-2", "x = 1  # second\n"),
            ("rank-3", "x = 2\n"),
        ]

        selected, skipped = select_distinct_code_programs(
            programs, 2, lambda program: program[1]
        )

        self.assertEqual([program[0] for program in selected], ["rank-1", "rank-3"])
        self.assertEqual([program[0] for program in skipped], ["rank-2"])


class TestCodeUtils(unittest.TestCase):
    """Tests for code utilities"""

    def test_extract_diffs(self):
        """Test extracting diffs from a response"""
        diff_text = """
        Let's improve this code:

        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        Another change:

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        diffs = extract_diffs(diff_text)
        self.assertEqual(len(diffs), 2)
        self.assertEqual(
            diffs[0][0],
            """        def hello():
            print(\"Hello\")""",
        )
        self.assertEqual(
            diffs[0][1],
            """        def hello():
            print(\"Hello, World!\")""",
        )
        self.assertEqual(diffs[1][0], "        x = 1")
        self.assertEqual(diffs[1][1], "        x = 2")

    def test_apply_diff(self):
        """Test applying diffs to code"""
        original_code = """
        def hello():
            print("Hello")

        x = 1
        y = 2
        """

        diff_text = """
        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        expected_code = """
        def hello():
            print("Hello, World!")

        x = 2
        y = 2
        """

        result = apply_diff(original_code, diff_text)

        # Normalize whitespace for comparison
        self.assertEqual(
            result,
            expected_code,
        )


class TestFormatDiffSummary(unittest.TestCase):
    """Tests for format_diff_summary showing actual diff content"""

    def test_single_line_changes(self):
        """Single-line changes should show inline format"""
        diff_blocks = [("x = 1", "x = 2")]
        result = format_diff_summary(diff_blocks)
        self.assertEqual(result, "Change 1: 'x = 1' to 'x = 2'")

    def test_multi_line_changes_show_actual_content(self):
        """Multi-line changes should show actual SEARCH/REPLACE content"""
        diff_blocks = [
            (
                "def old():\n    return False",
                "def new():\n    return True",
            )
        ]
        result = format_diff_summary(diff_blocks)
        # Should contain actual code, not "2 lines"
        self.assertIn("def old():", result)
        self.assertIn("return False", result)
        self.assertIn("def new():", result)
        self.assertIn("return True", result)
        self.assertIn("Replace:", result)
        self.assertIn("with:", result)
        # Should NOT contain generic line count
        self.assertNotIn("2 lines", result)

    def test_multiple_diff_blocks(self):
        """Multiple diff blocks should be numbered"""
        diff_blocks = [
            ("a = 1", "a = 2"),
            ("def foo():\n    pass", "def bar():\n    return 1"),
        ]
        result = format_diff_summary(diff_blocks)
        self.assertIn("Change 1:", result)
        self.assertIn("Change 2:", result)
        self.assertIn("'a = 1' to 'a = 2'", result)
        self.assertIn("def foo():", result)
        self.assertIn("def bar():", result)

    def test_configurable_max_line_len(self):
        """max_line_len parameter should control line truncation"""
        long_line = "x" * 50
        # Must be multi-line to trigger block format (single-line uses inline format)
        diff_blocks = [(long_line + "\nline2", "short\nline2")]
        # With default (100), no truncation
        result_default = format_diff_summary(diff_blocks)
        self.assertNotIn("...", result_default)
        # With max_line_len=30, should truncate the long line
        result_short = format_diff_summary(diff_blocks, max_line_len=30)
        self.assertIn("...", result_short)

    def test_configurable_max_lines(self):
        """max_lines parameter should control block truncation"""
        many_lines = "\n".join([f"line{i}" for i in range(20)])
        diff_blocks = [(many_lines, "replacement")]
        # With max_lines=10, should truncate
        result = format_diff_summary(diff_blocks, max_lines=10)
        self.assertIn("... (10 more lines)", result)

    def test_block_lines_basic_formatting(self):
        """Lines should be indented with 2 spaces"""
        lines = ["line1", "line2"]
        result = _format_block_lines(lines)
        self.assertEqual(result, "  line1\n  line2")

    def test_block_lines_long_line_truncation(self):
        """Lines over 100 chars should be truncated by default"""
        long_line = "x" * 150
        result = _format_block_lines([long_line])
        self.assertIn("...", result)
        self.assertLess(len(result.split("\n")[0]), 110)

    def test_block_lines_many_lines_truncation(self):
        """More than 30 lines should show truncation message by default"""
        lines = [f"line{i}" for i in range(50)]
        result = _format_block_lines(lines)
        self.assertIn("... (20 more lines)", result)
        self.assertEqual(len(result.split("\n")), 31)

    def test_block_lines_empty_input(self):
        """Empty input should return '(empty)'"""
        result = _format_block_lines([])
        self.assertEqual(result, "  (empty)")


if __name__ == "__main__":
    unittest.main()
