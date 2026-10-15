"""
Utilities for code parsing, diffing, and manipulation
"""

import ast
import re
import tokenize
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union


_ProgramT = TypeVar("_ProgramT")


@lru_cache(maxsize=4096)
def code_without_comments_and_docstrings(source: str) -> str:
    """Return Python source with commentary units removed.

    This is an exact-source comparison key, not a normalized or semantic
    representation. Comments and genuine leading docstring statements are
    removed directly from the original string; every other character is
    retained character-for-character. If ``source`` cannot be parsed as
    Python, it is returned unchanged. The bounded cache avoids reparsing the
    same program during pairwise diversity checks.

    Commentary units include the whole physical line for a comment-only or
    docstring-only line. For inline comments, the horizontal whitespace that
    separates the comment from surviving code is also removed. A same-line
    docstring followed by another statement consumes its following semicolon
    and adjacent horizontal whitespace.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, UnicodeError):
        return source

    # Unlike ``str.splitlines()``, this recognizes only the universal-newline
    # sequences that Python treats as physical line endings. In particular, a
    # form feed is horizontal indentation, not a new source line.
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z", source)
    try:
        line_iterator = iter(lines)
        tokens = list(tokenize.generate_tokens(lambda: next(line_iterator, "")))
    except (IndentationError, SyntaxError, tokenize.TokenError, UnicodeError):
        # Parsing has already succeeded, so this is only a defensive fallback:
        # never partially strip source when Python-aware analysis is incomplete.
        return source

    line_starts: List[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line)

    def line_bounds(line_number: int) -> Tuple[int, int, int]:
        """Return a physical line's start, content end, and full end offsets."""
        line = lines[line_number - 1]
        start = line_starts[line_number - 1]
        full_end = start + len(line)
        if line.endswith("\r\n"):
            content_end = full_end - 2
        elif line.endswith(("\r", "\n")):
            content_end = full_end - 1
        else:
            content_end = full_end
        return start, content_end, full_end

    def token_offset(position: Tuple[int, int]) -> int:
        line_number, column = position
        return line_starts[line_number - 1] + column

    def ast_offset(line_number: int, byte_column: int) -> int:
        # AST columns are UTF-8 byte offsets; tokenizer columns and Python
        # string slicing use Unicode character offsets.
        line = lines[line_number - 1]
        character_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        return line_starts[line_number - 1] + character_column

    removals: List[Tuple[int, int]] = []

    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        start = token_offset(token.start)
        end = token_offset(token.end)
        line_start, _, line_end = line_bounds(token.start[0])
        before_comment = source[line_start:start]
        if not before_comment.strip(" \t\f"):
            removals.append((line_start, line_end))
            continue

        # The separating horizontal whitespace is part of an inline comment's
        # commentary unit. Do not touch any other surviving whitespace.
        while start > line_start and source[start - 1] in " \t\f":
            start -= 1
        removals.append((start, end))

    docstring_nodes: List[ast.Expr] = []
    containers = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, containers) or not node.body:
            continue
        first_statement = node.body[0]
        if (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            docstring_nodes.append(first_statement)

    for node in docstring_nodes:
        if node.end_lineno is None or node.end_col_offset is None:
            # All supported Python versions provide end positions. Falling back
            # to raw source is safer than constructing a partially stripped key.
            return source

        start = ast_offset(node.lineno, node.col_offset)
        end = ast_offset(node.end_lineno, node.end_col_offset)
        start_line, _, _ = line_bounds(node.lineno)
        _, end_line_content, end_line = line_bounds(node.end_lineno)
        prefix = source[start_line:start]
        suffix = source[end:end_line_content]

        # A statement that occupies its physical line is removed together with
        # indentation and newline. A trailing semicolon and/or comment is still
        # part of that otherwise commentary-only line.
        suffix_is_commentary_only = re.fullmatch(
            r"[ \t\f]*(?:;[ \t\f]*)?(?:\#.*)?", suffix
        )
        if not prefix.strip(" \t\f") and suffix_is_commentary_only:
            removals.append((start_line, end_line))
            continue

        # For ``def f(): "doc"; return 1``, consume the right-hand separator
        # and its whitespace, leaving ``def f(): return 1`` without formatting
        # or otherwise changing the surrounding executable source.
        separator = re.match(r"[ \t\f]*;[ \t\f]*", suffix)
        if separator:
            end += separator.end()
        removals.append((start, end))

    if not removals:
        return source

    merged: List[List[int]] = []
    for start, end in sorted(removals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    parts: List[str] = []
    previous_end = 0
    for start, end in merged:
        parts.append(source[previous_end:start])
        previous_end = end
    parts.append(source[previous_end:])
    return "".join(parts)


def select_distinct_code_programs(
    ranked_programs: Sequence[_ProgramT],
    limit: int,
    code_getter: Callable[[_ProgramT], str],
) -> Tuple[List[_ProgramT], List[_ProgramT]]:
    """Select the first ``limit`` commentary-distinct Python programs.

    The input is assumed to already be fitness-ranked. This helper never sorts
    or considers IDs, lineage, metrics, or metadata; it only preserves the
    input order while skipping sources that differ only in Python comments or
    genuine docstrings. Sources that fail Python parsing retain raw equality.

    Returns:
        A pair containing the selected programs and duplicate-code programs
        skipped while filling the requested slots.
    """
    selected: List[_ProgramT] = []
    skipped_duplicates: List[_ProgramT] = []
    seen_code_keys: Set[str] = set()
    requested_count = max(0, int(limit))

    for program in ranked_programs:
        if len(selected) >= requested_count:
            break
        comparison_code = code_without_comments_and_docstrings(code_getter(program))
        if comparison_code in seen_code_keys:
            skipped_duplicates.append(program)
            continue
        seen_code_keys.add(comparison_code)
        selected.append(program)

    return selected, skipped_duplicates


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text, diff_pattern)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(
    diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def _format_block_lines(lines: List[str], max_line_len: int = 100, max_lines: int = 30) -> str:
    """Format a block of lines for diff summary: show all lines (truncated per line, optional cap)."""
    truncated = []
    for line in lines[:max_lines]:
        s = line.rstrip()
        if len(s) > max_line_len:
            s = s[: max_line_len - 3] + "..."
        truncated.append("  " + s)
    if len(lines) > max_lines:
        truncated.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(truncated) if truncated else "  (empty)"


def format_diff_summary(
    diff_blocks: List[Tuple[str, str]],
    max_line_len: int = 100,
    max_lines: int = 30,
) -> str:
    """
    Create a human-readable summary of the diff.
    For multi-line blocks, shows the full search and replace content (all lines).

    Args:
        diff_blocks: List of (search_text, replace_text) tuples
        max_line_len: Maximum characters per line before truncation (default: 100)
        max_lines: Maximum lines per SEARCH/REPLACE block (default: 30)

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_block = _format_block_lines(search_lines, max_line_len, max_lines)
            replace_block = _format_block_lines(replace_lines, max_line_len, max_lines)
            summary.append(f"Change {i+1}: Replace:\n{search_block}\nwith:\n{replace_block}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"


def _can_apply_linewise(haystack_lines: List[str], needle_lines: List[str]) -> bool:
    if not needle_lines:
        return False

    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        if haystack_lines[i : i + len(needle_lines)] == needle_lines:
            return True

    return False


def apply_diff_blocks(original_text: str, diff_blocks: List[Tuple[str, str]]) -> Tuple[str, int]:
    """
    Apply diff blocks line-wise and return (new_text, applied_count)
    """
    lines = original_text.split("\n")
    applied = 0

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        for i in range(len(lines) - len(search_lines) + 1):
            if lines[i : i + len(search_lines)] == search_lines:
                lines[i : i + len(search_lines)] = replace_lines
                applied += 1
                break

    return "\n".join(lines), applied


def split_diffs_by_target(
    diff_blocks: List[Tuple[str, str]],
    *,
    code_text: str,
    changes_description_text: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Route diff blocks to either code or changes_description based on exact line-wise match
    of SEARCH text. Returns (code_blocks, changes_desc_blocks, unmatched_blocks)

    If a SEARCH matches both targets, it's ambiguous and we raise error
    """
    code_lines = code_text.split("\n")
    desc_lines = changes_description_text.split("\n")

    code_blocks: List[Tuple[str, str]] = []
    desc_blocks: List[Tuple[str, str]] = []
    unmatched: List[Tuple[str, str]] = []

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")

        matches_code = _can_apply_linewise(code_lines, search_lines)
        matches_desc = _can_apply_linewise(desc_lines, search_lines)

        if matches_code and matches_desc:
            raise ValueError(
                "Ambiguous diff block: SEARCH matches both code and changes_description"
            )
        if matches_code:
            code_blocks.append((search_text, replace_text))
        elif matches_desc:
            desc_blocks.append((search_text, replace_text))
        else:
            unmatched.append((search_text, replace_text))

    return code_blocks, desc_blocks, unmatched
