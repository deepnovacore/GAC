"""Normalize legacy and current code completions before execution scoring.

The public evaluator historically asked for a function body while the
BigCode executor expects a runnable program.  This module makes that boundary
explicit and supports both formats without executing model output.
"""

from __future__ import annotations

import ast
import copy
import re
import textwrap
from typing import Any


PY_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def clean_completion(completion: str) -> str:
    """Remove an optional markdown fence without executing the completion."""
    match = PY_BLOCK_RE.search(completion)
    if match:
        return match.group(1).strip()
    return completion.strip()


def _find_function(source: str, name: str | None = None) -> ast.AST:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if name is None or node.name == name:
                return node
    raise ValueError(f"Could not find function {name!r}")


def _has_function(source: str, name: str) -> bool:
    try:
        _find_function(source, name)
    except (SyntaxError, ValueError):
        return False
    return True


def _body_block(candidate: str) -> str:
    body = textwrap.dedent(candidate).strip("\n")
    if not body:
        body = "pass"
    block = textwrap.indent(body, "    ")
    try:
        ast.parse("def __gac_protocol_probe__():\n" + block)
        return block
    except IndentationError:
        # Some legacy body-only generations keep the indentation from the
        # canonical function body on every line except the first one, e.g.
        # ``x = ...`` followed by four-space-indented assignments.  If the
        # first statement is not a compound statement, repair that specific
        # malformed baseline while leaving valid nested blocks untouched.
        lines = body.splitlines()
        first = next((line for line in lines if line.strip()), "")
        if first and not first[:1].isspace() and not first.rstrip().endswith(":"):
            repaired = [lines[0]]
            repaired.extend(
                line[4:] if line.startswith("    ") else line
                for line in lines[1:]
            )
            repaired_body = "\n".join(repaired)
            repaired_block = textwrap.indent(repaired_body, "    ")
            try:
                ast.parse("def __gac_protocol_probe__():\n" + repaired_block)
            except (IndentationError, SyntaxError):
                # Invalid candidates are intentionally left for the isolated scorer.
                # They must count as failed programs, not abort the whole batch.
                return block
            return repaired_block
        raise
    except SyntaxError:
        # Natural-language or otherwise malformed generations are invalid
        # code; keep them unchanged so the sandbox records a failed test.
        return block


def _mbpp_tested_functions(item: dict[str, Any]) -> list[str]:
    """Resolve entrypoints from actual tests, not the first reference helper.

    Only names/signatures are used. Never inject a reference function body.
    """
    reference = ast.parse(item['code'])
    definitions = {node.name for node in reference.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    tests = item.get('test_list')
    if tests is None:
        tests = [item['test']]
    names = []
    for test in tests:
        for node in ast.walk(ast.parse(test)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in definitions and node.func.id not in names):
                names.append(node.func.id)
    if not names:
        raise ValueError('MBPP tests do not identify a reference entrypoint')
    return names


def _mbpp_header(reference_code: str, entry_point: str) -> str:
    """Extract only the tested function signature from MBPP reference code.

    The reference implementation is used for its signature only; its body is
    never included in the candidate program.
    """
    function = _find_function(reference_code, entry_point)
    stub = copy.deepcopy(function)
    stub.decorator_list = []
    stub.body = [ast.Pass()]
    stub.type_comment = None
    ast.fix_missing_locations(stub)
    rendered = ast.unparse(stub).strip()
    marker = "\n    pass"
    if not rendered.endswith(marker):
        raise ValueError("Could not isolate MBPP function signature")
    return rendered[: -len(marker)]


def _humaneval_program_from_body(
    raw_prompt: str, entry_point: str, body: str
) -> str:
    function = _find_function(raw_prompt, entry_point)
    lines = raw_prompt.splitlines()
    body_start = function.body[0].lineno - 1
    body_end = function.end_lineno
    prefix = "\n".join(lines[:body_start]).rstrip()
    suffix = "\n".join(lines[body_end:]).strip()
    program = prefix + "\n" + _body_block(body)
    if suffix:
        program += "\n" + suffix
    return program + "\n"


def _humaneval_program_from_function(
    raw_prompt: str, entry_point: str, candidate: str
) -> str:
    function = _find_function(raw_prompt, entry_point)
    lines = raw_prompt.splitlines()
    target_start = function.lineno - 1
    target_end = function.end_lineno
    prefix = "\n".join(lines[:target_start]).rstrip()
    suffix = "\n".join(lines[target_end:]).strip()
    program = candidate.strip()
    if prefix:
        program = prefix + "\n\n" + program
    if suffix:
        program += "\n" + suffix
    return program + "\n"


def normalize_candidate(
    benchmark: str, completion: str, item: dict[str, Any]
) -> tuple[str, str]:
    """Return a runnable candidate and the normalization mode used.

    ``body_wrapped`` and ``body_inserted`` are for the old evaluator outputs;
    new generations should normally be ``full_function`` or
    ``full_function_with_context``.
    """
    candidate = clean_completion(completion)

    if benchmark == "mbpp":
        entries = _mbpp_tested_functions(item)
        if all(_has_function(candidate, entry) for entry in entries):
            return candidate + "\n", "full_function"
        if len(entries) != 1:
            raise ValueError('Cannot wrap a body-only MBPP completion with multiple tested entrypoints')
        header = _mbpp_header(item["code"], entries[0])
        return header + "\n" + _body_block(candidate) + "\n", "body_wrapped"

    if benchmark == "humaneval":
        entry_point = item["entry_point"]
        if _has_function(candidate, entry_point):
            return (
                _humaneval_program_from_function(
                    item["raw_prompt"], entry_point, candidate
                ),
                "full_function_with_context",
            )
        return (
            _humaneval_program_from_body(
                item["raw_prompt"], entry_point, candidate
            ),
            "body_inserted",
        )

    raise ValueError(f"Unsupported code benchmark: {benchmark}")
