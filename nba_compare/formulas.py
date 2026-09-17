"""
Lets you define custom stats as formulas over existing ones, e.g.
"PTS/G / USG Vol/G" for "points per used possession" -- using the EXACT
same stat labels shown in the comparison table (table.STAT_DEFS), not a
separate internal variable-naming scheme. That includes labels with
characters that aren't valid in a bare identifier, like "MIN/G", "TS%",
"+/-", or "MIN Floor (P10)".

This is NOT Python's eval() -- known stat labels are swapped for safe
placeholder identifiers BEFORE parsing (so "MIN/G" is treated as one
atomic value, not read as MIN divided by G), then the substituted
expression is walked as an AST that only allows numbers, those
placeholders, and +, -, *, /, ** (and parentheses, which the parser
handles automatically). No function calls, no attribute access, no
indexing, nothing else. A formula like "__import__('os')" is a syntax the
walker doesn't recognize and rejects, not something it executes.
"""
from __future__ import annotations
import ast
import operator
import re

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class FormulaError(ValueError):
    pass


def _eval_node(node, variables: dict):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise FormulaError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise FormulaError(f"Unknown stat name: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if left is None or right is None:
            return None
        try:
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        except ZeroDivisionError:
            return None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        val = _eval_node(node.operand, variables)
        return None if val is None else _ALLOWED_UNARY[type(node.op)](val)
    raise FormulaError(f"Unsupported expression near: {ast.dump(node)}")


def _tokenize_stat_labels(expr: str, variables: dict) -> tuple[str, dict]:
    """
    Stat labels (the keys of `variables`) can contain characters that
    aren't valid in a bare Python identifier -- '/', '%', '+', '-',
    spaces, parentheses (e.g. "MIN/G", "TS%", "+/-", "MIN Floor (P10)").
    Swaps every occurrence of a known label for a safe placeholder
    identifier BEFORE handing the expression to ast.parse, longest labels
    first so e.g. "TS% %ile" isn't chopped up by a "TS%" match winning
    first at the same starting position. Returns the substituted
    expression plus a {placeholder: value} dict for _eval_node.
    """
    labels = sorted(variables.keys(), key=len, reverse=True)
    if not labels:
        return expr, {}
    pattern = re.compile("|".join(re.escape(label) for label in labels))
    placeholder_values: dict[str, object] = {}

    def _replace(match: re.Match) -> str:
        placeholder = f"__v{len(placeholder_values)}__"
        placeholder_values[placeholder] = variables[match.group(0)]
        return placeholder

    return pattern.sub(_replace, expr), placeholder_values


def safe_eval(expr: str, variables: dict):
    """
    Evaluates expr using only the given variables, keyed by stat label
    (e.g. "MIN/G", "TS%", "PTS %ile") exactly as shown in the comparison
    table. Raises FormulaError on anything outside +,-,*,/,**,
    parentheses, numbers, and known stat labels.
    """
    substituted, placeholder_values = _tokenize_stat_labels(expr, variables)
    try:
        tree = ast.parse(substituted, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Invalid formula syntax: {e}")
    return _eval_node(tree, placeholder_values)


def validate_formula(expr: str, sample_variables: dict) -> str | None:
    """Returns an error message, or None if the formula is valid against a sample namespace."""
    if not expr or not expr.strip():
        return "Formula is empty."
    try:
        safe_eval(expr, sample_variables)
        return None
    except FormulaError as e:
        return str(e)


def flatten_block_for_formula(block: dict | None, stat_defs: dict) -> dict:
    """
    {label: value} for every stat in stat_defs, evaluated against block --
    the exact labels table.py shows in the comparison table, so a custom
    formula can reference any built-in stat exactly as it's displayed
    there (no separate internal naming scheme to learn, and any stat
    added to STAT_DEFS becomes formula-usable automatically). A getter
    that raises (missing data for this block) contributes None rather
    than failing the whole formula -- same as table.py's build_stat_table.

    stat_defs is deliberately the caller's BUILT-IN stat_defs (e.g.
    table.STAT_DEFS), not a dict that also includes other custom
    formulas -- custom formulas can't reference each other, only
    built-ins, which avoids chaining/self-reference/evaluation-order
    issues entirely.
    """
    if block is None:
        return {}
    variables = {}
    for label, (getter, _fmt, _lower) in stat_defs.items():
        try:
            variables[label] = getter(block)
        except Exception:
            variables[label] = None
    return variables
