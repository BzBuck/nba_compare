"""
Lets you define custom stats as formulas over existing ones, e.g.
"PTS / USG_VOL_G" for "points per used possession".

This is NOT Python's eval() -- it walks the parsed expression tree and only
allows numbers, known variable names, and +, -, *, /, ** (and parentheses,
which the parser handles automatically). No function calls, no attribute
access, no indexing, nothing else. A formula like "__import__('os')" is a
syntax the walker doesn't recognize and rejects, not something it executes.
"""
from __future__ import annotations
import ast
import operator

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


def safe_eval(expr: str, variables: dict):
    """Evaluates expr using only the given variables. Raises FormulaError on
    anything outside +,-,*,/,**, parentheses, numbers, and known names."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Invalid formula syntax: {e}")
    return _eval_node(tree, variables)


def validate_formula(expr: str, sample_variables: dict) -> str | None:
    """Returns an error message, or None if the formula is valid against a sample namespace."""
    if not expr or not expr.strip():
        return "Formula is empty."
    try:
        safe_eval(expr, sample_variables)
        return None
    except FormulaError as e:
        return str(e)


# Variables available in formulas, pulled from a single span's stat block for
# one season type. Missing/inapplicable values come through as None, and
# safe_eval propagates None rather than raising (so PTS/USG_VOL_G on a span
# with no team data just shows "--" instead of crashing).
def flatten_block_for_formula(block: dict | None) -> dict:
    if block is None:
        return {}
    variables = {}
    variables.update(block.get("per_game", {}))       # PTS, REB, AST, STL, BLK, TOV, PF, FGM, FGA, FG3M, FG3A, FTM, FTA
    variables.update(block.get("shooting", {}))        # FG_PCT, FG3_PCT, FT_PCT, EFG_PCT, TS_PCT

    variables["TSA_G"] = block.get("tsa_per_game")
    variables["MIN_G"] = block.get("minutes_per_game")
    variables["PLUS_MINUS"] = block.get("plus_minus_per_game")
    variables["PLUS_MINUS_STD"] = block.get("plus_minus_std")
    variables["W"] = block.get("wins")
    variables["L"] = block.get("losses")

    usage = block.get("usage") or {}
    variables["USG_PCT"] = usage.get("usg_pct")
    variables["USG_VOL_G"] = usage.get("usage_per_game")
    variables["MIN_PCT"] = usage.get("min_pct")

    team = block.get("team") or {}
    variables["TEAM_PTS_G"] = team.get("team_pts_per_game")
    variables["TEAM_POSS_G"] = team.get("team_poss_per_game")
    variables["TEAM_PACE"] = team.get("team_pace")
    variables["TEAM_ORTG"] = team.get("team_ortg")
    variables["TEAM_DRTG"] = team.get("team_drtg")
    variables["TEAM_NET_RTG"] = team.get("team_net_rtg")

    for stat, vals in block.get("consistency", {}).items():
        variables[f"{stat}_CV"] = vals.get("cv_pct")
        variables[f"{stat}_FLOOR"] = vals.get("floor")

    variables["GP"] = block.get("games")
    wins, losses = block.get("wins"), block.get("losses")
    variables["WIN_PCT"] = (
        wins / (wins + losses) if wins is not None and losses is not None and (wins + losses) > 0 else None
    )

    depth = block.get("depth") or {}
    variables["CHAMPIONSHIPS"] = depth.get("championships")
    variables["FINALS_APPS"] = depth.get("finals_apps")
    variables["SERIES_W"] = depth.get("series_w")
    variables["SERIES_L"] = depth.get("series_l")
    variables["BEST_ROUND"] = depth.get("best_round_num")
    variables["PLAYOFF_SEASONS"] = depth.get("seasons_in_playoffs")

    pctile = block.get("percentiles") or {}
    variables["PTS_PCTILE"] = pctile.get("PTS")
    variables["REB_PCTILE"] = pctile.get("REB")
    variables["AST_PCTILE"] = pctile.get("AST")
    variables["STL_PCTILE"] = pctile.get("STL")
    variables["BLK_PCTILE"] = pctile.get("BLK")
    variables["TOV_PCTILE"] = pctile.get("TOV")
    variables["FG_PCT_PCTILE"] = pctile.get("FG_PCT")
    variables["FG3_PCT_PCTILE"] = pctile.get("FG3_PCT")
    variables["FT_PCT_PCTILE"] = pctile.get("FT_PCT")
    variables["EFG_PCT_PCTILE"] = pctile.get("EFG_PCT")
    variables["TS_PCT_PCTILE"] = pctile.get("TS_PCT")

    return variables


AVAILABLE_VARIABLES = [
    "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "TS_PCT", "TSA_G", "MIN_G",
    "PLUS_MINUS", "PLUS_MINUS_STD", "W", "L", "GP", "WIN_PCT",
    "USG_PCT", "USG_VOL_G", "MIN_PCT",
    "TEAM_PTS_G", "TEAM_POSS_G", "TEAM_PACE", "TEAM_ORTG", "TEAM_DRTG", "TEAM_NET_RTG",
    "PTS_CV", "REB_CV", "AST_CV", "STL_CV", "BLK_CV", "TOV_CV", "FG3M_CV", "FGM_CV", "FTM_CV",
    "TSA_CV", "USG_EVENTS_CV", "TS_PCT_CV", "MIN_CV",
    "PTS_FLOOR", "REB_FLOOR", "AST_FLOOR", "STL_FLOOR", "BLK_FLOOR", "TOV_FLOOR",
    "FG3M_FLOOR", "FGM_FLOOR", "FTM_FLOOR", "TSA_FLOOR", "USG_EVENTS_FLOOR", "TS_PCT_FLOOR", "MIN_FLOOR",
    "CHAMPIONSHIPS", "FINALS_APPS", "SERIES_W", "SERIES_L", "BEST_ROUND", "PLAYOFF_SEASONS",
    "PTS_PCTILE", "REB_PCTILE", "AST_PCTILE", "STL_PCTILE", "BLK_PCTILE", "TOV_PCTILE",
    "FG_PCT_PCTILE", "FG3_PCT_PCTILE", "FT_PCT_PCTILE", "EFG_PCT_PCTILE", "TS_PCT_PCTILE",
]