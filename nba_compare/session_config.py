"""
Save/restore the current comparison setup (players, seasons, chosen stats,
custom formulas) as a compact, versioned string -- designed so a saved
setup keeps working even after the app's code has changed.

Why this survives code iterations, specifically:
- Plain JSON, not pickle. Pickle ties a save file to the exact class
  definitions that existed when it was created; any refactor (renaming a
  field, moving a class) can silently break or corrupt old saves. JSON is
  just data -- nothing here depends on Python object structure.
- Every field is read with .get() and a default, never direct indexing --
  a save from an older version missing a field just gets a sensible
  default instead of raising.
- Stat labels and custom formulas are NOT validated here -- that requires
  knowing what the current app supports, which this module intentionally
  doesn't know about (keeps it decoupled from table.py/formulas.py, so
  changes to the stat registry can't break this module). The caller
  (app.py) does that validation and reports what got dropped.
- CONFIG_VERSION exists so a future breaking format change has something
  to branch on, instead of guessing from field presence.
"""
from __future__ import annotations
import base64
import json

CONFIG_VERSION = 1


class ConfigError(ValueError):
    pass


def serialize_config(
    spans: list[dict],
    stat_order: list[str],
    custom_formulas: list[dict],
    accolade_path: str = "",
    duos: list[dict] | None = None,
) -> str:
    """
    spans: list of {"player_id": int, "player_name": str, "seasons": [int,...],
                     "label": str|None, "include_regular": bool, "include_playoffs": bool}
    duos: list of {"player_a_id": int, "player_a_name": str, "player_b_id": int,
                    "player_b_name": str, "seasons": [int,...], "label": str|None,
                    "include_regular": bool, "include_playoffs": bool}
    Returns a compact base64 string, safe to copy/paste or save to a file.
    """
    payload = {
        "config_version": CONFIG_VERSION,
        "spans": spans,
        "duos": duos or [],
        "stat_order": stat_order,
        "custom_formulas": custom_formulas,
        "accolade_path": accolade_path,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def deserialize_config(code: str) -> dict:
    """
    Returns {"spans": [...], "stat_order": [...], "custom_formulas": [...],
    "accolade_path": str, "config_version_found": int|None} with every field
    type-checked and defaulted. Raises ConfigError (with a message safe to
    show the user) only if the code can't be read as a save at all --
    garbage input, corrupted base64/JSON. Never raises for a well-formed
    but outdated/incomplete payload.
    """
    code = code.strip()
    if not code:
        raise ConfigError("Paste a save code first.")
    try:
        raw = base64.urlsafe_b64decode(code.encode("ascii"))
        payload = json.loads(raw)
    except Exception as e:
        raise ConfigError(f"Couldn't read that as a save code ({type(e).__name__}). "
                           f"Check you copied the whole thing.")

    if not isinstance(payload, dict):
        raise ConfigError("That code doesn't contain a valid setup.")

    spans_raw = payload.get("spans", [])
    if not isinstance(spans_raw, list):
        spans_raw = []
    spans = []
    for s in spans_raw:
        if not isinstance(s, dict):
            continue
        pid = s.get("player_id")
        seasons = s.get("seasons")
        if pid is None or not isinstance(seasons, list) or not seasons:
            continue
        try:
            spans.append({
                "player_id": int(pid),
                "player_name": str(s.get("player_name", "")),
                "seasons": [int(x) for x in seasons],
                "label": s.get("label") if isinstance(s.get("label"), str) else None,
                "include_regular": bool(s.get("include_regular", True)),
                "include_playoffs": bool(s.get("include_playoffs", True)),
            })
        except (TypeError, ValueError):
            continue  # malformed entry -- skip it, don't fail the whole load

    duos_raw = payload.get("duos", [])
    if not isinstance(duos_raw, list):
        duos_raw = []
    duos = []
    for d in duos_raw:
        if not isinstance(d, dict):
            continue
        pid_a, pid_b = d.get("player_a_id"), d.get("player_b_id")
        seasons = d.get("seasons")
        if pid_a is None or pid_b is None or not isinstance(seasons, list) or not seasons:
            continue
        try:
            duos.append({
                "player_a_id": int(pid_a),
                "player_a_name": str(d.get("player_a_name", "")),
                "player_b_id": int(pid_b),
                "player_b_name": str(d.get("player_b_name", "")),
                "seasons": [int(x) for x in seasons],
                "label": d.get("label") if isinstance(d.get("label"), str) else None,
                "include_regular": bool(d.get("include_regular", True)),
                "include_playoffs": bool(d.get("include_playoffs", True)),
            })
        except (TypeError, ValueError):
            continue  # malformed entry -- skip it, don't fail the whole load

    stat_order = payload.get("stat_order", [])
    stat_order = [s for s in stat_order if isinstance(s, str)] if isinstance(stat_order, list) else []

    formulas_raw = payload.get("custom_formulas", [])
    formulas = []
    if isinstance(formulas_raw, list):
        for f in formulas_raw:
            if isinstance(f, dict) and isinstance(f.get("label"), str) and isinstance(f.get("expr"), str):
                formulas.append({"label": f["label"], "expr": f["expr"]})

    accolade_path = payload.get("accolade_path", "")
    if not isinstance(accolade_path, str):
        accolade_path = ""

    return {
        "spans": spans,
        "duos": duos,
        "stat_order": stat_order,
        "custom_formulas": formulas,
        "accolade_path": accolade_path,
        "config_version_found": payload.get("config_version"),
    }