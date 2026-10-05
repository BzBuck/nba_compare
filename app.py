"""
Interactive player/span comparison UI.

Run from the project root:
    streamlit run app.py

Lets you add any number of player + year-range "spans" (including multiple
spans of the same player), pick which stats show up, drag to reorder them,
and define your own custom stat formulas from the existing ones.
"""
import dataclasses
import uuid
import streamlit as st
from streamlit_sortables import sort_items

import pandas as pd

from nba_compare import PlayerSpan, DuoSpan, NBADataStore, compare_spans
from nba_compare import config as data_config
from nba_compare.models import season_str
from nba_compare.players import search_players
from nba_compare.table import (
    build_stat_table, build_stat_flags, render_stat_table_html,
    formats_and_lower_is_better, STAT_DEFS, DEFAULT_STAT_LABELS,
)
from nba_compare.formulas import safe_eval, validate_formula, flatten_block_for_formula
from nba_compare.session_config import serialize_config, deserialize_config, ConfigError
from nba_compare.presets import (
    BUILTIN_PRESETS, BUILTIN_PRESET_NAMES, DEFAULT_FORMULA_FMT,
    apply_preset, matching_preset, preset_from_current,
)
from nba_compare.filters import (
    GameFilter, Condition, GAME_FIELDS, SEASON_FIELDS, OPS, OP_SYMBOLS,
    LOCATIONS, RESULTS, GAME_LENGTHS, OPPONENT_MODES, OPP_CONFERENCES,
    PLAYOFF_ROUNDS, SERIES_HOME_COURT, SERIES_SITUATIONS,
)
from nba_compare.playoffs import (
    render_series_table_html, build_series_table, SERIES_COLUMN_DEFS, DEFAULT_SERIES_COLUMNS,
    round_labels_present, build_round_comparison_table,
)

st.set_page_config(page_title="Basketball Compare", layout="wide")


# ---------- cached data access ----------

# "Both" puts every league's players in one search, each row still read
# from its own league's store.
LEAGUE_MODES = list(data_config.LEAGUES) + ["Both"]


@st.cache_resource
def get_stores() -> dict[str, NBADataStore]:
    """One store per league -- see NBADataStore for why they aren't merged.
    Building them touches no data; each league's files load on first use."""
    return {league: NBADataStore.from_config(league) for league in data_config.LEAGUES}


@st.cache_resource
def get_league_directory(league: str) -> pd.DataFrame:
    """
    One league's player directory, with a LEAGUE column. Building it is what
    pulls that league's regular-season game logs, which on a cold start
    means downloading them from the Hugging Face dataset (see config.py) --
    hence the spinner; every run after that is served from the local HF
    cache. Cached per league, so an NBA-only session never downloads WNBA
    data.
    """
    with st.spinner(f"Loading {league} game logs from {data_config.describe_source()}..."):
        return get_stores()[league].all_players("regular").assign(LEAGUE=league)


def directory_for(mode: str) -> pd.DataFrame:
    leagues = data_config.LEAGUES if mode == "Both" else [mode]
    frames = [get_league_directory(league) for league in leagues]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


@st.cache_resource
def get_team_abbreviations(league: str) -> list[str]:
    return get_stores()[league].team_abbreviations()


@st.cache_data
def get_seasons(_store, player_id: int) -> list[int]:
    return _store.seasons_played(player_id)


@st.cache_data
def get_seasons_together(_store, player_a_id: int, player_b_id: int) -> list[int]:
    """Seasons these two were actually teammates (regular OR playoffs) --
    see NBADataStore.seasons_together. Narrower than each player's own
    seasons_played intersected, which would also include seasons they were
    both merely active in the league on different teams."""
    regular = _store.seasons_together(player_a_id, player_b_id, "regular")
    playoffs = _store.seasons_together(player_a_id, player_b_id, "playoffs")
    return sorted(set(regular) | set(playoffs))


stores = get_stores()

# A loaded save switches league before the picker renders (Streamlit won't
# let a widget's value be set after it's drawn in the same run).
if "_pending_league_mode" in st.session_state:
    st.session_state["league_mode"] = st.session_state.pop("_pending_league_mode")
if st.session_state.get("league_mode") not in LEAGUE_MODES:
    st.session_state["league_mode"] = data_config.DEFAULT_LEAGUE


# ---------- session state ----------

if "spans" not in st.session_state:
    st.session_state.spans = [{"id": str(uuid.uuid4())}]
if "custom_formulas" not in st.session_state:
    st.session_state.custom_formulas = []  # list of {"label": str, "expr": str}
if "stat_order" not in st.session_state:
    st.session_state.stat_order = list(DEFAULT_STAT_LABELS)
if "series_col_order" not in st.session_state:
    st.session_state.series_col_order = list(DEFAULT_SERIES_COLUMNS)
if "span_order" not in st.session_state:
    st.session_state.span_order = []
if "user_presets" not in st.session_state:
    st.session_state.user_presets = []  # list of {"name", "stats", "formulas"}
# ---------- game filter state ----------
# Two kinds of condition row: game conditions and whole-season qualifiers
# (see filters.py). Each kind keeps its row list under its own state key, and
# each row's widgets are keyed by kind prefix + row id. The value key
# includes the field, so switching a row's field starts it at that field's
# default (a W% of .500 makes no sense as a minutes cutoff).
CONDITION_KINDS = {
    # kind: (fields, row-list state key, widget key prefix, field a new row starts on)
    "game": (GAME_FIELDS, "gf_conditions", "gfc", "MIN"),
    "season": (SEASON_FIELDS, "gf_season_conditions", "gfs", "GP%"),
}
for _fields, _list_key, _prefix, _default in CONDITION_KINDS.values():
    if _list_key not in st.session_state:
        st.session_state[_list_key] = []  # list of {"id"}; field/op/value live in widget keys


def _condition_value_key(prefix: str, cid: str, field: str) -> str:
    return f"{prefix}_val_{cid}_{field}"


def add_filter_condition(field: str | None = None, op: str | None = None, value: float | None = None,
                         kind: str = "game"):
    fields, list_key, prefix, default_field = CONDITION_KINDS[kind]
    field = field or default_field
    fd = fields[field]
    cid = str(uuid.uuid4())
    st.session_state[f"{prefix}_field_{cid}"] = field
    st.session_state[f"{prefix}_op_{cid}"] = op or fd.default_op
    st.session_state[_condition_value_key(prefix, cid, field)] = float(fd.default_value if value is None else value)
    st.session_state[list_key].append({"id": cid})


def remove_filter_condition(cid: str, kind: str = "game"):
    list_key = CONDITION_KINDS[kind][1]
    st.session_state[list_key] = [c for c in st.session_state[list_key] if c["id"] != cid]


def _conditions_from_state(kind: str) -> tuple[Condition, ...]:
    fields, list_key, prefix, _ = CONDITION_KINDS[kind]
    out = []
    for c in st.session_state[list_key]:
        field = st.session_state.get(f"{prefix}_field_{c['id']}")
        op = st.session_state.get(f"{prefix}_op_{c['id']}")
        value = st.session_state.get(_condition_value_key(prefix, c["id"], field))
        if field in fields and op in OPS and value is not None:
            out.append(Condition(field, op, float(value)))
    return tuple(out)


def render_condition_rows(kind: str, add_label: str):
    fields, list_key, prefix, _ = CONDITION_KINDS[kind]
    for c in st.session_state[list_key]:
        cid = c["id"]
        row = st.columns([3, 1, 2, 1])
        field = row[0].selectbox(
            "Stat", list(fields), key=f"{prefix}_field_{cid}", label_visibility="collapsed",
        )
        fd = fields[field]
        row[0].caption(fd.help)
        row[1].selectbox(
            "Op", list(OPS), key=f"{prefix}_op_{cid}", format_func=OP_SYMBOLS.get,
            label_visibility="collapsed",
        )
        value_key = _condition_value_key(prefix, cid, field)
        if value_key not in st.session_state:
            st.session_state[value_key] = float(fd.default_value)
        row[2].number_input(
            "Value", key=value_key, step=fd.step, format=fd.fmt, label_visibility="collapsed",
        )
        row[3].button("✕", key=f"{prefix}_del_{cid}", on_click=remove_filter_condition, args=(cid, kind))
    st.button(add_label, key=f"{prefix}_add", on_click=add_filter_condition, kwargs={"kind": kind})


def _matching_condition_ids(field: str, op: str, value: float, kind: str) -> list[str]:
    """Rows currently set to exactly this condition -- what makes a quick-add
    button show as on. Editing a row's value makes it a custom condition, and
    the button goes back to off."""
    fields, list_key, prefix, _ = CONDITION_KINDS[kind]
    return [
        c["id"] for c in st.session_state[list_key]
        if st.session_state.get(f"{prefix}_field_{c['id']}") == field
        and st.session_state.get(f"{prefix}_op_{c['id']}") == op
        and st.session_state.get(_condition_value_key(prefix, c["id"], field)) == float(value)
    ]


def toggle_quick_condition(field: str, op: str, value: float, kind: str):
    matches = _matching_condition_ids(field, op, value, kind)
    if matches:
        for cid in matches:
            remove_filter_condition(cid, kind)
    else:
        add_filter_condition(field, op, value, kind)


def render_quick_buttons(quick: list[tuple], kind: str, per_row: int = 5):
    """Toggle buttons: highlighted with a check while their condition is in
    the list, and a second click takes it back out."""
    for start in range(0, len(quick), per_row):
        cols = st.columns(per_row)
        for col, (text, field, op, value) in zip(cols, quick[start:start + per_row]):
            on = bool(_matching_condition_ids(field, op, value, kind))
            col.button(
                f"✓ {text}" if on else text, key=f"gf_quick_{text}",
                type="primary" if on else "secondary",
                on_click=toggle_quick_condition, args=(field, op, value, kind),
                help="Click again to remove." if on else None,
                use_container_width=True,
            )


def set_game_filter_state(gf: GameFilter, enabled: bool = True):
    """Puts a GameFilter into the filter widgets' state -- how Clear and Load
    setup reach widgets that render further down the page."""
    st.session_state.gf_conditions = []
    st.session_state.gf_season_conditions = []
    for c in gf.conditions:
        add_filter_condition(c.field, c.op, c.value, "game")
    for c in gf.season_conditions:
        add_filter_condition(c.field, c.op, c.value, "season")
    st.session_state.gf_location = gf.location
    st.session_state.gf_result = gf.result
    st.session_state.gf_length = gf.game_length
    st.session_state.gf_opp_conf = gf.opp_conference
    st.session_state.gf_opp_mode = gf.opponents_mode
    st.session_state.gf_opponents = list(gf.opponents)
    st.session_state.gf_po_rounds = list(gf.po_rounds)
    st.session_state.gf_po_home = gf.po_home_court
    st.session_state.gf_po_situation = gf.po_situation
    st.session_state.gf_on = enabled


def game_filter_from_state() -> GameFilter:
    return GameFilter(
        conditions=_conditions_from_state("game"),
        season_conditions=_conditions_from_state("season"),
        location=st.session_state.get("gf_location", "Any"),
        result=st.session_state.get("gf_result", "Any"),
        game_length=st.session_state.get("gf_length", "Any"),
        opp_conference=st.session_state.get("gf_opp_conf", "Any"),
        opponents=tuple(st.session_state.get("gf_opponents", [])),
        opponents_mode=st.session_state.get("gf_opp_mode", "Only"),
        po_rounds=tuple(sorted(st.session_state.get("gf_po_rounds", []))),
        po_home_court=st.session_state.get("gf_po_home", "Any"),
        po_situation=st.session_state.get("gf_po_situation", "Any"),
    )


# One click adds the condition; it's then editable like any other row.
QUICK_FILTERS = [
    ("Close games (≤ 10)", "|Margin|", "<=", 10),
    ("Blowouts (≥ 20)", "|Margin|", ">=", 20),
    ("vs. top-8 seeds", "Opp Seed", "<=", 8),
    ("vs. winning teams", "Opp W%", ">=", 0.5),
    ("Team top-8 seed", "Team Seed", "<=", 8),
    ("Team .500+", "Team W%", ">=", 0.5),
    ("30+ minutes", "MIN", ">=", 30),
    ("Back-to-backs", "Rest days", "=", 0),
    ("Rested (2+ days)", "Rest days", ">=", 2),
]
QUICK_PLAYOFF_FILTERS = [
    ("Game 7s", "Series game #", "=", 7),
    ("Trailing in series", "Series lead", "<", 0),
    ("Game 1s", "Series game #", "=", 1),
]
QUICK_SEASON_FILTERS = [
    ("Healthy seasons (GP% ≥ 60)", "GP%", ">=", 60),
    ("Starter minutes (30+ MIN/G)", "MIN/G", ">=", 30),
    ("Bench minutes (≤ 20 MIN/G)", "MIN/G", "<=", 20),
]


def add_span():
    st.session_state.spans.append({"id": str(uuid.uuid4())})


def remove_span(span_id: str):
    st.session_state.spans = [s for s in st.session_state.spans if s["id"] != span_id]


league_mode = st.session_state["league_mode"]
st.title(f"{'NBA / WNBA' if league_mode == 'Both' else league_mode} Player / Span Comparison")
st.radio(
    "League", LEAGUE_MODES, key="league_mode", horizontal=True,
    help="Which league's players to search. **Both** lets an NBA player sit next to a WNBA "
         "player -- each is still measured against their own league (%ile, relative "
         "shooting, pace, playoff rounds).",
)
directory = directory_for(league_mode)


def player_row(player_id: int, directory: pd.DataFrame = directory):
    """The directory row for this player id, or None if they aren't in it
    (e.g. a WNBA player picked under "Both", after switching to NBA)."""
    row = directory[directory.PLAYER_ID == player_id]
    return None if row.empty else row.iloc[0]


def player_option_label(player_id: int) -> str:
    row = player_row(player_id)
    if row is None:
        return str(player_id)
    return f"{row.PLAYER_NAME} ({row.LEAGUE})" if league_mode == "Both" else row.PLAYER_NAME


def league_of(player_id: int, directory: pd.DataFrame = directory) -> str:
    return player_row(player_id, directory).LEAGUE


def seasons_for(player_id: int, directory: pd.DataFrame = directory) -> list[int]:
    return get_seasons(stores[league_of(player_id, directory)], player_id)


def seasons_together_for(player_a_id: int, player_b_id: int, directory: pd.DataFrame = directory) -> list[int]:
    """[] for players from different leagues -- they were never teammates."""
    league = league_of(player_a_id, directory)
    if league != league_of(player_b_id, directory):
        return []
    return get_seasons_together(stores[league], player_a_id, player_b_id)


# ---------- save / load setup ----------
# Placed before the span builder loop below, because loading must set each
# span's widget state (q_/match_/range_/reg_/po_/label_) BEFORE those
# widgets render this run -- that's how the loaded values end up shown.

def _build_span_dicts_from_session() -> list[dict]:
    """Reads the currently-filled-in Player-mode rows back out of
    session_state, in the plain-dict shape session_config.py expects.
    Skips any row that never got a player picked (an empty '+ Add player /
    span' row) or is in Duo mode."""
    out = []
    for cfg in st.session_state.spans:
        sid = cfg["id"]
        if st.session_state.get(f"mode_{sid}", "Player") != "Player":
            continue
        match_key, range_key = f"match_{sid}", f"range_{sid}"
        if match_key not in st.session_state or range_key not in st.session_state:
            continue
        player_id = st.session_state[match_key]
        row = player_row(player_id)
        if row is None:
            continue
        name = row.PLAYER_NAME
        seasons_played = seasons_for(player_id)
        lo, hi = st.session_state[range_key]
        season_list = [s for s in seasons_played if lo <= s <= hi]
        if not season_list:
            continue
        out.append({
            "player_id": player_id,
            "player_name": name,
            "seasons": season_list,
            "label": st.session_state.get(f"label_{sid}") or None,
            "include_regular": st.session_state.get(f"reg_{sid}", True),
            "include_playoffs": st.session_state.get(f"po_{sid}", True),
        })
    return out


def _build_duo_dicts_from_session() -> list[dict]:
    """Same idea as _build_span_dicts_from_session(), but for Duo-mode rows."""
    out = []
    for cfg in st.session_state.spans:
        sid = cfg["id"]
        if st.session_state.get(f"mode_{sid}", "Player") != "Duo":
            continue
        matcha_key, matchb_key, range_key = f"matcha_{sid}", f"matchb_{sid}", f"range_duo_{sid}"
        if matcha_key not in st.session_state or matchb_key not in st.session_state or range_key not in st.session_state:
            continue
        player_a_id, player_b_id = st.session_state[matcha_key], st.session_state[matchb_key]
        row_a, row_b = player_row(player_a_id), player_row(player_b_id)
        if row_a is None or row_b is None or player_a_id == player_b_id:
            continue
        name_a, name_b = row_a.PLAYER_NAME, row_b.PLAYER_NAME
        overlap = seasons_together_for(player_a_id, player_b_id)
        lo, hi = st.session_state[range_key]
        season_list = [s for s in overlap if lo <= s <= hi]
        if not season_list:
            continue
        out.append({
            "player_a_id": player_a_id,
            "player_a_name": name_a,
            "player_b_id": player_b_id,
            "player_b_name": name_b,
            "seasons": season_list,
            "label": st.session_state.get(f"label_duo_{sid}") or None,
            "include_regular": st.session_state.get(f"reg_duo_{sid}", True),
            "include_playoffs": st.session_state.get(f"po_duo_{sid}", True),
        })
    return out


with st.sidebar.expander("Save / Load setup", expanded=False):
    st.caption("Save your current players, years, and chosen stats as a code to paste back in later.")
    if st.button("Generate save code"):
        code = serialize_config(
            spans=_build_span_dicts_from_session(),
            duos=_build_duo_dicts_from_session(),
            stat_order=st.session_state.get("stat_order", []),
            custom_formulas=st.session_state.get("custom_formulas", []),
            user_presets=st.session_state.get("user_presets", []),
            head_to_head=st.session_state.get("h2h_mode", False),
            league_mode=league_mode,
            game_filter={"enabled": st.session_state.get("gf_on", True), **game_filter_from_state().to_dict()},
        )
        st.session_state["_last_save_code"] = code
    if st.session_state.get("_last_save_code"):
        st.text_area("Save code (copy this)", value=st.session_state["_last_save_code"], height=100)
        st.download_button(
            "Download as file", data=st.session_state["_last_save_code"],
            file_name="nba_compare_setup.txt", mime="text/plain",
        )

    st.divider()
    load_code = st.text_area("Paste a save code here", key="load_code_input", height=100)
    uploaded = st.file_uploader("...or upload a saved file", type=["txt", "json"])
    if uploaded is not None:
        load_code = uploaded.read().decode("utf-8")

    if st.button("Load setup"):
        try:
            cfg = deserialize_config(load_code)
        except ConfigError as e:
            st.error(str(e))
        else:
            # Players are looked up in the SAVED league's directory -- the
            # league picker switches to it on the rerun below.
            load_dir = directory_for(cfg["league_mode"])
            skipped_players, new_span_cfgs = [], []
            for s in cfg["spans"]:
                match = player_row(s["player_id"], load_dir)
                if match is None:
                    skipped_players.append(s.get("player_name") or f"player id {s['player_id']}")
                    continue
                new_sid = str(uuid.uuid4())
                st.session_state[f"q_{new_sid}"] = match.PLAYER_NAME
                st.session_state[f"match_{new_sid}"] = s["player_id"]
                seasons_played = seasons_for(s["player_id"], load_dir)
                valid_seasons = [yr for yr in s["seasons"] if yr in seasons_played] or seasons_played
                st.session_state.setdefault("_pending_ranges", {})[new_sid] = (min(valid_seasons), max(valid_seasons))
                st.session_state[f"reg_{new_sid}"] = s["include_regular"]
                st.session_state[f"po_{new_sid}"] = s["include_playoffs"]
                st.session_state[f"label_{new_sid}"] = s["label"] or ""
                new_span_cfgs.append({"id": new_sid})

            skipped_duos = []
            for d in cfg["duos"]:
                match_a = player_row(d["player_a_id"], load_dir)
                match_b = player_row(d["player_b_id"], load_dir)
                if match_a is None or match_b is None:
                    label = d.get("label") or f"{d.get('player_a_name', '?')} & {d.get('player_b_name', '?')}"
                    skipped_duos.append(label)
                    continue
                new_sid = str(uuid.uuid4())
                st.session_state[f"mode_{new_sid}"] = "Duo"
                st.session_state[f"qa_{new_sid}"] = match_a.PLAYER_NAME
                st.session_state[f"matcha_{new_sid}"] = d["player_a_id"]
                st.session_state[f"qb_{new_sid}"] = match_b.PLAYER_NAME
                st.session_state[f"matchb_{new_sid}"] = d["player_b_id"]
                overlap = seasons_together_for(d["player_a_id"], d["player_b_id"], load_dir)
                valid_seasons = [yr for yr in d["seasons"] if yr in overlap] or overlap
                if valid_seasons:
                    st.session_state.setdefault("_pending_ranges", {})[new_sid] = (
                        min(valid_seasons), max(valid_seasons)
                    )
                st.session_state[f"reg_duo_{new_sid}"] = d["include_regular"]
                st.session_state[f"po_duo_{new_sid}"] = d["include_playoffs"]
                st.session_state[f"label_duo_{new_sid}"] = d["label"] or ""
                new_span_cfgs.append({"id": new_sid})

            if new_span_cfgs:
                st.session_state.spans = new_span_cfgs

            # Re-validate custom formulas against the CURRENT formula engine
            # before trusting them -- a variable the save relied on might
            # have been renamed/removed since. Invalid ones are dropped,
            # not silently kept broken.
            sample_vars = {v: 1.0 for v in STAT_DEFS}
            valid_formulas, dropped_formulas = [], []
            for f in cfg["custom_formulas"]:
                if validate_formula(f["expr"], sample_vars):
                    dropped_formulas.append(f["label"])
                else:
                    valid_formulas.append(f)
            st.session_state.custom_formulas = valid_formulas

            # Saved presets get the same check on their formulas, and keep
            # only the rows that still exist. A preset left with no rows, or
            # one named like a built-in (which always wins), is dropped.
            restored_presets, dropped_presets = [], []
            for p in cfg["user_presets"]:
                p_formulas = [f for f in p["formulas"] if not validate_formula(f["expr"], sample_vars)]
                p_valid = set(STAT_DEFS) | {f["label"] for f in p_formulas}
                p_stats = [x for x in p["stats"] if x in p_valid]
                if not p_stats or p["name"] in BUILTIN_PRESET_NAMES:
                    dropped_presets.append(p["name"])
                    continue
                restored_presets.append({**p, "stats": p_stats, "formulas": p_formulas})
            st.session_state.user_presets = restored_presets

            # Stats: only keep labels that still exist (built-ins current
            # code defines + the custom formulas just restored) -- anything
            # else (a stat renamed/removed since the save) is dropped rather
            # than crashing the picker on an option that no longer exists.
            valid_stat_names = set(STAT_DEFS.keys()) | {f["label"] for f in valid_formulas}
            restored_order = [s for s in cfg["stat_order"] if s in valid_stat_names]
            st.session_state.stat_order = restored_order or list(DEFAULT_STAT_LABELS)
            st.session_state.selected_stats = list(st.session_state.stat_order)

            st.session_state["h2h_mode"] = cfg["head_to_head"]
            st.session_state["_pending_league_mode"] = cfg["league_mode"]
            set_game_filter_state(
                GameFilter.from_dict(cfg["game_filter"]), enabled=cfg["game_filter"].get("enabled", True) is not False,
            )

            msg = f"Loaded {len(new_span_cfgs)} span(s)."
            if skipped_players:
                msg += f" Skipped player(s) (not found in current data): {', '.join(skipped_players)}."
            if skipped_duos:
                msg += f" Skipped duo(s) (a player not found in current data): {', '.join(skipped_duos)}."
            if dropped_formulas:
                msg += f" Dropped invalid custom formula(s): {', '.join(dropped_formulas)}."
            if dropped_presets:
                msg += f" Dropped preset(s) (no stats left, or named like a built-in): {', '.join(dropped_presets)}."
            st.success(msg)
            st.rerun()

# ---------- custom formulas ----------

st.sidebar.header("Custom stat formulas")
st.sidebar.caption(
    "Combine existing stats -- using the exact names shown in the table below "
    "(e.g. `PTS/G`, `TS%`, `USG Vol/G`) -- with + - * / and parentheses, e.g. "
    "`PTS/G / USG Vol/G` for points per used possession."
)
with st.sidebar.expander("Available variable names"):
    st.code(", ".join(STAT_DEFS), language=None)

with st.sidebar.form("add_formula_form", clear_on_submit=True):
    new_label = st.text_input("Stat name", placeholder="Pts per Use")
    new_expr = st.text_input("Formula", placeholder="PTS/G / USG Vol/G")
    fmt_cols = st.columns(2)
    new_decimals = fmt_cols[0].selectbox("Decimals", [0, 1, 2, 3], index=3)
    new_lower = fmt_cols[1].checkbox("Lower is better", help="Highlight the lowest value as best, like TOV/G.")
    submitted = st.form_submit_button("Add formula")
    if submitted:
        sample_vars = {v: 1.0 for v in STAT_DEFS}  # syntax/name check only
        error = validate_formula(new_expr, sample_vars)
        if not new_label.strip():
            st.sidebar.error("Give the stat a name.")
        elif new_label in STAT_DEFS or any(f["label"] == new_label for f in st.session_state.custom_formulas):
            st.sidebar.error(f"'{new_label}' already exists — pick a different name.")
        elif error:
            st.sidebar.error(error)
        else:
            st.session_state.custom_formulas.append({
                "label": new_label, "expr": new_expr,
                "fmt": f"{{:.{new_decimals}f}}", "lower": new_lower,
            })
            st.session_state.stat_order.append(new_label)
            # Auto-show the new stat immediately, not just in stat_order --
            # the multiselect widget has its own persisted state (keyed by
            # "selected_stats") that must be updated directly, since passing
            # `default=` again has no effect once the widget already has state.
            if "selected_stats" in st.session_state:
                st.session_state.selected_stats = list(st.session_state.selected_stats) + [new_label]


def _formula_rows(container, formulas: list[dict], key_prefix: str):
    """One caption + remove button per formula. Removal is by label, since
    the user's and the preset's formulas are listed separately."""
    for f in formulas:
        cols = container.columns([4, 1])
        cols[0].caption(f"**{f['label']}** = `{f['expr']}`")
        if cols[1].button("✕", key=f"{key_prefix}_{f['label']}"):
            removed_label = f["label"]
            st.session_state.custom_formulas = [
                g for g in st.session_state.custom_formulas if g["label"] != removed_label
            ]
            st.session_state.stat_order = [s for s in st.session_state.stat_order if s != removed_label]
            if "selected_stats" in st.session_state:
                st.session_state.selected_stats = [
                    s for s in st.session_state.selected_stats if s != removed_label
                ]
            st.rerun()


own_formulas = [f for f in st.session_state.custom_formulas if not f.get("managed")]
preset_formulas = [f for f in st.session_state.custom_formulas if f.get("managed")]
if own_formulas:
    st.sidebar.write("Your custom stats:")
    _formula_rows(st.sidebar, own_formulas, "del_formula")
if preset_formulas:
    with st.sidebar.expander(f"Added by preset ({len(preset_formulas)})"):
        st.caption("These come and go with the stat preset you pick.")
        _formula_rows(st, preset_formulas, "del_preset_formula")

# Combined registry: built-ins + custom formulas, each custom formula's
# getter closing over its own expr (default val=expr avoids late-binding bugs).
combined_stat_defs = dict(STAT_DEFS)
for f in st.session_state.custom_formulas:
    combined_stat_defs[f["label"]] = (
        lambda block, expr=f["expr"]: safe_eval(expr, flatten_block_for_formula(block, STAT_DEFS)),
        f.get("fmt", DEFAULT_FORMULA_FMT),
        f.get("lower", False),
    )

# ---------- span builder ----------

valid_spans: list[PlayerSpan | DuoSpan] = []

for cfg in st.session_state.spans:
    sid = cfg["id"]
    with st.container(border=True):
        mode = st.radio(
            "Type", ["Player", "Duo"], key=f"mode_{sid}", horizontal=True,
            label_visibility="collapsed",
        )

        if mode == "Player":
            cols = st.columns([3, 2, 2, 1, 1, 1])

            query = cols[0].text_input("Player", key=f"q_{sid}", placeholder="Type a name…")
            matches = search_players(query, directory)

            player_id = None
            player_name = None
            if not matches.empty:
                player_id = cols[0].selectbox(
                    "Match", [int(pid) for pid in matches.PLAYER_ID], key=f"match_{sid}",
                    format_func=player_option_label, label_visibility="collapsed",
                )
                player_name = player_row(player_id).PLAYER_NAME
                league = league_of(player_id)
            elif query:
                cols[0].caption("No matches")

            if player_id is not None:
                seasons = seasons_for(player_id)
                if seasons:
                    # select_slider needs an explicit value= on first render to
                    # know it's a RANGE slider at all -- pre-seeding its session
                    # state key alone (without value=) silently drops back to
                    # single-value mode and discards whatever was pre-seeded.
                    # So loaded ranges go through this one-time side-channel
                    # instead of writing directly into range_{sid}.
                    pending_ranges = st.session_state.setdefault("_pending_ranges", {})
                    default_range = pending_ranges.pop(sid, (seasons[0], seasons[-1]))
                    lo, hi = cols[1].select_slider(
                        "Seasons",
                        options=seasons,
                        value=default_range,
                        key=f"range_{sid}",
                        format_func=lambda y, league=league: season_str(y, league),
                    )
                    season_list = [s for s in seasons if lo <= s <= hi]
                else:
                    cols[1].caption("No seasons found")
                    season_list = []

                reg_key, po_key = f"reg_{sid}", f"po_{sid}"
                if reg_key not in st.session_state:
                    st.session_state[reg_key] = True
                if po_key not in st.session_state:
                    st.session_state[po_key] = True
                include_reg = cols[2].checkbox("Reg. season", key=reg_key)
                include_po = cols[2].checkbox("Playoffs", key=po_key)
                label = cols[3].text_input("Label", key=f"label_{sid}", placeholder="auto")

                if season_list:
                    valid_spans.append(
                        PlayerSpan(
                            player_id=player_id,
                            player_name=player_name,
                            seasons=season_list,
                            label=label or None,
                            include_regular=include_reg,
                            include_playoffs=include_po,
                            league=league,
                        )
                    )

            if cols[5].button("Remove", key=f"remove_{sid}"):
                remove_span(sid)
                st.rerun()

        else:  # Duo mode -- combined numbers for games these two played together as teammates
            cols = st.columns([3, 3, 2, 1, 1, 1])

            query_a = cols[0].text_input("Player A", key=f"qa_{sid}", placeholder="Type a name…")
            matches_a = search_players(query_a, directory)
            player_a_id = player_a_name = None
            if not matches_a.empty:
                player_a_id = cols[0].selectbox(
                    "Match A", [int(pid) for pid in matches_a.PLAYER_ID], key=f"matcha_{sid}",
                    format_func=player_option_label, label_visibility="collapsed",
                )
                player_a_name = player_row(player_a_id).PLAYER_NAME
            elif query_a:
                cols[0].caption("No matches")

            query_b = cols[1].text_input("Player B", key=f"qb_{sid}", placeholder="Type a name…")
            matches_b = search_players(query_b, directory)
            player_b_id = player_b_name = None
            if not matches_b.empty:
                player_b_id = cols[1].selectbox(
                    "Match B", [int(pid) for pid in matches_b.PLAYER_ID], key=f"matchb_{sid}",
                    format_func=player_option_label, label_visibility="collapsed",
                )
                player_b_name = player_row(player_b_id).PLAYER_NAME
            elif query_b:
                cols[1].caption("No matches")

            if player_a_id is not None and player_b_id is not None:
                if player_a_id == player_b_id:
                    cols[2].caption("Pick two different players")
                elif league_of(player_a_id) != league_of(player_b_id):
                    cols[2].caption("Different leagues -- never teammates")
                else:
                    league = league_of(player_a_id)
                    overlap = seasons_together_for(player_a_id, player_b_id)
                    if not overlap:
                        cols[2].caption("Never teammates")
                    else:
                        pending_ranges = st.session_state.setdefault("_pending_ranges", {})
                        default_range = pending_ranges.pop(sid, (overlap[0], overlap[-1]))
                        lo, hi = cols[2].select_slider(
                            "Seasons",
                            options=overlap,
                            value=default_range,
                            key=f"range_duo_{sid}",
                            format_func=lambda y, league=league: season_str(y, league),
                        )
                        season_list = [s for s in overlap if lo <= s <= hi]

                        reg_key, po_key = f"reg_duo_{sid}", f"po_duo_{sid}"
                        if reg_key not in st.session_state:
                            st.session_state[reg_key] = True
                        if po_key not in st.session_state:
                            st.session_state[po_key] = True
                        include_reg = cols[3].checkbox("Reg. season", key=reg_key)
                        include_po = cols[3].checkbox("Playoffs", key=po_key)
                        label = cols[4].text_input("Label", key=f"label_duo_{sid}", placeholder="auto")

                        if season_list:
                            valid_spans.append(
                                DuoSpan(
                                    player_a_id=player_a_id,
                                    player_a_name=player_a_name,
                                    player_b_id=player_b_id,
                                    player_b_name=player_b_name,
                                    seasons=season_list,
                                    label=label or None,
                                    include_regular=include_reg,
                                    include_playoffs=include_po,
                                    league=league,
                                )
                            )

            if cols[5].button("Remove", key=f"remove_{sid}"):
                remove_span(sid)
                st.rerun()

st.button("+ Add player / span", on_click=add_span)

head_to_head = st.toggle(
    "Head-to-head", key="h2h_mode",
    help="Only count games the two rows actually played against each other -- each row is a "
         "player or a duo, and everyone involved needs 10+ minutes, with the two sides on "
         "opposite teams. Needs exactly two rows.",
)
if head_to_head:
    if len(valid_spans) != 2:
        st.warning(
            f"Head-to-head compares exactly two rows (players or duos) -- you have {len(valid_spans)}. "
            "Showing the normal comparison instead."
        )
    elif valid_spans[0].league != valid_spans[1].league:
        st.warning(
            "Head-to-head needs both rows from the same league -- an NBA and a WNBA player "
            "never played each other. Showing the normal comparison instead."
        )
    elif set(valid_spans[0].player_ids) & set(valid_spans[1].player_ids):
        st.warning(
            "Head-to-head needs the two rows to share no players -- someone can't play against "
            "themselves. Showing the normal comparison instead."
        )
    else:
        a, b = valid_spans
        # Both sides use the seasons BOTH ranges cover, so they're built from
        # the exact same games -- otherwise one side could include meetings
        # the other side's range leaves out.
        shared = sorted(set(a.seasons) & set(b.seasons))
        if not shared:
            st.warning("Those two season ranges don't overlap -- no games to compare head-to-head.")
            valid_spans = []
        else:
            def _is_auto_label(span) -> bool:
                return span.label == dataclasses.replace(span, label=None).label

            # A label typed in the row stays as typed; an auto one is rebuilt
            # for the shared seasons and gains "vs. <opponent>".
            valid_spans = [
                dataclasses.replace(
                    me, seasons=shared, label=None if _is_auto_label(me) else me.label,
                    vs_player_ids=opp.player_ids, vs_name=opp.names,
                )
                for me, opp in ((a, b), (b, a))
            ]

with st.expander("Game filters", expanded=False):
    st.caption(
        "Only count games that pass **every** condition below -- for all rows, regular season "
        "and playoffs alike. Opponent W% and seed are the opponent's regular-season record "
        "that season. In head-to-head, conditions are read from the first row's side "
        "(a \"W\" is a game it won), and player conditions like MIN must hold for both rows."
    )
    if "gf_on" not in st.session_state:
        st.session_state.gf_on = True
    st.toggle("Apply game filters", key="gf_on",
              help="Switch off to see the unfiltered numbers without losing the filters set up here.")

    st.markdown("**Season qualifiers**")
    st.caption(
        "Keep or drop **whole seasons**, judged on the full regular season before any game "
        "filter below (a duo: the games they shared). A dropped season loses its playoff games "
        "too. Only availability and role are offered -- a cutoff on production like PTS/G would "
        "guarantee the very number being compared."
    )
    render_quick_buttons(QUICK_SEASON_FILTERS, "season", per_row=3)
    render_condition_rows("season", "+ Add season qualifier")

    st.divider()
    st.markdown("**Game conditions**")
    st.caption("Quick add:")
    render_quick_buttons(QUICK_FILTERS, "game")
    render_condition_rows("game", "+ Add condition")

    st.divider()
    cat_cols = st.columns(3)
    cat_cols[0].radio("Location", LOCATIONS, key="gf_location", horizontal=True)
    cat_cols[1].radio("Result", RESULTS, key="gf_result", horizontal=True)
    cat_cols[2].radio("Game length", GAME_LENGTHS, key="gf_length", horizontal=True)

    opp_cols = st.columns([2, 1, 4])
    opp_cols[0].radio(
        "Opponent conference", OPP_CONFERENCES, key="gf_opp_conf", horizontal=True,
        format_func=lambda c: {"Same": "Own conf.", "Other": "Other conf."}.get(c, c),
        help="Uses today's conference alignment for every season (realignments like Milwaukee "
             "moving East in 1980 aren't modeled), and teams from before the conference table's "
             "coverage fail East/West.",
    )
    opp_cols[1].radio("Opponents", OPPONENT_MODES, key="gf_opp_mode",
                      format_func=lambda m: "Only vs." if m == "Only" else "Exclude")
    filter_leagues = {s.league for s in valid_spans} or (
        set(data_config.LEAGUES) if league_mode == "Both" else {league_mode}
    )
    opp_options = sorted(
        {abbr for lg in filter_leagues for abbr in get_team_abbreviations(lg)}
        | set(st.session_state.get("gf_opponents", []))
    )
    opp_cols[2].multiselect(
        "Teams", opp_options, key="gf_opponents", placeholder="Any opponent",
        help="Abbreviations as they appeared at the time -- a relocated franchise is listed "
             "under each of its names (e.g. SEA and OKC), so pick all you mean.",
    )

    st.divider()
    st.markdown("**Playoff games**")
    st.caption(
        "These only narrow **playoff** games -- the regular-season table is untouched. Rounds count "
        "back from the Finals, so *Last 4* is the conference finals in a 4-round NBA year and the "
        "semifinals in the WNBA. A *closeout* game is one a win would clinch, an *elimination* game "
        "one a loss would end; a Game 7 is both (*winner-take-all*). Series length is read off the "
        "result (a 4-2 series was best-of-7)."
    )
    po_cols = st.columns([2, 1, 2])
    po_cols[0].multiselect(
        "Round", list(PLAYOFF_ROUNDS), key="gf_po_rounds", format_func=PLAYOFF_ROUNDS.get,
        placeholder="Every round",
    )
    po_cols[1].radio("Series home court", SERIES_HOME_COURT, key="gf_po_home",
                     help="Whether this row's team had home-court advantage in the series "
                          "(hosted Game 1) -- not whether this game was at home; that's Location above.")
    po_cols[2].radio("Series situation", SERIES_SITUATIONS, key="gf_po_situation", horizontal=True)
    st.caption("Quick add -- each one adds a playoffs-only row under Game conditions above:")
    render_quick_buttons(QUICK_PLAYOFF_FILTERS, "game", per_row=3)

    st.button("Clear all filters", on_click=set_game_filter_state, args=(GameFilter(),))

game_filter = game_filter_from_state() if st.session_state.get("gf_on", True) else GameFilter()

if len(valid_spans) > 1:
    current_labels = [s.label for s in valid_spans]
    if len(current_labels) != len(set(current_labels)):
        st.caption(
            "Give spans distinct labels above (the 'Label' field per row) to enable drag-reordering "
            "-- duplicate names found, and dragging can't tell which one you mean."
        )
    else:
        st.session_state.span_order = (
            [l for l in st.session_state.span_order if l in current_labels]
            + [l for l in current_labels if l not in st.session_state.span_order]
        )
        st.caption("Drag to reorder players/spans:")
        span_sorter_key = "span_order_sorter_" + "|".join(sorted(st.session_state.span_order))
        new_span_order = sort_items(st.session_state.span_order, key=span_sorter_key)
        if new_span_order and set(new_span_order) == set(st.session_state.span_order):
            st.session_state.span_order = new_span_order
        label_to_span = {s.label: s for s in valid_spans}
        valid_spans = [label_to_span[l] for l in st.session_state.span_order if l in label_to_span]

st.divider()



def _all_presets() -> list[dict]:
    return BUILTIN_PRESETS + st.session_state.user_presets


def _on_preset_pick():
    name = st.session_state.get("preset_pick")
    preset = next((p for p in _all_presets() if p["name"] == name), None)
    if preset is None:
        return  # clicking the active preset again just deselects it -- nothing to change
    formulas, order = apply_preset(preset, st.session_state.custom_formulas)
    st.session_state.custom_formulas = formulas
    st.session_state.stat_order = order
    st.session_state.selected_stats = list(order)


def _save_user_preset():
    name = st.session_state.get("new_preset_name", "").strip()
    if not name:
        st.session_state["_preset_msg"] = ("error", "Give the preset a name.")
    elif name in BUILTIN_PRESET_NAMES:
        st.session_state["_preset_msg"] = ("error", f"'{name}' is a built-in preset — pick a different name.")
    elif not st.session_state.stat_order:
        st.session_state["_preset_msg"] = ("error", "Pick some stats to save first.")
    else:
        preset = preset_from_current(name, st.session_state.stat_order, st.session_state.custom_formulas)
        others = [p for p in st.session_state.user_presets if p["name"] != name]
        replaced = len(others) != len(st.session_state.user_presets)
        st.session_state.user_presets = others + [preset]
        st.session_state.new_preset_name = ""
        st.session_state["_preset_msg"] = ("success", f"{'Updated' if replaced else 'Saved'} preset '{name}'.")


def _delete_user_preset(name: str):
    st.session_state.user_presets = [p for p in st.session_state.user_presets if p["name"] != name]


if len(valid_spans) == 0:
    st.info("Add at least one player above to see a comparison.")
else:
    # Reflect which preset (if any) the current rows match, so the pill
    # shows as active until the stats are edited by hand. Set before the
    # widget renders, which is the only point Streamlit allows it.
    presets = _all_presets()
    st.session_state.preset_pick = matching_preset(st.session_state.stat_order, presets)
    st.pills(
        "Stat presets", [p["name"] for p in presets], selection_mode="single",
        key="preset_pick", on_change=_on_preset_pick,
        help="Swap the table to a themed set of stats. Presets add the custom formulas they "
             "need and remove them again when you switch; your own formulas are never touched.",
    )

    with st.expander("Customize stats shown", expanded=False):
        available_labels = list(combined_stat_defs.keys())

        if "selected_stats" not in st.session_state:
            st.session_state.selected_stats = [s for s in st.session_state.stat_order if s in available_labels]

        selected = st.multiselect(
            "Stats to show (box score, usage, team, consistency, and your custom stats)",
            options=available_labels,
            key="selected_stats",
        )

        # Keep stat_order in sync with the current selection (add newly
        # picked stats at the end, drop deselected ones) before reordering.
        ordered_selection = [s for s in st.session_state.stat_order if s in selected]
        ordered_selection += [s for s in selected if s not in ordered_selection]
        st.session_state.stat_order = ordered_selection

        st.caption("Drag to reorder:")
        if ordered_selection:
            # Key depends on WHICH stats are selected (not their order), so
            # the component only remounts -- and re-syncs to the correct
            # item list -- when you add/remove a stat, not on every drag.
            sorter_key = "stat_order_sorter_" + "|".join(sorted(ordered_selection))
            new_order = sort_items(ordered_selection, key=sorter_key)
            # Defensive: only trust the component's output if it's actually
            # a reordering of what we gave it. A stale response (e.g. the
            # frontend hasn't caught up with a selection change yet) would
            # otherwise get written into stat_order and corrupt every
            # future run, since nothing else resets it.
            if new_order and set(new_order) == set(ordered_selection):
                st.session_state.stat_order = new_order
            else:
                st.session_state.stat_order = ordered_selection
        stat_labels = st.session_state.stat_order

        st.divider()
        preset_cols = st.columns([3, 1])
        preset_cols[0].text_input(
            "Save these stats as a preset", key="new_preset_name", placeholder="Preset name",
            help="Saves the stats above, in this order, plus any custom formulas they use. "
                 "Saving under an existing name updates it. Presets are kept in your save code.",
        )
        preset_cols[1].button("Save preset", on_click=_save_user_preset, use_container_width=True)
        msg = st.session_state.pop("_preset_msg", None)
        if msg:
            (st.error if msg[0] == "error" else st.success)(msg[1])
        for p in st.session_state.user_presets:
            cols = st.columns([4, 1])
            cols[0].caption(f"**{p['name']}** — {len(p['stats'])} stats")
            cols[1].button("✕", key=f"del_user_preset_{p['name']}",
                           on_click=_delete_user_preset, args=(p["name"],))

    result = compare_spans(valid_spans, stores, game_filter)
    if game_filter.is_active:
        unit = {"seasons": "seasons", "regular": "RS", "playoffs": "PO"}
        kept = []
        for agg in result.aggregates:
            parts = [
                f"{n:,} of {total:,} {unit[key]}"
                for key, (n, total) in sorted(agg["filter_counts"].items(), key=lambda kv: list(unit).index(kv[0]))
            ]
            kept.append(f"{agg['label']}: {', '.join(parts)}")
        if game_filter.narrows("regular"):
            caveat = (
                "%ile rows show — while game conditions are on, since they rank whole seasons, and the "
                "playoff series breakdown lists only series with at least one game that passed."
            )
        elif game_filter.narrows("playoffs"):
            caveat = (
                "The playoff filters leave the regular season untouched. Playoff %ile rows show —, and "
                "the playoff series breakdown lists only series with at least one game that passed."
            )
        else:
            caveat = (
                "Season qualifiers keep whole seasons, so %ile rows and the playoff series breakdown "
                "still apply to the seasons kept."
            )
        st.info(
            "**Game filter on** — " + " · ".join(game_filter.describe()) + "  \n"
            + "Kept — " + "; ".join(kept) + "  \n" + caveat
        )
    has_regular = any(agg["regular"] for agg in result.aggregates)
    has_playoffs = any(agg["playoffs"] for agg in result.aggregates)
    formats, lower_is_better = formats_and_lower_is_better(combined_stat_defs)

    if not stat_labels:
        st.caption("No stats selected — pick some above.")
    else:
        if has_regular:
            reg_table = build_stat_table(result, "regular", stat_labels=stat_labels, stat_defs=combined_stat_defs)
            reg_flags = build_stat_flags(result, "regular", stat_labels=stat_labels, stat_defs=combined_stat_defs)
            st.markdown(
                render_stat_table_html(reg_table, "Regular Season", formats=formats,
                                       lower_is_better=lower_is_better, flags=reg_flags),
                unsafe_allow_html=True,
            )
        if has_playoffs:
            st.markdown("<br>", unsafe_allow_html=True)
            po_table = build_stat_table(result, "playoffs", stat_labels=stat_labels, stat_defs=combined_stat_defs)
            po_flags = build_stat_flags(result, "playoffs", stat_labels=stat_labels, stat_defs=combined_stat_defs)
            st.markdown(
                render_stat_table_html(po_table, "Playoffs", formats=formats,
                                       lower_is_better=lower_is_better, flags=po_flags),
                unsafe_allow_html=True,
            )
        if any(lbl.endswith("/100") or lbl == "Poss/G" for lbl in stat_labels):
            st.caption(
                "**/100** = per 100 team possessions the player was on the floor for, using that "
                "team's own measured pace over these exact games — the pace-aware counterpart to "
                "/36, which can't tell a 100-possession game from an 85-possession one. Possessions "
                "come out of the team box score, so these rows are blank for NBA seasons before 1977: "
                "turnovers weren't recorded, and without them there's no possession estimate to make."
            )
        if any(lbl.endswith("/75") for lbl in stat_labels):
            st.caption(
                "**/75** = the /100 rate scaled to 75 possessions — the same pace adjustment, "
                "at roughly a starter's per-game possession count, which is the usual baseline "
                "for comparing players across eras."
            )
        if any(lbl.startswith(("Team ", "Opp ")) and lbl.endswith("/G") and lbl not in ("Team Poss/G",)
               for lbl in stat_labels):
            st.caption(
                "**Team/Opp …/G** = the player's team's and its opponents' box score, per game, over "
                "the games in this span. Each is averaged over the games that have it recorded; "
                "before 1985 (NBA) most columns are rebuilt from player logs where possible, as are "
                "WNBA team turnovers in 1997, 2000 and 2003 (footnoted)."
            )
        if len({s.league for s in valid_spans}) > 1:
            st.caption(
                "**NBA and WNBA side by side**: each row is measured against its own league -- %ile "
                "and relative shooting rank a WNBA season among WNBA players, and pace counts "
                "possessions per 48 minutes in the NBA but per 40 in the WNBA. Raw counting stats "
                "are straight comparisons, but WNBA games are 8 minutes shorter, so per-game "
                "numbers favor the NBA; /36 and /100 even that out."
            )
        if any(lbl.endswith("CV%") for lbl in stat_labels):
            st.caption(
                "CV% = game-to-game standard deviation \u00f7 mean, as a percent. Lower means more "
                "predictable output game to game; it isn't a judgment of good or bad for a given role."
            )
        if head_to_head and all(s.vs_player_ids for s in valid_spans):
            st.caption(
                "**Head-to-head**: only games where every player in both rows logged 10+ minutes, "
                "with the two rows on opposite teams, within the seasons both rows cover -- so both "
                "columns come from the same games, and W/L is the head-to-head record. A duo's "
                "column is the two players' combined numbers, as in Duo mode, and only counts games "
                "where both of them played. %ile rows show — here, since they rank whole seasons "
                "and don't describe these games. The playoff series breakdown lists only the "
                "series they met in."
            )
        if any(lbl.endswith("%ile") for lbl in stat_labels):
            st.caption(
                "%ile = where this stat ranks among qualifying players **in that same season and league** "
                "(min. 10 games regular season, 1 game playoffs) -- 90 means better than ~90% of the "
                "league that year. A span covering multiple seasons shows a games-weighted average "
                "across those seasons. Turnovers are inverted so higher %ile always means better, "
                "same as every other stat here. Duo spans show — for %ile rows -- percentiles "
                "rank against individual players, so a combined duo total isn't a meaningful comparison."
            )

    if has_playoffs:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Playoff series breakdown", expanded=False):
            st.caption(
                "Series and rounds come from the TEAM's full game log (not just this player's games), "
                "then this player's own games are matched into that structure -- so a round they missed "
                "to injury still shows up correctly (marked **(DNP)**) instead of mis-numbering every "
                "round after it or costing them championship credit. Championship detection uses that "
                "season's actual deepest round league-wide (not a hardcoded round count), so it stays "
                "correct across playoff formats with different numbers of rounds. "
                "**Home Court** is exact -- it's just whoever hosted Game 1. "
                "**Seed is an approximation** (regular-season win% rank within conference -- league-wide "
                "for WNBA seasons from 2016, when it stopped seeding by conference) -- it doesn't "
                "apply real tiebreakers or account for play-in games, so treat it as a rough signal, not "
                "an official seed. Rounds are numbered league-wide, so a team coming off a bye "
                "starts in the round it actually entered."
            )

            series_available_labels = list(SERIES_COLUMN_DEFS.keys())
            if "selected_series_cols" not in st.session_state:
                st.session_state.selected_series_cols = [
                    c for c in st.session_state.series_col_order if c in series_available_labels
                ]
            selected_series_cols = st.multiselect(
                "Series columns to show (Season/Round/Opponent/Result are always shown)",
                options=series_available_labels,
                key="selected_series_cols",
            )

            ordered_series_cols = [c for c in st.session_state.series_col_order if c in selected_series_cols]
            ordered_series_cols += [c for c in selected_series_cols if c not in ordered_series_cols]
            st.session_state.series_col_order = ordered_series_cols

            if ordered_series_cols:
                st.caption("Drag to reorder:")
                sorter_key = "series_col_sorter_" + "|".join(sorted(ordered_series_cols))
                new_series_order = sort_items(ordered_series_cols, key=sorter_key)
                if new_series_order and set(new_series_order) == set(ordered_series_cols):
                    st.session_state.series_col_order = new_series_order
                else:
                    st.session_state.series_col_order = ordered_series_cols
            series_cols = st.session_state.series_col_order

            spans_records = {
                agg["label"]: agg["playoffs"]["series_records"]
                for agg in result.aggregates
                if agg["playoffs"] is not None and "series_records" in agg["playoffs"]
            }
            rounds = round_labels_present(spans_records)
            comparison_cols = [c for c in series_cols if c not in ("Home Court", "Seed (approx)")]

            if rounds and comparison_cols:
                st.subheader("Round-by-round comparison")
                st.caption(
                    "Grouped by round LABEL (\"Finals\" always means the championship round, regardless "
                    "of how many total rounds existed that season) -- so a span with multiple runs to a "
                    "round shows its combined performance across all of them, not per-appearance. "
                    "Home Court and Seed aren't shown here since they don't aggregate across multiple "
                    "series the same way a stat does."
                )
                stat_formats, stat_lower_is_better = formats_and_lower_is_better(SERIES_COLUMN_DEFS)
                for round_label in rounds:
                    round_df = build_round_comparison_table(spans_records, round_label, comparison_cols)
                    if round_df.empty or round_df.isna().all(axis=None):
                        continue
                    st.markdown(
                        render_stat_table_html(round_df, round_label, formats=stat_formats, lower_is_better=stat_lower_is_better),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                st.divider()

            st.subheader("By player")
            for agg in result.aggregates:
                block = agg["playoffs"]
                if block is None or "series_records" not in block:
                    continue
                display_df = build_series_table(block["series_records"], columns=series_cols)
                st.markdown(
                    render_series_table_html(display_df, columns=series_cols, title=agg["label"]),
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)