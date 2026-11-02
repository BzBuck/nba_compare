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
from st_aggrid import AgGrid, JsCode

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
    apply_preset, matching_preset, preset_from_current, restore_user_presets,
    load_user_presets, save_user_presets, user_presets_path, presets_to_json, presets_from_json,
)
from nba_compare.filters import (
    GameFilter, Condition, GAME_FIELDS, SEASON_FIELDS, OPS, OP_SYMBOLS,
    LOCATIONS, RESULTS, GAME_LENGTHS, OPPONENT_MODES, OPP_CONFERENCES,
    PLAYOFF_ROUNDS, SERIES_HOME_COURT, SERIES_SITUATIONS, VS_PLAYER_MODES, SEASON_FINISHES,
)
from nba_compare.playoffs import (
    render_series_table_html, build_series_table, SERIES_COLUMN_DEFS, DEFAULT_SERIES_COLUMNS,
    round_labels_present, build_round_comparison_table,
)
from nba_compare.league import (
    ROW_UNITS, LeagueQuery, LeagueRows, league_rows, family_values, families_for, family_is_slow,
    family_applies, plan, estimate_seconds, build_league_table, FAMILY_NAMES,
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
    # list of {"name", "stats", "formulas"}, read from the presets file once
    # per session -- see presets.py.
    _loaded, _dropped, _err = load_user_presets()
    st.session_state.user_presets = _loaded
    st.session_state["_presets_file_ok"] = _err is None
    if _err:
        st.session_state["_preset_msg"] = ("error", _err)
    elif _dropped:
        st.session_state["_preset_msg"] = (
            "warning", f"Dropped saved preset(s) whose stats no longer exist: {', '.join(_dropped)}.",
        )


# ---------- view: compare players or league table ----------

APP_MODES = {"Individual": "Compare players", "League": "League table"}
if "_pending_app_mode" in st.session_state:
    st.session_state["app_mode"] = st.session_state.pop("_pending_app_mode")
if st.session_state.get("app_mode") not in APP_MODES:
    st.session_state["app_mode"] = "Individual"
app_mode = st.session_state["app_mode"]

# The comparison's widgets aren't drawn in the league table, and Streamlit
# drops an undrawn widget's state -- which would empty every player row. So
# their values are stashed on the way into the league table and put back on
# the way out (ranges through _pending_ranges, the same way Load setup does).
SPAN_WIDGET_PREFIXES = (
    "mode", "q", "match", "reg", "po", "label", "range",
    "qa", "matcha", "qb", "matchb", "range_duo", "reg_duo", "po_duo", "label_duo",
)
if app_mode == "League":
    if "_individual_stash" not in st.session_state:
        _keys = ["league_mode", "h2h_mode"] + [
            f"{prefix}_{cfg['id']}" for cfg in st.session_state.spans for prefix in SPAN_WIDGET_PREFIXES
        ]
        st.session_state["_individual_stash"] = {k: st.session_state[k] for k in _keys if k in st.session_state}
elif "_individual_stash" in st.session_state:
    for _key, _value in st.session_state.pop("_individual_stash").items():
        if _key.startswith("range_"):
            st.session_state.setdefault("_pending_ranges", {})[_key.rsplit("_", 1)[1]] = _value
        else:
            st.session_state[_key] = _value


def set_user_presets(presets: list[dict]) -> str | None:
    """Replaces the session's presets and writes them to the presets file.
    Returns the write error, if any -- the session keeps them either way. An
    unreadable presets file found at startup is never overwritten."""
    st.session_state.user_presets = presets
    if not st.session_state.get("_presets_file_ok", True):
        return f"Not saving to {user_presets_path()} -- it couldn't be read at startup; fix or remove it first."
    return save_user_presets(presets)
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
    st.session_state.gf_vs_players = [f"{lg}:{pid}" for lg, pid, _ in gf.vs_players]
    st.session_state.gf_vs_names = {f"{lg}:{pid}": name for lg, pid, name in gf.vs_players}
    st.session_state.gf_vs_mode = gf.vs_players_mode
    st.session_state.gf_opp_finish = gf.opp_finish
    st.session_state.gf_team_finish = gf.team_finish
    st.session_state.gf_on = enabled


def game_filter_from_state() -> GameFilter:
    # Opponent players are picked as "LEAGUE:ID" keys; the names the picker
    # last showed for them are kept in gf_vs_names (see the filter expander).
    names = st.session_state.get("gf_vs_names", {})
    vs_players = tuple(
        (key.split(":")[0], int(key.split(":")[1]), names.get(key, key))
        for key in st.session_state.get("gf_vs_players", [])
    )
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
        vs_players=vs_players,
        vs_players_mode=st.session_state.get("gf_vs_mode", "Any"),
        opp_finish=st.session_state.get("gf_opp_finish", "Any"),
        team_finish=st.session_state.get("gf_team_finish", "Any"),
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
if st.session_state.get("lg_league") not in data_config.LEAGUES:
    st.session_state["lg_league"] = st.session_state.get("_lg_saved", {}).get("lg_league") or (
        league_mode if league_mode in data_config.LEAGUES else data_config.DEFAULT_LEAGUE
    )
if app_mode == "League":
    st.title(f"{st.session_state['lg_league']} League Table")
else:
    st.title(f"{'NBA / WNBA' if league_mode == 'Both' else league_mode} Player / Span Comparison")
view_cols = st.columns([1, 2])
view_cols[0].radio(
    "View", list(APP_MODES), key="app_mode", horizontal=True, format_func=APP_MODES.get,
    help="**Compare players**: the players and spans you pick, side by side. **League table**: "
         "every player in a league, one row each, sortable by any stat.",
)
if app_mode == "League":
    view_cols[1].radio(
        "League", data_config.LEAGUES, key="lg_league", horizontal=True,
        help="One league at a time -- NBA and WNBA rows in one table would need their numbers "
             "put on one scale first.",
    )
    st.session_state.setdefault("_lg_saved", {})["lg_league"] = st.session_state["lg_league"]
    directory = directory_for(st.session_state["lg_league"])
else:
    view_cols[1].radio(
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

def _comparison_state() -> dict:
    """The comparison's widget values -- from session state, or from the
    stash while the league table is showing (see _individual_stash)."""
    return {**st.session_state.to_dict(), **st.session_state.get("_individual_stash", {})}


def _build_span_dicts_from_session() -> list[dict]:
    """Reads the currently-filled-in Player-mode rows back out of
    session_state, in the plain-dict shape session_config.py expects.
    Skips any row that never got a player picked (an empty '+ Add player /
    span' row) or is in Duo mode."""
    state = _comparison_state()
    out = []
    for cfg in st.session_state.spans:
        sid = cfg["id"]
        if state.get(f"mode_{sid}", "Player") != "Player":
            continue
        match_key, range_key = f"match_{sid}", f"range_{sid}"
        if match_key not in state or range_key not in state:
            continue
        player_id = state[match_key]
        row = player_row(player_id)
        if row is None:
            continue
        name = row.PLAYER_NAME
        seasons_played = seasons_for(player_id)
        lo, hi = state[range_key]
        season_list = [s for s in seasons_played if lo <= s <= hi]
        if not season_list:
            continue
        out.append({
            "player_id": player_id,
            "player_name": name,
            "seasons": season_list,
            "label": state.get(f"label_{sid}") or None,
            "include_regular": state.get(f"reg_{sid}", True),
            "include_playoffs": state.get(f"po_{sid}", True),
        })
    return out


def _build_duo_dicts_from_session() -> list[dict]:
    """Same idea as _build_span_dicts_from_session(), but for Duo-mode rows."""
    state = _comparison_state()
    out = []
    for cfg in st.session_state.spans:
        sid = cfg["id"]
        if state.get(f"mode_{sid}", "Player") != "Duo":
            continue
        matcha_key, matchb_key, range_key = f"matcha_{sid}", f"matchb_{sid}", f"range_duo_{sid}"
        if matcha_key not in state or matchb_key not in state or range_key not in state:
            continue
        player_a_id, player_b_id = state[matcha_key], state[matchb_key]
        row_a, row_b = player_row(player_a_id), player_row(player_b_id)
        if row_a is None or row_b is None or player_a_id == player_b_id:
            continue
        name_a, name_b = row_a.PLAYER_NAME, row_b.PLAYER_NAME
        overlap = seasons_together_for(player_a_id, player_b_id)
        lo, hi = state[range_key]
        season_list = [s for s in overlap if lo <= s <= hi]
        if not season_list:
            continue
        out.append({
            "player_a_id": player_a_id,
            "player_a_name": name_a,
            "player_b_id": player_b_id,
            "player_b_name": name_b,
            "seasons": season_list,
            "label": state.get(f"label_duo_{sid}") or None,
            "include_regular": state.get(f"reg_duo_{sid}", True),
            "include_playoffs": state.get(f"po_duo_{sid}", True),
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
            head_to_head=_comparison_state().get("h2h_mode", False),
            league_mode=_comparison_state().get("league_mode", league_mode),
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
            # only the rows that still exist (see restore_user_presets). They're
            # merged into the presets already saved -- one in the code replaces
            # a saved one of the same name -- rather than replacing them all,
            # since those also live in the presets file.
            restored_presets, dropped_presets = restore_user_presets(cfg["user_presets"])
            preset_save_err = None
            if restored_presets:
                preset_save_err = set_user_presets(
                    restore_user_presets(st.session_state.user_presets + restored_presets)[0]
                )

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
            # A save is a set of players: show them, and don't let rows stashed
            # by the league table (see above) overwrite what was just loaded.
            st.session_state["_pending_app_mode"] = "Individual"
            st.session_state.pop("_individual_stash", None)
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
            if preset_save_err:
                msg += f" {preset_save_err}"
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

# ---------- sections shared by both views ----------
# The game filters, stat presets/picker and table notes are the same widgets
# in the comparison and the league table -- same keys, so whatever is set up
# in one view is still set up in the other.

INDIVIDUAL_FILTER_INTRO = (
    "Only count games that pass **every** condition below -- for all rows, regular season "
    "and playoffs alike. Opponent W% and seed are the opponent's regular-season record "
    "that season. In head-to-head, conditions are read from the first row's side "
    "(a \"W\" is a game it won), and player conditions like MIN must hold for both rows."
)


def current_game_filter() -> GameFilter:
    return game_filter_from_state() if st.session_state.get("gf_on", True) else GameFilter()


def render_game_filters(filter_leagues: set[str], directory: pd.DataFrame, show_league: bool, intro: str):
    """The "Game filters" expander. filter_leagues: whose teams the opponent
    picker lists; show_league: tag opponent players with their league (when
    the player search covers both)."""
    with st.expander("Game filters", expanded=False):
        st.caption(intro)
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
        st.markdown("**Opponent players & season finish**")
        st.caption(
            "Like head-to-head, but for any number of rows: only games where the players picked here "
            "were on the **other** team with 10+ minutes. *Season finish* is how far a team went in "
            "that season's playoffs, counted back from the Finals like the rounds below (*Last 4* = "
            "the conference finals in a 4-round NBA year) and cumulative (*Made last 4* includes the "
            "Finalists) -- so \"opponent won title\" also keeps regular-season games against the "
            "eventual champion."
        )
        # Picker keys are "LEAGUE:ID", since the two leagues' ids are separate
        # lists. A player picked under another league setting stays selected,
        # named from gf_vs_names.
        vs_names = dict(zip(directory.LEAGUE + ":" + directory.PLAYER_ID.astype(str), directory.PLAYER_NAME))
        vs_names |= {k: v for k, v in st.session_state.get("gf_vs_names", {}).items() if k not in vs_names}
        vs_options = list(vs_names)
        vs_cols = st.columns([1, 4])
        vs_cols[0].radio(
            "Match", VS_PLAYER_MODES, key="gf_vs_mode",
            format_func={"Any": "vs. any of", "All": "vs. all of", "None": "vs. none of"}.get,
            help="*All of* keeps games where every player picked was on the other team at once.",
        )
        vs_cols[1].multiselect(
            "Opponent players", vs_options, key="gf_vs_players", placeholder="Type a name…",
            format_func=lambda k: f"{vs_names[k]} ({k.split(':')[0]})" if show_league else vs_names[k],
        )
        st.session_state.gf_vs_names = {k: vs_names[k] for k in st.session_state.get("gf_vs_players", [])}
        fin_cols = st.columns(2)
        fin_cols[0].selectbox(
            "Opponent's season", list(SEASON_FINISHES), key="gf_opp_finish", format_func=SEASON_FINISHES.get,
            help="Only games against teams that went this far that season.",
        )
        fin_cols[1].selectbox(
            "Own team's season", list(SEASON_FINISHES), key="gf_team_finish", format_func=SEASON_FINISHES.get,
            help="Only games for a team (this row's team) that went this far that season -- e.g. a "
                 "player's title runs.",
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
        err = set_user_presets(others + [preset])
        st.session_state.new_preset_name = ""
        st.session_state["_preset_msg"] = ("error", err) if err else (
            "success", f"{'Updated' if replaced else 'Saved'} preset '{name}'.",
        )


def _delete_user_preset(name: str):
    err = set_user_presets([p for p in st.session_state.user_presets if p["name"] != name])
    if err:
        st.session_state["_preset_msg"] = ("error", err)


def _import_presets_file():
    upload = st.session_state.get("preset_import")
    if upload is None:
        return
    try:
        imported, dropped = presets_from_json(upload.getvalue())
    except ValueError as e:
        st.session_state["_preset_msg"] = ("error", str(e))
        return
    err = set_user_presets(restore_user_presets(st.session_state.user_presets + imported)[0])
    msg = f"Imported {len(imported)} preset(s)" + (f": {', '.join(p['name'] for p in imported)}." if imported else ".")
    if dropped:
        msg += f" Dropped (no stats left, or named like a built-in): {', '.join(dropped)}."
    st.session_state["_preset_msg"] = ("error", err) if err else ("success", msg)


def render_stat_picker(view: str, format_stat=None, picker_note: str | None = None) -> list[str]:
    """Stat presets and the "Customize stats shown" picker. Returns the chosen
    stats in order. format_stat: how an option reads in the picker (the
    league table marks slow ones); picker_note: a caption under it."""
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

        # Options that read differently (the league table marks slow stats)
        # are a new widget to Streamlit, which would start it empty -- so it's
        # re-seeded from stat_order whenever the view changes, the same way
        # adding a formula re-seeds it.
        if "selected_stats" not in st.session_state or st.session_state.get("_picker_view") != view:
            st.session_state.selected_stats = [s for s in st.session_state.stat_order if s in available_labels]
            st.session_state["_picker_view"] = view

        selected = st.multiselect(
            "Stats to show (box score, usage, team, consistency, and your custom stats)",
            options=available_labels,
            key="selected_stats",
            format_func=format_stat or str,
        )
        if picker_note:
            st.caption(picker_note)

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
                 f"Saving under an existing name updates it. Presets are saved to "
                 f"{user_presets_path()} and come back after a refresh; they're also kept in your save code.",
        )
        preset_cols[1].button("Save preset", on_click=_save_user_preset, use_container_width=True)
        msg = st.session_state.pop("_preset_msg", None)
        if msg:
            {"error": st.error, "warning": st.warning}.get(msg[0], st.success)(msg[1])
        for p in st.session_state.user_presets:
            cols = st.columns([4, 1])
            cols[0].caption(f"**{p['name']}** — {len(p['stats'])} stats")
            cols[1].button("✕", key=f"del_user_preset_{p['name']}",
                           on_click=_delete_user_preset, args=(p["name"],))
        io_cols = st.columns(2)
        io_cols[0].download_button(
            "Export presets", data=presets_to_json(st.session_state.user_presets),
            file_name="nba_compare_presets.json", mime="application/json",
            disabled=not st.session_state.user_presets, use_container_width=True,
            help="Download your presets as a file, to back them up or move them to another machine.",
        )
        io_cols[1].file_uploader(
            "Import presets", type=["json"], key="preset_import", on_change=_import_presets_file,
            help="Add presets from an exported file. One with the same name as a saved preset replaces it.",
        )
    return stat_labels


def render_stat_notes(stat_labels: list[str], mixed_leagues: bool = False, head_to_head: bool = False):
    """The explanatory notes under a stat table, for whichever stat families it shows."""
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
    if any(lbl.startswith("Lg ") for lbl in stat_labels):
        st.caption(
            "**Lg …** = the league-average team's box score per game (and pace, TS%), over the whole "
            "league for this span's seasons, weighted by the span's games in each season. Never "
            "narrowed by game filters, and not highlighted -- it's context, mostly for formulas."
        )
    if any(lbl.endswith(("(P25)", "(P75)")) for lbl in stat_labels):
        st.caption(
            "**(P25)/(P75)** = the 25th and 75th percentile of the game log: half of the span's "
            "games fall between them."
        )
    if any(lbl in ("OWS", "DWS", "WS", "WS/48") for lbl in stat_labels):
        st.caption(
            "**Win Shares** = Basketball-Reference's method: credit for team wins split into "
            "offense (OWS, from Dean Oliver's points produced) and defense (DWS, from his "
            "individual defensive rating), against that season's league averages. A plain "
            "season or career uses the team's full-season totals, as Basketball-Reference does; "
            "a filtered span, head-to-head or duo uses only its own games, so it's the wins "
            "earned in those games. A duo's is the two players' own Win Shares added. WS/48 is "
            "per 48 minutes in the WNBA too (Basketball-Reference shows WS/40 there). Blank for "
            "NBA seasons before 1977-78, which have no player turnovers."
        )
    if mixed_leagues:
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
    if head_to_head:
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


# ---------- league table ----------
# The whole league in one sortable table, one row per player (or per
# player-season) -- see league.py for how it's built and priced.

LEAGUE_FILTER_INTRO = (
    "Only count games that pass **every** condition below, for every player in the table. "
    "Season qualifiers keep or drop each player's seasons one at a time. Opponent W% and "
    "seed are the opponent's regular-season record that season."
)
SEASON_TYPE_NAMES = {"regular": "Regular season", "playoffs": "Playoffs"}

# (query, family) -> family_values. One dict shared by every session rather
# than a cached function per family, because plan() has to know which
# families are already computed -- those cost nothing to show again.
LEAGUE_FAMILY_CACHE_SIZE = 64


@st.cache_resource
def get_league_family_cache() -> dict:
    return {}


@st.cache_resource(max_entries=4, show_spinner=False)
def get_league_rows(query: LeagueQuery) -> LeagueRows:
    return league_rows(get_stores()[query.league], query)


@st.cache_resource
def get_league_seasons(league: str) -> list[int]:
    return get_stores()[league].seasons("regular")


def league_family_values(rows: LeagueRows, family: str, on_progress=None) -> dict:
    cache = get_league_family_cache()
    key = (rows.query, family)
    if key not in cache:
        cache[key] = family_values(rows, family, on_progress)
        while len(cache) > LEAGUE_FAMILY_CACHE_SIZE:
            cache.pop(next(iter(cache)))
    return cache[key]


def _restore_league_widget(key: str, default):
    """League-table widgets aren't drawn in the comparison view, and Streamlit
    drops an undrawn widget's state -- so each one's value is kept in
    _lg_saved and put back before it's drawn again."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state.setdefault("_lg_saved", {}).get(key, default)


def _remember_league_widgets(*keys: str):
    st.session_state.setdefault("_lg_saved", {}).update({k: st.session_state[k] for k in keys})


def _confirm_league_families(query: LeagueQuery, held: frozenset):
    st.session_state["_lg_confirmed"] = (query, held)


# The league table is an AG Grid rather than st.dataframe for one reason: its
# Rk column. st.dataframe sorts in the browser without telling Python the
# order, so a rank computed here would ride along with its row when a header
# is clicked. AG Grid's is read off each row's displayed position, and
# refreshed after every sort -- 1 is always the top row, however it's sorted.
_RANK_GETTER = JsCode("function(p) { return p.node.rowIndex == null ? null : p.node.rowIndex + 1; }")
_RENUMBER = JsCode("function(e) { e.api.refreshCells({columns: ['rk'], force: true}); }")
# A title row (league.LeagueRows: a player-season the player won it in, or
# a series in a title run) is shaded gold. Translucent, so it reads in the
# light and dark themes alike.
CHAMPION_GOLD = "rgba(212, 175, 55, 0.30)"
# Blank cells (a stat not recorded, or not loaded yet) sort last either way.
# AG Grid flips the result for a descending sort, hence isDesc.
_BLANKS_LAST = JsCode("""function(a, b, nodeA, nodeB, isDesc) {
    var aBlank = a == null || isNaN(a), bBlank = b == null || isNaN(b);
    if (aBlank && bBlank) return 0;
    if (aBlank) return isDesc ? -1 : 1;
    if (bBlank) return isDesc ? 1 : -1;
    return a - b;
}""")


def _number_formatter(fmt: str) -> JsCode:
    """A table format ("{:.1f}", "{:+.3f}") as an AG Grid valueFormatter."""
    spec = fmt[2:-1] if fmt.startswith("{:") and fmt.endswith("}") else ".3f"
    signed = spec.startswith("+")
    decimals = int(spec.lstrip("+").lstrip(".").rstrip("f") or 0)
    sign = "(p.value > 0 ? '+' : '') + " if signed else ""
    return JsCode(
        f"function(p) {{ return (p.value == null || isNaN(p.value)) ? '\u2014' : {sign}p.value.toFixed({decimals}); }}"
    )


def render_league_grid(table: pd.DataFrame, stat_cols: list[str], formats: dict, sort_col: str | None,
                       descending: bool, height: int):
    # Columns go to AG Grid under plain ids: a stat label like "PTS/G" or a
    # formula name with a dot would read as a nested field path.
    ids = {col: f"c{i}" for i, col in enumerate(table.columns)}
    champion = ids["Champion"]
    column_defs = [{
        "headerName": "Rk", "colId": "rk", "valueGetter": _RANK_GETTER, "sortable": False,
        "pinned": "left", "width": 64, "suppressMovable": True, "cellStyle": {"opacity": 0.6},
    }]
    for col in table.columns:
        spec = {"headerName": col, "field": ids[col], "headerTooltip": col}
        if col in stat_cols:
            spec |= {
                "type": "rightAligned", "comparator": _BLANKS_LAST,
                "valueFormatter": _number_formatter(formats.get(col, DEFAULT_FORMULA_FMT)),
                # Wide enough for the header plus its sort arrow.
                "width": max(72, 8 * len(col) + 44),
            }
            if col == sort_col:
                spec["sort"] = "desc" if descending else "asc"
        elif col == "Champion":
            spec["hide"] = True  # read by getRowStyle, not shown
        else:
            # Rk and Player stay put while the rest scrolls sideways.
            spec |= {
                "pinned": "left" if col == "Player" else None,
                "width": {"Player": 190, "Season": 90, "Round": 120, "Opp": 70, "Series": 80}.get(col, 100),
            }
        column_defs.append(spec)
    AgGrid(
        table.rename(columns=ids),
        gridOptions={
            "columnDefs": column_defs,
            "defaultColDef": {"sortable": True, "resizable": True},
            # Every column in the DOM, not just those scrolled into view --
            # a table is at most a few dozen columns.
            "suppressColumnVirtualisation": True,
            "onSortChanged": _RENUMBER,
            "onFilterChanged": _RENUMBER,
            "getRowStyle": JsCode(
                f"function(p) {{ return p.data && p.data.{champion} ? {{background: '{CHAMPION_GOLD}'}} : null; }}"
            ),
            "suppressCellFocus": True,
        },
        height=height, theme="streamlit", allow_unsafe_jscode=True, update_on=[],
        # A new set of columns starts a fresh grid (and its default sort).
        key="lg_grid_" + "|".join(table.columns),
    )


def render_league_view():
    league = st.session_state["lg_league"]
    all_seasons = get_league_seasons(league)
    saved = st.session_state.setdefault("_lg_saved", {})
    st.caption(
        "Every player's numbers for the seasons below, one row each -- click a column header to "
        "sort. Stats, presets, custom formulas and game filters are the same ones the comparison uses."
    )

    cols = st.columns([2, 4, 2])
    _restore_league_widget("lg_season_type", "regular")
    season_type = cols[0].radio(
        "Games", list(SEASON_TYPE_NAMES), key="lg_season_type", horizontal=True, format_func=SEASON_TYPE_NAMES.get,
    )
    # One remembered range per league -- their seasons aren't the same list.
    range_key = f"lg_seasons_{league}"
    default_range = saved.get(range_key)
    if not default_range or not set(default_range) <= set(all_seasons):
        default_range = (all_seasons[-1], all_seasons[-1])
    lo, hi = cols[1].select_slider(
        "Seasons", options=all_seasons, value=default_range, key=range_key,
        format_func=lambda y: season_str(y, league),
    )
    seasons = tuple(s for s in all_seasons if lo <= s <= hi)
    # Series rows only exist for playoff games -- back to player-seasons
    # when the table switches to the regular season.
    units = [u for u in ROW_UNITS if u != "series" or season_type == "playoffs"]
    _restore_league_widget("lg_unit", "player")
    if st.session_state["lg_unit"] not in units:
        st.session_state["lg_unit"] = "season"
    unit = cols[2].radio(
        "Rows", units, key="lg_unit", format_func={
            "player": "Player", "season": "Season", "series": "Series",
        }.get,
        help="**Player**: each player's games across the seasons picked, in one row. **Season**: "
             "one row per player-season. **Series** (playoffs): one row per playoff series a player "
             "played in. A gold row won the title: a player-season of a champion, or a series in a "
             "title run (a Player row only when the table is one season).",
    )

    gp_key = f"lg_min_gp_{season_type}"
    _restore_league_widget(gp_key, 10 if season_type == "regular" else 1)
    _restore_league_widget("lg_min_mpg", 0.0)
    _restore_league_widget("lg_search", "")
    row_cols = st.columns([1, 1, 2, 3])
    min_games = row_cols[0].number_input(
        "Min games", min_value=0, step=1, key=gp_key,
        help="Leave out rows with fewer games (counted after the game filter) -- otherwise a "
             "one-game 1.000 TS% tops every rate column.",
    )
    min_minutes = row_cols[1].number_input("Min MIN/G", min_value=0.0, step=1.0, key="lg_min_mpg")
    search = row_cols[2].text_input("Find player", key="lg_search", placeholder="Type a name…")
    _remember_league_widgets("lg_season_type", "lg_unit", gp_key, "lg_min_mpg", "lg_search")
    saved[range_key] = (lo, hi)

    render_game_filters({league}, directory, False, LEAGUE_FILTER_INTRO)
    game_filter = current_game_filter()

    formulas = {f["label"]: f["expr"] for f in st.session_state.custom_formulas}

    def mark_slow(label: str) -> str:
        return f"{label}  ⏱" if any(family_is_slow(f) for f in families_for([label], formulas)) else label

    stat_labels = render_stat_picker(
        "League", format_stat=mark_slow,
        picker_note="⏱ = slow to compute for a whole league. A table that would take more than a "
                    f"few seconds holds those back until you ask for them.",
    )
    if not stat_labels:
        st.caption("No stats selected — pick some above.")
        return

    query = LeagueQuery(league, seasons, season_type, unit, game_filter, int(min_games), float(min_minutes))
    with st.spinner(f"Gathering every {league} player's games…"):
        rows = get_league_rows(query)
    if not len(rows):
        st.info("No players left -- loosen the minimums or the game filter.")
        return

    # Families already computed for these rows cost nothing; plan() holds
    # back the expensive rest when they'd make the table slow.
    needed = families_for(stat_labels, formulas)
    cache = get_league_family_cache()
    done = {f for f in needed if (query, f) in cache}
    run_now, held, held_seconds = plan(rows, needed, done)
    if held and st.session_state.get("_lg_confirmed") == (query, frozenset(held)):
        run_now, held = run_now | held, set()

    parts = {}
    to_compute = sorted(run_now, key=lambda f: estimate_seconds([f], len(rows), rows.player_seasons))
    bar = st.progress(0.0) if to_compute else None
    for i, family in enumerate(to_compute):
        text = f"Computing {FAMILY_NAMES[family]} for {len(rows):,} rows…"
        bar.progress(i / len(to_compute), text=text)
        parts[family] = league_family_values(
            rows, family, lambda frac, i=i, text=text: bar.progress((i + frac) / len(to_compute), text=text),
        )
    if bar is not None:
        bar.empty()
    for family in done:
        parts[family] = league_family_values(rows, family)

    if held:
        held_stats = [label for label in stat_labels if families_for([label], formulas) & held]
        with st.container(border=True):
            st.warning(
                f"**Not loaded yet:** {', '.join(held_stats)} -- they need "
                f"{', '.join(FAMILY_NAMES[f] for f in sorted(held))}, about **{held_seconds:.0f} s** "
                f"for {len(rows):,} rows. Their columns are blank until loaded; everything else is ready."
            )
            st.button(
                f"Load them (~{held_seconds:.0f} s)", type="primary",
                on_click=_confirm_league_families, args=(query, frozenset(held)),
            )

    # Series logs only exist in the playoffs and seeds only in the regular
    # season -- blank by definition here, so say where they are instead.
    other_type = {f for f in needed if f in ("depth", "avg_seed") and not family_applies(f, query)}
    if other_type:
        other = "playoffs" if season_type == "regular" else "regular"
        st.caption(
            f"{', '.join(l for l in stat_labels if families_for([l], formulas) & other_type)} "
            f"only exist for {SEASON_TYPE_NAMES[other].lower()} games -- switch **Games** to "
            f"{SEASON_TYPE_NAMES[other]} to fill them in."
        )

    table = build_league_table(rows, parts, stat_labels, formulas).reset_index(drop=True)
    stat_cols = [c for c in table.columns if c not in rows.info.columns]

    teams = sorted({t for cell in table["Team"] for t in cell.split("/") if t})
    picked_teams = row_cols[3].multiselect("Teams", teams, key=f"lg_teams_{league}", placeholder="Every team")
    shown = table
    if search:
        shown = shown[shown["Player"].str.contains(search, case=False, na=False, regex=False)]
    if picked_teams:
        shown = shown[shown["Team"].map(lambda cell: bool(set(cell.split("/")) & set(picked_teams)))]

    formats, lower_is_better = formats_and_lower_is_better(combined_stat_defs)
    sort_col = ("PTS/G" if "PTS/G" in stat_cols else stat_cols[0]) if stat_cols else None
    if sort_col:
        shown = shown.sort_values(sort_col, ascending=sort_col in lower_is_better, na_position="last")

    what = f"{SEASON_TYPE_NAMES[season_type].lower()}, " + (
        season_str(seasons[0], league) if len(seasons) == 1
        else f"{season_str(seasons[0], league)} to {season_str(seasons[-1], league)}"
        + (", seasons combined" if unit == "player" else "")
    )
    noun = {"player": "players", "season": "player-seasons", "series": "player-series"}[unit]
    count = f"{len(shown):,} of {len(rows):,} {noun}" if len(shown) != len(rows) else f"{len(rows):,} {noun}"
    titles = int(shown["Champion"].sum())
    st.caption(f"**{count}** · {what} · min {int(min_games)} GP" + (
        f", {min_minutes:g} MIN/G" if min_minutes else "") + (
        f" · :orange-background[gold] = won the title ({titles:,} rows)" if titles else ""))
    if unit == "series":
        st.caption(
            "Each row is one series: %ile is blank (it ranks whole seasons), and Win Shares and the "
            "series columns count only that series' games."
        )
    if game_filter.is_active:
        note = (
            " %ile rows show — while game conditions are on, since they rank whole seasons."
            if query.narrowed else ""
        )
        st.info(
            "**Game filter on** — " + " · ".join(game_filter.describe())
            + f"  \nKept {rows.games_kept:,} of {rows.games_total:,} games." + note
        )

    render_league_grid(
        shown, stat_cols, formats, sort_col, descending=sort_col not in lower_is_better,
        height=min(42 * (len(shown) + 1) + 10, 740),
    )
    render_stat_notes(stat_labels)
    st.caption(
        "The estimated / partial-coverage footnote marks aren't shown here -- add the player under "
        "*Compare players* to see which of their numbers carry one."
    )


if app_mode == "League":
    render_league_view()
    st.stop()

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

render_game_filters(
    {s.league for s in valid_spans} or (set(data_config.LEAGUES) if league_mode == "Both" else {league_mode}),
    directory, league_mode == "Both", INDIVIDUAL_FILTER_INTRO,
)

game_filter = current_game_filter()

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



if len(valid_spans) == 0:
    st.info("Add at least one player above to see a comparison.")
else:
    stat_labels = render_stat_picker("Individual")

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
        render_stat_notes(
            stat_labels, mixed_leagues=len({s.league for s in valid_spans}) > 1,
            head_to_head=head_to_head and all(s.vs_player_ids for s in valid_spans),
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