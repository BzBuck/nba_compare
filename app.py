"""
Interactive player/span comparison UI.

Run from the project root:
    streamlit run app.py

Lets you add any number of player + year-range "spans" (including multiple
spans of the same player), pick which stats show up, drag to reorder them,
and define your own custom stat formulas from the existing ones.
"""
import uuid
import streamlit as st
from streamlit_sortables import sort_items

from nba_compare import PlayerSpan, DuoSpan, NBADataStore, compare_spans, AccoladeStore
from nba_compare.players import search_players
from nba_compare.table import (
    build_stat_table, render_stat_table_html, build_awards_table,
    formats_and_lower_is_better, STAT_DEFS, DEFAULT_STAT_LABELS,
)
from nba_compare.formulas import safe_eval, validate_formula, flatten_block_for_formula, AVAILABLE_VARIABLES
from nba_compare.session_config import serialize_config, deserialize_config, ConfigError
from nba_compare.playoffs import (
    render_series_table_html, build_series_table, SERIES_COLUMN_DEFS, DEFAULT_SERIES_COLUMNS,
    round_labels_present, build_round_comparison_table,
)

st.set_page_config(page_title="NBA Compare", layout="wide")


# ---------- cached data access ----------

@st.cache_resource
def get_store_and_directory():
    store = NBADataStore.from_config()
    directory = store.all_players("regular")
    return store, directory


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


store, directory = get_store_and_directory()


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


def add_span():
    st.session_state.spans.append({"id": str(uuid.uuid4())})


def remove_span(span_id: str):
    st.session_state.spans = [s for s in st.session_state.spans if s["id"] != span_id]


st.title("NBA Player / Span Comparison")

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
        name = st.session_state[match_key]
        row = directory[directory.PLAYER_NAME == name]
        if row.empty:
            continue
        player_id = int(row.iloc[0].PLAYER_ID)
        seasons_played = get_seasons(store, player_id)
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
        name_a, name_b = st.session_state[matcha_key], st.session_state[matchb_key]
        row_a = directory[directory.PLAYER_NAME == name_a]
        row_b = directory[directory.PLAYER_NAME == name_b]
        if row_a.empty or row_b.empty:
            continue
        player_a_id, player_b_id = int(row_a.iloc[0].PLAYER_ID), int(row_b.iloc[0].PLAYER_ID)
        if player_a_id == player_b_id:
            continue
        overlap = get_seasons_together(store, player_a_id, player_b_id)
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
            accolade_path=st.session_state.get("accolade_path_input", "") or "",
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
            skipped_players, new_span_cfgs = [], []
            for s in cfg["spans"]:
                match = directory[directory.PLAYER_ID == s["player_id"]]
                if match.empty:
                    skipped_players.append(s.get("player_name") or f"player id {s['player_id']}")
                    continue
                canonical_name = match.iloc[0].PLAYER_NAME
                new_sid = str(uuid.uuid4())
                st.session_state[f"q_{new_sid}"] = canonical_name
                st.session_state[f"match_{new_sid}"] = canonical_name
                seasons_played = get_seasons(store, s["player_id"])
                valid_seasons = [yr for yr in s["seasons"] if yr in seasons_played] or seasons_played
                st.session_state.setdefault("_pending_ranges", {})[new_sid] = (min(valid_seasons), max(valid_seasons))
                st.session_state[f"reg_{new_sid}"] = s["include_regular"]
                st.session_state[f"po_{new_sid}"] = s["include_playoffs"]
                st.session_state[f"label_{new_sid}"] = s["label"] or ""
                new_span_cfgs.append({"id": new_sid})

            skipped_duos = []
            for d in cfg["duos"]:
                match_a = directory[directory.PLAYER_ID == d["player_a_id"]]
                match_b = directory[directory.PLAYER_ID == d["player_b_id"]]
                if match_a.empty or match_b.empty:
                    label = d.get("label") or f"{d.get('player_a_name', '?')} & {d.get('player_b_name', '?')}"
                    skipped_duos.append(label)
                    continue
                name_a, name_b = match_a.iloc[0].PLAYER_NAME, match_b.iloc[0].PLAYER_NAME
                new_sid = str(uuid.uuid4())
                st.session_state[f"mode_{new_sid}"] = "Duo"
                st.session_state[f"qa_{new_sid}"] = name_a
                st.session_state[f"matcha_{new_sid}"] = name_a
                st.session_state[f"qb_{new_sid}"] = name_b
                st.session_state[f"matchb_{new_sid}"] = name_b
                overlap = get_seasons_together(store, d["player_a_id"], d["player_b_id"])
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
            sample_vars = {v: 1.0 for v in AVAILABLE_VARIABLES}
            valid_formulas, dropped_formulas = [], []
            for f in cfg["custom_formulas"]:
                if validate_formula(f["expr"], sample_vars):
                    dropped_formulas.append(f["label"])
                else:
                    valid_formulas.append(f)
            st.session_state.custom_formulas = valid_formulas

            # Stats: only keep labels that still exist (built-ins current
            # code defines + the custom formulas just restored) -- anything
            # else (a stat renamed/removed since the save) is dropped rather
            # than crashing the picker on an option that no longer exists.
            valid_stat_names = set(STAT_DEFS.keys()) | {f["label"] for f in valid_formulas}
            restored_order = [s for s in cfg["stat_order"] if s in valid_stat_names]
            st.session_state.stat_order = restored_order or list(DEFAULT_STAT_LABELS)
            st.session_state.selected_stats = list(st.session_state.stat_order)

            if cfg["accolade_path"]:
                st.session_state["accolade_path_input"] = cfg["accolade_path"]

            msg = f"Loaded {len(new_span_cfgs)} span(s)."
            if skipped_players:
                msg += f" Skipped player(s) (not found in current data): {', '.join(skipped_players)}."
            if skipped_duos:
                msg += f" Skipped duo(s) (a player not found in current data): {', '.join(skipped_duos)}."
            if dropped_formulas:
                msg += f" Dropped invalid custom formula(s): {', '.join(dropped_formulas)}."
            st.success(msg)
            st.rerun()

st.sidebar.header("Accolades data (optional)")
accolade_path = st.sidebar.text_input(
    "Path to accolades CSV",
    key="accolade_path_input",
    help="See accolades.py for the expected columns. Leave blank to skip awards.",
)
accolade_store = AccoladeStore(accolade_path) if accolade_path else None

# ---------- custom formulas ----------

st.sidebar.header("Custom stat formulas")
st.sidebar.caption(
    "Combine existing stats with + - * / and parentheses, e.g. `PTS / USG_VOL_G` "
    "for points per used possession."
)
with st.sidebar.expander("Available variable names"):
    st.code(", ".join(AVAILABLE_VARIABLES), language=None)

with st.sidebar.form("add_formula_form", clear_on_submit=True):
    new_label = st.text_input("Stat name", placeholder="Pts per Use")
    new_expr = st.text_input("Formula", placeholder="PTS / USG_VOL_G")
    submitted = st.form_submit_button("Add formula")
    if submitted:
        sample_vars = {v: 1.0 for v in AVAILABLE_VARIABLES}  # syntax/name check only
        error = validate_formula(new_expr, sample_vars)
        if not new_label.strip():
            st.sidebar.error("Give the stat a name.")
        elif new_label in STAT_DEFS or any(f["label"] == new_label for f in st.session_state.custom_formulas):
            st.sidebar.error(f"'{new_label}' already exists — pick a different name.")
        elif error:
            st.sidebar.error(error)
        else:
            st.session_state.custom_formulas.append({"label": new_label, "expr": new_expr})
            st.session_state.stat_order.append(new_label)
            # Auto-show the new stat immediately, not just in stat_order --
            # the multiselect widget has its own persisted state (keyed by
            # "selected_stats") that must be updated directly, since passing
            # `default=` again has no effect once the widget already has state.
            if "selected_stats" in st.session_state:
                st.session_state.selected_stats = list(st.session_state.selected_stats) + [new_label]

if st.session_state.custom_formulas:
    st.sidebar.write("Your custom stats:")
    for i, f in enumerate(st.session_state.custom_formulas):
        cols = st.sidebar.columns([4, 1])
        cols[0].caption(f"**{f['label']}** = `{f['expr']}`")
        if cols[1].button("✕", key=f"del_formula_{i}"):
            removed_label = st.session_state.custom_formulas.pop(i)["label"]
            st.session_state.stat_order = [s for s in st.session_state.stat_order if s != removed_label]
            if "selected_stats" in st.session_state:
                st.session_state.selected_stats = [
                    s for s in st.session_state.selected_stats if s != removed_label
                ]
            st.rerun()

# Combined registry: built-ins + custom formulas, each custom formula's
# getter closing over its own expr (default val=expr avoids late-binding bugs).
combined_stat_defs = dict(STAT_DEFS)
for f in st.session_state.custom_formulas:
    combined_stat_defs[f["label"]] = (
        lambda block, expr=f["expr"]: safe_eval(expr, flatten_block_for_formula(block)),
        "{:.3f}",
        False,
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
                options = {f"{row.PLAYER_NAME}": row.PLAYER_ID for row in matches.itertuples()}
                chosen = cols[0].selectbox(
                    "Match", list(options.keys()), key=f"match_{sid}", label_visibility="collapsed"
                )
                player_id = int(options[chosen])
                player_name = chosen
            elif query:
                cols[0].caption("No matches")

            if player_id is not None:
                seasons = get_seasons(store, player_id)
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
                        format_func=lambda y: f"{y}-{str(y + 1)[-2:]}",
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
                options_a = {row.PLAYER_NAME: row.PLAYER_ID for row in matches_a.itertuples()}
                chosen_a = cols[0].selectbox(
                    "Match A", list(options_a.keys()), key=f"matcha_{sid}", label_visibility="collapsed"
                )
                player_a_id = int(options_a[chosen_a])
                player_a_name = chosen_a
            elif query_a:
                cols[0].caption("No matches")

            query_b = cols[1].text_input("Player B", key=f"qb_{sid}", placeholder="Type a name…")
            matches_b = search_players(query_b, directory)
            player_b_id = player_b_name = None
            if not matches_b.empty:
                options_b = {row.PLAYER_NAME: row.PLAYER_ID for row in matches_b.itertuples()}
                chosen_b = cols[1].selectbox(
                    "Match B", list(options_b.keys()), key=f"matchb_{sid}", label_visibility="collapsed"
                )
                player_b_id = int(options_b[chosen_b])
                player_b_name = chosen_b
            elif query_b:
                cols[1].caption("No matches")

            if player_a_id is not None and player_b_id is not None:
                if player_a_id == player_b_id:
                    cols[2].caption("Pick two different players")
                else:
                    overlap = get_seasons_together(store, player_a_id, player_b_id)
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
                            format_func=lambda y: f"{y}-{str(y + 1)[-2:]}",
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
                                )
                            )

            if cols[5].button("Remove", key=f"remove_{sid}"):
                remove_span(sid)
                st.rerun()

st.button("+ Add player / span", on_click=add_span)

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

    result = compare_spans(valid_spans, store)
    has_regular = any(agg["regular"] for agg in result.aggregates)
    has_playoffs = any(agg["playoffs"] for agg in result.aggregates)
    formats, lower_is_better = formats_and_lower_is_better(combined_stat_defs)

    if not stat_labels:
        st.caption("No stats selected — pick some above.")
    else:
        if has_regular:
            reg_table = build_stat_table(result, "regular", stat_labels=stat_labels, stat_defs=combined_stat_defs)
            st.markdown(
                render_stat_table_html(reg_table, "Regular Season", formats=formats, lower_is_better=lower_is_better),
                unsafe_allow_html=True,
            )
        if has_playoffs:
            st.markdown("<br>", unsafe_allow_html=True)
            po_table = build_stat_table(result, "playoffs", stat_labels=stat_labels, stat_defs=combined_stat_defs)
            st.markdown(
                render_stat_table_html(po_table, "Playoffs", formats=formats, lower_is_better=lower_is_better),
                unsafe_allow_html=True,
            )
        if any(lbl.endswith("CV%") for lbl in stat_labels):
            st.caption(
                "CV% = game-to-game standard deviation \u00f7 mean, as a percent. Lower means more "
                "predictable output game to game; it isn't a judgment of good or bad for a given role."
            )
        if any(lbl.endswith("%ile") for lbl in stat_labels):
            st.caption(
                "%ile = where this stat ranks among qualifying players **in that same season** "
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
                "**Seed is an approximation** (regular-season win% rank within conference) -- it doesn't "
                "apply real tiebreakers or account for play-in games, so treat it as a rough signal, not "
                "an official seed."
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

    awards_table = build_awards_table(valid_spans, accolade_store)
    if awards_table is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(render_stat_table_html(awards_table, "Awards & Honors"), unsafe_allow_html=True)
    elif accolade_store is None:
        st.caption("No accolades CSV loaded — add one in the sidebar to show Awards & Honors.")