"""
Win Shares, Basketball-Reference's method (basketball-reference.com/about/ws.html):
credit for team wins split between the players, offensive and defensive.

    OWS = (PProd - 0.92 * lg_PPP * TotPoss) / marginal_pts_per_win
    DWS = (MP / Team_MP) * Team_Poss * (1.08 * lg_PPP - DRtg / 100) / marginal_pts_per_win
    marginal_pts_per_win = 0.32 * lg_PPG * (team_pace / lg_pace)

PProd and TotPoss (points produced, offensive possessions) and DRtg
(individual defensive rating) are Dean Oliver's, exactly as
basketball-reference.com/about/ratings.html writes them out. They need the
player's box line, the team's and the opponent's, and three league
constants -- points per possession, points per game and pace.

Basketball-Reference applies all of it to SEASON totals, one team at a time
(a traded player gets one line per team). Same here: a span's games are
grouped by (season, team), each group runs the formulas on its own summed
box lines, and the groups are added up. Team and opponent totals are the
team's whole season for a plain span, as on Basketball-Reference, but only
the span's own games for a filtered span, head-to-head or duo -- the same
convention as Team ORtg/DRtg there (see compute_win_shares). League
constants always come from the whole season of that type (regular or
playoffs), never from filtered games -- "league average" means the league.

The modern method needs player turnovers, which the NBA only recorded from
1977-78. Basketball-Reference fills 1973-77 with an age-based turnover
regression and uses cruder formulas before that; neither is attempted here
(no ages in the data), so those games get no Win Shares at all.
"""
from __future__ import annotations
from functools import lru_cache

import pandas as pd

from .data import NBADataStore, COUNTING_STATS

# Player columns every game needs for the modern method. 3PM/3PA aren't
# here: before the 3-point line they're blank but truly zero, and are filled.
PLAYER_COLS = ["MIN_NUM", "PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
BOX_COLS = ["PTS", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
TEAM_COLS = ["TEAM_MIN"] + [f"TEAM_{c}" for c in BOX_COLS]
OPP_COLS = [f"OPP_{c}" for c in BOX_COLS]


def _possessions(t: dict, o: dict) -> float:
    """Basketball-Reference's team possession estimate, averaged over both
    sides -- the one its pace, league ORtg and team DRtg are built on (and
    a little different from the simpler FGA - ORB + TOV + .44*FTA used
    elsewhere in the app, which would leave lg_PPP visibly off theirs)."""
    def side(a, b):
        orb_pct = a["OREB"] / (a["OREB"] + b["DREB"])
        return a["FGA"] + 0.4 * a["FTA"] - 1.07 * orb_pct * (a["FGA"] - a["FGM"]) + a["TOV"]
    return 0.5 * (side(t, o) + side(o, t))


def _any_estimated(rows: pd.DataFrame) -> bool:
    """Whether any of these rows' team lines were rebuilt (see data.TEAM_EST)."""
    return "TEAM_EST" in rows.columns and bool(rows["TEAM_EST"].fillna(False).astype(bool).any())


def _sums(rows: pd.DataFrame, prefix: str = "") -> dict:
    return {c: rows[f"{prefix}{c}"].sum() for c in BOX_COLS}


def _fill_pre_three(games: pd.DataFrame) -> pd.DataFrame:
    """No 3-point line before 1979-80: a blank 3PM there is a zero."""
    cols = [c for c in ("FG3M", "TEAM_FG3M", "OPP_FG3M") if c in games.columns]
    return games.assign(**{c: games[c].fillna(0) for c in cols})


@lru_cache(maxsize=128)
def league_constants(store: NBADataStore, season: int, season_type: str) -> dict | None:
    """
    lg_PPP (points per possession), lg_PPG (points per team-game) and
    lg_pace for one season, from every team game that has a full box line
    on both sides. Possessions are figured per team over its season totals
    and then added, as Basketball-Reference does, rather than game by game.
    """
    games = _fill_pre_three(store.team_games_with_opponents(season, season_type))
    valid = _with_full_lines(games)
    if valid.empty:
        return None
    poss, minutes = 0.0, 0.0
    for _, team in valid.groupby("TEAM_ID"):
        poss += _possessions(_sums(team, "TEAM_"), _sums(team, "OPP_"))
        minutes += team["TEAM_MIN"].sum()
    if not poss or not minutes:
        return None
    return {
        "lg_ppp": valid["TEAM_PTS"].sum() / poss,
        "lg_ppg": valid["TEAM_PTS"].mean(),
        "lg_pace": store.game_minutes * poss / (minutes / 5),
    }


def _group_win_shares(p: dict, t: dict, o: dict, mp: float, team_mp: float, opp_mp: float,
                      lg: dict, game_minutes: float) -> dict:
    """OWS/DWS for one (season, team) group of summed box lines: p the
    player's, t the team's, o the opponent's. Variable names follow
    Basketball-Reference's write-up so each line can be checked against it."""
    # -- Offense: points produced and individual possessions ---------------
    q_ast = ((mp / (team_mp / 5)) * (1.14 * ((t["AST"] - p["AST"]) / t["FGM"]))
             + ((((t["AST"] / team_mp) * mp * 5 - p["AST"])
                 / ((t["FGM"] / team_mp) * mp * 5 - p["FGM"]))
                * (1 - (mp / (team_mp / 5)))))
    shot_share = 0.5 * ((p["PTS"] - p["FTM"]) / (2 * p["FGA"])) * q_ast if p["FGA"] else 0.0
    fg_part = p["FGM"] * (1 - shot_share)
    team_two_pt_rate = ((t["PTS"] - t["FTM"]) - (p["PTS"] - p["FTM"])) / (2 * (t["FGA"] - p["FGA"]))
    ast_part = 0.5 * team_two_pt_rate * p["AST"]
    ft_miss_sq = (1 - p["FTM"] / p["FTA"]) ** 2 if p["FTA"] else 1.0
    ft_part = (1 - ft_miss_sq) * 0.4 * p["FTA"]

    team_scoring_poss = t["FGM"] + (1 - (1 - t["FTM"] / t["FTA"]) ** 2) * t["FTA"] * 0.4
    team_orb_pct = t["OREB"] / (t["OREB"] + o["DREB"])
    team_play_pct = team_scoring_poss / (t["FGA"] + t["FTA"] * 0.4 + t["TOV"])
    team_orb_weight = (((1 - team_orb_pct) * team_play_pct)
                       / ((1 - team_orb_pct) * team_play_pct + team_orb_pct * (1 - team_play_pct)))
    orb_part = p["OREB"] * team_orb_weight * team_play_pct
    orb_discount = 1 - (t["OREB"] / team_scoring_poss) * team_orb_weight * team_play_pct

    sc_poss = (fg_part + ast_part + ft_part) * orb_discount + orb_part
    fgx_poss = (p["FGA"] - p["FGM"]) * (1 - 1.07 * team_orb_pct)
    ftx_poss = ft_miss_sq * 0.4 * p["FTA"] if p["FTA"] else 0.0
    tot_poss = sc_poss + fgx_poss + ftx_poss + p["TOV"]

    pprod_fg = 2 * (p["FGM"] + 0.5 * p["FG3M"]) * (1 - shot_share)
    team_other_fgm = t["FGM"] - p["FGM"]
    pprod_ast = (2 * ((team_other_fgm + 0.5 * (t["FG3M"] - p["FG3M"])) / team_other_fgm)
                 * team_two_pt_rate * 0.5 * p["AST"]) if team_other_fgm else 0.0
    pprod_orb = orb_part * (t["PTS"] / team_scoring_poss)
    pprod = (pprod_fg + pprod_ast + p["FTM"]) * orb_discount + pprod_orb

    # -- Defense: stops and individual defensive rating --------------------
    dor_pct = o["OREB"] / (o["OREB"] + t["DREB"])
    dfg_pct = o["FGM"] / o["FGA"]
    fm_wt = (dfg_pct * (1 - dor_pct)) / (dfg_pct * (1 - dor_pct) + (1 - dfg_pct) * dor_pct)
    stops1 = p["STL"] + p["BLK"] * fm_wt * (1 - 1.07 * dor_pct) + p["DREB"] * (1 - fm_wt)
    opp_ft_miss_sq = (1 - o["FTM"] / o["FTA"]) ** 2
    stops2 = ((((o["FGA"] - o["FGM"] - t["BLK"]) / team_mp) * fm_wt * (1 - 1.07 * dor_pct)
               + ((o["TOV"] - t["STL"]) / team_mp)) * mp
              + (p["PF"] / t["PF"]) * 0.4 * o["FTA"] * opp_ft_miss_sq)
    stops = stops1 + stops2

    team_poss = _possessions(t, o)
    stop_pct = (stops * opp_mp) / (team_poss * mp)
    team_drtg = 100 * o["PTS"] / team_poss
    d_pts_per_scposs = o["PTS"] / (o["FGM"] + (1 - opp_ft_miss_sq) * o["FTA"] * 0.4)
    drtg = team_drtg + 0.2 * (100 * d_pts_per_scposs * (1 - stop_pct) - team_drtg)

    # -- Credit, in wins -------------------------------------------------
    team_pace = game_minutes * team_poss / (team_mp / 5)
    pts_per_win = 0.32 * lg["lg_ppg"] * (team_pace / lg["lg_pace"])
    marginal_off = pprod - 0.92 * lg["lg_ppp"] * tot_poss
    marginal_def = (mp / team_mp) * team_poss * (1.08 * lg["lg_ppp"] - drtg / 100)
    return {
        "ows": marginal_off / pts_per_win,
        "dws": marginal_def / pts_per_win,
        "pprod": pprod, "tot_poss": tot_poss, "drtg": drtg,
    }


def _with_full_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows with a complete team AND opponent box line (team minutes
    recorded as more than zero -- a 0 there is a blank, see data.py)."""
    rows = rows.dropna(subset=TEAM_COLS + OPP_COLS + ["OPP_MIN"])
    return rows[rows["TEAM_MIN"] > 0]


def _player_view(games: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """One half of a duo's combined rows as if they were that player's own:
    their A_*/B_* box columns (see NBADataStore.games_together) moved into
    the player slots, team and opponent context left as is."""
    cols = COUNTING_STATS + ["MIN_NUM"]
    stashed = [f"{side}_{c}" for side in ("A", "B") for c in cols if f"{side}_{c}" != f"{prefix}_{c}"]
    return (games.drop(columns=cols + stashed, errors="ignore")
            .rename(columns={f"{prefix}_{c}": c for c in cols}))


def compute_win_shares(
    games: pd.DataFrame, store: NBADataStore, season_type: str, full_team_seasons: bool = False,
) -> dict | None:
    """
    {"ows", "dws", "ws_per_48", "minutes", "covered_games", "games",
    "estimated"} for a span's games, or None when not one of them can carry
    Win Shares.

    full_team_seasons: take each (season, team) group's team and opponent
    totals from that team's whole season of this type, the way
    Basketball-Reference does, rather than from these games alone. Right for
    a plain season or career (it reproduces their numbers to about 0.1 WS,
    where the games-only version drifts by a few tenths for a player who
    missed games); wrong for a filtered span, head-to-head or duo, whose
    team context should be the games actually in it.

    A group whose games only partly have full team and opponent lines -- the
    1977-84 seasons, where those lines are rebuilt from player logs for some
    games and not others -- is computed over the games that have them,
    scaled up to all the player's minutes in that group, and marked
    estimated. Games with no possible Win Shares at all (before player
    turnovers were recorded) aren't extrapolated across eras: they're left
    out, and covered_games says how many games the totals stand on.

    A duo's Win Shares are each player's own, computed separately over
    their shared games and added -- WS is additive across players, but the
    formulas aren't linear in the box score, so they can't be run on the
    combined line.
    """
    if games.empty or "TEAM_MIN" not in games.columns:
        return None
    if "A_PTS" in games.columns:
        halves = [
            compute_win_shares(_player_view(games, side), store, season_type, full_team_seasons)
            for side in ("A", "B")
        ]
        if any(h is None for h in halves):
            return None
        a, b = halves
        minutes = a["minutes"] + b["minutes"]
        ows, dws = a["ows"] + b["ows"], a["dws"] + b["dws"]
        return {
            "ows": ows, "dws": dws, "minutes": minutes,
            "ws_per_48": 48 * (ows + dws) / minutes if minutes else None,
            "covered_games": min(a["covered_games"], b["covered_games"]),
            "games": len(games),
            "estimated": a["estimated"] or b["estimated"],
        }

    games = _fill_pre_three(games)
    ows = dws = covered_minutes = 0.0
    covered_games, estimated = 0, False
    for (season, team_id), group in games.groupby(["SEASON", "TEAM_ID"]):
        lg = league_constants(store, int(season), season_type)
        playable = group.dropna(subset=PLAYER_COLS)
        valid = _with_full_lines(playable)
        mp = valid["MIN_NUM"].sum()
        if lg is None or valid.empty or not mp:
            continue
        if full_team_seasons:
            team_games = _fill_pre_three(store.team_games_with_opponents(int(season), season_type))
            context = _with_full_lines(team_games[team_games["TEAM_ID"] == team_id])
        else:
            context = valid
        team_mp = context["TEAM_MIN"].sum()
        opp_mp = context["OPP_MIN"].where(context["OPP_MIN"] > 0, context["TEAM_MIN"]).sum()
        try:
            ws = _group_win_shares(
                _sums(valid), _sums(context, "TEAM_"), _sums(context, "OPP_"),
                mp, team_mp, opp_mp, lg, store.game_minutes,
            )
        except ZeroDivisionError:
            continue
        # Scale a partly covered group up to all the player's minutes in it.
        group_minutes = playable["MIN_NUM"].sum()
        scale = group_minutes / mp
        ows += ws["ows"] * scale
        dws += ws["dws"] * scale
        covered_minutes += group_minutes
        covered_games += len(playable)
        estimated |= scale > 1.0001 or _any_estimated(valid) or _any_estimated(context)
    if not covered_games:
        return None
    return {
        "ows": ows, "dws": dws, "minutes": covered_minutes,
        # Per 48 minutes in both leagues, so NBA and WNBA rows read on one
        # scale (Basketball-Reference's WNBA pages use WS/40 instead).
        "ws_per_48": 48 * (ows + dws) / covered_minutes if covered_minutes else None,
        "covered_games": covered_games,
        "games": len(games),
        "estimated": estimated,
    }
