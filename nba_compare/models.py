"""
Core data model for the comparison tool.

The atomic unit being compared is a PlayerSpan: a specific player over a
specific set of seasons. This is what lets you compare two different
players, two different eras of the SAME player, or any mix, through the
exact same code path.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PlayerSpan:
    player_id: int
    player_name: str
    seasons: list[int]          # season START years, e.g. [2015, 2016, 2017] = 2015-16, 2016-17, 2017-18 (WNBA: 2015, 2016, 2017)
    label: str | None = None    # display name, e.g. "LeBron (2015-2018)". Auto-generated if None.
    include_regular: bool = True
    include_playoffs: bool = True
    # Head-to-head: when set, only games this player played AGAINST these
    # player(s) count -- one player or a duo (see NBADataStore.games_head_to_head).
    vs_player_ids: tuple[int, ...] = ()
    vs_name: str | None = None
    # Which league's data this span is read from (see config.LEAGUES).
    league: str = "NBA"

    def __post_init__(self):
        self.seasons = sorted(set(self.seasons))
        if self.label is None:
            if len(self.seasons) == 1:
                self.label = f"{self.player_name} {season_str(self.seasons[0], self.league)}"
            else:
                self.label = (
                    f"{self.player_name} "
                    f"{season_str(self.seasons[0], self.league)}\u2013{season_str(self.seasons[-1], self.league)}"
                )
            if self.vs_name:
                self.label += f" vs. {self.vs_name}"

    @property
    def player_ids(self) -> tuple[int, ...]:
        return (self.player_id,)

    @property
    def names(self) -> str:
        return self.player_name

    @classmethod
    def single_season(cls, player_id: int, player_name: str, season: int, **kwargs) -> "PlayerSpan":
        return cls(player_id, player_name, [season], **kwargs)

    @classmethod
    def range(cls, player_id: int, player_name: str, start: int, end: int, **kwargs) -> "PlayerSpan":
        """Inclusive range of season start-years, e.g. range(..., 2015, 2018) -> 2015-16 .. 2018-19."""
        return cls(player_id, player_name, list(range(start, end + 1)), **kwargs)

    @classmethod
    def career(cls, player_id: int, player_name: str, all_seasons: list[int], **kwargs) -> "PlayerSpan":
        """Pass the full list of seasons this player appears in (e.g. from data.seasons_played)."""
        kwargs.setdefault("label", f"{player_name} (Career)")
        return cls(player_id, player_name, all_seasons, **kwargs)


@dataclass
class DuoSpan:
    """
    Two teammates over a set of seasons, compared as their COMBINED numbers
    for the games they actually shared (same GAME_ID + same TEAM_ID) --
    not their individual careers added together. `seasons` narrows which
    seasons to consider; the actual "were they teammates that game" filter
    happens at the game level (see NBADataStore.games_together).
    """
    player_a_id: int
    player_a_name: str
    player_b_id: int
    player_b_name: str
    seasons: list[int]
    label: str | None = None
    include_regular: bool = True
    include_playoffs: bool = True
    # Head-to-head, same as PlayerSpan's: only games the duo played against these player(s).
    vs_player_ids: tuple[int, ...] = ()
    vs_name: str | None = None
    league: str = "NBA"

    def __post_init__(self):
        self.seasons = sorted(set(self.seasons))
        if self.label is None:
            if len(self.seasons) == 1:
                self.label = f"{self.names} {season_str(self.seasons[0], self.league)}"
            else:
                self.label = (
                    f"{self.names} {season_str(self.seasons[0], self.league)}–"
                    f"{season_str(self.seasons[-1], self.league)}"
                )
            if self.vs_name:
                self.label += f" vs. {self.vs_name}"

    @property
    def player_ids(self) -> tuple[int, ...]:
        return (self.player_a_id, self.player_b_id)

    @property
    def names(self) -> str:
        return f"{self.player_a_name} & {self.player_b_name}"


def season_str(start_year: int, league: str = "NBA") -> str:
    """Display form of a SEASON value. The NBA season straddles two years
    ("2015-16"); the WNBA plays inside one calendar year, so its SEASON is
    just that year ("2015")."""
    if league == "WNBA":
        return str(start_year)
    return f"{start_year}-{str(start_year + 1)[-2:]}"