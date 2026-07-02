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
    seasons: list[int]          # season START years, e.g. [2015, 2016, 2017] = 2015-16, 2016-17, 2017-18
    label: str | None = None    # display name, e.g. "LeBron (2015-2018)". Auto-generated if None.
    include_regular: bool = True
    include_playoffs: bool = True

    def __post_init__(self):
        self.seasons = sorted(set(self.seasons))
        if self.label is None:
            if len(self.seasons) == 1:
                self.label = f"{self.player_name} {_season_str(self.seasons[0])}"
            else:
                self.label = (
                    f"{self.player_name} "
                    f"{_season_str(self.seasons[0])}\u2013{_season_str(self.seasons[-1])}"
                )

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


def _season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"