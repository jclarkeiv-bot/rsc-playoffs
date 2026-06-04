"""Team-vs-team and player-vs-player comparison.

Players are compared on per-game production with percentile context within their
tier (so a Premier player isn't unfairly measured against Amateur counting
stats). Teams are compared on record, Elo strength, head-to-head this season,
and a matchup win-probability derived from Elo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .predict import train_elo, _expected, ELO_BASE

# Player stat -> (per-game?, higher-is-better)
PLAYER_STATS = {
    "Pts": (True, True), "G": (True, True), "A": (True, True),
    "S": (True, True), "SH": (True, True), "DM": (True, True),
    "SH%": (False, True), "MVP": (True, True),
}


# ---- players -----------------------------------------------------------------

def _per_game(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gp = df["GP"].replace(0, np.nan)
    for stat, (pg, _) in PLAYER_STATS.items():
        if pg and stat in df:
            df[f"{stat}/g"] = df[stat] / gp
    df["WP"] = df["W"] / gp
    return df


@dataclass
class PlayerComparison:
    a: str
    b: str
    rows: list          # per-stat dicts
    tally: dict         # {a: n_wins, b: n_wins, tie: n}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def compare_players(players: pd.DataFrame, name_a: str, name_b: str
                    ) -> PlayerComparison:
    df = _per_game(players)

    def row_for(name):
        m = df[df["Player"].astype(str) == name]
        if m.empty:
            raise KeyError(f"player {name!r} not found")
        return m.iloc[0]
    ra, rb = row_for(name_a), row_for(name_b)

    rows, tally = [], {name_a: 0, name_b: 0, "tie": 0}
    for stat, (pg, higher_better) in PLAYER_STATS.items():
        key = f"{stat}/g" if pg else stat
        va, vb = ra.get(key), rb.get(key)
        # percentile within each player's own tier
        pa = _pctile(df, ra["Tier"], key, va)
        pb = _pctile(df, rb["Tier"], key, vb)
        if pd.isna(va) or pd.isna(vb):
            edge = "?"
        elif va == vb:
            edge = "tie"; tally["tie"] += 1
        else:
            winner = name_a if (va > vb) == higher_better else name_b
            edge = winner; tally[winner] += 1
        rows.append({
            "stat": key,
            name_a: round(float(va), 3) if pd.notna(va) else None,
            f"{name_a} %ile": pa,
            name_b: round(float(vb), 3) if pd.notna(vb) else None,
            f"{name_b} %ile": pb,
            "edge": edge,
        })
    return PlayerComparison(name_a, name_b, rows, tally)


def _pctile(df, tier, col, val) -> int | None:
    if pd.isna(val) or col not in df:
        return None
    pool = df[df["Tier"] == tier][col].dropna()
    if pool.empty:
        return None
    return int(round((pool < val).mean() * 100))


# ---- teams -------------------------------------------------------------------

@dataclass
class TeamComparison:
    a: str
    b: str
    tier: str
    record_a: str
    record_b: str
    elo_a: float
    elo_b: float
    p_a_game: float          # P(A wins one game)
    series_split: tuple      # (expected A games, expected B games) of 4
    h2h: list                # past meetings this season
    summary: str


def compare_teams(season, tier: str, team_a: str, team_b: str) -> TeamComparison:
    played = season.matches[(season.matches["tier"] == tier)
                            & (season.matches["is_regular"])
                            & (season.matches["played"])]
    ratings = train_elo(played)
    ra = ratings.get((tier, team_a), ELO_BASE)
    rb = ratings.get((tier, team_b), ELO_BASE)

    from .standings import compute_standings
    st = compute_standings(season.matches)
    st = st[st["tier"] == tier].set_index("team")

    def rec(t):
        if t in st.index:
            r = st.loc[t]
            return f"{int(r.w)}-{int(r.l)} ({r.wp:.3f})"
        return "0-0"

    p_a = _expected(ra, rb)

    # head-to-head this season (regular)
    h2h = []
    for m in played.itertuples():
        if {m.away, m.home} == {team_a, team_b}:
            a_g = m.away_g if m.away == team_a else m.home_g
            b_g = m.away_g if m.away == team_b else m.home_g
            h2h.append({"day": m.match_day, "a_games": a_g, "b_games": b_g})

    fav = team_a if p_a >= 0.5 else team_b
    summary = (f"{fav} favored: {max(p_a, 1-p_a):.1%} per game; "
               f"expected 4-game split {round(4*p_a,1)}-{round(4*(1-p_a),1)} "
               f"in {team_a}'s favor."
               if p_a >= 0.5 else
               f"{fav} favored: {max(p_a, 1-p_a):.1%} per game; "
               f"expected 4-game split {round(4*(1-p_a),1)}-{round(4*p_a,1)} "
               f"in {team_b}'s favor.")

    return TeamComparison(
        a=team_a, b=team_b, tier=tier,
        record_a=rec(team_a), record_b=rec(team_b),
        elo_a=round(ra, 1), elo_b=round(rb, 1),
        p_a_game=p_a, series_split=(round(4 * p_a, 2), round(4 * (1 - p_a), 2)),
        h2h=h2h, summary=summary,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    mode = sys.argv[1] if len(sys.argv) > 1 else "players"
    if mode == "players":
        from rsc.api import RSCClient
        players = RSCClient().player_stats()
        cmp = compare_players(players, sys.argv[2], sys.argv[3])
        print(f"\n{cmp.a}  vs  {cmp.b}\n")
        print(cmp.to_frame().to_string(index=False))
        print(f"\ncategory edges -> {cmp.a}: {cmp.tally[cmp.a]}   "
              f"{cmp.b}: {cmp.tally[cmp.b]}   tie: {cmp.tally['tie']}")
    elif mode == "teams":
        from rsc.ingest import load_season
        root = Path(__file__).resolve().parents[2]
        season = load_season(root / "data" / f"{sys.argv[2]}_standings.xlsx", sys.argv[2])
        c = compare_teams(season, sys.argv[3], sys.argv[4], sys.argv[5])
        print(f"\n{c.a} vs {c.b}  ({c.tier})\n")
        print(f"  {c.a:22} {c.record_a:16} Elo {c.elo_a}")
        print(f"  {c.b:22} {c.record_b:16} Elo {c.elo_b}")
        print(f"\n  {c.summary}")
        if c.h2h:
            print(f"  head-to-head this season: {c.h2h}")
