"""Recompute standings from raw match results.

We deliberately derive W/L/WP ourselves from the schedule rather than trusting
the sheet's standings columns, so the engine can simulate hypothetical results.
The validation script checks our recompute against the league's own numbers.
"""
from __future__ import annotations

import pandas as pd


def compute_standings(matches: pd.DataFrame) -> pd.DataFrame:
    """Game-based standings per (tier, team) from PLAYED matches.

    Each game counts: in a series Away a - b Home, Away earns `a` wins and `b`
    losses, Home earns `b` wins and `a` losses.
    """
    # Regular-season games only: playoffs (Bo5/Bo7) don't count toward standings.
    played = matches[matches["played"] & matches["is_regular"]].copy()
    tallies: dict[tuple[str, str], list[int]] = {}

    def add(tier, team, w, l):
        key = (tier, team)
        if key not in tallies:
            tallies[key] = [0, 0]
        tallies[key][0] += w
        tallies[key][1] += l

    for row in played.itertuples(index=False):
        add(row.tier, row.away, row.away_g, row.home_g)
        add(row.tier, row.home, row.home_g, row.away_g)

    recs = []
    for (tier, team), (w, l) in tallies.items():
        gp = w + l
        recs.append({
            "tier": tier, "team": team, "w": w, "l": l,
            "gp": gp, "wp": (w / gp) if gp else 0.0,
        })
    df = pd.DataFrame(recs)
    # Rank within tier by win pct (1 = best). Ties share the min rank.
    df["rank"] = (
        df.groupby("tier")["wp"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return df.sort_values(["tier", "rank", "team"]).reset_index(drop=True)
