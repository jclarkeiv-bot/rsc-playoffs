"""Build a Season live from the rscna.com API instead of the xlsx snapshot.

The matches endpoint exposes the full schedule (played + upcoming), so the
schedule-aware clinch/sim engine works the same. Score is reported Home-Away.
Per-tier config (match days, games played) is derived from the schedule;
playoff cut sizes are stable league config (PLAYOFF_SPOTS).

Team sheet metrics (RPI / strength-of-schedule / the league's own magic number)
live only in the spreadsheet, so they're absent in live mode - the team page
shows '-' for those. Everything else is recomputed from the live schedule.
"""
from __future__ import annotations

import re

import pandas as pd

from .api import RSCClient
from .ingest import Season, TIER_ORDER

# stable league config: playoff spots per tier (matches the Variables sheet)
PLAYOFF_SPOTS = {
    "Premier": 3, "Master": 5, "Elite": 8, "Veteran": 8, "Rival": 8,
    "Challenger": 8, "Prospect": 5, "Contender": 6, "Amateur": 3,
}
_SCORE = re.compile(r"(\d+)\s*-\s*(\d+)")
_TEAMS_COLS = ["tier", "team", "rpi", "past_sos", "fut_sos", "last5",
               "magic_number", "ov_rank", "ov_w", "ov_l", "ov_wp",
               "gf", "ga", "gd", "shots", "franchise"]


def _build_teams(standings: pd.DataFrame) -> pd.DataFrame:
    """Live standings -> teams frame (adds goals for/against/diff; sheet-only
    metrics like RPI/SOS stay empty)."""
    if standings is None or standings.empty:
        return pd.DataFrame(columns=_TEAMS_COLS)
    df = standings.rename(columns={
        "Team": "team", "Tier": "tier", "W": "ov_w", "L": "ov_l",
        "GP": "gp", "Win%": "ov_wp", "GF": "gf", "GA": "ga", "GD": "gd",
        "Shots": "shots", "Franchise": "franchise"})
    for c in _TEAMS_COLS:
        if c not in df.columns:
            df[c] = None
    return df[[c for c in _TEAMS_COLS if c in df.columns]]


def _parse_matches(raw: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in raw.itertuples(index=False):
        tier = getattr(r, "Tier", None)
        if not tier:
            continue
        typ = str(getattr(r, "Type", "") or "")
        is_regular = typ.strip().lower() == "regular season"
        home, away = str(r.Home).strip(), str(r.Away).strip()
        m = _SCORE.search(str(getattr(r, "Score", "") or ""))
        played = m is not None
        home_g = int(m.group(1)) if played else None   # Score = Home - Away
        away_g = int(m.group(2)) if played else None
        day = getattr(r, "Day", None)
        try:
            md = int(day)
        except (TypeError, ValueError):
            md = None
        recs.append({
            "tier": tier,
            "phase": "regular" if is_regular else (typ or "playoff"),
            "is_regular": is_regular,
            "match_day": md if is_regular else None,
            "round_label": None if is_regular else (typ or "Playoff"),
            "date": pd.to_datetime(getattr(r, "Date", None), errors="coerce"),
            "away": away, "home": home,
            "away_g": away_g, "home_g": home_g,
            "total_g": (away_g + home_g) if played else None,
            "played": played,
        })
    return pd.DataFrame(recs)


def _derive_variables(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reg = matches[matches["is_regular"]]
    for tier in TIER_ORDER:
        sub = reg[reg["tier"] == tier]
        if sub.empty:
            continue
        teams = len(set(sub["away"]).union(sub["home"]))
        match_days = int(sub["match_day"].max())
        # a match day counts as "played" only once all its matches have results
        by_day = sub.groupby("match_day")["played"].agg(["sum", "count"])
        mds_played = int((by_day["sum"] == by_day["count"]).sum())
        spots = PLAYOFF_SPOTS.get(tier, max(1, teams // 2))
        rows.append({
            "tier": tier, "teams": teams, "match_days": match_days,
            "mds_played": mds_played,
            "games_left": (match_days - mds_played) * 4,
            "playoff_spots": spots, "first_team_out": spots + 1,
        })
    return pd.DataFrame(rows)


def load_live_season(label: str = "S26", client: RSCClient | None = None) -> Season:
    api = client or RSCClient()
    matches = _parse_matches(api.matches(tier="all"))
    variables = _derive_variables(matches)
    try:
        teams = _build_teams(api.standings())
    except Exception:
        teams = pd.DataFrame(columns=_TEAMS_COLS)
    return Season(label=label, variables=variables, matches=matches, teams=teams)


if __name__ == "__main__":
    s = load_live_season()
    print("LIVE season built")
    print(s.variables.to_string(index=False))
    print(f"\nmatches: {len(s.matches)} "
          f"({int(s.matches['played'].sum())} played, "
          f"{int((~s.matches['played']).sum())} upcoming)")
