"""Team and player profiles built from the live player feed.

The rscna.com player feed carries every player's Tier + Team, so rosters,
team stat totals, player pages and search all derive from one cached pull.
"""
from __future__ import annotations

import pandas as pd

from .api import RSCClient
from .engine.compare import _per_game, _pctile

_client = None
_players = None

_PERGAME = ["Pts", "G", "A", "S", "SH", "DM"]
_LABELS = {"Pts": "Points", "G": "Goals", "A": "Assists",
           "S": "Saves", "SH": "Shots", "DM": "Demos"}


_rated = None


def players(refresh: bool = False) -> pd.DataFrame:
    global _client, _players, _rated
    if _players is None or refresh:
        _client = _client or RSCClient()
        _players = _client.player_stats()
        _rated = None
    return _players


def rated() -> pd.DataFrame:
    """Player feed augmented with composite rating columns (OVR, tier_pct,
    overskilled). Cached."""
    global _rated
    if _rated is None:
        from . import rating
        r = rating.compute_ratings(players())
        key = ["Player", "Tier", "Team"]
        _rated = players().merge(
            r[key + ["OVR", "tier_pct", "tier_z", "overskilled",
                     "projected_tier", "tier_delta", "next_tier"]],
            on=key, how="left")
    return _rated


# ---- teams -------------------------------------------------------------------

def team_roster(tier: str, team: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _per_game(players() if df is None else df)
    r = df[(df["Tier"] == tier) & (df["Team"] == team)]
    return r.sort_values("Pts", ascending=False)


def roster_with_ratings(tier: str, team: str) -> list[dict]:
    """Roster rows enriched with OVR and role, for team comparison."""
    from . import project
    r = rated()
    sub = r[(r["Tier"] == tier) & (r["Team"] == team)].sort_values(
        "OVR", ascending=False, na_position="last")
    out = []
    for rec in sub.itertuples():
        role = project.player_role(players(), rec.Player)
        out.append({
            "Player": rec.Player, "GP": int(rec.GP),
            "OVR": (int(rec.OVR) if pd.notna(rec.OVR) else None),
            "role": role["role"] if role else "-",
        })
    return out


def team_totals(tier: str, team: str, df: pd.DataFrame | None = None) -> dict:
    src = players() if df is None else df
    r = src[(src["Tier"] == tier) & (src["Team"] == team)]
    if r.empty:
        return {}
    gp = int(r["GP"].max())  # team games ~ a player's GP
    out = {"n_players": len(r), "gp": gp}
    for s in _PERGAME + ["MVP"]:
        out[s] = int(r[s].sum())
    # offense/defense rates per team-game
    out["goals_for_pg"] = round(r["G"].sum() / gp, 2) if gp else 0
    out["saves_pg"] = round(r["S"].sum() / gp, 2) if gp else 0
    return out


# ---- players -----------------------------------------------------------------

# stats offerable on the leaderboard: key -> (label, can-be-per-game)
LEADERBOARD_STATS = {
    "OVR": ("Overall Rating", False),
    "Pts": ("Points", True), "G": ("Goals", True), "A": ("Assists", True),
    "S": ("Saves", True), "SH": ("Shots", True), "DM": ("Demos", True),
    "MVP": ("MVPs", True), "SH%": ("Shot %", False), "W": ("Wins", False),
}


def leaderboard(tier: str = "all", stat: str = "OVR", per_game: bool = True,
                limit: int = 100, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Players ranked by a stat, optionally filtered to one tier and/or scored
    per game. Returns columns: rank, Player, Tier, Team, GP, value."""
    src = _per_game(rated() if df is None else df)
    if tier and tier != "all":
        src = src[src["Tier"] == tier]
    if stat not in LEADERBOARD_STATS:
        stat = "Pts"
    can_pg = LEADERBOARD_STATS[stat][1]
    col = f"{stat}/g" if (per_game and can_pg and f"{stat}/g" in src) else stat
    out = src[["Player", "Tier", "Team", "GP"]].copy()
    out["value"] = src[col]
    out = out.dropna(subset=["value"])
    out = out.sort_values("value", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out.head(limit)


def find_players(query: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    src = players() if df is None else df
    q = (query or "").strip().lower()
    if not q:
        return src.iloc[0:0]
    mask = src["Player"].astype(str).str.lower().str.contains(q, regex=False)
    return src[mask].sort_values("Pts", ascending=False)


def find_teams(query: str) -> list[dict]:
    """Distinct (Tier, Team) whose team name matches the query."""
    q = (query or "").strip().lower()
    if not q:
        return []
    df = players()[["Tier", "Team"]].drop_duplicates()
    mask = df["Team"].astype(str).str.lower().str.contains(q, regex=False)
    hits = df[mask].sort_values(["Team", "Tier"])
    return [{"tier": r.Tier, "team": r.Team} for r in hits.itertuples()]


def player_profile(name: str, df: pd.DataFrame | None = None) -> dict | None:
    full = _per_game(players() if df is None else df)
    rows = full[full["Player"].astype(str) == name]
    if rows.empty:
        return None
    # if a name appears more than once, take the most-played entry.
    rec = rows.sort_values("GP", ascending=False).iloc[0]
    stats = []
    for s in _PERGAME:
        key = f"{s}/g"
        val = float(rec[key]) if pd.notna(rec[key]) else None
        stats.append({
            "key": s, "label": _LABELS[s],
            "per_game": round(val, 2) if val is not None else None,
            "total": int(rec[s]),
            "pctile": _pctile(full, rec["Tier"], key, rec[key]),
        })
    gp = int(rec["GP"]) or 1
    return {
        "name": name, "tier": rec["Tier"], "team": rec["Team"],
        "gp": int(rec["GP"]), "w": int(rec["W"]), "l": int(rec["L"]),
        "wp": round(float(rec["W"]) / gp, 3),
        "mvp": int(rec["MVP"]), "mvp_rate": round(float(rec["MVP"]) / gp, 3),
        "sh_pct": round(float(rec["SH%"]), 3) if pd.notna(rec["SH%"]) else None,
        "pts": int(rec["Pts"]),
        "stats": stats,
        "duplicate_entries": len(rows),
    }
