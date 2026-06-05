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
        from . import store
        store.save("players", _players)
    return _players


def invalidate_ratings() -> None:
    """Drop the cached ratings so they recompute (e.g. after a stats refresh)."""
    global _rated
    _rated = None


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
    r = df[(df["Tier"] == tier) & (df["Team"] == team) & (df["GP"] >= 1)]
    return r.sort_values("Pts", ascending=False)


TEAM_METRICS = {
    "avg_ovr": ("Avg player rating", "Average Overall Rating of the roster (cross-tier comparable)."),
    "wp": ("Win %", "Game win percentage (within-tier)."),
    "goals_pg": ("Goals / game", "Team goals per game played."),
    "saves_pg": ("Saves / game", "Team saves per game played."),
    "assists_pg": ("Assists / game", "Team assists per game played."),
    "rpi": ("Rating index", "Ratings Percentage Index from the league sheet."),
    "sos": ("Schedule strength", "Average win% of opponents faced so far."),
}


def team_strength() -> dict:
    """Average player Overall Rating per (tier, team) - a roster-skill measure
    used to blend player skill into match prediction."""
    r = rated()
    g = r[r["OVR"].notna()].groupby(["Tier", "Team"])["OVR"].mean()
    return {(t, tm): float(v) for (t, tm), v in g.items()}


def team_metrics(season) -> pd.DataFrame:
    """One row per team with record, per-game production, average player rating,
    and the league sheet's RPI / strength-of-schedule."""
    from .engine.standings import compute_standings
    st = compute_standings(season.matches)[["tier", "team", "w", "l", "gp", "wp"]]
    pl = rated()
    agg = pl.groupby(["Tier", "Team"], as_index=False).agg(
        goals=("G", "sum"), assists=("A", "sum"), saves=("S", "sum"),
        avg_ovr=("OVR", "mean")).rename(columns={"Tier": "tier", "Team": "team"})
    df = st.merge(agg, on=["tier", "team"], how="left")
    gp = df["gp"].replace(0, pd.NA)
    df["goals_pg"] = (df["goals"] / gp).round(2)
    df["saves_pg"] = (df["saves"] / gp).round(2)
    df["assists_pg"] = (df["assists"] / gp).round(2)
    df["avg_ovr"] = df["avg_ovr"].round(0)
    df["record"] = df["w"].astype(int).astype(str) + "-" + df["l"].astype(int).astype(str)
    sheet = season.teams[["tier", "team", "rpi", "past_sos"]].copy()
    for c in ("rpi", "past_sos"):
        sheet[c] = pd.to_numeric(sheet[c], errors="coerce").round(3)
    df = df.merge(sheet, on=["tier", "team"], how="left").rename(columns={"past_sos": "sos"})
    return df


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
                limit: int = 100, min_games: int = 1,
                df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Players ranked by a stat, optionally filtered to one tier and/or scored
    per game. `min_games` excludes players below that many games played."""
    src = _per_game(rated() if df is None else df)
    src = src[src["GP"] >= max(min_games, 1)]
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
    return _played_only(src[mask]).sort_values("Pts", ascending=False)


def _played_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rostered-but-never-played entries (0 games)."""
    return df[df["GP"] >= 1]


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
