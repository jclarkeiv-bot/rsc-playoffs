"""Comparables / analogs (Phase 9).

Builds a pool of player-seasons from the harvested history (data/history/*.csv)
and finds, for any player, their most statistically similar players across RSC
history (nearest neighbours in standardized per-game stat space). Also exposes a
player's own cross-season history and a soft "what similar players did next"
trajectory.

Honest scope: this is descriptive similarity + observed cross-season change, not
a hard forecast - cross-season stat scales can shift with roster/rule changes,
and player names can change between seasons (so some history won't link).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HIST = Path(__file__).resolve().parent.parent / "data" / "history"
# per-game features that characterise a player's production + style
FEATURES = ["goals", "assists", "saves", "shots", "boost_per_min", "avg_speed",
            "pct_supersonic", "pct_offensive_third", "demos_inflicted"]
MIN_GAMES = 8
_cache: dict = {}


def available() -> bool:
    return HIST.exists() and any(HIST.glob("*.csv"))


def load_pool(min_games: int = MIN_GAMES) -> pd.DataFrame:
    if "pool" not in _cache:
        frames = [pd.read_csv(f) for f in sorted(HIST.glob("*.csv"))]
        pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not pool.empty:                       # normalize tier names ("1Premier" -> "Premier")
            pool["tier"] = pool["tier"].astype(str).str.replace(r"^\s*\d+\s*", "", regex=True)
        _cache["pool"] = pool
    pool = _cache["pool"]
    return pool[pool["games"] >= min_games].copy() if not pool.empty else pool


def reload() -> None:
    _cache.clear()


def _zcols(pool: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    z = pool.copy()
    cols = []
    for f in FEATURES:
        if f not in z:
            continue
        sd = z[f].std(ddof=0)
        zc = f"{f}_z"
        z[zc] = (z[f] - z[f].mean()) / sd if sd and sd > 0 else 0.0
        cols.append(zc)
    return z, cols


def find_comparables(name: str, season: str = "S26", k: int = 10) -> dict | None:
    pool = load_pool()
    if pool.empty:
        return None
    z, zcols = _zcols(pool)
    tgt = z[(z["Player"].astype(str) == name) & (z["season"] == season)]
    if tgt.empty:                       # fall back to any season for this player
        tgt = z[z["Player"].astype(str) == name]
    if tgt.empty:
        return None
    t = tgt.sort_values("games", ascending=False).iloc[0]
    others = z[~((z["Player"].astype(str) == t["Player"]) & (z["season"] == t["season"]))].copy()
    M = others[zcols].fillna(0).to_numpy(dtype=float)
    v = pd.to_numeric(t[zcols], errors="coerce").fillna(0).to_numpy(dtype=float)
    others["dist"] = np.sqrt(((M - v) ** 2).sum(axis=1))
    others = others.sort_values("dist").head(k)
    keep = ["Player", "season", "tier", "games", "goals", "assists", "saves",
            "shots", "boost_per_min", "avg_speed", "demos_inflicted", "dist"]
    comps = others[keep].round(2).to_dict("records")
    return {
        "name": t["Player"], "season": t["season"], "tier": t["tier"],
        "n_pool": len(pool), "comps": comps,
        "tightness": round(float(others["dist"].head(5).mean()), 2),
    }


FORECAST_STATS = [("goals", "Goals/game"), ("assists", "Assists/game"),
                  ("saves", "Saves/game"), ("shots", "Shots/game")]
CURRENT_SEASON = "S26"


def forecast(name: str, k: int = 25) -> dict | None:
    """Comparable-based outlook: from the player's nearest analogs in COMPLETED
    seasons (full, stable seasons), the median + inter-quartile range for each
    per-game stat. A descriptive 'players like you finished here' range, not a
    point forecast; confidence reflects how tight/numerous the analogs are."""
    pool = load_pool()
    if pool.empty:
        return None
    z, zcols = _zcols(pool)
    tgt = z[(z["Player"].astype(str) == name) & (z["season"] == CURRENT_SEASON)]
    if tgt.empty:
        tgt = z[z["Player"].astype(str) == name]
    if tgt.empty:
        return None
    t = tgt.sort_values("games", ascending=False).iloc[0]
    past = z[z["season"] != CURRENT_SEASON].copy()      # completed seasons only
    if len(past) < 10:
        return None
    M = past[zcols].fillna(0).to_numpy(dtype=float)
    v = pd.to_numeric(t[zcols], errors="coerce").fillna(0).to_numpy(dtype=float)
    past["dist"] = np.sqrt(((M - v) ** 2).sum(axis=1))
    near = past.sort_values("dist").head(k)
    rows = []
    for col, label in FORECAST_STATS:
        if col not in near:
            continue
        s = near[col].dropna()
        rows.append({"label": label,
                     "current": round(float(t[col]), 2) if col in t else None,
                     "median": round(float(s.median()), 2),
                     "lo": round(float(s.quantile(0.25)), 2),
                     "hi": round(float(s.quantile(0.75)), 2)})
    tightness = float(near["dist"].head(10).mean())
    conf = ("High" if tightness < 1.5 and len(near) >= 20 else
            "Medium" if tightness < 2.5 and len(near) >= 10 else "Low")
    return {"name": t["Player"], "k": len(near), "confidence": conf,
            "n_seasons": int(past["season"].nunique()), "rows": rows}


def historical_skill() -> dict:
    """Career production level per player, 0-100, from COMPLETED past seasons:
    the average of their per-game 'score' percentile within each past season.
    Used as a prior so returning players aren't rated from scratch each season."""
    pool = load_pool(min_games=8)
    if pool.empty or "score" not in pool:
        return {}
    past = pool[pool["season"] != CURRENT_SEASON].copy()
    if past.empty:
        return {}
    past["pct"] = past.groupby("season")["score"].rank(pct=True) * 100
    g = past.groupby(past["Player"].astype(str).str.lower())["pct"].mean()
    return {k: float(v) for k, v in g.items()}


def player_history(name: str) -> list[dict]:
    """A player's per-season lines (any games), oldest first."""
    pool = load_pool(min_games=1)
    if pool.empty:
        return []
    r = pool[pool["Player"].astype(str) == name].sort_values("season")
    cols = ["season", "tier", "games", "goals", "assists", "saves", "shots",
            "score", "boost_per_min", "avg_speed"]
    return r[[c for c in cols if c in r.columns]].round(2).to_dict("records")
