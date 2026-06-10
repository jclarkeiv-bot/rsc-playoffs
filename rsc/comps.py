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

from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pandas as pd

HIST = Path(__file__).resolve().parent.parent / "data" / "history"

# request-scoped data filter: (play, seasons) where play is "official"|"all" and
# seasons is None (all) or a tuple of season labels. Set per-request from app.
_scope: ContextVar = ContextVar("scope", default=("official", None))


def set_scope(play: str = "official", seasons=None) -> None:
    play = play if play in ("official", "all") else "official"
    _scope.set((play, tuple(seasons) if seasons else None))


def get_scope():
    return _scope.get()
# per-game features that characterise a player's production + style
FEATURES = ["goals", "assists", "saves", "shots", "boost_per_min", "avg_speed",
            "pct_supersonic", "pct_offensive_third", "demos_inflicted"]
MIN_GAMES = 8
_cache: dict = {}


def available() -> bool:
    return HIST.exists() and any(HIST.glob("*.csv"))


def _read(files) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in files]
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not pool.empty:                           # normalize tier names ("1Premier" -> "Premier")
        pool["tier"] = pool["tier"].astype(str).str.replace(r"^\s*\d+\s*", "", regex=True)
    return pool


def _merge_play(reg: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Combine regular + pre-season rows per person-season: games-weighted rates,
    summed games. Key by id (or name) + season."""
    both = pd.concat([reg, pre], ignore_index=True)
    sid = both["sid"].astype(str) if "sid" in both.columns else pd.Series([""] * len(both))
    both["_k"] = (sid.where(sid.str.len() > 2, both["Player"].astype(str).str.lower())
                  + "|" + both["season"].astype(str))
    keep = {"Player", "sid", "season", "games", "tier", "_k"}
    metrics = [c for c in both.columns if c not in keep]
    rows = []
    for _, g in both.groupby("_k"):
        gm = g["games"].astype(float)
        tot = gm.sum()
        top = g.sort_values("games", ascending=False).iloc[0]
        row = {"Player": top["Player"], "sid": top.get("sid", ""),
               "season": top["season"], "games": int(tot), "tier": top["tier"]}
        for c in metrics:
            row[c] = float((g[c] * gm).sum() / tot) if tot else None
        rows.append(row)
    return pd.DataFrame(rows)


def _build_pool(play: str, seasons) -> pd.DataFrame:
    reg = _read([f for f in sorted(HIST.glob("*.csv")) if "_" not in f.stem])
    if reg.empty:
        return reg
    if play == "all":
        pre = _read(sorted(HIST.glob("*_pre.csv")))
        if not pre.empty:
            reg = _merge_play(reg, pre)
    if seasons:
        reg = reg[reg["season"].isin(seasons)]
    return reg


def load_pool(min_games: int = MIN_GAMES, play: str | None = None,
              seasons="__scope__") -> pd.DataFrame:
    """History pool under the active data scope. play/seasons override the
    request scope (used by identity/career-prior code that must stay full)."""
    sp_play, sp_seasons = _scope.get()
    play = sp_play if play is None else play
    seasons = sp_seasons if seasons == "__scope__" else (tuple(seasons) if seasons else None)
    key = (play, seasons)
    pools = _cache.setdefault("pools", {})
    if key not in pools:
        pools[key] = _build_pool(play, seasons)
    pool = pools[key]
    return pool[pool["games"] >= min_games].copy() if not pool.empty else pool


def reload() -> None:
    _cache.clear()


def _has_sid(pool: pd.DataFrame) -> bool:
    return "sid" in pool.columns and pool["sid"].astype(str).str.len().gt(5).any()


def _name_to_sid() -> dict:
    """Current-season display name (lower) -> steam id, for linking a current
    player to their (steam-id-keyed) history across seasons."""
    if "n2s" not in _cache:
        pool = load_pool(min_games=1, play="official", seasons=None)
        if pool.empty or not _has_sid(pool):
            _cache["n2s"] = {}
        else:
            cur = pool[pool["season"] == CURRENT_SEASON].sort_values(
                "games", ascending=False).drop_duplicates("Player")
            _cache["n2s"] = {str(r.Player).lower(): str(r.sid)
                             for r in cur.itertuples()
                             if r.sid and len(str(r.sid)) > 5}
    return _cache["n2s"]


def _rows_for(pool: pd.DataFrame, name: str, past_only: bool):
    """Rows for a player across seasons - by steam id when available, else name."""
    sid = _name_to_sid().get(name.lower())
    if sid and _has_sid(pool):
        r = pool[pool["sid"].astype(str) == sid]
    else:
        r = pool[pool["Player"].astype(str).str.lower() == name.lower()]
    if past_only:
        r = r[r["season"] != CURRENT_SEASON]
    return r


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


SEASON_STAT_OPTS = {"goals": "Goals/g", "assists": "Assists/g",
                    "saves": "Saves/g", "shots": "Shots/g", "score": "Score/g",
                    "demos_inflicted": "Demos/g", "boost_per_min": "Boost/min",
                    "avg_speed": "Avg speed"}


def seasons() -> list[str]:
    pool = load_pool(min_games=1, play="official", seasons=None)   # full list, scope-independent
    if pool.empty:
        return []
    return sorted(pool["season"].unique(),
                  key=lambda s: int(s[1:]) if s[1:].isdigit() else 0, reverse=True)


def season_leaderboard(season: str, stat: str = "goals", tier: str = "all",
                       min_games: int = 8, limit: int = 100) -> list[dict]:
    """Per-game leaderboard for any harvested season, from the historical pool."""
    pool = load_pool(min_games=min_games)
    sub = pool[pool["season"] == season].copy()
    if tier != "all":
        sub = sub[sub["tier"] == tier]
    if stat not in sub.columns:
        stat = "goals"
    sub = sub.sort_values(stat, ascending=False).head(limit).reset_index(drop=True)
    sub.insert(0, "rank", sub.index + 1)
    out = sub[["rank", "Player", "tier", "games", stat]].rename(
        columns={stat: "value", "tier": "Tier", "games": "GP"})
    return out.round(2).to_dict("records")


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
    pool = load_pool(min_games=8, play="official", seasons=None)
    if pool.empty or "score" not in pool:
        return {}
    past = pool[pool["season"] != CURRENT_SEASON].copy()
    if past.empty:
        return {}
    past["pct"] = past.groupby("season")["score"].rank(pct=True) * 100
    if _has_sid(pool):                          # link by steam id (name-change proof)
        by_sid = past[past["sid"].astype(str).str.len() > 5].groupby(
            past["sid"].astype(str))["pct"].mean()
        return {nm: float(by_sid[sid]) for nm, sid in _name_to_sid().items()
                if sid in by_sid.index}
    g = past.groupby(past["Player"].astype(str).str.lower())["pct"].mean()
    return {k: float(v) for k, v in g.items()}


# history column -> projection stat column
_HIST_TO_PROJ = {"goals": "G", "assists": "A", "saves": "S",
                 "shots": "SH", "demos_inflicted": "DM"}


def career_rates(name: str) -> dict:
    """A player's games-weighted per-game rates across COMPLETED past seasons,
    keyed by projection stat (G/A/S/SH/DM). Empty if they have no history."""
    pool = load_pool(min_games=8, play="official", seasons=None)
    if pool.empty:
        return {}
    past = _rows_for(pool, name, past_only=True)
    if past.empty:
        return {}
    g = past["games"].astype(float)
    rates = {}
    for hist_col, proj_col in _HIST_TO_PROJ.items():
        if hist_col in past:
            rates[proj_col] = float((past[hist_col] * g).sum() / g.sum())
    return {"rates": rates, "games": int(g.sum()),
            "seasons": int(past["season"].nunique())}


ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "alias_overrides.json"


def _alias_overrides() -> dict:
    """Manual same-person merges the platform id can't catch (new accounts, etc).
    JSON: {"<alias id-or-name>": "<canonical id-or-name>", ...}."""
    if "aliases" not in _cache:
        import json
        try:
            _cache["aliases"] = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache["aliases"] = {}
    return _cache["aliases"]


def _pool_with_cid() -> pd.DataFrame:
    """History pool with a career id (cid) per row: the steam/platform id when
    present, else the lowercased name, then remapped through alias overrides."""
    if "cidpool" not in _cache:
        pool = load_pool(min_games=1, play="official", seasons=None).copy()
        if not pool.empty:
            sid = (pool["sid"].astype(str) if "sid" in pool.columns
                   else pd.Series([""] * len(pool), index=pool.index))
            name = pool["Player"].astype(str).str.lower()
            cid = sid.where(sid.str.len() > 2, name)
            ov = _alias_overrides()
            pool["cid"] = cid.map(lambda c: ov.get(c, c))
            pool["pct"] = pool.groupby("season")["score"].rank(pct=True) * 100
        _cache["cidpool"] = pool
    return _cache["cidpool"]


def _alias_rows(grp: pd.DataFrame) -> list[dict]:
    out = []
    for nm, g in grp.groupby("Player"):
        gm = g["games"].astype(float)
        out.append({
            "name": nm, "games": int(gm.sum()),
            "seasons": int(g["season"].nunique()),
            "goals": round(float((g["goals"] * gm).sum() / gm.sum()), 2),
            "saves": round(float((g["saves"] * gm).sum() / gm.sum()), 2),
            "last_season": sorted(g["season"], key=_season_key)[-1],
        })
    out.sort(key=lambda a: -a["games"])
    return out


def _season_key(s: str) -> int:
    return int(s[1:]) if str(s)[1:].isdigit() else 0


def career_for(cid: str) -> dict | None:
    """Full career detail for one person (grouped across all their RL names)."""
    pool = _pool_with_cid()
    if pool.empty:
        return None
    grp = pool[pool["cid"].astype(str) == str(cid)]
    if grp.empty:
        return None
    aliases = _alias_rows(grp)
    seasons = sorted(grp["season"].unique(), key=_season_key)
    tiers = grp.groupby("tier")["games"].sum().sort_values(ascending=False)
    return {
        "cid": str(cid), "primary": aliases[0]["name"], "rating": round(float(grp["pct"].mean())),
        "aliases": aliases, "n_names": len(aliases),
        "seasons": seasons, "n_seasons": len(seasons),
        "tiers": [(t, int(v)) for t, v in tiers.items()],
        "total_games": int(grp["games"].sum()),
        "has_id": bool(str(cid).isdigit()),
    }


def career_cid_for_name(name: str) -> str | None:
    """The career id a given RL name belongs to (most-played, if ambiguous)."""
    pool = _pool_with_cid()
    if pool.empty:
        return None
    r = pool[pool["Player"].astype(str).str.lower() == name.lower()]
    if r.empty:
        return None
    return str(r.groupby("cid")["games"].sum().idxmax())


def multi_name_careers(limit: int = 300) -> list[dict]:
    """Careers that span more than one RL name - the interesting groupings."""
    pool = _pool_with_cid()
    if pool.empty:
        return []
    g = pool.groupby("cid").agg(names=("Player", "nunique"),
                                games=("games", "sum"),
                                seasons=("season", "nunique"))
    g = g[g["names"] > 1].sort_values(["names", "games"], ascending=False).head(limit)
    out = []
    for cid, row in g.iterrows():
        sub = pool[pool["cid"] == cid]
        order = sub.groupby("Player")["games"].sum().sort_values(ascending=False)
        out.append({
            "cid": str(cid), "primary": order.index[0],
            "names": order.index.tolist(), "n_names": int(row["names"]),
            "rating": round(float(sub["pct"].mean())),
            "seasons": int(row["seasons"]), "games": int(row["games"]),
            "has_id": str(cid).isdigit(),
        })
    return out


def rising_players(limit: int = 50, min_seasons: int = 2) -> list[dict]:
    """Players improving fastest over time: the slope of their per-season
    production percentile across seasons (percentile points gained per season).
    Linked by account id, restricted to players active in the last two seasons.
    Always uses the full official dataset (a trajectory needs all seasons)."""
    pool = load_pool(min_games=8, play="official", seasons=None)
    if pool.empty or "score" not in pool.columns:
        return []
    pool = pool.copy()
    pool["pct"] = pool.groupby("season")["score"].rank(pct=True) * 100
    sid = (pool["sid"].astype(str) if "sid" in pool.columns
           else pd.Series([""] * len(pool), index=pool.index))
    cid = sid.where(sid.str.len() > 2, pool["Player"].astype(str).str.lower())
    ov = _alias_overrides()
    pool["cid"] = cid.map(lambda c: ov.get(c, c))
    recent = _season_key(CURRENT_SEASON) - 1          # active in current or prior season
    out = []
    for cid, g in pool.groupby("cid"):
        if g["season"].nunique() < min_seasons:
            continue
        g = g.assign(_k=g["season"].map(_season_key)).sort_values("_k")
        if g["_k"].max() < recent:
            continue
        x = g["_k"].to_numpy(dtype=float)
        y = g["pct"].to_numpy(dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else 0.0
        if slope <= 0:
            continue
        order = g.sort_values("games", ascending=False)
        out.append({
            "cid": str(cid), "primary": order["Player"].iloc[0],
            "n_seasons": int(g["season"].nunique()),
            "slope": round(slope, 1),
            "first": round(float(y[0])), "latest": round(float(y[-1])),
            "seasons": list(g["season"]), "pcts": [round(float(v)) for v in y],
            "tier": order["tier"].iloc[0], "games": int(g["games"].sum()),
        })
    out.sort(key=lambda o: -o["slope"])
    return out[:limit]


def player_history(name: str) -> list[dict]:
    """A player's per-season lines (any games), oldest first."""
    pool = load_pool(min_games=1)
    if pool.empty:
        return []
    r = _rows_for(pool, name, past_only=False).sort_values("season")
    cols = ["season", "tier", "games", "goals", "assists", "saves", "shots",
            "score", "boost_per_min", "avg_speed"]
    return r[[c for c in cols if c in r.columns]].round(2).to_dict("records")
