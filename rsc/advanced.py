"""Advanced (ballchasing) stats layer.

Reads the offline snapshot built by scripts/bc_build.py. Provides each covered
player's advanced profile and a SEPARATE 'Advanced OVR' - a percentile among the
~450 players who have replay data, weighted by what correlates with winning.
This is kept apart from the universal box-score OVR because it only covers ~37%
of the league; mixing them would bias rankings toward players who upload.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "bc_advanced.csv"
MIN_GAMES = 10   # need this many replay-games to get an Advanced OVR

# display profile: (col, label, fmt) - fmt in {num, pct, speed}
PROFILE = [
    ("boost_per_min", "Boost / min", "num"),
    ("avg_boost", "Avg boost", "num"),
    ("boost_stolen", "Boost stolen / game", "num"),
    ("pct_zero_boost", "% time at 0 boost", "pct"),
    ("avg_speed", "Avg speed (uu/s)", "num"),
    ("pct_supersonic", "% supersonic", "pct"),
    ("pct_high_air", "% time high in air", "pct"),
    ("dist_to_ball", "Avg distance to ball", "num"),
    ("pct_offensive_third", "% time attacking third", "pct"),
    ("demos_inflicted", "Demos / game", "num"),
    ("demos_taken", "Demos taken / game", "num"),
    ("shooting_pct", "Shooting %", "pct"),
]

# features for Advanced OVR (sign comes from correlation with winning)
ADV_FEATURES = ["boost_per_min", "avg_speed", "pct_supersonic", "boost_stolen",
                "demos_inflicted", "demos_taken", "pct_zero_boost",
                "shooting_pct", "score", "dist_to_ball"]

_cache = {}


def _df() -> pd.DataFrame:
    if "df" not in _cache:
        _cache["df"] = pd.read_csv(_CSV) if _CSV.exists() else pd.DataFrame()
    return _cache["df"]


def available() -> bool:
    return not _df().empty


def _ratings() -> pd.DataFrame:
    if "rated" in _cache:
        return _cache["rated"]
    df = _df()
    if df.empty:
        _cache["rated"] = df
        return df
    pool = df[df["bc_games"] >= MIN_GAMES].copy()
    if len(pool) < 10:
        _cache["rated"] = pool.assign(adv_ovr=np.nan)
        return _cache["rated"]
    win = pool["win_pct"].fillna(0.5)
    composite = pd.Series(0.0, index=pool.index)
    for f in ADV_FEATURES:
        col = pool[f].astype(float)
        sd = col.std(ddof=0)
        z = (col - col.mean()) / sd if sd > 0 else 0.0
        c = np.corrcoef(col.fillna(col.mean()), win)[0, 1]
        composite += (0.0 if np.isnan(c) else c) * z          # signed weight
    pool["adv_ovr"] = composite.rank(pct=True).mul(100).round(0)
    _cache["rated"] = pool
    return pool


def player_advanced(name: str) -> dict | None:
    df = _df()
    if df.empty:
        return None
    m = df[df["Player"].astype(str) == name]
    if m.empty:
        return None
    rec = m.iloc[0]
    metrics = []
    for col, label, fmt in PROFILE:
        v = rec.get(col)
        if pd.isna(v):
            continue
        metrics.append({"label": label, "value": float(v), "fmt": fmt})
    rated = _ratings()
    rr = rated[rated["Player"].astype(str) == name] if not rated.empty else None
    adv_ovr = (int(rr.iloc[0]["adv_ovr"])
               if rr is not None and len(rr) and not pd.isna(rr.iloc[0]["adv_ovr"])
               else None)
    pool_n = int((df["bc_games"] >= MIN_GAMES).sum())
    return {"name": name, "bc_games": int(rec["bc_games"]),
            "metrics": metrics, "adv_ovr": adv_ovr, "pool_n": pool_n}


def advanced_importance() -> list[dict]:
    """Which advanced metrics correlate most with winning (among covered players)."""
    df = _df()
    pool = df[df["bc_games"] >= MIN_GAMES]
    if len(pool) < 10:
        return []
    win = pool["win_pct"].fillna(0.5)
    rows = []
    for col, label, _ in PROFILE:
        if col in pool:
            c = np.corrcoef(pool[col].astype(float).fillna(pool[col].mean()), win)[0, 1]
            rows.append({"stat": label, "corr": round(0.0 if np.isnan(c) else c, 3)})
    return sorted(rows, key=lambda r: abs(r["corr"]), reverse=True)


if __name__ == "__main__":
    print("snapshot available:", available(), "| covered:", len(_df()))
    print("\nAdvanced metrics most tied to winning:")
    for r in advanced_importance():
        print(f"  {r['stat']:24} corr={r['corr']:+.3f}")
    for n in ["GeneralLeigh91", "bradyisadopted"]:
        a = player_advanced(n)
        if a:
            print(f"\n{n}: {a['bc_games']} replays, Advanced OVR={a['adv_ovr']} "
                  f"(among {a['pool_n']} covered players)")
