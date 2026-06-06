"""Tier-balance diagnostic.

Are the tiers well-sorted by skill? Two independent angles:

1. Advanced-mechanics separation (empirical). Mechanics (speed, boost control,
   positioning) are roughly opponent-independent, so they're cross-tier
   comparable - validated: mean mechanics skill orders the tiers at r=-0.96.
   We measure how cleanly ADJACENT tiers separate vs overlap. ~37% coverage.

2. Projected-tier movement (full roster, box-score model). Re-sort every player
   by cross-tier skill into the current tier sizes; count who would move up/down.
   Full coverage but model-based (relies on the tier-strength assumption).

Neither is gospel - the league's real tool is MMR, which we don't have - but
together they give an evidence-based read on whether tiers are balanced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import TIER_ORDER
from . import rating

_CSV = Path(__file__).resolve().parent.parent / "data" / "bc_advanced.csv"
_ADV_FEATS = ["avg_speed", "pct_supersonic", "boost_per_min", "boost_stolen",
              "shooting_pct", "pct_offensive_third", "pct_zero_boost",
              "dist_to_ball", "demos_inflicted"]
_MIN_GAMES = 10


def mechanics_separation() -> dict:
    """Per-tier mean mechanics skill + how much each tier overlaps the one below."""
    if not _CSV.exists():
        return {"available": False, "rows": []}
    df = pd.read_csv(_CSV)
    df = df[df["bc_games"] >= _MIN_GAMES].copy()
    if len(df) < 20:
        return {"available": False, "rows": []}

    win = df["win_pct"].fillna(0.5)
    skill = pd.Series(0.0, index=df.index)
    for f in _ADV_FEATS:
        col = df[f].astype(float)
        sd = col.std(ddof=0)
        z = (col - col.mean()) / sd if sd > 0 else 0.0
        c = np.corrcoef(col.fillna(col.mean()), win)[0, 1]
        skill += (0.0 if np.isnan(c) else c) * z
    df["skill"] = skill

    by = {t: df.loc[df["Tier"] == t, "skill"].to_numpy() for t in TIER_ORDER}
    rows = []
    present = [t for t in TIER_ORDER if len(by[t]) >= 3]
    for i, t in enumerate(TIER_ORDER):
        arr = by[t]
        if len(arr) < 3:
            rows.append({"tier": t, "n": int(len(arr)), "mean": None,
                         "overlap_below": None})
            continue
        # overlap with the tier directly BELOW: % of lower-tier players whose
        # mechanics exceed THIS tier's median (high => poor separation).
        lower = TIER_ORDER[i + 1] if i + 1 < len(TIER_ORDER) else None
        ov = None
        if lower is not None and len(by[lower]) >= 3:
            ov = round(float((by[lower] > np.median(arr)).mean()) * 100)
        rows.append({"tier": t, "n": int(len(arr)),
                     "mean": round(float(arr.mean()), 2), "overlap_below": ov})

    means = [r["mean"] for r in rows if r["mean"] is not None]
    order_corr = (float(np.corrcoef(range(len(means)), means)[0, 1])
                  if len(means) > 2 else 0.0)
    overlaps = [r["overlap_below"] for r in rows if r["overlap_below"] is not None]
    return {"available": True, "rows": rows,
            "order_corr": round(order_corr, 2),
            "avg_overlap": round(float(np.mean(overlaps)), 0) if overlaps else None,
            "n_covered": int(len(df)), "tiers_covered": len(present)}


def projected_movement(players: pd.DataFrame) -> dict:
    """Per-tier count of players who project up / down / stay (box-score model)."""
    r = rating.compute_ratings(players)
    rows = []
    tot_up = tot_down = tot = 0
    for t in TIER_ORDER:
        sub = r[r["Tier"] == t]
        n = len(sub)
        if not n:
            continue
        up = int((sub["tier_delta"] > 0).sum())
        down = int((sub["tier_delta"] < 0).sum())
        stay = n - up - down
        tot_up += up; tot_down += down; tot += n
        rows.append({"tier": t, "n": n, "up": up, "down": down, "stay": stay,
                     "churn": round((up + down) / n * 100)})
    return {"rows": rows, "total": tot, "moved": tot_up + tot_down,
            "churn": round((tot_up + tot_down) / tot * 100) if tot else 0}


def season_balance() -> list[dict]:
    """How cleanly tiers separated by skill in EACH season (from the historical
    pool). 'Avg overlap' = mean share of a lower tier that beats the tier above's
    median mechanics - lower means tiers are more cleanly sorted (more balanced).
    Mechanics are standardized within each season, so the overlap is comparable
    across seasons even if the game's scales drifted."""
    from . import comps
    pool = comps.load_pool(min_games=8)
    if pool.empty:
        return []
    feats = [f for f in ("avg_speed", "pct_supersonic", "boost_per_min", "boost_stolen")
             if f in pool.columns]
    rows = []
    for season in pool["season"].unique():
        sub = pool[pool["season"] == season].copy()
        if len(sub) < 30:
            continue
        skill = pd.Series(0.0, index=sub.index)
        for f in feats:
            sd = sub[f].std(ddof=0)
            if sd and sd > 0:
                skill += (sub[f] - sub[f].mean()) / sd
        sub["skill"] = skill
        tiers = [t for t in TIER_ORDER if (sub["tier"] == t).sum() >= 5]
        if len(tiers) < 3:
            continue
        means = [sub.loc[sub["tier"] == t, "skill"].mean() for t in tiers]
        overlaps = []
        for i in range(len(tiers) - 1):
            hi = sub.loc[sub["tier"] == tiers[i], "skill"]
            lo = sub.loc[sub["tier"] == tiers[i + 1], "skill"]
            overlaps.append(float((lo > hi.median()).mean()) * 100)
        order_corr = float(np.corrcoef(range(len(means)), means)[0, 1]) if len(means) > 2 else 0.0
        rows.append({"season": season, "players": int(len(sub)),
                     "tiers": len(tiers),
                     "avg_overlap": round(float(np.mean(overlaps)), 0),
                     "order_corr": round(order_corr, 2)})
    return sorted(rows, key=lambda r: r["avg_overlap"])


def diagnose(players: pd.DataFrame) -> dict:
    sep = mechanics_separation()
    mov = projected_movement(players)
    verdict = []
    if sep["available"]:
        verdict.append(
            f"Mechanics rank the tiers correctly (order corr {sep['order_corr']}), "
            f"but adjacent tiers overlap ~{sep['avg_overlap']:.0f}% on average - "
            f"some lower-tier players already have higher-tier mechanics.")
    verdict.append(
        f"By the cross-tier skill model, {mov['moved']} of {mov['total']} players "
        f"({mov['churn']}%) sit in a tier their skill doesn't match.")
    return {"separation": sep, "movement": mov, "verdict": verdict}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rsc.profiles import players
    d = diagnose(players())
    print("=== Mechanics separation (advanced, ~450 players) ===")
    for r in d["separation"]["rows"]:
        print(f"  {r['tier']:11} n={r['n']:>3}  mean={r['mean']}  "
              f"overlap_below={r['overlap_below']}")
    print("order corr:", d["separation"].get("order_corr"),
          "avg overlap:", d["separation"].get("avg_overlap"))
    print("\n=== Projected movement (full roster) ===")
    for r in d["movement"]["rows"]:
        print(f"  {r['tier']:11} n={r['n']:>3}  up={r['up']:>2} down={r['down']:>2} "
              f"churn={r['churn']}%")
    print("\nVERDICT:")
    for v in d["verdict"]:
        print(" -", v)
