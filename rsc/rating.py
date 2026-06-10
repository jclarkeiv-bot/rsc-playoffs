"""One composite player rating, plus an 'overskilled for tier' signal.

Per-game counting stats aren't comparable across tiers (production inflates in
weaker tiers), so we standardize each stat WITHIN its tier. The composite then
measures how much a player dominates their own tier - fair to compare across
tiers. Each stat is weighted by how strongly it correlates with winning (W/GP),
mirroring the league's own SBV philosophy (auditable, no black box). Small
samples are shrunk toward average so a 7-game hot streak doesn't top the chart.

OVR (Overall Rating, 0-100): the single "who's best overall" number. It adds a
tier-strength term to within-tier dominance, so a Premier player outranks an
equally-dominant Amateur, while a truly dominant lower-tier player can still
outrank a weak higher-tier one. The tier-strength step is a stated assumption
(no cross-tier games exist to measure it directly).

Overskilled: a player dominating their tier (top of their tier's dominance
distribution) is flagged as a promotion candidate - too good for their tier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .engine.compare import _per_game
from .ingest import TIER_ORDER

# per-game contributions that feed the composite (Pts excluded: it's a function
# of these, so including it would double-count)
FEATURES = ["G/g", "A/g", "S/g", "SH/g", "DM/g", "SH%", "MVP/g"]
# advanced ballchasing mechanics folded into OVR (coverage now ~all active players)
ADV_FEATURES = ["boost_per_min", "avg_speed", "pct_supersonic", "boost_stolen",
                "pct_offensive_third", "demos_inflicted"]
_BC_CSV = Path(__file__).resolve().parent.parent / "data" / "bc_advanced.csv"
MIN_GP = 4
SHRINK_K = 8          # games of regression toward tier-average
TIER_STEP = 1.5       # OVR boost per tier of difficulty (in dominance-z units)
OVERSKILLED_PCT = 92  # top of tier's dominance distribution => promotion candidate


def _merge_advanced(pool: pd.DataFrame) -> list[str]:
    """Merge advanced mechanics onto the pool by player; fill gaps with the
    tier average so uncovered players are neutral. Returns the columns added."""
    if not _BC_CSV.exists():
        return []
    adv = pd.read_csv(_BC_CSV)
    cols = [c for c in ADV_FEATURES if c in adv.columns]
    if not cols:
        return []
    # one row per player (a name can appear in multiple tiers as a sub)
    if "bc_games" in adv.columns:
        adv = adv.sort_values("bc_games", ascending=False)
    adv_u = adv.drop_duplicates("Player")[["Player"] + cols]
    m = pool[["Player", "Tier"]].merge(adv_u, on="Player", how="left")
    m.index = pool.index
    for f in cols:
        filled = m.groupby("Tier")[f].transform(lambda x: x.fillna(x.mean()))
        pool[f] = filled.fillna(m[f].mean())
    return cols


def compute_ratings(players: pd.DataFrame) -> pd.DataFrame:
    df = _per_game(players).copy()
    df["MVP/g"] = df["MVP"] / df["GP"].replace(0, np.nan)
    pool = df[df["GP"] >= MIN_GP].copy()
    adv_cols = _merge_advanced(pool)
    feats = FEATURES + adv_cols

    # weights: correlation of each stat with winning, computed league-wide
    winrate = (pool["W"] / pool["GP"]).fillna(0.5)
    weights = {}
    for f in feats:
        col = pool[f].astype(float).fillna(pool[f].mean())
        c = np.corrcoef(col, winrate)[0, 1]
        weights[f] = max(0.0, 0.0 if np.isnan(c) else c)
    wsum = sum(weights.values()) or 1.0
    weights = {f: w / wsum for f, w in weights.items()}

    # standardize each feature WITHIN tier, then weight-combine
    ztier = pd.DataFrame(index=pool.index)
    for f in feats:
        g = pool.groupby("Tier")[f]
        mean, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
        ztier[f] = ((pool[f] - mean) / sd).fillna(0.0)
    dom = sum(ztier[f] * weights[f] for f in feats)

    # shrink small samples toward tier average (0)
    reliability = pool["GP"] / (pool["GP"] + SHRINK_K)
    pool["composite"] = dom * reliability      # within-tier dominance

    # within-tier dominance MARGIN: how many SDs above the tier average a player
    # sits. This is the outlier signal - separates a runaway tier-buster from a
    # player who's merely top-of-a-balanced-tier.
    g2 = pool.groupby("Tier")["composite"]
    pool["tier_z"] = ((pool["composite"] - g2.transform("mean"))
                      / g2.transform("std").replace(0, np.nan)).fillna(0.0)
    pool["tier_pct"] = g2.rank(pct=True).mul(100).round(0)

    # cross-tier skill = within-tier dominance + a tier-strength term, so players
    # from different tiers sit on one scale. OVR = its league percentile.
    tier_idx = {t: i for i, t in enumerate(TIER_ORDER)}
    tier_level = pool["Tier"].map(
        lambda t: (len(TIER_ORDER) - 1 - tier_idx.get(t, len(TIER_ORDER) - 1)) * TIER_STEP)
    pool["overall_score"] = pool["composite"] + tier_level
    pool["OVR"] = pool["overall_score"].rank(pct=True).mul(100).round(0)

    # PROJECTED TIER: re-bin every player by cross-tier skill into the current
    # per-tier population sizes. Where a player lands = the tier their skill fits.
    # Projecting ABOVE your current tier => genuinely overskilled (not just "best
    # in tier"). This needs cross-tier-comparable skill, which only the tier-
    # strength model provides, so it's a model estimate (TIER_STEP is the key
    # assumption), reported as such.
    counts = pool["Tier"].value_counts()
    order = pool.sort_values("overall_score", ascending=False).index.tolist()
    proj, pos = {}, 0
    for t in TIER_ORDER:
        n = int(counts.get(t, 0))
        for idx in order[pos:pos + n]:
            proj[idx] = t
        pos += n
    for idx in order[pos:]:
        proj[idx] = TIER_ORDER[-1]
    pool["projected_tier"] = pd.Series(proj)
    pool["cur_idx"] = pool["Tier"].map(tier_idx)
    pool["proj_idx"] = pool["projected_tier"].map(tier_idx)
    pool["tier_delta"] = pool["cur_idx"] - pool["proj_idx"]   # +ve = projects higher
    higher = {t: (TIER_ORDER[i - 1] if i > 0 else None)
              for i, t in enumerate(TIER_ORDER)}
    pool["next_tier"] = pool["Tier"].map(higher)
    # genuinely overskilled: projects up a tier AND a clear within-tier outlier
    pool["overskilled"] = (pool["tier_delta"] >= 1) & (pool["tier_z"] >= 1.0)
    pool["underskilled"] = (pool["tier_delta"] <= -1) & (pool["tier_z"] <= -1.0)

    # career skill from past seasons -> confirm overskilled flags vs hot streaks
    try:
        from . import comps
        career = comps.historical_skill()
    except Exception:
        career = {}
    key = pool["Player"].astype(str).str.lower()
    pool["career_pct"] = key.map(career) if career else np.nan
    pool["career_confirmed"] = pool["overskilled"] & (pool["career_pct"] >= 65)

    pool["_weights"] = [weights] * len(pool)
    return pool


_STAT_LABELS = {"G/g": "Goals/game", "A/g": "Assists/game", "S/g": "Saves/game",
                "SH/g": "Shots/game", "DM/g": "Demos/game", "SH%": "Shot %",
                "MVP/g": "MVPs/game"}


def stat_importance(players: pd.DataFrame) -> list[dict]:
    """Which stats best predict winning. Returns each feature's correlation
    with win rate and its (normalized) weight in OVR, most important first."""
    df = _per_game(players).copy()
    df["MVP/g"] = df["MVP"] / df["GP"].replace(0, np.nan)
    pool = df[df["GP"] >= MIN_GP]
    winrate = (pool["W"] / pool["GP"]).fillna(0.5)
    rows = []
    raw = {}
    for f in FEATURES:
        col = pool[f].astype(float).fillna(pool[f].mean())
        c = np.corrcoef(col, winrate)[0, 1]
        raw[f] = 0.0 if np.isnan(c) else c
    wsum = sum(max(0.0, c) for c in raw.values()) or 1.0
    for f in FEATURES:
        rows.append({"stat": _STAT_LABELS[f], "key": f,
                     "corr_with_winning": round(raw[f], 3),
                     "weight": round(max(0.0, raw[f]) / wsum, 3)})
    return sorted(rows, key=lambda r: r["weight"], reverse=True)


def player_rating(players: pd.DataFrame, name: str) -> dict | None:
    r = compute_ratings(players)
    m = r[r["Player"].astype(str) == name]
    if m.empty:
        return None
    rec = m.sort_values("GP", ascending=False).iloc[0]
    return {
        "name": name, "tier": rec["Tier"],
        "ovr": int(rec["OVR"]),
        "tier_pct": int(rec["tier_pct"]),
        "tier_z": round(float(rec["tier_z"]), 1),
        "projected_tier": rec["projected_tier"],
        "tier_delta": int(rec["tier_delta"]),
        "next_tier": rec["next_tier"],
        "overskilled": bool(rec["overskilled"]),
        "underskilled": bool(rec["underskilled"]),
        "career_pct": (None if pd.isna(rec.get("career_pct"))
                       else int(rec["career_pct"])),
        "career_confirmed": bool(rec.get("career_confirmed")),
        "weights": {k: round(v, 3) for k, v in rec["_weights"].items()},
    }


def best_overall(players: pd.DataFrame, tier: str = "all",
                 limit: int = 100) -> pd.DataFrame:
    r = compute_ratings(players)
    if tier and tier != "all":
        r = r[r["Tier"] == tier]
    r = r.sort_values("OVR", ascending=False).head(limit)
    return r[["Player", "Tier", "Team", "GP", "OVR", "tier_pct",
              "overskilled"]].reset_index(drop=True)


def tier_misplaced(players: pd.DataFrame, tier: str):
    """Players in `tier` whose cross-tier skill projects them up or down."""
    r = compute_ratings(players)
    r = r[r["Tier"] == tier]
    cols = ["Player", "Team", "GP", "OVR", "tier_z", "tier_pct", "projected_tier"]
    up = (r[r["tier_delta"] > 0].sort_values("tier_z", ascending=False)[cols]
          .to_dict("records"))
    down = (r[r["tier_delta"] < 0].sort_values("tier_z")[cols]
            .to_dict("records"))
    return up, down


_MISPLACED_COLS = ["Player", "Tier", "Team", "GP", "OVR", "tier_z", "tier_pct",
                   "projected_tier", "tier_delta", "career_pct", "career_confirmed"]


def misplaced_candidates(players: pd.DataFrame, direction: str = "up",
                         limit: int = 80) -> pd.DataFrame:
    """Players whose cross-tier skill doesn't match their tier.
    direction='up'   -> too good for their tier (project a tier higher).
    direction='down' -> placed too high (project a tier lower, struggling)."""
    r = compute_ratings(players)
    if direction == "down":
        r = r[r["underskilled"]].sort_values(["tier_delta", "tier_z"],
                                             ascending=True)
    else:
        r = r[r["overskilled"]].sort_values(["career_confirmed", "tier_delta",
                                             "tier_z"], ascending=False)
    return r.head(limit)[_MISPLACED_COLS].reset_index(drop=True)


def overskilled_candidates(players: pd.DataFrame, limit: int = 80) -> pd.DataFrame:
    return misplaced_candidates(players, "up", limit)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rsc.api import RSCClient
    players = RSCClient().player_stats()

    print("=== Which stats predict winning most? ===")
    for s in stat_importance(players):
        print(f"  {s['stat']:14} corr={s['corr_with_winning']:+.3f}  weight={s['weight']:.3f}")
    print("\n=== Top 12 players overall (OVR) ===")
    print(best_overall(players, limit=12).to_string(index=False))
    print("\n=== Most overskilled for their tier ===")
    print(overskilled_candidates(players, limit=12).to_string(index=False))
