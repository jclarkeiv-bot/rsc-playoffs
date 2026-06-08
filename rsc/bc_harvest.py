"""Harvest comprehensive advanced stats from the official RSC ballchasing
account and write data/bc_advanced.csv. Importable so the app can auto-refresh
it in the background; scripts/bc_build_official.py is a thin CLI wrapper.

Incremental by design: match-group details are cached ~30 days (immutable), so a
re-run only fetches newly-played matches; the group traversal uses a short TTL
to discover them.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .ballchasing import Ballchasing, TRAVERSE_TTL

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "bc_advanced.csv"

# official tree roots
RSC_3S = "3-s-league-klanmsplvp"
SEASON_ROOT = "season-26-zynncbqlcx"   # Season 26

METRICS = [
    ("shooting_pct", "core", "shooting_percentage"), ("score", "core", "score"),
    ("boost_per_min", "boost", "bpm"), ("avg_boost", "boost", "avg_amount"),
    ("boost_stolen", "boost", "amount_stolen"), ("pct_zero_boost", "boost", "percent_zero_boost"),
    ("avg_speed", "movement", "avg_speed"), ("pct_supersonic", "movement", "percent_supersonic_speed"),
    ("pct_high_air", "movement", "percent_high_air"),
    ("dist_to_ball", "positioning", "avg_distance_to_ball"),
    ("pct_offensive_third", "positioning", "percent_offensive_third"),
    ("demos_inflicted", "demo", "inflicted"), ("demos_taken", "demo", "taken"),
]


# season group ids (from the official tree); harvest auto-descends to Regular Season
SEASON_REG = {
    "S26": "season-26-zynncbqlcx", "S25": "season-25-rnrpfdu2kn",
    "S24": "season-24-izbkvyjn02", "S23": "season-23-a4i5njgo4g",
    "S22": "season-22-y61vh6vdcu", "S21": "season-21-6ull7qkld1",
    "S20": "season-20-ik5n62p3lz", "S19": "season-19-h481hlr3wt",
    "S18": "season-18-zaawe8k7qt", "S17": "season-17-jd9fbcprwb",
    "S14": "season-14-24vu82iuxi", "S13": "season-13-hvk711vfn0",
}
HISTORY_DIR = ROOT / "data" / "history"

# full per-player profile (per-game) for the comparables model
HIST_METRICS = [
    ("goals", "core", "goals"), ("assists", "core", "assists"),
    ("saves", "core", "saves"), ("shots", "core", "shots"),
    ("score", "core", "score"), ("shooting_pct", "core", "shooting_percentage"),
    ("boost_per_min", "boost", "bpm"), ("avg_boost", "boost", "avg_amount"),
    ("boost_stolen", "boost", "amount_stolen"),
    ("avg_speed", "movement", "avg_speed"),
    ("pct_supersonic", "movement", "percent_supersonic_speed"),
    ("dist_to_ball", "positioning", "avg_distance_to_ball"),
    ("pct_offensive_third", "positioning", "percent_offensive_third"),
    ("demos_inflicted", "demo", "inflicted"),
]


def _tier_leaves(bc, reg_id, log=lambda *_: None):
    """(leaf match group, tier) pairs - tier comes from the tree level under
    Regular Season, so it works for past seasons with no rscna roster."""
    out = []
    tiers = bc._get(f"/groups?group={reg_id}&count=80", ttl=TRAVERSE_TTL).get("list", [])
    for tg in tiers:
        tier = tg["name"].strip()
        stack, seen = [tg["id"]], 0
        while stack:
            gid = stack.pop()
            for c in bc._get(f"/groups?group={gid}&count=200", ttl=TRAVERSE_TTL).get("list", []):
                if c.get("direct_replays", 0) > 0:
                    out.append((c["id"], tier))
                elif c.get("indirect_replays", 0) > 0:
                    stack.append(c["id"])
        log(f"  {tier}: {len(out)} matches so far")
    return out


def harvest_season(label: str, log=lambda *_: None) -> pd.DataFrame:
    """Per-player season profile (per-game core + advanced stats + tier) for one
    season, derived entirely from the official ballchasing tree. Writes
    data/history/<label>.csv. Used to build the comparables pool."""
    from collections import Counter
    bc = Ballchasing()
    reg_id = SEASON_REG[label]
    deeper = _child(bc, reg_id, "Regular Season")   # descend if given a season root
    if deeper:
        reg_id = deeper
    leaves = _tier_leaves(bc, reg_id, log)
    log(f"{label}: {len(leaves)} match groups")
    acc = defaultdict(lambda: {"games": 0, "name": None, "sid": "",
                               "tiers": Counter(),
                               **{m[0]: 0.0 for m in HIST_METRICS}})
    for i, (gid, tier) in enumerate(leaves, 1):
        try:
            d = bc.group(gid)
        except Exception:
            continue
        for p in d.get("players", []):
            name = p.get("name")
            games = p.get("cumulative", {}).get("games", 0) or 0
            if not name or games == 0:
                continue
            sid = p.get("id")                    # steam id - stable across seasons
            sid = str(sid) if sid else ""
            key = sid or name.lower()            # key by steam id when present
            ga = p.get("game_average", {})
            rec = acc[key]
            rec["name"] = name
            rec["sid"] = sid
            rec["games"] += games
            rec["tiers"][tier] += games
            for col, cat, field in HIST_METRICS:
                v = ga.get(cat, {}).get(field)
                if v is not None:
                    rec[col] += v * games
        if i % 100 == 0:
            log(f"  pulled {i}/{len(leaves)} matches, {len(acc)} players...")
    rows = []
    for rec in acc.values():
        g = rec["games"]
        if g < 1:
            continue
        row = {"Player": rec["name"], "sid": rec["sid"], "season": label,
               "games": g, "tier": rec["tiers"].most_common(1)[0][0]}
        for col, _, _ in HIST_METRICS:
            row[col] = round(rec[col] / g, 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_DIR / f"{label}.csv", index=False)
    return df


def _child(bc, gid, name):
    for c in bc._get(f"/groups?group={gid}&count=80", ttl=TRAVERSE_TTL).get("list", []):
        if c["name"].strip().lower() == name.lower():
            return c["id"]
    return None


def _leaves(bc, root_id, log=lambda *_: None):
    leaves, stack, seen = [], [root_id], 0
    while stack:
        gid = stack.pop()
        for c in bc._get(f"/groups?group={gid}&count=200", ttl=TRAVERSE_TTL).get("list", []):
            if c.get("direct_replays", 0) > 0:
                leaves.append(c["id"])
            elif c.get("indirect_replays", 0) > 0:
                stack.append(c["id"])
        seen += 1
        if seen % 25 == 0:
            log(f"  traversed {seen} groups, {len(leaves)} matches...")
    return leaves


def build(season_root: str = SEASON_ROOT, players_df: pd.DataFrame | None = None,
          log=lambda *_: None) -> dict:
    bc = Ballchasing()
    reg = _child(bc, season_root, "Regular Season") or season_root
    leaves = _leaves(bc, reg, log)
    log(f"leaf match groups: {len(leaves)}")

    acc = defaultdict(lambda: {"games": 0, "name": None, **{m[0]: 0.0 for m in METRICS}})
    for i, gid in enumerate(leaves, 1):
        try:
            d = bc.group(gid)
        except Exception:
            continue
        for p in d.get("players", []):
            name = p.get("name")
            games = p.get("cumulative", {}).get("games", 0) or 0
            if not name or games == 0:
                continue
            ga = p.get("game_average", {})
            rec = acc[name.lower()]
            rec["name"] = name
            rec["games"] += games
            for col, cat, field in METRICS:
                v = ga.get(cat, {}).get(field)
                if v is not None:
                    rec[col] += v * games
        if i % 100 == 0:
            log(f"  pulled {i}/{len(leaves)} matches, {len(acc)} players...")

    rows = []
    for rec in acc.values():
        g = rec["games"]
        row = {"bc_name": rec["name"], "bc_games": g}
        for col, _, _ in METRICS:
            row[col] = rec[col] / g if g else None
        rows.append(row)
    adv = pd.DataFrame(rows)

    if players_df is None:
        from .profiles import players as rscna_players
        players_df = rscna_players()
    rp = players_df[["Player", "Tier", "Team", "GP", "W", "L"]].copy()
    rp["_k"] = rp["Player"].astype(str).str.lower()
    adv["_k"] = adv["bc_name"].astype(str).str.lower()
    merged = rp.merge(adv, on="_k", how="inner").drop(columns="_k")
    merged["win_pct"] = merged["W"] / merged["GP"].replace(0, 1)
    merged.to_csv(CSV, index=False)
    return {"matched": len(merged), "total": len(rp), "matches": len(leaves)}
