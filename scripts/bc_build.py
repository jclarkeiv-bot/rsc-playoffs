"""Build the ballchasing advanced-stats snapshot for S26.

Discovers S26 RSC groups, aggregates per-player advanced stats (weighting each
group's per-game averages by games played), matches players to the rscna roster
by name, and writes data/bc_advanced.csv. The Flask app reads that file - it
never calls ballchasing live. Re-run this when you want fresh replay data.

Coverage is partial (only teams whose members upload replays), so the snapshot
records each player's `bc_games` sample size for honest labeling downstream.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from rsc.ballchasing import Ballchasing  # noqa: E402
from rsc.profiles import players as rscna_players  # noqa: E402

SEARCH_TERMS = ["RSC S26", "RSC S26 MD", "RSC Season 26", "S26 RSC"]

# advanced per-game metrics to keep: (output_col, category, field)
METRICS = [
    ("shooting_pct", "core", "shooting_percentage"),
    ("score", "core", "score"),
    ("boost_per_min", "boost", "bpm"),
    ("avg_boost", "boost", "avg_amount"),
    ("boost_stolen", "boost", "amount_stolen"),
    ("pct_zero_boost", "boost", "percent_zero_boost"),
    ("avg_speed", "movement", "avg_speed"),
    ("pct_supersonic", "movement", "percent_supersonic_speed"),
    ("pct_high_air", "movement", "percent_high_air"),
    ("dist_to_ball", "positioning", "avg_distance_to_ball"),
    ("pct_offensive_third", "positioning", "percent_offensive_third"),
    ("demos_inflicted", "demo", "inflicted"),
    ("demos_taken", "demo", "taken"),
]


def main():
    bc = Ballchasing()
    print("auth:", bc.ping().get("name"))

    groups = {}
    for term in SEARCH_TERMS:
        for g in bc.search_groups(term):
            groups[g["id"]] = g
    print(f"S26 groups discovered: {len(groups)}")

    # accumulate weighted per-game sums + games per player (by lowercased name)
    acc = defaultdict(lambda: {"games": 0, "name": None,
                               **{m[0]: 0.0 for m in METRICS}})
    n_groups_ok = 0
    for i, g in enumerate(groups.values(), 1):
        try:
            d = bc.group(g["id"])
        except Exception:
            continue
        n_groups_ok += 1
        for p in d.get("players", []):
            name = p.get("name")
            if not name:
                continue
            games = p.get("cumulative", {}).get("games", 0) or 0
            if games == 0:
                continue
            ga = p.get("game_average", {})
            key = name.lower()
            rec = acc[key]
            rec["name"] = name
            rec["games"] += games
            for col, cat, field in METRICS:
                val = ga.get(cat, {}).get(field)
                if val is not None:
                    rec[col] += val * games   # weight by games
        if i % 15 == 0:
            print(f"  ...{i}/{len(groups)} groups")
    print(f"groups aggregated: {n_groups_ok}; distinct players: {len(acc)}")

    # to per-game averages
    rows = []
    for rec in acc.values():
        g = rec["games"]
        row = {"bc_name": rec["name"], "bc_games": g}
        for col, _, _ in METRICS:
            row[col] = rec[col] / g if g else None
        rows.append(row)
    adv = pd.DataFrame(rows)

    # match to rscna roster by lowercased name
    rp = rscna_players()[["Player", "Tier", "Team", "GP", "W", "L"]].copy()
    rp["_k"] = rp["Player"].astype(str).str.lower()
    adv["_k"] = adv["bc_name"].astype(str).str.lower()
    merged = rp.merge(adv, on="_k", how="inner").drop(columns="_k")
    merged["win_pct"] = merged["W"] / merged["GP"].replace(0, 1)

    out = ROOT / "data" / "bc_advanced.csv"
    merged.to_csv(out, index=False)
    print(f"\nmatched players: {len(merged)} / {len(rp)} "
          f"({100*len(merged)/len(rp):.1f}%)  -> {out}")
    print(merged.sort_values("bc_games", ascending=False)
          [["Player", "Tier", "bc_games", "boost_per_min", "avg_speed",
            "demos_inflicted"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
