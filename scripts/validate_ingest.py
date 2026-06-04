"""Phase 1 validation: prove the parser + standings recompute are correct.

Compares our game-based W/L (recomputed from raw match results) against the
league's own `All Teams Data` numbers. Any mismatch means the parser or the
standings logic is wrong and must be fixed before building anything on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from rsc.ingest import load_season  # noqa: E402
from rsc.engine.standings import compute_standings  # noqa: E402


def validate(label: str) -> bool:
    season = load_season(ROOT / "data" / f"{label}_standings.xlsx", label)
    ours = compute_standings(season.matches)
    truth = season.teams[["tier", "team", "ov_w", "ov_l", "ov_wp"]].copy()
    truth = truth.dropna(subset=["team"])

    merged = ours.merge(truth, on=["tier", "team"], how="outer", indicator=True)

    print(f"\n{'='*70}\n{label}: ingest + standings validation\n{'='*70}")
    print(f"teams parsed (schedule):   {len(ours)}")
    print(f"teams in All Teams Data:   {len(truth)}")

    only_sched = merged[merged["_merge"] == "left_only"]["team"].tolist()
    only_truth = merged[merged["_merge"] == "right_only"]["team"].tolist()
    if only_sched:
        print(f"  ! in schedule but not in standings sheet: {only_sched}")
    if only_truth:
        print(f"  ! in standings sheet but not in schedule: {only_truth}")

    both = merged[merged["_merge"] == "both"].copy()
    both["w_ok"] = both["w"] == both["ov_w"]
    both["l_ok"] = both["l"] == both["ov_l"]
    both["wp_ok"] = (both["wp"] - both["ov_wp"]).abs() < 1e-4

    n = len(both)
    w_bad = both[~both["w_ok"]]
    l_bad = both[~both["l_ok"]]
    wp_bad = both[~both["wp_ok"]]

    print(f"\nmatched teams: {n}")
    print(f"  wins match:    {n - len(w_bad)}/{n}")
    print(f"  losses match:  {n - len(l_bad)}/{n}")
    print(f"  win%% match:    {n - len(wp_bad)}/{n}")

    bad = both[~(both["w_ok"] & both["l_ok"] & both["wp_ok"])]
    if len(bad):
        print("\n  MISMATCHES (ours vs sheet):")
        for r in bad.itertuples(index=False):
            print(f"    [{r.tier}] {r.team:24} "
                  f"ours {r.w}-{r.l} ({r.wp:.3f})  "
                  f"sheet {r.ov_w}-{r.ov_l} ({r.ov_wp:.3f})")
    ok = len(bad) == 0 and not only_sched and not only_truth
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    labels = sys.argv[1:] or ["S26", "S25"]
    results = {lbl: validate(lbl) for lbl in labels}
    print(f"\n{'='*70}")
    print("SUMMARY:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)
