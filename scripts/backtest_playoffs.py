"""End-to-end playoff-prediction backtest on a COMPLETED season (S25).

Reconstruct each tier as of match day N (default 6 = where S26 is now), train
Elo on only the games known by then, Monte-Carlo the rest, and compare the
predicted playoff field to who actually finished top-K. Baseline = just taking
the top-K of the as-of-N standings (no simulation). This tells us whether the
simulator adds real value and how trustworthy the playoff odds are.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from rsc.ingest import load_season  # noqa: E402
from rsc.engine.clinch import build_tier_state  # noqa: E402
from rsc.engine.simulate import simulate_tier  # noqa: E402
from rsc.engine.predict import train_elo, make_elo_model  # noqa: E402


def actual_topk(season, tier, k):
    """Top-K teams by FINAL regular-season standings (the playoff seeds)."""
    final = build_tier_state(season, tier, asof_md=None)
    ranked = sorted(final.teams.values(), key=lambda t: (-t.wp, t.team))
    return [t.team for t in ranked[:k]], final


def run(label="S25", asof=6, n_sims=10000):
    season = load_season(ROOT / "data" / f"{label}_standings.xlsx", label)
    tiers = season.tiers

    agg = {"sim_hits": 0, "base_hits": 0, "spots": 0,
           "brier": 0.0, "logloss": 0.0, "n_teams": 0,
           "base_brier": 0.0}
    print(f"\n{label}: predict playoffs from match day {asof}  "
          f"({n_sims} sims/tier)\n")
    print(f"{'tier':11} {'K':>2} {'sim hits':>8} {'base hits':>9} "
          f"{'sim Brier':>9} {'base Brier':>10}")
    print("-" * 56)

    for tier in tiers:
        ts = build_tier_state(season, tier, asof_md=asof)
        k = ts.k
        # train Elo only on info available at `asof`
        played = season.matches[(season.matches["tier"] == tier)
                                & (season.matches["is_regular"])
                                & (season.matches["played"])
                                & (season.matches["match_day"] <= asof)]
        ratings = train_elo(played, k=16.0)
        model = make_elo_model(ratings, tier)
        sim = simulate_tier(ts, n_sims=n_sims, model=model)

        summ = sim.summary().set_index("team")
        prob = summ["playoff_prob"]

        actual, _ = actual_topk(season, tier, k)
        actual_set = set(actual)

        # predicted field = top-K by simulated playoff probability
        pred = list(prob.sort_values(ascending=False).index[:k])
        sim_hits = len(set(pred) & actual_set)

        # baseline = top-K by as-of standings
        base_rank = sorted(ts.teams.values(), key=lambda t: (-t.wp, t.team))
        base_pred = [t.team for t in base_rank[:k]]
        base_hits = len(set(base_pred) & actual_set)

        # calibration: Brier/logloss of playoff prob vs actual membership
        teams = list(prob.index)
        y = np.array([1.0 if t in actual_set else 0.0 for t in teams])
        p = np.clip(prob.loc[teams].to_numpy(), 1e-6, 1 - 1e-6)
        brier = float(np.mean((p - y) ** 2))
        logloss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
        # baseline brier: 1.0 for predicted-in, 0.0 else (hard call)
        base_p = np.array([1.0 if t in set(base_pred) else 0.0 for t in teams])
        base_brier = float(np.mean((np.clip(base_p, 1e-6, 1 - 1e-6) - y) ** 2))

        agg["sim_hits"] += sim_hits; agg["base_hits"] += base_hits
        agg["spots"] += k
        agg["brier"] += brier * len(teams); agg["base_brier"] += base_brier * len(teams)
        agg["logloss"] += logloss * len(teams); agg["n_teams"] += len(teams)

        print(f"{tier:11} {k:>2} {f'{sim_hits}/{k}':>8} {f'{base_hits}/{k}':>9} "
              f"{brier:>9.3f} {base_brier:>10.3f}")

    s, b, sp = agg["sim_hits"], agg["base_hits"], agg["spots"]
    print("-" * 56)
    print(f"TOTAL playoff spots: {sp}")
    print(f"  simulator correct:  {s}/{sp}  ({s/sp:.1%})")
    print(f"  baseline  correct:  {b}/{sp}  ({b/sp:.1%})")
    print(f"  sim  Brier (calibration): {agg['brier']/agg['n_teams']:.3f}")
    print(f"  base Brier (hard call):   {agg['base_brier']/agg['n_teams']:.3f}")
    print(f"  -> simulator {'beats' if s >= b else 'trails'} the naive "
          f"standings baseline on hit-rate; lower Brier = better calibrated.")


if __name__ == "__main__":
    asof = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run("S25", asof=asof)
