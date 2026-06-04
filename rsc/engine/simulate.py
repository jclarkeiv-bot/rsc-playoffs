"""Monte Carlo simulation of the remaining schedule.

Exact top-K clinch accounting for who-plays-whom is NP-hard, so we simulate.
We play out every remaining regular-season series (4 independent games each)
using a per-game win-probability model, many times, and aggregate:

  - playoff_prob   how often each team lands in the tier's top-K
  - avg_seed       mean final seed
  - the conditional curve: P(team makes playoffs | team finishes on W wins),
    which answers "exactly how they need to perform, given everyone else".

The win-probability model is pluggable. Default is Log5 from season game win%
(regressed toward .500 for small samples); Phase 4 swaps in Elo/Pythagorean
trained on S25 and backtests it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clinch import TierState, build_tier_state


def log5(p_a: float, p_b: float) -> float:
    """Probability A beats B in one game given each team's base win prob."""
    denom = p_a + p_b - 2 * p_a * p_b
    if denom == 0:
        return 0.5
    return (p_a - p_a * p_b) / denom


def make_log5_model(ts: TierState, regress_games: int = 8):
    """Per-game P(a beats b) using Log5 on regressed season win%.

    Regression pulls a team's win% toward .500 by `regress_games` phantom .500
    games, so a 4-0 start doesn't read as a 100% team.
    """
    base = {}
    for name, t in ts.teams.items():
        w = t.w + 0.5 * regress_games
        g = t.gp + regress_games
        base[name] = w / g if g else 0.5
    return lambda a, b: log5(base[a], base[b])


@dataclass
class SimResult:
    tier: str
    k: int
    n_sims: int
    teams: list[str]
    current_w: np.ndarray          # (n_teams,)
    final_w: np.ndarray            # (n_sims, n_teams) total wins
    made_playoffs: np.ndarray      # (n_sims, n_teams) bool
    seed: np.ndarray               # (n_sims, n_teams) int rank 1..N
    champion: np.ndarray | None = None   # (n_sims,) winning team index

    def summary(self) -> pd.DataFrame:
        title = (None if self.champion is None else
                 np.array([(self.champion == i).mean() for i in range(len(self.teams))]))
        data = {
            "team": self.teams,
            "cur_w": self.current_w.astype(int),
            "playoff_prob": self.made_playoffs.mean(axis=0),
            "avg_final_w": self.final_w.mean(axis=0),
            "avg_seed": self.seed.mean(axis=0),
            "p_top_seed": (self.seed == 1).mean(axis=0),
        }
        if title is not None:
            data["title_prob"] = title
        return pd.DataFrame(data).sort_values("playoff_prob", ascending=False).reset_index(drop=True)

    def title_board(self) -> pd.DataFrame:
        s = self.summary()
        if "title_prob" not in s:
            return pd.DataFrame()
        return s[s["title_prob"] > 0].sort_values("title_prob", ascending=False)[
            ["team", "cur_w", "avg_seed", "playoff_prob", "title_prob"]].reset_index(drop=True)

    def title_confidence(self) -> str:
        """How decisive the title race is: concentration of the favourite."""
        if self.champion is None:
            return "n/a"
        probs = np.array([(self.champion == i).mean() for i in range(len(self.teams))])
        top = probs.max()
        if top >= 0.40:
            return "High"
        if top >= 0.22:
            return "Medium"
        return "Low"


def _bracket_seed_order(b: int) -> list[int]:
    """Standard single-elimination seed slots for a bracket of size b
    (power of 2). e.g. b=8 -> [1,8,4,5,2,7,3,6] so #1 plays #8, #4 plays #5..."""
    order = [1, 2]
    while len(order) < b:
        n = len(order) * 2
        order = [s for x in order for s in (x, n + 1 - x)]
    return order


def _series_win_prob(p: np.ndarray, games: int) -> np.ndarray:
    """P(win a best-of-`games` series) given per-game win prob p (vectorized)."""
    from math import comb
    need = games // 2 + 1
    out = np.zeros_like(p, dtype=float)
    for k in range(need, games + 1):
        out += comb(games, k) * p ** k * (1 - p) ** (games - k)
    return out


def _simulate_playoffs(ranks: np.ndarray, k: int, P: np.ndarray,
                       rng) -> np.ndarray:
    """Given per-sim seeds, play a seeded single-elim bracket and return the
    champion team index per sim. Bo5 in early rounds, Bo7 in semis + final."""
    n_sims = ranks.shape[0]
    order = np.argsort(ranks, axis=1)        # team idx by seed (col0 = #1 seed)
    b = 1
    while b < k:
        b *= 2
    slot_seeds = _bracket_seed_order(b)
    slots = np.full((n_sims, b), -1, dtype=np.int64)
    for s, seed_no in enumerate(slot_seeds):
        if seed_no <= k:                     # seeds beyond k are byes (-1)
            slots[:, s] = order[:, seed_no - 1]

    size = b
    while size > 1:
        bo = 7 if size <= 4 else 5           # semis (size 4) + final (2) are Bo7
        nxt = np.empty((n_sims, size // 2), dtype=np.int64)
        for m in range(size // 2):
            a, bb = slots[:, 2 * m], slots[:, 2 * m + 1]
            pa = P[np.clip(a, 0, None), np.clip(bb, 0, None)]
            a_wins = rng.random(n_sims) < _series_win_prob(pa, bo)
            win = np.where(a_wins, a, bb)
            win = np.where(bb < 0, a, win)   # opponent is a bye -> a advances
            win = np.where(a < 0, bb, win)
            nxt[:, m] = win
        slots, size = nxt, size // 2
    return slots[:, 0]


def simulate_tier(ts: TierState, n_sims: int = 20000,
                  model=None, seed: int = 12345,
                  playoffs: bool = True) -> SimResult:
    rng = np.random.default_rng(seed)
    model = model or make_log5_model(ts)

    teams = list(ts.teams)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    cur = np.array([ts.teams[t].w for t in teams], dtype=np.int32)

    wins = np.tile(cur, (n_sims, 1)).astype(np.int32)

    # Each remaining series -> 4 Bernoulli games for the away team.
    for a, opps in ts.remaining_vs.items():
        ia = idx[a]
        for b, games in opps.items():
            # remaining_vs is symmetric; count each unordered pair once (a<b).
            if a >= b:
                continue
            ib = idx[b]
            series = games  # total games left between a and b (multiple of 4)
            p = model(a, b)
            a_wins = rng.binomial(series, p, size=n_sims).astype(np.int32)
            wins[:, ia] += a_wins
            wins[:, ib] += series - a_wins

    # Rank within each sim (1 = most wins). Random jitter breaks ties uniformly,
    # standing in for tiebreakers we don't fully model yet.
    jitter = rng.random((n_sims, n))
    order_key = wins + jitter  # break ties randomly
    # seed = 1 + number of teams strictly ranked above
    ranks = (order_key[:, None, :] > order_key[:, :, None]).sum(axis=2) + 1
    made = ranks <= ts.k

    champion = None
    if playoffs and ts.k >= 1:
        # fixed pairwise per-game win-prob matrix (model is constant across sims)
        P = np.array([[model(teams[i], teams[j]) for j in range(n)]
                      for i in range(n)], dtype=float)
        champion = _simulate_playoffs(ranks, ts.k, P, rng)

    return SimResult(tier=ts.tier, k=ts.k, n_sims=n_sims, teams=teams,
                     current_w=cur, final_w=wins, made_playoffs=made, seed=ranks,
                     champion=champion)


def playoff_curve(sim: SimResult, team: str) -> pd.DataFrame:
    """P(make playoffs | team finishes on exactly W wins), from the sim.

    This is the "what do we need to do" answer: each row is a possible final
    win total and the playoff probability conditional on hitting it.
    """
    j = sim.teams.index(team)
    fw = sim.final_w[:, j]
    made = sim.made_playoffs[:, j]
    rows = []
    for w in range(int(fw.min()), int(fw.max()) + 1):
        mask = fw == w
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        rows.append({
            "final_wins": w,
            "extra_wins": w - int(sim.current_w[j]),
            "p_playoffs": made[mask].mean(),
            "sample": cnt,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rsc.ingest import load_season

    root = Path(__file__).resolve().parents[2]
    label = sys.argv[1] if len(sys.argv) > 1 else "S26"
    tier = sys.argv[2] if len(sys.argv) > 2 else "Premier"
    season = load_season(root / "data" / f"{label}_standings.xlsx", label)
    ts = build_tier_state(season, tier)
    sim = simulate_tier(ts, n_sims=30000)

    print(f"\n{label} - {tier}  (top {ts.k} make playoffs; {sim.n_sims} sims)\n")
    s = sim.summary()
    s["playoff_prob"] = (s["playoff_prob"] * 100).round(1)
    s["p_top_seed"] = (s["p_top_seed"] * 100).round(1)
    s["avg_final_w"] = s["avg_final_w"].round(1)
    s["avg_seed"] = s["avg_seed"].round(1)
    print(s.to_string(index=False))

    if len(sys.argv) > 3:
        team = sys.argv[3]
        print(f"\nWhat {team} needs - P(playoffs | final wins):\n")
        c = playoff_curve(sim, team)
        c["p_playoffs"] = (c["p_playoffs"] * 100).round(1)
        print(c.to_string(index=False))
