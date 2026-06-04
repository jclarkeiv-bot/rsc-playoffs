"""Game win-probability models + honest backtesting.

Standings are game-based, so we model the probability of winning a single game.
Primary model is Elo, trained chronologically per tier (teams only play within
their tier, so each tier is its own rating pool). A series result "a-b" is
treated as `a` games won by Away and `b` by Home, updated as a batch using the
pre-series ratings (order-independent).

We never trust a model on faith: `backtest()` walks the season in date order,
predicting each game *before* updating, and scores Brier + log-loss against two
baselines (coin flip, cumulative win%/Log5). `confidence()` turns that into a
go/no-go gate so the app only shows predictions when the model beats a coin.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulate import log5

ELO_BASE = 1500.0
ELO_SCALE = 400.0


def _expected(r_a: float, r_b: float) -> float:
    """Per-game expected score (win prob) for A vs B."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / ELO_SCALE))


def _chrono(matches: pd.DataFrame) -> pd.DataFrame:
    """Played regular-season series in chronological order."""
    m = matches[matches["played"] & matches["is_regular"]].copy()
    # date may be missing for some rows; fall back to match_day as tiebreak.
    m["_d"] = pd.to_datetime(m["date"], errors="coerce")
    return m.sort_values(["_d", "match_day"], kind="stable")


def train_elo(matches: pd.DataFrame, k: float = 16.0,
              prior: dict[tuple[str, str], float] | None = None,
              ) -> dict[tuple[str, str], float]:
    """Elo ratings keyed by (tier, team). `k` is per-game.

    `prior` lets a previous season seed starting ratings (regressed toward base
    by the caller). Unknown teams start at ELO_BASE.
    """
    r: dict[tuple[str, str], float] = defaultdict(lambda: ELO_BASE)
    if prior:
        r.update(prior)
    for m in _chrono(matches).itertuples(index=False):
        ka, kb = (m.tier, m.away), (m.tier, m.home)
        ra, rb = r[ka], r[kb]
        e_a = _expected(ra, rb)
        # actual A game-wins = away_g; expected over 4 games = 4*e_a
        delta = k * (m.away_g - 4 * e_a)
        r[ka] = ra + delta
        r[kb] = rb - delta
    return dict(r)


def make_elo_model(ratings: dict, tier: str):
    """A model(a, b) -> P(a beats b in one game) for use in simulate_tier."""
    def model(a, b):
        return _expected(ratings.get((tier, a), ELO_BASE),
                         ratings.get((tier, b), ELO_BASE))
    return model


# ---- backtesting -------------------------------------------------------------

@dataclass
class Score:
    n_games: int
    brier: float
    logloss: float
    def __str__(self):
        return f"n={self.n_games:5d}  brier={self.brier:.4f}  logloss={self.logloss:.4f}"


def _accumulate(p: float, away_g: int, home_g: int, eps: float = 1e-6):
    """Brier + log-loss contributions for a series, scored per game from the
    Away perspective (p = P(away wins a game))."""
    p = min(max(p, eps), 1 - eps)
    brier = away_g * (p - 1) ** 2 + home_g * (p - 0) ** 2
    logloss = -(away_g * math.log(p) + home_g * math.log(1 - p))
    return brier, logloss


def backtest(matches: pd.DataFrame, k: float = 16.0) -> dict[str, Score]:
    """Walk-forward backtest. For each series, predict with info available
    *before* it, then update. Scores three models over all games.
    """
    r: dict[tuple[str, str], float] = defaultdict(lambda: ELO_BASE)
    wl: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [w, l]

    tot = {m: [0, 0.0, 0.0] for m in ("elo", "winpct", "coin")}  # n, brier, ll

    for m in _chrono(matches).itertuples(index=False):
        ka, kb = (m.tier, m.away), (m.tier, m.home)
        ng = m.away_g + m.home_g

        # --- predictions (pre-update) ---
        e_elo = _expected(r[ka], r[kb])

        wa, la = wl[ka]; wb, lb = wl[kb]
        pa = (wa + 4) / (wa + la + 8)   # regressed win% (8 phantom .500 games)
        pb = (wb + 4) / (wb + lb + 8)
        e_wp = log5(pa, pb)

        for name, p in (("elo", e_elo), ("winpct", e_wp), ("coin", 0.5)):
            br, ll = _accumulate(p, m.away_g, m.home_g)
            tot[name][0] += ng
            tot[name][1] += br
            tot[name][2] += ll

        # --- updates (post-prediction) ---
        delta = k * (m.away_g - 4 * e_elo)
        r[ka] += delta; r[kb] -= delta
        wl[ka][0] += m.away_g; wl[ka][1] += m.home_g
        wl[kb][0] += m.home_g; wl[kb][1] += m.away_g

    return {name: Score(int(n), brier / n, ll / n)
            for name, (n, brier, ll) in tot.items()}


def confidence(scores: dict[str, Score]) -> dict:
    """Skill of the model vs a coin flip; a simple go/no-go gate."""
    elo, coin = scores["elo"], scores["coin"]
    skill = 1 - elo.logloss / coin.logloss          # Brier/LL skill score
    return {
        "logloss_skill_vs_coin": skill,
        "brier_skill_vs_coin": 1 - elo.brier / coin.brier,
        "beats_winpct": elo.logloss < scores["winpct"].logloss,
        # enough signal to show predictions?
        "sufficient": skill > 0.02 and elo.logloss < scores["winpct"].logloss + 1e-9,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rsc.ingest import load_season

    root = Path(__file__).resolve().parents[2]
    for label in (sys.argv[1:] or ["S25", "S26"]):
        season = load_season(root / "data" / f"{label}_standings.xlsx", label)
        scores = backtest(season.matches)
        conf = confidence(scores)
        print(f"\n=== {label} walk-forward backtest (per-game) ===")
        for name in ("coin", "winpct", "elo"):
            print(f"  {name:7s} {scores[name]}")
        print(f"  Elo skill vs coin (logloss): {conf['logloss_skill_vs_coin']:+.3%}"
              f"   beats win%%: {conf['beats_winpct']}"
              f"   -> predictions {'ENABLED' if conf['sufficient'] else 'WITHHELD'}")
