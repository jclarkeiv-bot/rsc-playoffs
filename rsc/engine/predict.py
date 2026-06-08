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


def make_blended_model(ratings: dict, strengths: dict, tier: str,
                       skill_weight: float = 0.25, skill_scale: float = 30.0):
    """Per-game win prob blending team form (Elo) with roster skill (average
    player OVR). skill_weight is how much the roster-skill signal counts.

    Still a per-GAME probability, so it feeds the game-based simulation/standings
    exactly like the Elo model - match/series outcomes are never the target.
    """
    def model(a, b):
        p_elo = _expected(ratings.get((tier, a), ELO_BASE),
                          ratings.get((tier, b), ELO_BASE))
        sa, sb = strengths.get((tier, a)), strengths.get((tier, b))
        if sa is None or sb is None or skill_weight <= 0:
            return p_elo
        p_skill = 1.0 / (1.0 + 10 ** ((sb - sa) / skill_scale))
        return (1 - skill_weight) * p_elo + skill_weight * p_skill
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


def accuracy_by_matchday(matches: pd.DataFrame, k: float = 16.0,
                         prior: dict | None = None) -> dict:
    """Walk-forward accuracy bucketed by match day. For each played regular
    series, the Elo model (trained only on EARLIER games) predicts the winner;
    we score hit-rate + game-level Brier, grouped by match day, so you can see
    the model getting sharper as the season fills in.

    `prior` seeds starting Elo (e.g. from career skill) - a legitimate pre-season
    signal, so comparing prior vs no-prior is a fair backtest of "does history help".
    """
    r: dict[tuple[str, str], float] = defaultdict(lambda: ELO_BASE)
    if prior:
        r.update(prior)
    buckets: dict[int, list[float]] = {}   # md -> [series, hits, brier, games]

    for m in _chrono(matches).itertuples(index=False):
        md = m.match_day
        if md is None:
            continue
        ka, kb = (m.tier, m.away), (m.tier, m.home)
        e = _expected(r[ka], r[kb])             # P(away wins a game)
        b = buckets.setdefault(int(md), [0, 0, 0.0, 0, 0])
        ng = m.away_g + m.home_g
        fav_away = e >= 0.5
        # series-winner hit (skip ties - no winner)
        if m.away_g != m.home_g:
            hit = fav_away == (m.away_g > m.home_g)
            b[0] += 1
            b[1] += 1 if hit else 0
        # game-level: how many individual games the favored side actually won
        b[4] += m.away_g if fav_away else m.home_g
        # game-level Brier from the away perspective
        b[2] += m.away_g * (e - 1) ** 2 + m.home_g * (e - 0) ** 2
        b[3] += ng
        # update
        delta = k * (m.away_g - 4 * e)
        r[ka] += delta
        r[kb] -= delta

    rows, cum_s, cum_h, cum_g, cum_fg = [], 0, 0, 0, 0
    for md in sorted(buckets):
        series, hits, brier, games, fav_games = buckets[md]
        if games == 0:                          # skip empty/forfeit-only days
            continue
        cum_s += series
        cum_h += hits
        cum_g += games
        cum_fg += fav_games
        rows.append({
            "match_day": md,
            "n_series": int(series),
            "n_games": int(games),
            "game_acc": round(fav_games / games * 100, 1),     # game-level
            "cum_game_acc": round(cum_fg / cum_g * 100, 1),
            "hit_rate": round(hits / series * 100, 1) if series else None,
            "cum_hit_rate": round(cum_h / cum_s * 100, 1) if cum_s else None,
            "brier": round(brier / games, 3) if games else None,
        })
    return {"rows": rows,
            "overall_game_acc": round(cum_fg / cum_g * 100, 1) if cum_g else None,
            "overall_hit_rate": round(cum_h / cum_s * 100, 1) if cum_s else None,
            "total_series": cum_s, "total_games": cum_g, "coin_hit_rate": 50.0}


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
