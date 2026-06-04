"""High-level orchestration: the API the CLI and future Flask app call.

Fuses the three engine layers for a single team or tier:
  - certainty  (clinch.py)      -> CLINCHED / ELIMINATED guarantees
  - odds       (simulate.py)    -> calibrated playoff probability (Elo-driven)
  - what-it-needs (simulate.py) -> P(playoffs | final wins) curve

Elo is trained on the season's games to date and used as the simulation model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from .ingest import load_season, Season
from .engine.clinch import build_tier_state, evaluate_team, TierState
from .engine.simulate import simulate_tier, playoff_curve, SimResult
from .engine.predict import train_elo, make_elo_model, backtest, confidence

_DATA = Path(__file__).resolve().parent.parent / "data"


def load(label: str = "S26") -> Season:
    return load_season(_DATA / f"{label}_standings.xlsx", label)


def elo_model_for(season: Season, tier: str):
    played = season.matches[(season.matches["tier"] == tier)
                            & (season.matches["is_regular"])
                            & (season.matches["played"])]
    return make_elo_model(train_elo(played), tier)


def model_confidence(season: Season) -> dict:
    """Season-wide backtest -> is the model trustworthy enough to show?"""
    return confidence(backtest(season.matches))


@dataclass
class Outlook:
    team: str
    tier: str
    rank: int
    record: str
    playoff_prob: float
    avg_seed: float
    status: str             # CLINCHED / ELIMINATED / alive
    clinch_wins: int | None
    elim_losses: int | None
    headline: str
    curve: list             # [{final_wins, extra_wins, p_playoffs, sample}, ...]


def tier_state(season: Season, tier: str) -> TierState:
    return build_tier_state(season, tier)


def tier_odds(season: Season, tier: str, n_sims: int = 30000) -> SimResult:
    ts = build_tier_state(season, tier)
    return simulate_tier(ts, n_sims=n_sims, model=elo_model_for(season, tier))


def team_outlook(season: Season, tier: str, team: str,
                 sim: SimResult | None = None, n_sims: int = 30000) -> Outlook:
    ts = build_tier_state(season, tier)
    verdict = evaluate_team(season, tier, team, ts=ts)
    sim = sim or simulate_tier(ts, n_sims=n_sims, model=elo_model_for(season, tier))
    summ = sim.summary().set_index("team")
    curve = playoff_curve(sim, team).to_dict("records")
    status = ("CLINCHED" if verdict.clinched
              else "ELIMINATED" if verdict.eliminated else "alive")
    return Outlook(
        team=team, tier=tier, rank=verdict.rank, record=verdict.record,
        playoff_prob=float(summ.loc[team, "playoff_prob"]),
        avg_seed=float(summ.loc[team, "avg_seed"]),
        status=status,
        clinch_wins=verdict.clinch_wins, elim_losses=verdict.elim_losses,
        headline=verdict.headline(), curve=curve,
    )


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "S26"
    tier = sys.argv[2] if len(sys.argv) > 2 else "Premier"
    season = load(label)
    conf = model_confidence(season)
    gate = "ENABLED" if conf["sufficient"] else "LOW-CONFIDENCE"
    print(f"\n{label} / {tier}  | model: Elo, predictions {gate} "
          f"(skill vs coin {conf['logloss_skill_vs_coin']:+.2%})")

    sim = tier_odds(season, tier)
    s = sim.summary()
    s["playoff_prob"] = (s["playoff_prob"] * 100).round(1)
    s["avg_seed"] = s["avg_seed"].round(1)
    print("\n" + s[["team", "cur_w", "playoff_prob", "avg_seed"]].to_string(index=False))

    if len(sys.argv) > 3:
        team = sys.argv[3]
        o = team_outlook(season, tier, team, sim=sim)
        print(f"\n>>> {team}: {o.headline}")
        print(f"    playoff odds: {o.playoff_prob:.1%}   avg seed: {o.avg_seed:.1f}")
