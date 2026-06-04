"""Schedule-aware clinch / elimination engine.

Given the current standings and the *actual remaining matchups* for every team
in a tier, compute — for any team — what it takes to reach the tier's top-K
playoff cut, factoring in every other team.

Definitions (all standings are game-based win%; within a tier every team plays
the same total number of games, so ranking by wins == ranking by win%):

  worst_case_seed   T loses out, everyone else wins out  -> the lowest T can fall
  best_case_seed    T wins out, everyone else loses out  -> the highest T can rise
  clinched          worst_case_seed <= K  (in no matter what)
  eliminated        best_case_seed  >  K  (out no matter what)
  clinch_number     fewest additional game-wins that GUARANTEE a top-K finish
  elim_number       fewest additional game-losses that GUARANTEE elimination
  magic_number      traditional (wins_T + losses_chaser) vs the first-team-out

The guarantee uses independent worst-case bounds for rivals (each rival assumed
to win all its remaining games). This is the safe direction: it never tells a
team it has clinched when it hasn't. It can be very slightly conservative when
a rival's remaining games include games against T; that only ever makes
clinch_number a hair larger, never smaller. A future refinement can tighten it
with head-to-head accounting / max-flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .standings import compute_standings


@dataclass
class TeamState:
    team: str
    w: int
    l: int
    gp: int
    rem: int          # remaining regular-season games
    @property
    def wp(self) -> float:
        return self.w / self.gp if self.gp else 0.0
    @property
    def max_w(self) -> int:   # win out
        return self.w + self.rem
    @property
    def min_w(self) -> int:   # lose out
        return self.w


@dataclass
class TierState:
    tier: str
    k: int                                  # playoff spots
    first_out_rank: int                     # rank of first team out (usually k+1)
    total_games: int                        # per-team season length
    teams: dict[str, TeamState] = field(default_factory=dict)
    remaining_vs: dict[str, dict[str, int]] = field(default_factory=dict)  # head-to-head games left

    def ranked(self) -> list[TeamState]:
        return sorted(self.teams.values(), key=lambda t: (-t.wp, t.team))


def build_tier_state(season, tier: str, asof_md: int | None = None) -> TierState:
    """Current (or as-of) clinch state for a tier.

    `asof_md` reconstructs the state as it stood after match day N: regular
    games with match_day <= N count as played, the rest as remaining. This lets
    us backtest predictions against a completed season. When None, the real
    played/unplayed flags are used (the live, in-progress view).
    """
    var = season.variables.set_index("tier")
    k = int(var.loc[tier, "playoff_spots"])
    first_out = int(var.loc[tier, "first_team_out"])
    total_games = int(var.loc[tier, "match_days"]) * 4

    m = season.matches[(season.matches["tier"] == tier)
                       & (season.matches["is_regular"])]
    if asof_md is None:
        played = m[m["played"]]
        rem = m[~m["played"]]
    else:
        played = m[m["played"] & (m["match_day"] <= asof_md)]
        rem = m[m["match_day"] > asof_md]

    # Every team that appears anywhere in the tier's regular season.
    names = set(m["away"]).union(m["home"])
    st = {name: TeamState(name, 0, 0, 0, 0) for name in names}
    for r in played.itertuples():
        st[r.away].w += r.away_g; st[r.away].l += r.home_g
        st[r.home].w += r.home_g; st[r.home].l += r.away_g
    for t in st.values():
        t.gp = t.w + t.l

    # Remaining series -> games (4 each) + head-to-head counts.
    h2h: dict[str, dict[str, int]] = {t: {} for t in st}
    for r in rem.itertuples():
        for a, b in ((r.away, r.home), (r.home, r.away)):
            if a in st:
                st[a].rem += 4
                h2h[a][b] = h2h[a].get(b, 0) + 4
    return TierState(tier=tier, k=k, first_out_rank=first_out,
                     total_games=total_games, teams=st, remaining_vs=h2h)


# ---- core computations -------------------------------------------------------

def _seed_if(ts: TierState, team: str, team_final_w: int,
             rivals_win_out: bool) -> int:
    """Seed (1=best) for `team` finishing on `team_final_w` wins, when every
    rival either wins out (worst case for team) or loses out (best case)."""
    ahead = 0
    for name, r in ts.teams.items():
        if name == team:
            continue
        rival_w = r.max_w if rivals_win_out else r.min_w
        # Strictly-ahead counts for sure; a tie (>=) is resolved by tiebreaker,
        # which we treat pessimistically/optimistically to match the scenario.
        if rivals_win_out:
            if rival_w >= team_final_w:   # pessimistic: ties go against team
                ahead += 1
        else:
            if rival_w > team_final_w:    # optimistic: ties go to team
                ahead += 1
    return ahead + 1


def worst_case_seed(ts: TierState, team: str) -> int:
    t = ts.teams[team]
    return _seed_if(ts, team, t.min_w, rivals_win_out=True)


def best_case_seed(ts: TierState, team: str) -> int:
    t = ts.teams[team]
    return _seed_if(ts, team, t.max_w, rivals_win_out=False)


def clinch_number(ts: TierState, team: str) -> int | None:
    """Fewest additional game-wins that guarantee a top-K finish, or None if
    it can't be guaranteed within the remaining games (i.e. not yet clinchable)."""
    t = ts.teams[team]
    for k in range(0, t.rem + 1):
        final_w = t.w + k
        # team wins k of its games; rivals still assumed to win out.
        if _seed_if(ts, team, final_w, rivals_win_out=True) <= ts.k:
            return k
    return None


def elim_number(ts: TierState, team: str) -> int | None:
    """Fewest additional game-losses that guarantee elimination, or None if the
    team cannot be eliminated within its remaining games (i.e. already safe-ish)."""
    t = ts.teams[team]
    for losses in range(0, t.rem + 1):
        # team loses `losses` games -> its max remaining wins = rem - losses.
        final_max_w = t.w + (t.rem - losses)
        # eliminated if, even winning all the rest, >= K teams certainly finish ahead
        certain_ahead = sum(
            1 for n, r in ts.teams.items()
            if n != team and r.min_w > final_max_w
        )
        if certain_ahead >= ts.k:
            return losses
    return None


def magic_number_vs_first_out(ts: TierState, team: str) -> int | None:
    """Traditional magic number to clinch a spot over the first-team-out:
    M = (games left in the race) + 1 - wins_T - losses_chaser, where the chaser
    is the strongest team currently outside the cut. Reaches 0 when clinched
    over that chaser. Reported mainly to cross-check the sheet's `M#`."""
    ranked = ts.ranked()
    if len(ranked) <= ts.k:
        return None
    chaser = ranked[ts.k]  # 0-indexed: the (K+1)th team = first team out
    t = ts.teams[team]
    m = ts.total_games + 1 - t.w - chaser.l
    return max(m, 0)


# ---- reporting ---------------------------------------------------------------

@dataclass
class TeamVerdict:
    team: str
    tier: str
    rank: int
    record: str
    wp: float
    rem: int
    k: int
    clinched: bool
    eliminated: bool
    worst_seed: int
    best_seed: int
    clinch_wins: int | None
    elim_losses: int | None
    magic_vs_first_out: int | None

    def headline(self) -> str:
        if self.clinched:
            return f"CLINCHED — in the playoffs no matter what (worst-case seed #{self.worst_seed})."
        if self.eliminated:
            return f"ELIMINATED — cannot reach top {self.k} (best-case seed #{self.best_seed})."
        parts = []
        if self.clinch_wins is not None:
            parts.append(f"win {self.clinch_wins} of their last {self.rem} games to CLINCH")
        else:
            parts.append("cannot yet mathematically clinch")
        if self.elim_losses is not None:
            parts.append(f"lose {self.elim_losses} to be ELIMINATED")
        return f"IN THE HUNT — {', '.join(parts)}. Range: seed #{self.best_seed}–#{self.worst_seed}."


def evaluate_team(season, tier: str, team: str,
                  ts: TierState | None = None) -> TeamVerdict:
    ts = ts or build_tier_state(season, tier)
    if team not in ts.teams:
        raise KeyError(f"{team!r} not found in {tier}. "
                       f"Teams: {sorted(ts.teams)}")
    t = ts.teams[team]
    ranked = ts.ranked()
    rank = next(i for i, r in enumerate(ranked, 1) if r.team == team)
    ws = worst_case_seed(ts, team)
    bs = best_case_seed(ts, team)
    return TeamVerdict(
        team=team, tier=tier, rank=rank,
        record=f"{t.w}-{t.l}", wp=t.wp, rem=t.rem, k=ts.k,
        clinched=ws <= ts.k,
        eliminated=bs > ts.k,
        worst_seed=ws, best_seed=bs,
        clinch_wins=clinch_number(ts, team),
        elim_losses=elim_number(ts, team),
        magic_vs_first_out=magic_number_vs_first_out(ts, team),
    )


def tier_report(season, tier: str) -> pd.DataFrame:
    ts = build_tier_state(season, tier)
    rows = []
    for r in ts.ranked():
        v = evaluate_team(season, tier, r.team, ts=ts)
        rows.append({
            "rank": v.rank, "team": v.team, "record": v.record,
            "wp": round(v.wp, 3), "rem": v.rem,
            "status": ("CLINCHED" if v.clinched else
                       "ELIM" if v.eliminated else "alive"),
            "clinch_wins": v.clinch_wins, "elim_losses": v.elim_losses,
            "best_seed": v.best_seed, "worst_seed": v.worst_seed,
            "magic_vs_1stout": v.magic_vs_first_out,
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
    print(f"\n{label} — {tier}  (playoff cut: top "
          f"{int(season.variables.set_index('tier').loc[tier,'playoff_spots'])})\n")
    print(tier_report(season, tier).to_string(index=False))
    if len(sys.argv) > 3:
        team = sys.argv[3]
        print()
        print(evaluate_team(season, tier, team).headline())
