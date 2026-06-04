"""Current-season projections and rankings (no prior-season data required).

Pace projection: scale a player's production to a full season based on how much
of the season has been played, with an 80% prediction interval that widens for
small samples. Confidence reflects both how far into the season we are and how
many games the player personally has.

This is the "no old data needed" half of the prediction request. The analogs /
comparables half (how similar past players finished) waits on historical
player-season data.
"""
from __future__ import annotations

import math

import pandas as pd

from .engine.compare import _per_game

# counting stats we project (label, column)
_PROJ_STATS = [("Points", "Pts"), ("Goals", "G"), ("Assists", "A"),
               ("Saves", "S"), ("Shots", "SH"), ("Demos", "DM"), ("MVPs", "MVP")]
_Z80 = 1.2816  # 80% two-sided


REGRESS_K = 8   # games of "tier-average prior" mixed into each rate estimate


def _confidence(season_frac: float, gp: int, team_games: int) -> str:
    """High/Medium/Low from how much season is done and the player's game share."""
    share = gp / team_games if team_games else 0
    if season_frac >= 0.6 and gp >= 18 and share >= 0.7:
        return "High"
    if season_frac >= 0.35 and gp >= 10 and share >= 0.5:
        return "Medium"
    return "Low"


def project_player(players: pd.DataFrame, variables: pd.DataFrame,
                   name: str) -> dict | None:
    pg = _per_game(players).copy()
    pg["MVP/g"] = pg["MVP"] / pg["GP"].replace(0, float("nan"))
    rows = pg[pg["Player"].astype(str) == name]
    if rows.empty:
        return None
    rec = rows.sort_values("GP", ascending=False).iloc[0]
    tier = rec["Tier"]

    var = variables.set_index("tier")
    if tier not in var.index:
        return None
    match_days = int(var.loc[tier, "match_days"])
    mds_played = int(var.loc[tier, "mds_played"])
    season_frac = mds_played / match_days if match_days else 0
    team_games_played = mds_played * 4
    team_games_total = match_days * 4

    gp = int(rec["GP"])
    proj_gp = round(gp * (team_games_total / team_games_played)) if team_games_played else gp
    n_rem = max(proj_gp - gp, 0)

    pool = pg[pg["Tier"] == tier]            # tier peers, for the mean (prior)

    projections = []
    for label, col in _PROJ_STATS:
        x = float(rec[col])
        r = x / gp if gp else 0.0                          # raw per-game rate
        mu = float(pool[f"{col}/g"].mean())                # tier-average rate
        # regression to the mean: blend the player's rate with the tier prior,
        # weighted by sample size. Few games -> trust the tier average more.
        r_adj = (gp * r + REGRESS_K * mu) / (gp + REGRESS_K)
        pace = x + r * n_rem                               # naive extrapolation
        proj = x + r_adj * n_rem                           # regression-adjusted
        # variance: Poisson on remaining games at the adjusted rate + rate SE
        se_rate = math.sqrt(r_adj / (gp + REGRESS_K)) if r_adj > 0 else 0.0
        var_total = n_rem * r_adj + (n_rem * se_rate) ** 2
        sd = math.sqrt(var_total)
        lo = max(x, proj - _Z80 * sd)
        hi = proj + _Z80 * sd
        projections.append({
            "label": label, "current": int(x),
            "per_game": round(r, 2), "tier_avg": round(mu, 2),
            "pace": round(pace), "proj": round(proj),
            "low": round(lo), "high": round(hi),
        })

    return {
        "name": name, "tier": tier,
        "gp": gp, "proj_gp": proj_gp,
        "season_frac": season_frac,
        "season_pct": round(season_frac * 100),
        "confidence": _confidence(season_frac, gp, team_games_played),
        "regress_k": REGRESS_K,
        "projections": projections,
    }


def project_all(players: pd.DataFrame, variables: pd.DataFrame,
                stat: str = "G") -> pd.DataFrame:
    """Regression-adjusted projected FINAL totals for every player, for one
    counting stat. Used for projected stat-leader boards."""
    pg = _per_game(players).copy()
    pg["MVP/g"] = pg["MVP"] / pg["GP"].replace(0, float("nan"))
    var = variables.set_index("tier")
    col_g = f"{stat}/g"
    out = pg[["Player", "Tier", "Team", "GP", stat]].copy()
    # season fraction per tier
    md = var["match_days"]
    mp = var["mds_played"]
    out["mult"] = out["Tier"].map(lambda t: (int(md[t]) / int(mp[t]))
                                  if t in var.index and int(mp.get(t, 0)) else 1.0)
    out["proj_gp"] = (out["GP"] * out["mult"]).round()
    out["n_rem"] = (out["proj_gp"] - out["GP"]).clip(lower=0)
    mu = pg.groupby("Tier")[col_g].transform("mean")
    r = pg[stat] / pg["GP"].replace(0, float("nan"))
    r_adj = (pg["GP"] * r + REGRESS_K * mu) / (pg["GP"] + REGRESS_K)
    out["per_game"] = r.round(2)
    out["proj"] = (pg[stat] + r_adj * out["n_rem"]).round()
    return out.sort_values("proj", ascending=False).reset_index(drop=True)


# role classification from a player's stat mix vs tier peers
_ROLE_STATS = [("G/g", "Striker"), ("A/g", "Playmaker"), ("S/g", "Anchor")]


def player_role(players: pd.DataFrame, name: str) -> dict | None:
    pg = _per_game(players)
    rows = pg[pg["Player"].astype(str) == name]
    if rows.empty:
        return None
    rec = rows.sort_values("GP", ascending=False).iloc[0]
    pool = pg[pg["Tier"] == rec["Tier"]]
    pcts = {}
    for col, _role in _ROLE_STATS:
        p = pool[col].dropna()
        pcts[col] = int(round((p < rec[col]).mean() * 100)) if len(p) else 50
    top_col = max(pcts, key=pcts.get)
    top_role = dict(_ROLE_STATS)[top_col]
    high = [c for c, v in pcts.items() if v >= 70]
    if len(high) >= 3:
        role = "Star all-rounder"
    elif pcts[top_col] < 55:
        role = "Role player"
    elif len(high) == 2:
        role = " / ".join(dict(_ROLE_STATS)[c] for c in
                          sorted(high, key=lambda c: -pcts[c]))
    else:
        role = top_role
    return {"role": role,
            "goals_pct": pcts["G/g"], "assists_pct": pcts["A/g"],
            "saves_pct": pcts["S/g"]}


def player_rankings(players: pd.DataFrame, name: str) -> dict | None:
    """Where a player ranks among all current players and within their tier."""
    df = _per_game(players)
    rows = df[df["Player"].astype(str) == name]
    if rows.empty:
        return None
    rec = rows.sort_values("GP", ascending=False).iloc[0]
    tier = rec["Tier"]
    out = {"tier": tier, "n_all": len(df),
           "n_tier": int((df["Tier"] == tier).sum()), "ranks": []}
    for label, col in [("Points", "Pts"), ("Goals", "G"), ("Assists", "A"),
                       ("Saves", "S"), ("Demos", "DM")]:
        pg = f"{col}/g"
        val = rec[pg]
        # rank by per-game rate (1 = best); ties share the better rank
        overall = int((df[pg] > val).sum()) + 1
        tier_pool = df[df["Tier"] == tier]
        tier_rank = int((tier_pool[pg] > val).sum()) + 1
        out["ranks"].append({
            "label": label, "per_game": round(float(val), 2),
            "overall": overall, "tier_rank": tier_rank,
        })
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rsc.api import RSCClient
    from rsc.ingest import load_season

    root = Path(__file__).resolve().parents[1]
    name = sys.argv[1] if len(sys.argv) > 1 else "Reyes."
    players = RSCClient().player_stats()
    variables = load_season(root / "data" / "S26_standings.xlsx", "S26").variables

    r = player_rankings(players, name)
    p = project_player(players, variables, name)
    print(f"\n{name} ({p['tier']}) - {p['season_pct']}% of season played - "
          f"confidence: {p['confidence']}")
    print(f"rank among {r['n_all']} players (per-game):")
    for rk in r["ranks"]:
        print(f"  {rk['label']:8} {rk['per_game']:>6}/g   "
              f"#{rk['overall']} overall   #{rk['tier_rank']} in {r['tier']}")
    print(f"\nprojected final ({p['gp']} -> {p['proj_gp']} games):")
    for pr in p["projections"]:
        print(f"  {pr['label']:8} now {pr['current']:>5}  -> "
              f"{pr['proj']:>5}  [{pr['low']}-{pr['high']}]")
