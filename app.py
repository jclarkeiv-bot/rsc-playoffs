"""RSC Playoffs - Flask UI.

Sits on top of rsc.playoffs (clinch + Elo odds + curves) and rsc.engine.compare.
Sims are cached per (season, tier) since each takes a few seconds.

Run:  python app.py    ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import numpy as np
from flask import Flask, render_template, request, redirect, url_for

from rsc import playoffs as P
from rsc import profiles
from rsc import project
from rsc import rating
from rsc import advanced
from rsc.engine.clinch import build_tier_state, evaluate_team
from rsc.engine.simulate import playoff_curve
from rsc.engine.compare import compare_players, compare_teams

app = Flask(__name__)

SEASON_LABEL = "S26"
_season = None
_sim_cache: dict[str, object] = {}
_players_cache = None


def season():
    global _season
    if _season is None:
        _season = P.load(SEASON_LABEL)
    return _season


def tier_sim(tier: str):
    if tier not in _sim_cache:
        _sim_cache[tier] = P.tier_odds(season(), tier)
    return _sim_cache[tier]


def players_df():
    return profiles.players()


@app.route("/")
def index():
    s = season()
    conf = P.model_confidence(s)
    return render_template("index.html", tiers=s.tiers, label=SEASON_LABEL,
                           conf=conf)


@app.route("/tier/<tier>")
def tier(tier):
    s = season()
    if tier not in s.tiers:
        return redirect(url_for("index"))
    sim = tier_sim(tier)
    ts = build_tier_state(s, tier)
    summ = sim.summary().set_index("team")
    rows = []
    for r in ts.ranked():
        v = evaluate_team(s, tier, r.team, ts=ts)
        rows.append({
            "rank": v.rank, "team": r.team, "record": v.record,
            "wp": r.wp, "rem": r.rem,
            "prob": float(summ.loc[r.team, "playoff_prob"]),
            "title": float(summ.loc[r.team, "title_prob"]) if "title_prob" in summ else 0.0,
            "avg_seed": float(summ.loc[r.team, "avg_seed"]),
            "status": v.headline().split(" -")[0].split(" —")[0],
            "clinched": v.clinched, "eliminated": v.eliminated,
        })
    v = s.variables.set_index("tier")
    vr = v.loc[tier]
    sched = {"played": int(vr["mds_played"]), "total": int(vr["match_days"]),
             "left": int(vr["match_days"]) - int(vr["mds_played"])}

    # projected final standings (from the simulation)
    total_games = int(vr["match_days"]) * 4
    avg_w = sim.final_w.mean(axis=0)
    med_seed = np.median(sim.seed, axis=0)
    proj = []
    for i, t in enumerate(sim.teams):
        pw = int(round(avg_w[i]))
        proj.append({"team": t, "proj_w": pw, "proj_l": total_games - pw,
                     "proj_seed": int(round(med_seed[i])),
                     "prob": float(summ.loc[t, "playoff_prob"]),
                     "title": float(summ.loc[t, "title_prob"]) if "title_prob" in summ else 0.0,
                     "in": med_seed[i] <= ts.k})
    proj.sort(key=lambda r: -r["proj_w"])
    for n, r in enumerate(proj, 1):
        r["proj_rank"] = n

    tb = sim.title_board().head(6).to_dict("records")
    champ = {"team": tb[0]["team"], "prob": tb[0]["title_prob"],
             "conf": sim.title_confidence()} if tb else None
    return render_template("tier.html", tier=tier, k=ts.k, rows=rows,
                           tiers=s.tiers, label=SEASON_LABEL, sched=sched,
                           champ=champ, title_board=tb, proj=proj)


@app.route("/tier/<tier>/team/<team>")
def team(tier, team):
    s = season()
    sim = tier_sim(tier)
    o = P.team_outlook(s, tier, team, sim=sim)
    curve = playoff_curve(sim, team)
    curve = curve[curve["sample"] >= 20]  # drop noisy tails
    chart = {
        "labels": [int(x) for x in curve["final_wins"]],
        "probs": [round(float(x) * 100, 1) for x in curve["p_playoffs"]],
    }
    try:
        roster = profiles.team_roster(tier, team)
        roster_rows = roster.to_dict("records")
        totals = profiles.team_totals(tier, team)
    except Exception:
        roster_rows, totals = [], {}

    # league-sheet metrics (strength of schedule, RPI, last 5, magic #) + projection
    tmeta = {}
    try:
        tr = s.teams[(s.teams["tier"] == tier) & (s.teams["team"] == team)]
        if len(tr):
            r0 = tr.iloc[0]
            def num(v):
                try:
                    return round(float(v), 3)
                except (TypeError, ValueError):
                    return None
            tmeta = {
                "rpi": num(r0.get("rpi")),
                "past_sos": num(r0.get("past_sos")),
                "fut_sos": num(r0.get("fut_sos")),
                "last5": r0.get("last5"),
                "sheet_magic": num(r0.get("magic_number")),
            }
        summ = sim.summary().set_index("team")
        if team in summ.index:
            tmeta["proj_final_w"] = round(float(summ.loc[team, "avg_final_w"]), 1)
    except Exception:
        pass

    return render_template("team.html", o=o, tier=tier, chart=chart,
                           tiers=s.tiers, label=SEASON_LABEL,
                           roster=roster_rows, totals=totals, tmeta=tmeta)


@app.route("/player/<path:name>")
def player(name):
    prof = profiles.player_profile(name)
    if prof is None:
        return render_template("player.html", prof=None, name=name, proj=None,
                               ranks=None, tiers=season().tiers, label=SEASON_LABEL)
    pdf = profiles.players()
    proj = project.project_player(pdf, season().variables, name)
    ranks = project.player_rankings(pdf, name)
    rat = rating.player_rating(pdf, name)
    adv = advanced.player_advanced(name)
    role = project.player_role(pdf, name)
    return render_template("player.html", prof=prof, name=name, proj=proj,
                           ranks=ranks, rat=rat, adv=adv, role=role,
                           tiers=season().tiers, label=SEASON_LABEL)


@app.route("/overskilled")
def overskilled():
    tier = request.args.get("tier", "all")
    cand = rating.overskilled_candidates(profiles.players(), limit=300)
    if tier and tier != "all":
        cand = cand[cand["Tier"] == tier]
    return render_template("overskilled.html", rows=cand.head(80).to_dict("records"),
                           tier=tier, tiers=season().tiers, label=SEASON_LABEL)


@app.route("/stat-impact")
def stat_impact():
    imp = rating.stat_importance(profiles.players())
    adv_imp = advanced.advanced_importance() if advanced.available() else []
    return render_template("stat_impact.html", imp=imp, adv_imp=adv_imp,
                           tiers=season().tiers, label=SEASON_LABEL)


@app.route("/rankings")
def rankings():
    s = season()
    tier = request.args.get("tier", "all")
    stat = request.args.get("stat", "Pts")
    per_game = request.args.get("mode", "pg") == "pg"
    board = profiles.leaderboard(tier=tier, stat=stat, per_game=per_game, limit=100)
    return render_template("rankings.html", tiers=s.tiers, tier=tier, stat=stat,
                           per_game=per_game, rows=board.to_dict("records"),
                           stat_opts=profiles.LEADERBOARD_STATS,
                           stat_label=profiles.LEADERBOARD_STATS.get(stat, ("Points",))[0],
                           label=SEASON_LABEL)


_PROJ_STAT_OPTS = {"G": "Goals", "A": "Assists", "S": "Saves", "Pts": "Points",
                   "SH": "Shots", "DM": "Demos", "MVP": "MVPs"}


@app.route("/projections")
def projections():
    s = season()
    tier = request.args.get("tier", "all")
    stat = request.args.get("stat", "G")
    if stat not in _PROJ_STAT_OPTS:
        stat = "G"
    board = project.project_all(profiles.players(), s.variables, stat)
    if tier != "all":
        board = board[board["Tier"] == tier]
    return render_template("projections.html",
                           rows=board.head(60).to_dict("records"),
                           tier=tier, stat=stat, stat_col=stat,
                           stat_opts=_PROJ_STAT_OPTS,
                           stat_label=_PROJ_STAT_OPTS[stat],
                           tiers=s.tiers, label=SEASON_LABEL)


@app.route("/players")
def players_search():
    q = request.args.get("q", "")
    results = profiles.find_players(q).head(60).to_dict("records") if q else []
    teams = profiles.find_teams(q) if q else []
    return render_template("players.html", q=q, results=results, teams=teams,
                           tiers=season().tiers, label=SEASON_LABEL)


@app.route("/compare/teams", methods=["GET", "POST"])
def cmp_teams():
    s = season()
    result = None
    tier = request.values.get("tier", s.tiers[0])
    a = request.values.get("a")
    b = request.values.get("b")
    ts = build_tier_state(s, tier)
    teams = [t.team for t in ts.ranked()]
    roster_a = roster_b = None
    if a and b and a in teams and b in teams:
        result = compare_teams(s, tier, a, b)
        roster_a = profiles.roster_with_ratings(tier, a)
        roster_b = profiles.roster_with_ratings(tier, b)
    return render_template("compare_teams.html", tiers=s.tiers, tier=tier,
                           teams=teams, a=a, b=b, result=result,
                           roster_a=roster_a, roster_b=roster_b,
                           label=SEASON_LABEL)


@app.route("/compare/players", methods=["GET", "POST"])
def cmp_players():
    df = players_df()
    names = sorted(df["Player"].astype(str).tolist())
    a = request.values.get("a")
    b = request.values.get("b")
    result = None
    if a and b:
        try:
            result = compare_players(df, a, b)
        except KeyError:
            result = None
    return render_template("compare_players.html", names=names, a=a, b=b,
                           result=result, tiers=season().tiers,
                           label=SEASON_LABEL)


if __name__ == "__main__":
    # Port 5000 is taken by another local app, so default to 5055.
    app.run(host="127.0.0.1", port=5055, debug=True)
