"""RSC Playoffs - Flask UI.

Sits on top of rsc.playoffs (clinch + Elo odds + curves) and rsc.engine.compare.
Sims are cached per (season, tier) since each takes a few seconds.

Run:  python app.py    ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

from rsc import playoffs as P
from rsc import profiles
from rsc import project
from rsc import rating
from rsc import advanced
from rsc import comps
from rsc import balance as balance_mod
from rsc.engine.clinch import build_tier_state, evaluate_team
from rsc.engine.simulate import playoff_curve
from rsc.engine.compare import compare_players, compare_teams

app = Flask(__name__)

SEASON_LABEL = "S26"
LIVE_DATA = True            # pull standings/schedule live from rscna.com
SEASON_TTL = 3600           # rebuild the live season at most hourly
DATA_SOURCE = "xlsx snapshot"
_season = None
_season_ts = 0.0
_sim_cache: dict[str, object] = {}


def season():
    """Current season. Live from rscna.com (hourly), falling back to the xlsx
    snapshot if the API is unreachable."""
    global _season, _season_ts, _sim_cache, DATA_SOURCE
    now = time.time()
    if _season is None or (now - _season_ts) > SEASON_TTL:
        loaded = None
        if LIVE_DATA:
            try:
                from rsc import live
                loaded = live.load_live_season(SEASON_LABEL)
                DATA_SOURCE = "live (rscna.com)"
            except Exception:
                loaded = None
        if loaded is None:
            loaded = P.load(SEASON_LABEL)
            DATA_SOURCE = "xlsx snapshot"
        if _season is not None:           # data refreshed -> drop stale caches
            _sim_cache = {}
            profiles.players(refresh=True)
        _season, _season_ts = loaded, now
    _maybe_refresh_advanced()
    return _season


BC_REFRESH_INTERVAL = 18 * 3600    # rebuild ballchasing advanced stats ~daily
_bc_refreshing = {"on": False}


def _maybe_refresh_advanced():
    """If the advanced-stats snapshot is stale, rebuild it in the background
    (incremental + cached, so it only pulls newly-played matches)."""
    from rsc import bc_harvest
    csv = bc_harvest.CSV
    age = (time.time() - os.path.getmtime(csv)) if csv.exists() else float("inf")
    if _bc_refreshing["on"] or age < BC_REFRESH_INTERVAL:
        return
    _bc_refreshing["on"] = True

    def work():
        try:
            from rsc import advanced
            bc_harvest.build(players_df=profiles.players())
            advanced.reload()
            profiles.invalidate_ratings()
            try:                          # keep current-season comparables fresh
                bc_harvest.harvest_season("S26")
                comps.reload()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            _bc_refreshing["on"] = False

    threading.Thread(target=work, daemon=True).start()


def tier_sim(tier: str, mode: str = "all"):
    key = (tier, mode)
    if key not in _sim_cache:
        _sim_cache[key] = P.tier_odds(season(), tier, mode=mode)
    return _sim_cache[key]


def players_df():
    return profiles.players()


@app.context_processor
def inject_data_source():
    return {"data_source": DATA_SOURCE}


@app.route("/")
def index():
    s = season()
    conf = P.model_confidence(s)
    pdf = profiles.players()
    counts = {"total": int(len(pdf)), "played": int((pdf["GP"] >= 1).sum())}
    return render_template("index.html", tiers=s.tiers, label=SEASON_LABEL,
                           conf=conf, counts=counts)


@app.route("/tier/<tier>")
def tier(tier):
    s = season()
    if tier not in s.tiers:
        return redirect(url_for("index"))
    mode = "current" if request.args.get("model") == "current" else "all"
    sim = tier_sim(tier, mode)
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
                           champ=champ, title_board=tb, proj=proj, mode=mode)


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
            total_games = int(s.variables.set_index("tier").loc[tier, "match_days"]) * 4
            tmeta["proj_final_l"] = total_games - int(round(summ.loc[team, "avg_final_w"]))
    except Exception:
        pass

    team_proj = project.project_team(profiles.players(), s.variables, tier, team)

    return render_template("team.html", o=o, tier=tier, chart=chart,
                           tiers=s.tiers, label=SEASON_LABEL,
                           roster=roster_rows, totals=totals, tmeta=tmeta,
                           team_proj=team_proj)


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
    history = comps.player_history(name) if comps.available() else []
    comp = comps.find_comparables(name, "S26") if comps.available() else None
    fc = comps.forecast(name) if comps.available() else None
    return render_template("player.html", prof=prof, name=name, proj=proj,
                           ranks=ranks, rat=rat, adv=adv, role=role,
                           history=history, comp=comp, fc=fc,
                           tiers=season().tiers, label=SEASON_LABEL)


@app.route("/overskilled")
def overskilled():
    tier = request.args.get("tier", "all")
    cand = rating.overskilled_candidates(profiles.players(), limit=300)
    if tier and tier != "all":
        cand = cand[cand["Tier"] == tier]
    return render_template("overskilled.html", rows=cand.head(80).to_dict("records"),
                           tier=tier, tiers=season().tiers, label=SEASON_LABEL)


@app.route("/team-rankings")
def team_rankings():
    s = season()
    tier = request.args.get("tier", "all")
    metric = request.args.get("metric", "avg_ovr")
    if metric not in profiles.TEAM_METRICS:
        metric = "avg_ovr"
    t = profiles.team_metrics(s)
    if tier != "all":
        t = t[t["tier"] == tier]
    t = t.dropna(subset=[metric]).sort_values(metric, ascending=False).head(120)
    t.insert(0, "rank", range(1, len(t) + 1))
    return render_template("team_rankings.html", rows=t.to_dict("records"),
                           tier=tier, metric=metric,
                           metric_opts=profiles.TEAM_METRICS,
                           metric_label=profiles.TEAM_METRICS[metric][0],
                           tiers=s.tiers, label=SEASON_LABEL)


@app.route("/matches")
def matches():
    s = season()
    tier = request.args.get("tier", s.tiers[0])
    team = request.args.get("team", "")
    show = request.args.get("show", "all")  # all | played | upcoming
    m = s.matches[s.matches["tier"] == tier].copy()
    teams = sorted(set(m["away"]).union(m["home"]))
    if team:
        m = m[(m["away"] == team) | (m["home"] == team)]
    if show == "played":
        m = m[m["played"]]
    elif show == "upcoming":
        m = m[~m["played"]]
    m["_d"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.sort_values(["_d", "match_day"], na_position="last")
    rows = []
    for r in m.itertuples():
        winner = None
        if r.played:
            winner = (r.away if r.away_g > r.home_g
                      else r.home if r.home_g > r.away_g else "tie")
        rows.append({
            "md": (str(r.match_day) if r.is_regular else r.round_label),
            "is_regular": r.is_regular,
            "date": r.date.strftime("%b %d") if pd.notna(r.date) else "",
            "away": r.away, "home": r.home,
            "away_g": int(r.away_g) if r.played else None,
            "home_g": int(r.home_g) if r.played else None,
            "played": bool(r.played), "winner": winner,
        })
    return render_template("matches.html", tier=tier, team=team, show=show,
                           teams=teams, rows=rows, tiers=s.tiers,
                           label=SEASON_LABEL)


@app.route("/match")
def match_detail():
    s = season()
    tier = request.args.get("tier")
    md = request.args.get("md", "")
    away = request.args.get("away")
    home = request.args.get("home")
    m = s.matches[(s.matches["tier"] == tier) & (s.matches["away"] == away)
                  & (s.matches["home"] == home)]
    if md.isdigit():
        m = m[m["match_day"] == int(md)]
    else:
        m = m[m["round_label"] == md]
    if m.empty:
        return redirect(url_for("matches", tier=tier))
    r = m.iloc[0]
    played = bool(r["played"])
    winner = None
    if played:
        winner = (away if r["away_g"] > r["home_g"]
                  else home if r["home_g"] > r["away_g"] else "tie")

    from rsc.engine.standings import compute_standings
    st = compute_standings(s.matches)
    st = st[st["tier"] == tier].set_index("team")

    def rec(t):
        return (f"{int(st.loc[t].w)}-{int(st.loc[t].l)}" if t in st.index else "0-0")

    mu = P.team_matchup(s, tier, away, home, "all")      # all data (career priors)
    mu_cur = P.team_matchup(s, tier, away, home, "current")  # this season only
    fav = away if mu["exp_a"] >= mu["exp_b"] else home
    pred_correct = (winner == fav) if (played and winner != "tie") else None

    try:
        roster_a = profiles.team_roster(tier, away).head(5).to_dict("records")
        roster_h = profiles.team_roster(tier, home).head(5).to_dict("records")
    except Exception:
        roster_a = roster_h = []

    ctx = {
        "tier": tier, "away": away, "home": home,
        "md_label": ("Match day " + md) if md.isdigit() else md,
        "date": r["date"].strftime("%b %d, %Y") if pd.notna(r["date"]) else "",
        "played": played, "winner": winner,
        "away_g": int(r["away_g"]) if played else None,
        "home_g": int(r["home_g"]) if played else None,
        "rec_a": rec(away), "rec_h": rec(home),
        "mu": mu, "mu_cur": mu_cur, "fav": fav, "pred_correct": pred_correct,
        "roster_a": roster_a, "roster_h": roster_h,
    }
    return render_template("match.html", tiers=s.tiers, label=SEASON_LABEL, **ctx)


@app.route("/balance")
def tier_balance():
    d = balance_mod.diagnose(profiles.players())
    return render_template("balance.html", d=d, tiers=season().tiers,
                           label=SEASON_LABEL)


@app.route("/balance/<tier>")
def balance_tier(tier):
    s = season()
    if tier not in s.tiers:
        return redirect(url_for("tier_balance"))
    up, down = rating.tier_misplaced(profiles.players(), tier)
    return render_template("balance_tier.html", tier=tier, up=up, down=down,
                           tiers=s.tiers, label=SEASON_LABEL)


@app.route("/accuracy")
def accuracy():
    from rsc.engine.predict import accuracy_by_matchday
    s = season()
    acc = accuracy_by_matchday(s.matches)
    chart = {
        "labels": [r["match_day"] for r in acc["rows"]],
        "game": [r["game_acc"] for r in acc["rows"]],
        "cum_game": [r["cum_game_acc"] for r in acc["rows"]],
    }
    return render_template("accuracy.html", acc=acc, chart=chart,
                           tiers=s.tiers, label=SEASON_LABEL)


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
    min_games = request.args.get("min_games", 1, type=int) or 1
    board = profiles.leaderboard(tier=tier, stat=stat, per_game=per_game,
                                 limit=100, min_games=min_games)
    return render_template("rankings.html", tiers=s.tiers, tier=tier, stat=stat,
                           per_game=per_game, min_games=min_games,
                           rows=board.to_dict("records"),
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
    min_games = request.args.get("min_games", 1, type=int) or 1
    board = project.project_all(profiles.players(), s.variables, stat,
                                min_games=min_games)
    if tier != "all":
        board = board[board["Tier"] == tier]
    return render_template("projections.html",
                           rows=board.head(60).to_dict("records"),
                           tier=tier, stat=stat, stat_col=stat, min_games=min_games,
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
    # Set RSC_HOST=0.0.0.0 to allow remote access (e.g. over Tailscale).
    host = os.environ.get("RSC_HOST", "127.0.0.1")
    debug = host == "127.0.0.1"            # never run the debugger when exposed
    app.run(host=host, port=5055, debug=debug, threaded=True)
