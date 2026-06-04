# RSC Playoffs

A tool for the **Rocket Soccar Confederation (RSC)** amateur Rocket League league that answers:

- **Clinch/elimination:** for any team, exactly what they must do to make their tier's
  playoffs — factoring in every other team's remaining schedule.
- **Playoff prediction:** probability each team makes the playoffs (Monte Carlo over the
  remaining schedule), surfaced only when the data clears a confidence bar.
- **Comparison:** team-vs-team and player-vs-player.

## Run it

```bash
pip install -r requirements.txt
python app.py                       # -> http://127.0.0.1:5000
```

CLI entry points (no server):
```bash
python -m rsc.playoffs S26 Premier "Valkyries"     # odds + clinch + headline
python -m rsc.engine.simulate S26 Elite            # tier playoff odds
python -m rsc.engine.compare players "Reyes." "Whitlock7"
python -m rsc.engine.compare teams S26 Premier "Long Beards" "Gummy Bears"
python scripts/validate_ingest.py S26 S25          # parser proof (both PASS)
python scripts/backtest_playoffs.py 6              # prediction backtest on S25
```

The web app: home (tier grid + model-confidence badge) -> tier board (playoff
odds bars, clinch/elim badges, current standings) -> team page (clinch/elim
numbers + the "what you need" odds-by-final-record chart) -> compare teams /
compare players.

## How the league works (facts the engine depends on)

- **9 tiers** (highest→lowest MMR): Premier, Master, Elite, Veteran, Rival, Challenger,
  Prospect, Contender, Amateur.
- **Game-based standings.** Each match day is a **4-game series**; every game counts.
  A result of `3-1` = 3 game-wins for Away, 1 for Home. Ranking is by **game win %**.
- **Playoff cutoffs are per-tier** (from the `Variables` sheet), e.g. Premier top 3,
  Master top 5, Elite/Veteran/Rival/Challenger top 8, Prospect top 5, Contender top 6,
  Amateur top 3. (The rulebook's "≥50% qualifies" is looser; we trust the sheet.)
- **Playoffs are excluded from standings.** Regular-season rows have integer match days
  (1..N); playoff rows are labeled `Wild Card`/`Quarterfinals`/`Semifinals`/`Finals`
  and are Bo5/Bo7 (3–7 games).
- **Tiebreakers** (RSC rulebook, in order): (1) head-to-head win%, (2) vs in-division,
  (3) vs in-conference, (4) vs common opponents, (5) overall goal differential,
  (6) goal diff in-division, (7) goal diff in-conference, (8) goal diff vs common, (9) random.
  Tiebreakers 1–4 come from the xlsx; 5–8 need **goal data** (see sources).

## Data sources

| Source | What it gives | Freshness | Status |
|---|---|---|---|
| `data/S26_standings.xlsx`, `data/S25_standings.xlsx` | Schedule (played + future), standings, per-tier `Variables`, league's own `magic_number`/SOS columns | Snapshot | **Parsed & validated** |
| `S26 RSC SBV` Google Sheet (`158xj5...`) | Team & player **goal-level** stats + pre-computed PWE (Pythagorean) | Fresher snapshot | Identified; not yet wired |
| `api.rscna.com` (htmx fragments) | Live standings, matches, **per-player stats** (G/A/Sv/Sh/Demos/MVP) | Live | **Endpoints mapped & parseable** |

### Live API endpoints (htmx HTML fragments — parse with `pandas.read_html`)
- `/standings/overall/`, `/standings/playoffs/`
- `/matches/table/?tier=all&page=1&per_page=25&sort=day&dir=asc&q=`
- `/player-stats/table/?tier=all&page=1&per_page=25&sort=points&dir=desc&mode=total&q=`
  (`mode=total|average`; columns: Player, Tier, Team, GP, W, L, MVP, Pts, G, A, S, SH, SH%, DM)
- `/franchises/widgets/grid/`

**Etiquette:** read-only, cache aggressively, poll infrequently (≈once per match day),
identify via User-Agent. Consider pinging the maintainer (`monty._`) as a courtesy.

## Validation status

`python scripts/validate_ingest.py S26 S25` → **both PASS** (S26 200/200 teams, S25 180/180):
our game-based W/L/WP recomputed from raw results matches the league's own standings exactly.
This is the trust anchor the simulation engine builds on.

## Prediction validation (S25 backtest)

Exact top-K clinch (accounting for who-plays-whom) is NP-hard, so we use a
**certainty layer** (guarantee-based clinch/elimination) + a **simulation layer**
(Monte Carlo, Elo-driven) for realistic odds. Per-game outcomes in this league
are near coin-flips (Elo skill ~2% over a coin), so the model's value is
*calibration*, not point-prediction. Predicting the S25 playoff field from
mid-season:

| Season point | Sim hit-rate | Naive baseline | Sim Brier (calibration) |
|---|---|---|---|
| MD3 | 60% | 57% | 0.163 |
| MD6 (= S26 now) | 68% | 70% | 0.120 |
| MD8 | 81% | 83% | 0.084 |
| MD12 | 85% | 85% | 0.055 |

Takeaway: ~2/3 of the playoff field is identifiable at S26's current point,
rising to ~85% late; the simulator ties the standings baseline on hit-rate but
is far better *calibrated* (lower Brier), so its probabilities are trustworthy.
Run: `python scripts/backtest_playoffs.py 6`.

## Layout

```
data/                 season workbooks
rsc/
  ingest.py           xlsx -> Season(variables, matches, teams)
  engine/
    standings.py      recompute standings from raw results (validated)
    clinch.py         schedule-aware clinch/elimination (guarantee layer)
    simulate.py       Monte Carlo -> playoff odds + "what you need" curve
    predict.py        Elo model (K=16) + walk-forward backtest + confidence gate
  playoffs.py         high-level API: team_outlook / tier_odds (Elo-driven)
  api.py              (next) live rscna.com client
scripts/
  validate_ingest.py  parser/standings ground-truth check
  backtest_playoffs.py end-to-end playoff-prediction backtest on S25
templates/ static/    (next) Flask UI
```

## Decisions on record
- Stack: **Flask** web app. Predictions: **classical only** (Elo/log5/Pythagorean +
  Monte Carlo), no ML black box — keep it explainable and backtestable.
- Playoff cutoff: per-tier from the sheet (configurable later).
- Data: xlsx-validated foundation; SBV sheet for goals; live API for player data.
- Predictions are gated on a backtest confidence bar (train on S25, test on early S26).
