"""Lifetime replay harvest (list pass).

Cheap first pass: for each active player with a steam id, page their public
replays and record each one's metadata + per-player score + RSC classification
in the replay DB. No per-replay detail calls - that's the expensive second pass.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import replaydb
from .ballchasing import Ballchasing

ROOT = Path(__file__).resolve().parent.parent
LIST_TTL = 7 * 24 * 3600       # replay listings change slowly
MAX_PAGES = 30                 # cap ~6000 replays/player for the very-high-volume few


def active_steam_ids() -> list[str]:
    """Steam ids of current-season players (the harvestable ~half)."""
    df = pd.read_csv(ROOT / "data" / "history" / "S26.csv")
    return [s for s in df["sid"].astype(str) if s.isdigit() and len(s) >= 17]


def _norm_id(pid) -> str:
    if isinstance(pid, dict):
        return f"{pid.get('platform')}:{pid.get('id')}"
    return str(pid)


def _classify(groups) -> str:
    g = " ".join((x.get("id", "") or "") for x in (groups or [])).lower()
    return "official" if ("rsc" in g or "season-" in g) else "non_rsc"


def list_pass(player_ids=None, log=lambda *_: None) -> dict:
    bc = Ballchasing()
    conn = replaydb.connect()
    replaydb.init(conn)
    ids = player_ids if player_ids is not None else active_steam_ids()
    log(f"list pass: {len(ids)} players")
    for i, pid in enumerate(ids, 1):
        url = f"/replays?player-id=steam:{pid}&count=200"
        pages = total = 0
        last_date = ""
        while url and pages < MAX_PAGES:
            try:
                d = bc._get(url, ttl=LIST_TTL)
            except Exception as e:
                log(f"  {pid}: {str(e)[:60]}")
                break
            rrows, prows = [], []
            for rp in d.get("list", []):
                rid = rp.get("id")
                if not rid:
                    continue
                cls = _classify(rp.get("groups"))
                blue, orange = rp.get("blue", {}), rp.get("orange", {})
                rrows.append((rid, rp.get("date"), rp.get("season"),
                              rp.get("playlist_name"), rp.get("season_type"),
                              cls, blue.get("goals"), orange.get("goals")))
                for team, side in (("blue", blue), ("orange", orange)):
                    for pl in side.get("players", []):
                        prows.append((rid, _norm_id(pl.get("id")), pl.get("name"),
                                      team, pl.get("score")))
                total += 1
                last_date = max(last_date, (rp.get("date") or "")[:10])
            replaydb.upsert_replays(conn, rrows)
            replaydb.upsert_players(conn, prows)
            conn.commit()
            url = d.get("next")
            pages += 1
        replaydb.set_state(conn, f"steam:{pid}", total, last_date)
        conn.commit()
        if i % 25 == 0 or i == len(ids):
            log(f"  {i}/{len(ids)} players, last had {total} replays")
    conn.close()
    return replaydb.stats()
