"""Replay-level data store (SQLite), the backend for lifetime / non-RSC stats.

Lives on a big drive (default H:\\rsc-data) since it scales to ~1M rows. The app
keeps its small curated CSVs in the repo; this is the large raw/derived store.
Deduped by replay_id (primary key), so a match is stored once no matter how many
players we reach it through.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DATA_DIR = Path(os.environ.get("RSC_DATA_DIR", r"H:\rsc-data"))
DB_PATH = DATA_DIR / "replays.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS replays(
  replay_id   TEXT PRIMARY KEY,
  date        TEXT,
  rl_season   INTEGER,
  playlist    TEXT,
  season_type TEXT,
  rsc_class   TEXT,           -- 'official' | 'inferred' | 'non_rsc'
  blue_goals  INTEGER,
  orange_goals INTEGER,
  detail      INTEGER DEFAULT 0   -- 0 = list only, 1 = full detail fetched
);
CREATE TABLE IF NOT EXISTS replay_players(
  replay_id TEXT,
  player_id TEXT,             -- normalized 'platform:id'
  name      TEXT,
  team      TEXT,
  score     INTEGER,
  PRIMARY KEY(replay_id, player_id)
);
CREATE TABLE IF NOT EXISTS harvest_state(
  player_id TEXT PRIMARY KEY,
  n_replays INTEGER,
  last_date TEXT
);
CREATE INDEX IF NOT EXISTS idx_rp_player ON replay_players(player_id);
CREATE INDEX IF NOT EXISTS idx_r_class ON replays(rsc_class);
CREATE INDEX IF NOT EXISTS idx_r_playlist ON replays(playlist);
"""


def available() -> bool:
    return DB_PATH.exists()


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def init(conn: sqlite3.Connection | None = None) -> None:
    c = conn or connect()
    c.executescript(_SCHEMA)
    c.commit()
    if conn is None:
        c.close()


def upsert_replays(conn, rows) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO replays"
        "(replay_id,date,rl_season,playlist,season_type,rsc_class,blue_goals,orange_goals)"
        " VALUES(?,?,?,?,?,?,?,?)", rows)


def upsert_players(conn, rows) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO replay_players"
        "(replay_id,player_id,name,team,score) VALUES(?,?,?,?,?)", rows)


def set_state(conn, player_id, n_replays, last_date) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO harvest_state(player_id,n_replays,last_date)"
        " VALUES(?,?,?)", (player_id, n_replays, last_date))


def stats() -> dict:
    if not available():
        return {}
    c = connect()
    out = {}
    out["replays"] = c.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    out["player_rows"] = c.execute("SELECT COUNT(*) FROM replay_players").fetchone()[0]
    out["by_class"] = dict(c.execute(
        "SELECT rsc_class, COUNT(*) FROM replays GROUP BY rsc_class").fetchall())
    out["players_harvested"] = c.execute("SELECT COUNT(*) FROM harvest_state").fetchone()[0]
    out["detail_done"] = c.execute("SELECT COUNT(*) FROM replays WHERE detail=1").fetchone()[0]
    c.close()
    return out
