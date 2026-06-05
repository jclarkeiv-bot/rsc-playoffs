"""Durable local store for everything we pull.

Beyond the HTTP caches (which expire), we keep clean CSV copies of every pull:
a 'latest' snapshot (overwritten) plus one dated snapshot per day under
history/, so the data accumulates locally and survives offline / API outages.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

STORE = Path(__file__).resolve().parent.parent / "data" / "store"
HISTORY = STORE / "history"


def save(name: str, df: pd.DataFrame, dated: bool = True) -> None:
    if df is None or len(df) == 0:
        return
    STORE.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(STORE / f"{name}.csv", index=False)
        if dated:
            HISTORY.mkdir(parents=True, exist_ok=True)
            day = _dt.date.today().isoformat()
            p = HISTORY / f"{name}_{day}.csv"
            if not p.exists():           # one snapshot per day, keep the first
                df.to_csv(p, index=False)
    except Exception:
        pass  # storing is best-effort; never break a request over it


def latest(name: str) -> pd.DataFrame | None:
    p = STORE / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None
