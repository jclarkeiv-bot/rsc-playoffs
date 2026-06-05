"""Live client for rscna.com.

The site is an htmx app: data lives in HTML table *fragments* served from a few
endpoints, which we fetch and parse with pandas.read_html. We are a polite
guest: a descriptive User-Agent, on-disk caching with a TTL so repeated runs
don't re-hit the server, and pagination at the server's max page size (100).

Endpoints (current season):
    /standings/overall/
    /matches/table/?tier=all&page=N&per_page=100&sort=day&dir=asc&q=
    /player-stats/table/?tier=all&page=N&per_page=100&sort=points&dir=desc&mode=total|average&q=

This complements the xlsx: it adds live, per-player stats (Goals/Assists/Saves/
Shots/Demos/MVP) the spreadsheets don't carry, for the comparison features.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.rscna.com"
UA = "rsc-playoffs/0.1 (personal analytics tool; contact: local user)"
PER_PAGE = 100
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_TTL = 3600  # seconds (hourly refresh - fresh but polite)


class RSCClient:
    def __init__(self, ttl: int = CACHE_TTL, cache_dir: Path = CACHE_DIR):
        self.ttl = ttl
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})

    # ---- low level ----------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.html"

    def _get(self, path: str, params: dict | None = None) -> str:
        url = path if path.startswith("http") else BASE + path
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        cp = self._cache_path(url)
        if cp.exists() and (time.time() - cp.stat().st_mtime) < self.ttl:
            return cp.read_text(encoding="utf-8")
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        cp.write_text(resp.text, encoding="utf-8")
        return resp.text

    @staticmethod
    def _read_table(html: str) -> pd.DataFrame:
        tables = pd.read_html(io.StringIO(html))
        if not tables:
            return pd.DataFrame()
        return tables[0]

    def _paginate(self, path: str, params: dict) -> pd.DataFrame:
        """Fetch all pages until a short page signals the end."""
        frames, page = [], 1
        while True:
            p = {**params, "page": page, "per_page": PER_PAGE}
            df = self._read_table(self._get(path, p))
            if df.empty:
                break
            frames.append(df)
            if len(df) < PER_PAGE:
                break
            page += 1
            if page > 100:  # safety stop
                break
        return (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame())

    # ---- public ------------------------------------------------------------
    def player_stats(self, tier: str = "all", mode: str = "total") -> pd.DataFrame:
        """All players. mode='total' or 'average' (per-game). Columns:
        Player, Tier, Team, GP, W, L, MVP, Pts, G, A, S, SH, SH%, DM."""
        df = self._paginate("/player-stats/table/",
                            {"tier": tier, "sort": "points", "dir": "desc",
                             "mode": mode, "q": ""})
        return _clean(df)

    def matches(self, tier: str = "all") -> pd.DataFrame:
        """All matches. Columns: Day, Date, Tier, Home, Away, Score, Type."""
        df = self._paginate("/matches/table/",
                            {"tier": tier, "sort": "day", "dir": "asc", "q": ""})
        return _clean(df)

    def standings(self) -> pd.DataFrame:
        return _clean(self._read_table(self._get("/standings/overall/")))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip %/whitespace and coerce obvious numeric columns."""
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        # pandas 3.0 uses StringDtype (not object) for text columns.
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(str).str.strip()
            # percent columns -> float fraction
            if s.str.endswith("%").mean() > 0.5:
                df[col] = pd.to_numeric(s.str.rstrip("%"), errors="coerce") / 100.0
            else:
                num = pd.to_numeric(s.str.replace(",", "", regex=False),
                                    errors="coerce")
                if num.notna().mean() > 0.9:  # mostly numeric -> convert
                    df[col] = num
    return df


if __name__ == "__main__":
    import sys
    c = RSCClient()
    what = sys.argv[1] if len(sys.argv) > 1 else "players"
    if what == "players":
        df = c.player_stats(mode="total")
        print(f"players: {len(df)} rows, cols={list(df.columns)}")
        print(df.head(8).to_string(index=False))
    elif what == "matches":
        df = c.matches()
        played = df["Score"].astype(str).str.contains("-").sum()
        print(f"matches: {len(df)} rows ({played} with scores)")
        print(df.head(5).to_string(index=False))
    elif what == "standings":
        df = c.standings()
        print(f"standings: {len(df)} rows, cols={list(df.columns)}")
        print(df.head(8).to_string(index=False))
