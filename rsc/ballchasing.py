"""Cached, rate-limited client for the ballchasing.com API.

Ballchasing has per-replay advanced Rocket League stats (boost, movement,
positioning, demos) far richer than the standings site. The free "regular" tier
is rate-limited (~2 req/s), so we sleep between calls and cache every response
to disk. The API token lives in .ballchasing_token (gitignored).

Key efficiency: a group's detail endpoint already returns per-player aggregated
stats across all replays in the group, so one call per match-group beats
fetching every replay individually.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://ballchasing.com/api"
ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT / ".ballchasing_token"
CACHE_DIR = ROOT / "data" / "bc_cache"
CACHE_TTL = 30 * 24 * 3600   # match replays are immutable once uploaded
TRAVERSE_TTL = 6 * 3600      # group listings refresh often to find new matches

# seconds between live calls per patron tier (with a small safety margin)
RATE_BY_TIER = {"regular": 0.55, "gold": 0.55, "diamond": 0.27,
                "champion": 0.13, "gc": 0.07}

_last_call = [0.0]
_interval = [0.55]           # current min seconds between calls (set from tier)
_tier = [None]               # detected patron tier, cached per process


class Ballchasing:
    def __init__(self, ttl: int = CACHE_TTL, detect_tier: bool = True):
        self.token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        self.ttl = ttl
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if detect_tier and _tier[0] is None:
            try:                                  # one cached ping -> set the rate
                t = (self._get("/", ttl=3600).get("type") or "regular").lower()
                _tier[0] = t
                _interval[0] = RATE_BY_TIER.get(t, 0.55)
            except Exception:
                _tier[0] = "regular"

    def _cache_path(self, url: str) -> Path:
        return CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest()[:18] + ".json")

    def _get(self, url: str, ttl: int | None = None) -> dict:
        if not url.startswith("http"):
            url = BASE + url
        ttl = self.ttl if ttl is None else ttl
        cp = self._cache_path(url)
        if cp.exists() and (time.time() - cp.stat().st_mtime) < ttl:
            return json.loads(cp.read_text(encoding="utf-8"))
        req = urllib.request.Request(url, headers={"Authorization": self.token})
        for attempt in range(6):
            wait = _interval[0] - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8"))
                _last_call[0] = time.time()
                break
            except urllib.error.HTTPError as e:
                _last_call[0] = time.time()
                if e.code == 429:                       # rate limited - back off
                    ra = e.headers.get("Retry-After")
                    delay = float(ra) if ra and ra.isdigit() else min(2 ** attempt, 30)
                    time.sleep(delay + 0.5)
                    continue
                raise
        else:
            raise RuntimeError(f"ballchasing 429: gave up after retries on {url}")
        cp.write_text(json.dumps(data), encoding="utf-8")
        return data

    def ping(self) -> dict:
        return self._get("/")

    def search_groups(self, name: str, count: int = 200) -> list[dict]:
        """All groups matching a name (follows pagination)."""
        out, url = [], f"/groups?name={urllib.parse.quote(name)}&count={count}"
        while url:
            d = self._get(url)
            out.extend(d.get("list", []))
            url = d.get("next")
            if url and not url.startswith("http"):
                url = url
            if len(out) >= 2000:
                break
        return out

    def child_groups(self, group_id: str) -> list[dict]:
        return self._get(f"/groups?group={group_id}&count=200").get("list", [])

    def group(self, group_id: str) -> dict:
        """Group detail, including per-player aggregated stats."""
        return self._get(f"/groups/{group_id}")

    def replays_in_group(self, group_id: str) -> list[dict]:
        return self._get(f"/replays?group={group_id}&count=200").get("list", [])

    def replay(self, replay_id: str) -> dict:
        return self._get(f"/replays/{replay_id}")


if __name__ == "__main__":
    bc = Ballchasing()
    me = bc.ping()
    print("auth OK as:", me.get("name"), "| tier:", me.get("type"))
    # inspect one S26 match group's player aggregation structure
    groups = bc.search_groups("RSC S26 MD6 Match")
    print(f"\n'RSC S26 MD6 Match' groups found: {len(groups)}")
    g = groups[0]
    print("sample group:", g["name"], "|", g["id"], "| replays:", g.get("direct_replays"))
    detail = bc.group(g["id"])
    players = detail.get("players", [])
    print(f"\ngroup-detail has {len(players)} aggregated players. keys per player:")
    if players:
        p = players[0]
        print("  top-level:", list(p.keys()))
        print("  stat groups:", list(p.get("game_average", p.get("cumulative", {})).keys())
              if isinstance(p.get("game_average", p.get("cumulative")), dict) else "n/a")
        print("  sample:", p.get("name"), "| games:", p.get("cumulative", {}).get("games"))
