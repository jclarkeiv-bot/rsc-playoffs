"""CLI: rebuild data/bc_advanced.csv from the official RSC ballchasing account.

    python scripts/bc_build_official.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsc import bc_harvest  # noqa: E402

if __name__ == "__main__":
    r = bc_harvest.build(log=print)
    print(f"\nDONE: {r['matched']}/{r['total']} players from {r['matches']} matches "
          f"-> data/bc_advanced.csv")
