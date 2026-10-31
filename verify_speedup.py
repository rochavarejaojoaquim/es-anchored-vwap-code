r"""
verify_speedup.py -- the acceptance test for the faster core_v2 (2026-09-24).

The speed-up must not move a single number. It re-runs searches your PC has
ALREADY finished and compares every field to the stored result, exactly (rule
D3: a known number, checked). Same standard the laptop had to meet.

    python verify_speedup.py           checks 10 finished searches
    python verify_speedup.py 40        checks 40 (slower, stronger)

Picks the largest finished cells first -- those exercise the bootstrap hardest.
Reads data\dump and cores_v2; writes nothing.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import core_v2                                                    # noqa: E402

FIELDS = ["col", "cut", "sense", "score", "eA", "eB", "seA", "seB", "nA", "nB", "pct"]

if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    want = int(pos[0]) if pos else 10
    files = []
    for f in glob.glob(os.path.join(core_v2.OUT, "*_side*.parquet")):
        if os.path.getsize(f) > 1200:                # skip the empty ones
            files.append((os.path.getsize(f), f))
    files = [f for _, f in sorted(files, reverse=True)][:want]
    # --part i/k or --part 1,2,3/k : take an interleaved share, as runner.py does.
    # Interleaved (not blocked) so every share gets a fair mix of big and small
    # searches. Shares are disjoint and cover everything, so k terminals together
    # check exactly the same set one terminal would.
    part = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--part=")), None)
    if part:
        sel, k = part.split("/")
        keep, k = {int(x) for x in sel.split(",")}, int(k)
        if not keep or max(keep) > k or min(keep) < 1:
            raise SystemExit(f"bad --part={part}: shares must be 1..{k}")
        files = [f for i, f in enumerate(files) if (i % k) + 1 in keep]
        print(f"  --part {part}: {len(files)} searches of the {want} largest", flush=True)
    if not files:
        raise SystemExit("no finished searches in cores_v2\\")
    bad = 0
    t0 = time.time()
    for f in files:
        t = pd.read_parquet(f)
        row = t.iloc[0]
        cell = str(row["cell"])
        side = None if pd.isna(row["side"]) else float(row["side"])
        s0 = time.time()
        r, c, T = core_v2.load_cell(cell, side)
        P = core_v2.prep(r, c)
        b = core_v2.best_core(r, c, P, core_v2.search_cols(r, side))
        diff = []
        for k in FIELDS:
            a, bb = row[k], b[k]
            if (a != bb) if isinstance(bb, str) else (float(a) != float(bb)):
                diff.append(f"{k}: stored {a} vs now {bb}")
        bad += bool(diff)
        print(f"  [{'PASS' if not diff else 'FAIL'}]  {os.path.basename(f):<44} "
              f"{time.time() - s0:5.1f}s  " + ("identical" if not diff else "; ".join(diff[:2])),
              flush=True)
    print()
    if bad:
        print(f"  {bad} of {len(files)} DIFFER -- do NOT run the floors on this version.")
    else:
        print(f"  ALL {len(files)} IDENTICAL in {time.time() - t0:.0f}s "
              f"({(time.time() - t0)/len(files):.1f}s per search; it was ~4 min).")
        print("  The floors can run on this version.")
