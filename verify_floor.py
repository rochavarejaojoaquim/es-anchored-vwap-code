r"""
verify_floor.py -- the acceptance test for the faster floor loop (2026-09-24).

A floor is a permutation test: if the patched code drew even slightly different
permutations, the null distribution would shift and every p-value with it. So
this recomputes floors you have ALREADY FINISHED and compares EVERY field,
including the full null distribution, permutation by permutation.

    python verify_floor.py            checks the 2 quickest finished floors
    python verify_floor.py 4          checks 4 (slower, stronger)

Each check costs one full floor (~100 searches), so start with 2. Reads
data\dump and floors_v2; writes nothing.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
import floor_v2                                                   # noqa: E402

FIELDS = ["col", "obs_score", "floor95", "edge_over_floor", "p", "eA", "eB",
          "seA", "seB", "nA", "nB", "nperm", "n_null_empty"]

if __name__ == "__main__":
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = []
    for f in glob.glob(os.path.join(floor_v2.FLOORS, "*.parquet")):
        t = pd.read_parquet(f)
        if len(t):
            rows.append(t.iloc[0])
    if not rows:
        raise SystemExit("no finished floors in floors_v2\\ yet")
    rows.sort(key=lambda r: int(r["nA"]) + int(r["nB"]))       # smallest = quickest
    bad = 0
    for row in rows[:want]:
        cell, side = str(row["cell"]), (None if pd.isna(row["side"]) else float(row["side"]))
        tag = f"{cell}_side{side}"
        rng = np.random.default_rng([floor_v2.SEED, floor_v2.zlib.crc32(tag.encode())])
        t0 = time.time()
        out = floor_v2.floor_cell(cell, side, float(row["obs_score"]), int(row["nperm"]), rng)
        if out is None:
            print(f"  [FAIL]  {tag:<40} the re-run did not reproduce the stored search")
            bad += 1
            continue
        diff = []
        for k in FIELDS:
            a, b_ = row[k], out[k]
            same = (a == b_) if isinstance(b_, str) else (float(a) == float(b_))
            if not same:
                diff.append(f"{k}: stored {a} vs now {b_}")
        n_old = np.array(json.loads(row["null_scores"]))
        n_new = np.array(json.loads(out["null_scores"]))
        same_null = len(n_old) == len(n_new) and bool(np.array_equal(n_old, n_new))
        if not same_null:
            k = int(np.argmax(n_old[:len(n_new)] != n_new)) if len(n_new) else 0
            diff.append(f"null distribution differs, first at permutation {k}")
        bad += bool(diff)
        print(f"  [{'PASS' if not diff else 'FAIL'}]  {tag:<40} {time.time() - t0:5.0f}s  "
              + ("identical, all 100 permutations" if not diff else "; ".join(diff[:2])),
              flush=True)
    print()
    print(f"  {bad} of {min(want, len(rows))} DIFFER -- do NOT install this version."
          if bad else
          f"  ALL {min(want, len(rows))} IDENTICAL, null distributions included. Safe to install.")
