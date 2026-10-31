r"""
floors_todo.py -- why is a cell still in the queue?

The runner treats a cell as finished when every side that HAS a search result
also has a floor. A side can be missing a floor for two very different reasons:

  not run yet          normal -- it is still queued or in flight
  NOT WRITTEN          floor_v2 refused: its re-run of the stored search did
                       not reproduce the stored score. That is a REAL problem
                       and the cell will be re-queued forever until it is
                       looked at.

    python floors_todo.py            summary + any refusals
    python floors_todo.py --all      list every unfinished cell

Reads cores_v2, floors_v2 and logs; writes nothing.
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORES, FLOORS, LOGS = (os.path.join(HERE, d) for d in ("cores_v2", "floors_v2", "logs"))

if __name__ == "__main__":
    want = {}
    for f in glob.glob(os.path.join(CORES, "*_side*.parquet")):
        tag = os.path.basename(f)[:-8]
        if len(pd.read_parquet(f)):                      # read it, don't guess by size
            want.setdefault(tag.rsplit("_side", 1)[0], []).append(tag)
    have = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(FLOORS, "*.parquet"))}
    refused, unfinished = [], []
    for cell, tags in sorted(want.items()):
        missing = [t for t in tags if t not in have]
        if not missing:
            continue
        log = os.path.join(LOGS, f"floors_{cell}.log")
        txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
        for t in missing:
            side = t.rsplit("_side", 1)[1]
            if "NOT WRITTEN" in txt and f"side {side}" in txt:
                refused.append(t)
        unfinished.append((cell, missing))
    done = sum(1 for c, t in want.items() if all(x in have for x in t))
    print(f"  cells with a searched side : {len(want)}")
    print(f"  cells fully floored        : {done}")
    print(f"  cells still unfinished     : {len(unfinished)}")
    print(f"  cell+sides floored         : {len(have)} of "
          f"{sum(len(v) for v in want.values())}")
    if refused:
        print(f"\n  !! {len(refused)} side(s) REFUSED (re-run did not match the stored "
              f"search). These will be re-queued forever until fixed:")
        for t in refused[:20]:
            print(f"       {t}")
        print("     Send this list to the author -- it means a stored search and its "
              "re-run disagree.")
    else:
        print("\n  no refusals: every missing floor is simply not run yet.")
    if "--all" in sys.argv:
        print("\n  unfinished cells:")
        for cell, missing in unfinished:
            print(f"    {cell:<26} missing {len(missing)}: "
                  f"{', '.join(m.rsplit('_side', 1)[1] for m in missing)}")
