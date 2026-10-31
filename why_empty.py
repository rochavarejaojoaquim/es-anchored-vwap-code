r"""
why_empty.py -- an empty result file in cores_v2\ means one of TWO things, and
the file does not say which (AUD-43):

  NOT SEARCHED  the cell+side never reached the search: under 1,000 setups, or
                under 1,000 twins, in block A or B (the AUD-12 gate)
  NO CUT        it WAS searched over every column and cut point, and nothing
                left enough episodes on both sides of the cut

Only the second is a test that was performed, so only the second counts when we
ask how many chances the search had. This rebuilds the gate from the dump and
splits them.

    python why_empty.py           (~1-2 minutes; reads 3 columns per cell)

Reads data\dump and cores_v2; writes nothing.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from loader import cell_files, DUMP_DIR                           # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cores_v2")
GATE = 1000                                                       # AUD-12

if __name__ == "__main__":
    rows = []
    for f in cell_files("N*", DUMP_DIR):
        cell = os.path.basename(f)[:-8]
        x = pd.read_parquet(f, columns=["is_real", "side", "block"])
        r = x[x.is_real == 1]; c = x[x.is_real == 0]
        for side in (None, 1.0, -1.0):
            rr = r if side is None else r[np.sign(r.side) == side]
            cc = c if side is None else c[np.sign(c.side) == side]
            nA, nB = (rr.block == "A").sum(), (rr.block == "B").sum()
            tA, tB = (cc.block == "A").sum(), (cc.block == "B").sum()
            thin = min(nA, nB) < GATE or min(tA, tB) < GATE
            p = os.path.join(OUT, f"{cell}_side{side}.parquet")
            if not os.path.exists(p):
                state = "missing"
            else:
                state = "result" if len(pd.read_parquet(p)) else "empty"
            rows.append(dict(cell=cell, side=str(side), nA=int(nA), nB=int(nB),
                             tA=int(tA), tB=int(tB), thin=bool(thin), state=state))
    t = pd.DataFrame(rows)
    res = (t.state == "result").sum(); emp = (t.state == "empty").sum()
    ns = ((t.state == "empty") & t.thin).sum()
    nc = ((t.state == "empty") & ~t.thin).sum()
    odd = ((t.state == "result") & t.thin).sum()
    print(f"\n  {len(t)} cell+side searches")
    print(f"    {res:>4}  produced a result")
    print(f"    {ns:>4}  NOT SEARCHED -- under {GATE:,} setups or twins in a block")
    print(f"    {nc:>4}  SEARCHED, no cut left enough episodes on both sides")
    print(f"    {int(t.state.eq('missing').sum()):>4}  missing file")
    if odd:
        print(f"    !! {odd} results from cell+sides that look thin -- tell the author")
    print(f"\n  tests actually performed: {res + nc} of {len(t)}")
    by = t[t.state == "empty"].groupby([t.cell.str.split("_").str[1], "thin"]).size()
    print("\n  empties by anchor (True = never searched):")
    for k, v in by.items():
        print(f"    {k[0]:<9} {'not searched' if k[1] else 'no cut     '}  {v}")
    t.to_csv(os.path.join(os.path.dirname(OUT), "search_coverage.csv"), index=False)
    print("\n  written: search_coverage.csv  (every cell+side, its counts and why)")
