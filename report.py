r"""
report.py -- read the searches, their floors and the coverage, and lay out what
was actually tested and what survived STAGE 1.

    python report.py                 the table + summary
    python report.py --csv           also write results_table.csv

WHAT IT DOES NOT DO: declare anything real. With 100 permutations the smallest
possible p is 1/101 = 0.0099, and with ~409 tests any multiple-testing
correction needs a smaller p than that. So this run is a SCREEN, not a verdict:

  STAGE 1 (this)   100 permutations on every cell+side. A cell SURVIVES if its
                   observed score beats the 95th percentile of its OWN null
                   (exact permutation p <= 0.05). Survivors are candidates, not results.
  STAGE 2 (later)  re-run a SHORTLIST at many more permutations. 24.8 fixes it
                   at the top 20 by Romano-Wolf adjusted p, K set by cost first.

The session family is counted ONCE (N=60 is the pre-registered primary; the
other lookbacks hold the same episodes and differ only in cross-anchor columns).
Both counts are printed, so nothing is hidden by the choice.
"""
import glob, json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_PRIMARY_N = 60          # pre-registered 2026-09-23 (option A)


def load(folder, cols=None):
    fs = glob.glob(os.path.join(HERE, folder, "*.parquet"))
    out = []
    for f in fs:
        t = pd.read_parquet(f)
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def is_duplicate_session(cell):
    p = cell.split("_")
    return p[1] == "session" and int(p[0][1:]) != SESSION_PRIMARY_N


if __name__ == "__main__":
    cores = load("cores_v2")
    floors = load("floors_v2")
    if cores.empty:
        raise SystemExit("no searches in cores_v2\\")
    print(f"  searches with a result : {len(cores)}")
    cov = os.path.join(HERE, "search_coverage.csv")
    if os.path.exists(cov):
        c = pd.read_csv(cov)
        print(f"  cell+sides attempted   : {len(c)}  "
              f"({int((c.state == 'empty').sum())} never searched -- too thin)")
    if floors.empty:
        print("\n  no floors yet -- nothing can be read from the searches alone.")
        raise SystemExit(0)
    t = floors.copy()
    t["dup_session"] = [is_duplicate_session(x) for x in t.cell]
    t["survives"] = t.p <= 0.05
    prim = t[~t.dup_session]
    print(f"  floors computed        : {len(t)}  ({len(prim)} after counting the "
          f"session family once, N={SESSION_PRIMARY_N} primary)")
    print(f"  p floor for {int(t.nperm.max())} permutations: {1/(1+t.nperm.max()):.4f}  "
          f"-- correction across {len(prim)} tests needs ~{0.05/max(len(prim),1):.5f}")
    print(f"\n  STAGE 1 survivors (exact permutation p <= 0.05, registered 24.8): "
          f"{int(prim.survives.sum())} of {len(prim)}")
    exp = len(prim) * ((int(prim.nperm.iloc[0]) + 1) // 20) / (int(prim.nperm.iloc[0]) + 1)
    print(f"  expected by chance at the registered criterion (true size {100*exp/max(len(prim),1):.2f}%): ~{exp:.1f}")
    try:
        import rw_stepdown as RW
    except ImportError as e:
        raise SystemExit(f"  rw_stepdown.py not importable: {e}")
    N = np.array([json.loads(x) for x in prim.null_scores]).T        # (B,S)
    Tz, Nz = RW.studentize(prim.obs_score.to_numpy(), N)
    prim = prim.assign(rw_p=RW.rw_stepdown(Tz, Nz))
    print(f"  ROMANO-WOLF adjusted p <= 0.05 (paper-level, 24.8): {int((prim.rw_p <= 0.05).sum())} of {len(prim)}   |  min {prim.rw_p.min():.4f}")
    s = prim[prim.survives].sort_values(["p", "edge_over_floor"], ascending=[True, False])
    if len(s):
        show = s.head(25)[["cell", "side", "col", "obs_score", "floor95",
                           "edge_over_floor", "p", "eA", "eB", "nA", "nB"]].copy()
        for k in ("obs_score", "floor95", "edge_over_floor", "eA", "eB"):
            show[k] = (100 * show[k]).round(2)
        print("\n  top survivors (edges in percentage points):")
        print(show.to_string(index=False))
    empt = int(t.n_null_empty.sum())
    if empt:
        print(f"\n  note: {empt} permutations produced no candidate at all "
              f"(counted as -inf, i.e. the null could not beat the observed).")
    if "--csv" in sys.argv:
        t.to_csv(os.path.join(HERE, "results_table.csv"), index=False)
        print("\n  written: results_table.csv")
    print("\n  NOT a verdict: with a p floor of "
          f"{1/(1+t.nperm.max()):.4f} no single cell can clear correction across "
          f"{len(prim)} tests -- that is what the Romano-Wolf line above does. Stage 2 re-runs the top 20.")
