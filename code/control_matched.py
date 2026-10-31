"""
control_matched.py -- the matched random-level null.

For every real episode at bar i (signed gap g0, distance d in frozen ATR),
draw K control bars j from the SAME stratum and place a level at

    level_j  =  close[j]  -  sign(g0) * d * atr[j]

so the control starts at the identical ATR distance, on the identical side.
Then walk it with the IDENTICAL touch and stop machinery (C65).

Two matching schemes, reported separately:
    M1  hour bucket x ATR quintile
    M2  hour bucket x ATR quintile x 240-bar return quintile

M2 additionally matches the trend state. Real episodes are, by construction,
situations where price has moved away from a recent extreme, so M1 leaves
trend unmatched; M2 closes that at the cost of a tighter, smaller pool.

Controls are drawn from a DIFFERENT CME trade date AND more than h_rec bars
away, so the episode and its control never share a price path (AUD-7/13;
the drawing itself is controls_fast.draw_controls).

2026-09-21: only strata() is live. run_controls() and the __main__ report
used the UTC date and hour, a 400-bar walk and close-only touches; they moved
to code\\legacy\\control_matched_runcontrols_legacy.py (AUD-32).
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import numpy as np
import pandas as pd



def strata(d, with_momentum):
    atr = d.atr.to_numpy()
    hour = d.ts.dt.tz_convert("America/Chicago").dt.hour.to_numpy()   # AUD-17
    hb = hour // 4                                   # 6 buckets of 4 hours (Chicago)
    ok = np.isfinite(atr)
    aq = np.full(len(d), -1)
    aq[ok] = pd.qcut(atr[ok], 5, labels=False, duplicates="drop")
    lab = hb.astype(np.int64) * 100 + aq
    if with_momentum:
        c = d.close.to_numpy()
        ret = np.full(len(d), np.nan)
        ret[240:] = (c[240:] - c[:-240]) / np.maximum(atr[240:], 1e-9)
        rok = np.isfinite(ret)
        rq = np.full(len(d), -1)
        rq[rok] = pd.qcut(ret[rok], 5, labels=False, duplicates="drop")
        lab = lab * 10 + rq
    return lab
