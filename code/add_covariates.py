"""
add_covariates.py -- the covariates missing from the dump.

RESTORED (were in v2, lost in the rewrite -- my error):
  L_all       max single-bar move of the anchored VWAP, anchor -> signal, in ATR
  L_settled   same, ignoring the first 10 bars (the line's violent birth)

NEW:
  slope_k     (V_t - V_{t-k}) / ATR for k in {15, 60}: where the line is going
  bars_since_touch   bars since price last crossed THIS anchor's line,
              counted from the anchor bar (-1 if never)          (AUD-22)
  prior_touches      crossings of THIS anchor's line since the anchor bar
              (moved here from volume_features, which counted crossings of
              whichever line was live at each bar)                (AUD-21)
  dow         day of week of the CME trade date (0=Mon)           (AUD-16)
  days_to_exp calendar days from the trade date to the front contract's
              expiry (3rd Friday), from the contract calendar     (R7/AUD-15)
  is_rth      1 if inside 9:30-16:00 New York local time, weekdays (AUD-14)
  rv_close    realised vol from CLOSES over 60 bars, in ATR units
              (ATR is a RANGE measure; the two diverge exactly when gaps matter)
  day_pos     position within the range so far of this trade date AND this
              contract, 0..1 (a roll happens at 00:00 UTC, inside a trade
              date; grouping by trade date alone mixed two contracts, AUD-23)
  anchor_spike  anchor bar's true range / rolling median TR -- was the anchor
              a genuine level or a one-tick print? (C21)

L, slope and the crossing counts are computed PER ANCHOR RUN: while the
anchor is unchanged the VWAP path is a single running mean, so the whole run is
one vectorised pass. The line is always rebuilt FROM THE ANCHOR BAR, even when
the run starts later (an older high expired and the anchor jumped), so every
bar's value refers to its own anchor's line only (C118).

2026-09-21: main() and its join loops moved to code\\legacy\\ (AUD-29: it
anchored hv as the rolling LOW and duplicated cov_phase). cov_phase is the
only join path.
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import glob
import os
import re
import sys
import numpy as np
import pandas as pd

from loader import trade_day
from spec import SETTLE, SLOPES



def line_features(d, a):
    """L_all, L_settled, slope_k, bars_since_touch, prior_touches -- one pass
    over anchor runs."""
    n = len(d)
    vp = d.vp.to_numpy(); v = d.volume.to_numpy().astype(float)
    close = d.close.to_numpy(); atr = d.atr.to_numpy()
    L_all = np.full(n, np.nan, np.float32); L_set = np.full(n, np.nan, np.float32)
    slope = {k: np.full(n, np.nan, np.float32) for k in SLOPES}
    since = np.full(n, -1.0, np.float32)
    ptouch = np.full(n, np.nan, np.float32)

    # anchor runs: maximal stretches where a[i] is constant
    chg = np.r_[True, a[1:] != a[:-1]]
    starts = np.where(chg)[0]
    ends = np.r_[starts[1:], n] - 1
    for s, e in zip(starts, ends):
        anc = a[s]
        if anc < 0 or anc > s:
            continue
        sl = slice(anc, e + 1)
        pv = np.cumsum(vp[sl] * v[sl]); vv = np.cumsum(v[sl])
        V = pv / np.maximum(vv, 1e-9)              # the line, anchor -> e
        step = np.abs(np.diff(V))
        rm = np.maximum.accumulate(step) if len(step) else np.array([])
        off = np.arange(s, e + 1) - anc            # index into V for each bar
        A = atr[s:e + 1]
        if len(rm):
            idx = np.clip(off - 1, 0, len(rm) - 1)
            L_all[s:e + 1] = np.where(off >= 1, rm[idx] / A, np.nan)
            if len(step) > SETTLE:
                rm2 = np.full(len(step), np.nan)
                rm2[SETTLE:] = np.maximum.accumulate(step[SETTLE:])
                L_set[s:e + 1] = np.where(off > SETTLE, rm2[idx] / A, np.nan)
        for k in SLOPES:
            pk = off - k
            ok = pk >= 0
            sv = np.full(len(off), np.nan)
            sv[ok] = (V[off[ok]] - V[pk[ok]]) / A[ok]
            slope[k][s:e + 1] = sv
        # crossings of THIS line, from the ANCHOR bar (AUD-21/22)
        sg = np.sign(close[anc:e + 1] - V)
        cr = np.r_[False, (sg[1:] != sg[:-1]) & (sg[1:] != 0) & (sg[:-1] != 0)]
        ii = np.arange(len(sg))
        lastix = np.maximum.accumulate(np.where(cr, ii, -1))
        since[s:e + 1] = np.where(lastix[off] >= 0, off - lastix[off], -1.0)
        ptouch[s:e + 1] = np.cumsum(cr)[off]
    return L_all, L_set, slope, since, ptouch


def bar_features(d):
    ts = d.ts
    # AUD-14: RTH = 9:30-16:00 NEW YORK local time, so daylight saving is
    # handled (was UTC 13:30-20:00: right in summer, 1h wrong in winter).
    ny = ts.dt.tz_convert("America/New_York")
    hny = ny.dt.hour.to_numpy() + ny.dt.minute.to_numpy() / 60.0
    is_rth = ((hny >= 9.5) & (hny < 16.0) & (ny.dt.dayofweek.to_numpy() < 5)).astype(float)
    # AUD-16: weekday and "the day" = CME TRADE DATE (the 17:00 CT open
    # starts the next trade date), not the UTC calendar day
    tday = trade_day(ts)                                   # int days since 1970
    dow = ((tday + 3) % 7).astype(float)                   # 1970-01-01 = Thursday
    close = d.close.to_numpy(); atr = d.atr.to_numpy()
    r = np.r_[0.0, np.diff(close)]
    rv = (pd.Series(r.astype(np.float32)).rolling(60, min_periods=20)
            .std().to_numpy() / np.maximum(atr, 1e-9))
    # position within the day's range so far (causal: expanding within the day)
    # integer day codes -- NOT python date objects (that cost 2GB, C100)
    seg = d.seg.to_numpy()
    newday = np.r_[True, (tday[1:] != tday[:-1]) | (seg[1:] != seg[:-1])]   # AUD-23
    grp = np.cumsum(newday) - 1
    hi = pd.Series(d.high.to_numpy()).groupby(grp).cummax().to_numpy()
    lo = pd.Series(d.low.to_numpy()).groupby(grp).cummin().to_numpy()
    day_pos = (close - lo) / np.maximum(hi - lo, 1e-9)
    # R7: calendar days from the trade date to the contract's expiry
    days_to_exp = (d.exp_day.to_numpy().astype(np.int64) - tday).astype(np.float32)
    # anchor bar character
    tr = d.tr.to_numpy()
    medtr = (pd.Series(tr.astype(np.float32))
             .rolling(1440, min_periods=200).median().to_numpy())
    spike = tr / np.maximum(medtr, 1e-9)
    return {k: v.astype(np.float32) for k, v in
            dict(is_rth=is_rth, dow=dow, rv_close=rv, day_pos=day_pos,
                 days_to_exp=days_to_exp, spike=spike).items()}
