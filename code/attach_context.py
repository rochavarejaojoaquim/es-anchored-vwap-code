"""
attach_context.py -- adds context columns to each dump cell, in place.

For every SETUP row (is_real==1) in a cell:

A. CROSS-ANCHOR, 1m. The other two anchor rules' line and POC:
     xa_<rule>_dPV = (price - that rule's VWAP) / ATR      signed
     xa_<rule>_dPC = (price - that rule's POC)  / ATR      signed

B. HIGHER TIMEFRAME. Same two, for 5m/15m x {4h, 1d} lookbacks, using the
   most recently CLOSED HTF bar (gate already passed in enrich.py):
     htf_<tf>_<dur>_dPV / _dPC
   R4 (2026-09-21): the HTF line uses the CELL'S OWN anchor rule
   (htf_<rule>_<tf>m_<dur>.npz). Column names are unchanged; their meaning
   for hh cells is unchanged, for ll / hv / session cells it is now their own.

C. "_xs" COLUMNS (R10 / AUD-19): for every column in spec.XS_BASE,
     <col>_xs = <col> x side
   + = pointing the way of the stretch; for a slope, the line moving TOWARD
   price. Example: price ABOVE the line (side +1), line rising (slope_15 =
   +0.2) -> slope_15_xs = +0.2 (toward). Price BELOW (side -1), line still
   rising -> -0.2 (away). Pooled tables cannot see a toward/away effect
   without these; per-side searches see the same information either way.
   Needs slope_15/60 from cov_phase B -> this step refuses to run before it.

All POCs computed ONLY at the signal bars (lazy).
Twin rows get NaN -- context is a property of the setup, and the twin is a
distance-matched placebo, not a second VWAP.
"""
import time
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

from loader import load
from pipeline import rolling_argextreme, session_anchor, vwap_and_sd, TICK

DUMP = DUMP_DIR
from spec import HTF, XS_BASE
import safeio
TF_MIN = {"5m": 5, "15m": 15}


def htf_files(rule):
    return {(f"{m}m", dur): ROOT_DIR + f"htf_{rule}_{m}m_{dur}.npz"
            for m, lbs in HTF for dur in lbs}


def poc_at(vp, v, anchor, bars):
    """Exact POC over [anchor[b], b] -- ONLY at the bars given."""
    out = np.full(len(bars), np.nan)
    for k, i in enumerate(bars):
        a = anchor[i]
        if a < 0 or a > i:
            continue
        b = np.round(vp[a:i + 1] / TICK).astype(np.int64)
        lo = b.min()
        h = np.bincount(b - lo, weights=v[a:i + 1])
        out[k] = (np.argmax(h) + lo) * TICK
    return out


def main(only_n=None):
    d = load("train")
    close = d.close.to_numpy(); atr = d.atr.to_numpy()
    vp = d.vp.to_numpy(); vol = d.volume.to_numpy().astype(float)
    seg = d.seg.to_numpy()
    sess = session_anchor(d)

    htf_by_rule = {}

    # HTF bar arrays (for the lazy HTF POC)
    from enrich import resample_seg
    rs = {tf: resample_seg(d, m) for tf, m in TF_MIN.items()}

    from loader import cell_files
    files = cell_files("N*", DUMP)
    for f in files:
        m = re.match(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)\.parquet", os.path.basename(f))
        N = int(m.group(1)); akey = m.group(2)
        if only_n and N not in only_n:
            continue
        _t0 = time.time()
        x = pd.read_parquet(f)
        if "htf_15m_1d_dPV" in x.columns:   # always added, unlike xa_<own rule>
            continue                                   # already attached
        if "slope_15" not in x.columns:
            raise SystemExit(f"STOP: {os.path.basename(f)} has no slope_15 -- run "
                             f"cov_phase B for N={N} {akey} before attach_context")
        if akey not in htf_by_rule:
            htf_by_rule.clear()                        # one rule in memory at a time
            htf_by_rule[akey] = {}
            for key, fn in htf_files(akey).items():
                z = safeio.load_npz(fn)                # refuses a stale file
                htf_by_rule[akey][key] = (z["map_idx"], z["vwap"], z["anchor"])
        htf = htf_by_rule[akey]
        real = x.index[x.is_real == 1]
        bars = x.loc[real, "bar"].to_numpy().astype(np.int64)
        atr0 = x.loc[real, "atr0"].to_numpy()

        # ---- A. cross-anchor, the two rules NOT used as the trigger
        for rule, col, mx in [("hh", "high", True), ("ll", "low", False),
                              ("hv", "volume", True),      # volume-anchored line
                              ("session", None, None)]:
            if rule == akey:
                continue
            a = sess if rule == "session" else rolling_argextreme(
                d[col].to_numpy().astype(float), N, seg, mx)
            vw, _sd, _W, *_ = vwap_and_sd(d, a)
            poc = poc_at(vp, vol, a, bars)
            x.loc[real, f"xa_{rule}_dPV"] = (close[bars] - vw[bars]) / atr0
            x.loc[real, f"xa_{rule}_dPC"] = (close[bars] - poc) / atr0
            del a, vw

        # ---- B. higher timeframe
        for (tf, dur), (map_idx, vw_h, anc_h) in htf.items():
            j = map_idx[bars]
            ok = j >= 0
            dpv = np.full(len(bars), np.nan)
            dpc = np.full(len(bars), np.nan)
            if ok.any():
                dpv[ok] = (close[bars[ok]] - vw_h[j[ok]]) / atr0[ok]
                r = rs[tf]
                hp = poc_at(r.vp.to_numpy(), r.volume.to_numpy().astype(float),
                            anc_h, j[ok])
                dpc[ok] = (close[bars[ok]] - hp) / atr0[ok]
            x.loc[real, f"htf_{tf}_{dur}_dPV"] = dpv
            x.loc[real, f"htf_{tf}_{dur}_dPC"] = dpc

        # ---- C. _xs columns (x side), setups only
        sd_ = x.loc[real, "side"].to_numpy(float)
        for col in XS_BASE:
            if col in x.columns:
                x.loc[real, f"{col}_xs"] = x.loc[real, col].to_numpy(float) * sd_
        safeio.write_parquet(x, f)
        print(f"  {os.path.basename(f):<34} +{len(x.columns)} cols  "
              f"{len(real):,} setups  {time.time()-_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or None)
