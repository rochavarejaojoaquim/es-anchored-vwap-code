"""
enrich.py -- builds the HIGHER-TIMEFRAME context files: each 1m bar mapped to
its last CLOSED HTF bar, with that timeframe's anchored VWAP. It writes NO cell
columns -- attach_context.py joins them into the cells.

R4 / AUD-11 (2026-09-21): the HTF line is anchored by the CELL'S OWN RULE
  hh -> rolling HTF high     ll -> rolling HTF low
  hv -> highest-volume HTF bar   session -> the Globex session open
  4 rules x 2 timeframes (5m, 15m) x 2 lookbacks (4h, 1d) = 16 files
  htf_<rule>_<tf>m_<dur>.npz   (session does not use the lookback, so its 4h
  and 1d files are identical -- kept so every rule has the same 4 files)
  Before: every cell got the rolling-HIGH line.
AUD-36: an HTF bar's price = the VOLUME-WEIGHTED mean of its 1m typical
  prices, so the HTF VWAP equals the true 1m VWAP at HTF-bar resolution.
  (Was the unweighted mean.) Example: two 1m bars, typical 100 on volume 900
  and 110 on volume 100 -> HTF price 101, not 105.

LOOK-AHEAD GATE: an HTF value at a 1m bar must come from an HTF bar that
CLOSED at or before that 1m bar. Verified before anything is written.
Files carry SPEC_VERSION and are written atomically (safeio).
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
import glob
import os
import re
import numpy as np
import pandas as pd

from loader import load
from pipeline import rolling_argextreme, vwap_and_sd, TICK
from loader import reopen_mask
from spec import HTF, RULES
import safeio

DUMP = DUMP_DIR


def poc_series(vp, v, anchor, bars):
    """Exact POC at the given bars, over [anchor, bar]."""
    out = np.empty(len(bars))
    for k, i in enumerate(bars):
        s = slice(anchor[i], i + 1)
        b = np.round(vp[s] / TICK).astype(np.int64)
        lo = b.min()
        h = np.bincount(b - lo, weights=v[s])
        out[k] = (np.argmax(h) + lo) * TICK
    return out


def resample_seg(d, minutes):
    parts = []
    for seg, sub in d.set_index("ts").groupby("seg"):
        sub = sub.assign(pv=sub.vp.astype(float) * sub.volume.astype(float))
        r = sub.resample(f"{minutes}min").agg(
            {"high": "max", "low": "min", "close": "last",
             "volume": "sum", "pv": "sum"})
        r = r[r.volume > 0].copy()
        r["vp"] = r.pv / r.volume                  # AUD-36: volume-weighted
        r = r.drop(columns="pv")
        r["seg"] = seg
        parts.append(r)
    return pd.concat(parts).reset_index().sort_values("ts").reset_index(drop=True)


def htf_anchor(r, rule, N):
    seg_h = r.seg.to_numpy()
    if rule == "session":
        reopen = np.array(reopen_mask(r.ts), copy=True)
        reopen |= np.r_[True, seg_h[1:] != seg_h[:-1]]
        return np.maximum.accumulate(np.where(reopen, np.arange(len(r)), -1))
    col, mx = {"hh": ("high", True), "ll": ("low", False), "hv": ("volume", True)}[rule]
    return rolling_argextreme(r[col].to_numpy().astype(float), N, seg_h, mx)


def htf_context(d, minutes, lookbacks, rule, r=None):
    """Map each 1m bar to the most recently CLOSED HTF bar (searchsorted, not
    merge_asof -- far lighter), and return the HTF VWAP plus everything needed
    to compute the POC LAZILY for only the bars a cell actually asks for."""
    if r is None:
        r = resample_seg(d, minutes)
    vp = r.vp.to_numpy(); v = r.volume.to_numpy().astype(float)
    seg_h = r.seg.to_numpy()
    close_ts = (r.ts + pd.Timedelta(minutes=minutes)).to_numpy()
    ts1 = d.ts.to_numpy()
    # index of the last HTF bar whose CLOSE is <= this 1m bar's timestamp
    j = np.searchsorted(close_ts, ts1, side="right") - 1
    same = (j >= 0) & (seg_h[np.maximum(j, 0)] == d.seg.to_numpy())
    j = np.where(same, j, -1)
    out = {}
    for name, N in lookbacks.items():
        a = htf_anchor(r, rule, N)
        vw, sd, W, *_ = vwap_and_sd(r, a)
        out[name] = dict(map_idx=j, vwap=vw, anchor=a, vp=vp, vol=v,
                         close_ts=close_ts)
    return out


def check_lookahead(d, minutes, ctx):
    """GATE: the mapped HTF bar must have CLOSED at or before the 1m bar."""
    ts1 = d.ts.to_numpy()
    bad = 0
    for name, c in ctx.items():
        j = c["map_idx"]; m = j >= 0
        bad += int((c["close_ts"][j[m]] > ts1[m]).sum())
    return bad == 0, f"{bad} look-ahead violations at {minutes}m"


def htf_file(rule, minutes, name):
    return ROOT_DIR + f"htf_{rule}_{minutes}m_{name}.npz"


if __name__ == "__main__":
    d = load("train")
    print("Building HTF context (4 rules x 2 tf x 2 lookbacks)...", flush=True)
    for minutes, lbs in HTF:
        r = resample_seg(d, minutes)
        for rule in RULES:
            c = htf_context(d, minutes, lbs, rule, r=r)
            ok, msg = check_lookahead(d, minutes, c)
            print(f"  [{'PASS' if ok else 'FAIL'}]  look-ahead gate {rule} {minutes}m  {msg}", flush=True)
            if not ok:
                raise SystemExit("GATE FAILURE -- nothing written for this rule")
            for name, cc in c.items():
                safeio.save_npz(htf_file(rule, minutes, name), map_idx=cc["map_idx"],
                                vwap=cc["vwap"], anchor=cc["anchor"])
                print(f"  wrote htf_{rule}_{minutes}m_{name}.npz  "
                      f"({(cc['map_idx'] >= 0).mean()*100:.1f}% of 1m bars mapped)", flush=True)
            del c
