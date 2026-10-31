r"""
sd_precision.py -- AUD-38, second round: does the SD's remaining rounding
change any SETUP on the real data?

Why this exists. The anchored SD is computed from cumulative sums, which is
fast but subtracts two large numbers (m2 - m1^2) to get a small one. On Linux
those sums are kept in 80-bit floats and the error is ~1e-15 points; on
WINDOWS numpy's longdouble IS float64, so gate G5b measures ~2e-5 points
there. A deviation only matters if it moves a bar across the eps threshold.

This script settles that by measuring, not by reasoning:
  for one (N, anchor) it computes the EXACT SD of every bar's window with a
  direct two-pass loop, rebuilds the candidates and the re-arm setups for all
  four SD-trigger cells both ways, and reports how many setups differ.

    python code\\sd_precision.py               (N=60, hh -- about 3-6 minutes)
    python code\\sd_precision.py 240 session

It reads only data\\outright.parquet and writes nothing.
"""
import sys
import time
import numpy as np

from loader import load, rolling_argextreme
from pipeline import session_anchor, vwap_and_sd
from spec import SD_EPS, ATR_EPS, H_REC_MULT, t_stop
import rearm


def exact_sd(d, a):
    """SD of every bar's window by a direct two-pass loop (no cumulative sums)."""
    vp = d.vp.to_numpy().astype(float); v = d.volume.to_numpy().astype(float)
    n = len(d); out = np.zeros(n); t0 = time.time()
    for i in range(n):
        j = a[i]
        if j < 0 or j == i:
            continue
        w = v[j:i + 1]; p = vp[j:i + 1]; W = w.sum()
        if W <= 0:
            continue
        m = (p * w).sum() / W
        out[i] = np.sqrt(max((w * (p - m) ** 2).sum() / W, 0.0))
        if i and i % 1_000_000 == 0:
            print(f"    {i:,} / {n:,} bars ({time.time() - t0:.0f}s)", flush=True)
    return out


def setups(d, a, vw, sd, atr, close, hi, lo, seg, trig, eps):
    n = len(d)
    T = t_stop(trig, eps); h = H_REC_MULT * T
    el = rearm.eligible(d, a, vw, sd, atr, trig)
    scale = sd if trig == "SD" else atr
    dist = np.where(el, np.abs(close - vw) / np.maximum(scale, 1e-9), np.nan)
    c = rearm.candidates(dist, a, el, eps, n, h)
    f = rearm.first_fill(lo, hi, seg, c, vw[c], np.sign(close[c] - vw[c]), T)
    return set(c.tolist()), set(c[rearm.greedy_keep(c, f, T)].tolist())


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    akey = sys.argv[2] if len(sys.argv) > 2 else "hh"
    d = load("train")
    seg = d.seg.to_numpy(); atr = d.atr.to_numpy(); close = d.close.to_numpy()
    hi = d.high.to_numpy().astype(float); lo = d.low.to_numpy().astype(float)
    a = session_anchor(d) if akey == "session" else rolling_argextreme(
        d["high" if akey == "hh" else ("low" if akey == "ll" else "volume")]
        .to_numpy().astype(float), N, seg, akey != "ll")
    vw, sd_fast, *_ = vwap_and_sd(d, a)
    print(f"  longdouble mantissa bits on this machine: {np.finfo(np.longdouble).nmant} "
          f"(float64 = 52, 80-bit = 63)", flush=True)
    print("  computing the exact SD of every window (slow loop)...", flush=True)
    sd_slow = exact_sd(d, a)
    ok = np.isfinite(sd_fast) & (a >= 0) & (a != np.arange(len(d)))
    dev = np.abs(sd_fast[ok] - sd_slow[ok])
    print(f"\n  SD deviation on REAL data: max {dev.max():.2e} pts | "
          f"99.9th pct {np.percentile(dev, 99.9):.2e} | median {np.median(dev):.2e}")
    print(f"  bars where the two disagree about sd > 0: "
          f"{int(((sd_fast[ok] > 0) != (sd_slow[ok] > 0)).sum())}")
    print(f"\n  cell                      candidates differ   setups differ")
    worst = 0
    for trig, grid in (("SD", SD_EPS), ("ATR", ATR_EPS)):
        for eps in grid:
            c0, k0 = setups(d, a, vw, sd_fast, atr, close, hi, lo, seg, trig, eps)
            c1, k1 = setups(d, a, vw, sd_slow, atr, close, hi, lo, seg, trig, eps)
            worst = max(worst, len(k0 ^ k1))
            print(f"  N{N}_{akey}_{trig}_{eps:<6}  {len(c0):>8,} setups  "
                  f"{len(c0 ^ c1):>6}          {len(k0 ^ k1):>6}")
    print(f"\n  [{'PASS' if worst == 0 else 'FAIL'}]  the SD rounding changes no setup "
          f"in any cell of this group (worst cell: {worst})")
