"""
rearm_reference.py -- R9 / rule D3: an INDEPENDENT, slow, loop-based build of
one cell's setups under the re-arm rule R1. It shares NOTHING with the fast
path except the loader (the data constraints, rule C3):

  anchor     brute force over the window (latest bar wins a tie), not the deque
  VWAP, SD   direct sums over [anchor, bar] each bar, not cumulative sums
  trigger    a bar-by-bar state machine (open / closed), not a greedy pass
  fill       a bar-by-bar loop, not the matrix walk

The setup list and the primary hit rate must match the fast dump EXACTLY.
Whatever it prints becomes selfcheck's known number (reference_rearm.json).

    python code\\rearm_reference.py                 (default cell N60_hh_ATR_12.0)
    python code\\rearm_reference.py 60 hh ATR 12.0

Two honest diagnostics are printed with the answer:
  near_eps   bars whose distance is within 1e-9 of eps: the two builds compute
             VWAP by different arithmetic, so only these could legitimately flip
  one_bar    bars that would fire on a ONE-BAR line (anchor = signal bar). The
             reference treats that band as zero-width (not eligible); the fast
             build's sd can be ~1e-6 there from rounding (AUD-37).
Speed: ~1-3 minutes per million bars on a Ryzen 5 3600.
"""
import json
import os
import sys
import time
import numpy as np

from loader import load
from spec import SD_MIN_AGE, SD_MIN_BAND, H_REC_MULT, t_stop, SPEC_VERSION

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_FILE = os.path.join(HERE, "reference_rearm.json")


def reference_cell(d, N, akey, trigger, eps, progress=False):
    close = d.close.to_numpy().astype(float); hi = d.high.to_numpy().astype(float)
    lo = d.low.to_numpy().astype(float); vp = d.vp.to_numpy().astype(float)
    v = d.volume.to_numpy().astype(float); atr = d.atr.to_numpy().astype(float)
    seg = d.seg.to_numpy(); elig0 = d.eligible.to_numpy()
    n = len(d)
    T = t_stop(trigger, eps); h_rec = H_REC_MULT * T
    src = {"hh": hi, "ll": lo, "hv": v}.get(akey)
    if akey == "session":
        import pandas as pd
        h = d.ts.dt.tz_convert("America/Chicago").dt.hour.to_numpy()
    kept, fills = [], []
    near_eps = one_bar = 0
    prev_ab = False
    prev_a = -2
    seg_start = 0
    sess_a = -1
    open_until = -1
    t0 = time.time()
    for i in range(n):
        if i > 0 and seg[i] != seg[i - 1]:
            seg_start = i
        # ---- anchor, from first principles
        if akey == "session":
            if i == seg_start or (h[i] == 17 and (i == 0 or h[i - 1] != 17)):
                sess_a = i
            a = sess_a
        elif i - seg_start >= N - 1:
            w = src[i - N + 1:i + 1]
            rev = w[::-1]
            j = int(np.argmax(rev)) if akey != "ll" else int(np.argmin(rev))
            a = i - j                                   # latest bar wins a tie
        else:
            a = -1
        chg = (i > 0) and (a != prev_a)
        prev_a = a
        # ---- VWAP and SD by direct sums
        ab = False
        if a >= 0 and elig0[i] and np.isfinite(atr[i]):
            ww = v[a:i + 1]; W = ww.sum()
            if W > 0:
                p = vp[a:i + 1] - vp[a]
                m = (p * ww).sum() / W
                var = (ww * (p - m) ** 2).sum() / W if i > a else 0.0
                sd = np.sqrt(max(var, 0.0))
                vw = m + vp[a]
                ok = sd > 0 and (i - a) >= 1        # AUD-37: >= 2 bars in the line
                if trigger == "SD":
                    ok = ok and sd >= SD_MIN_BAND * atr[i] and (i - a) >= SD_MIN_AGE
                scale = sd if trigger == "SD" else atr[i]
                if i == a and trigger == "ATR" and abs(close[i] - vw) / atr[i] >= eps:
                    one_bar += 1                   # would have fired on a one-bar line
                if ok:
                    dist = abs(close[i] - vw) / max(scale, 1e-9)
                    if abs(dist - eps) < 1e-9:
                        near_eps += 1
                    ab = dist >= eps
        fire = ab and (not prev_ab or chg) and i > 0
        prev_ab = ab
        if fire and i < n - h_rec - 2 and i > open_until:
            # ---- the primary fill, bar by bar
            level = vw; up = close[i] > vw
            f = -1
            for j in range(1, T + 1):
                t = i + j
                if t >= n or seg[t] != seg[i]:
                    break
                if (lo[t] <= level) if up else (hi[t] >= level):
                    f = j
                    break
            kept.append(i); fills.append(f)
            open_until = i + (f if f >= 1 else T)
        if progress and i and i % 500_000 == 0:
            print(f"    {i:,} / {n:,} bars  ({time.time() - t0:.0f}s)", flush=True)
    kept = np.array(kept, dtype=np.int64); fills = np.array(fills)
    hit = float(((fills >= 1) & (fills <= T)).mean()) if len(kept) else float("nan")
    return dict(kept=kept, fills=fills, hit=hit, T=T, near_eps=near_eps, one_bar=one_bar)


from rearm import bars_hash      # the same fingerprint function, nothing else shared


if __name__ == "__main__":
    a = sys.argv[1:] or ["60", "hh", "ATR", "12.0"]
    N, akey, trig, eps = int(a[0]), a[1], a[2], float(a[3])
    d = load("train")
    t0 = time.time()
    r = reference_cell(d, N, akey, trig, eps, progress=True)
    cell = f"N{N}_{akey}_{trig}_{eps}"
    out = dict(cell=cell, spec=SPEC_VERSION, n_setups=int(len(r["kept"])),
               hit_rate_tau0=r["hit"], bars_sha=bars_hash(r["kept"]),
               first_bars=[int(x) for x in r["kept"][:5]],
               near_eps=r["near_eps"], one_bar=r["one_bar"],
               seconds=round(time.time() - t0))
    tmp = REF_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, REF_FILE)
    print(json.dumps(out, indent=1))
    print(f"written -> {REF_FILE}")
