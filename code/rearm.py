"""
rearm.py -- R1 / AUD-3 (the author, 2026-09-21): within a cell, a new setup may
fire only after the previous one CLOSES.

  CLOSE  = the first of (a) the PRIMARY fill -- frozen VWAP, WICK touch,
           tau = 0 -- or (b) T bars after the signal.  (sec. 21.20)
           The tau=0 fill is used for EVERY tau, so there is ONE setup set:
           the tau nesting and the band identity (21.5) survive.
  RE-ARM = after a close, a new setup still needs the existing trigger:
           a FRESH eps crossing or an anchor change (C89). A price that just
           stays beyond eps does not re-fire by itself.
  While an episode is open, NO setup of either side fires in that cell.

  Example (eps = 3 ATR, T = 54): price crosses 3 ATR at bar 100 -> setup.
  It wobbles 2.9 <-> 3.01 ATR and re-crosses at 104, 109, 131: under the old
  rule those were 4 setups; now they are candidates inside an open episode
  and are dropped. The primary fill comes at bar 140 -> closed at 140; the
  next candidate after 140 may become the next setup.

Implementation: candidates exactly as before -> the primary fill of every
candidate -> greedy keep (a candidate is kept iff it is AFTER the close of the
last kept one). The candidate count is the OLD rule's setup count, so the
burst rate old/new is MEASURED, not lost (R1 "record before overwriting").

Independent check: rearm_reference.py (slow, loop-based, own anchor and own
VWAP) must give the identical kept set -- rule D3.
"""
import numpy as np

from spec import MIN_LINE_BARS, SD_MIN_AGE, SD_MIN_BAND


def eligible(d, a, vwap, sd, atr, trigger):
    """Which bars may fire at all. ONE definition (C3), used by the dump and by
    the gates; rearm_reference.py re-derives it independently.

      the bar passed the loader's own warm-up / roll rules
      an anchor exists, the line and the ATR are finite
      the line has at least MIN_LINE_BARS bars, so its band has real width
        (AUD-37: a one-bar band was 'positive' only through rounding)
      SD trigger only: band >= SD_MIN_BAND ATR and age >= SD_MIN_AGE (R19)
    """
    age = np.arange(len(d)) - a
    ok = (d.eligible.to_numpy() & (a >= 0) & np.isfinite(vwap) & np.isfinite(atr)
          & (sd > 0) & (age >= MIN_LINE_BARS - 1))
    if trigger == "SD":
        ok &= (sd >= SD_MIN_BAND * atr) & (age >= SD_MIN_AGE)
    return ok


def candidates(dist, a, elig, eps, n, h_rec):
    """The unchanged trigger (C89): at/after eps, and not at/after eps on the
    previous bar OR the anchor just changed. End-of-data rows dropped."""
    ab = dist >= eps                                      # NaN -> False
    chg = np.r_[False, a[1:] != a[:-1]]
    fire = ab & (~np.roll(ab, 1) | chg)
    fire[0] = False
    fire &= elig
    sig = np.where(fire)[0]
    return sig[sig < n - h_rec - 2]


def first_fill(lo, hi, seg, sig, level, side, T, chunk_elems=20_000_000):
    """First bar j in 1..T whose WICK touches the frozen level from the
    correct side (low <= level if price was above, high >= level if below),
    without leaving the segment. 1-based; -1 if none within T."""
    n = len(lo)
    out = np.full(len(sig), -1, dtype=np.int64)
    off = np.arange(1, T + 1)
    step = max(100, chunk_elems // max(T, 1))
    for s in range(0, len(sig), step):
        S = sig[s:s + step]
        t = S[:, None] + off[None, :]
        tc = np.minimum(t, n - 1)
        valid = np.logical_and.accumulate((t < n) & (seg[tc] == seg[S][:, None]), axis=1)
        L = level[s:s + step][:, None]
        up = side[s:s + step][:, None] > 0
        hit = np.where(up, lo[tc] <= L, hi[tc] >= L) & valid
        anyh = hit.any(axis=1)
        out[s:s + step] = np.where(anyh, np.argmax(hit, axis=1) + 1, -1)
    return out


def greedy_keep(cand, fill, T):
    """Indices (into cand) of the setups kept under R1."""
    keep = []
    close_at = -1
    for k in range(len(cand)):
        s = int(cand[k])
        if s > close_at:
            keep.append(k)
            f = int(fill[k])
            close_at = s + (f if 1 <= f <= T else T)
    return np.array(keep, dtype=np.int64)


def bars_hash(b):
    """Short fingerprint of a list of bar indices (used to prove the new
    candidate stage == the OLD rule's setups, cell by cell)."""
    import hashlib
    return hashlib.sha256(np.asarray(b, dtype=np.int64).tobytes()).hexdigest()[:16]
