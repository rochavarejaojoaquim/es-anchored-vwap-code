"""
pipeline.py -- ONE clean module. Replaces grid_stage1.py (C94: that file had
drifted through ~12 in-place patches and a sed silently disabled a filter).

Every decision is stated here, once. Reuses only modules that have passed a
correctness gate: loader.py, walk_fast.py, controls_fast.py.

LOCKED SPEC v5 (2026-09-10)
  data      ES front month, spreads dropped, anchors reset at rolls
  train     A 2010-2018 | B 2022-2026        sealed 2018-2022 (untouched)
  ATR       true range, 14, Wilder, frozen at signal, reopen bar excluded
  anchor    rolling extreme over N bars (wick), or session open; frozen at signal
  trigger   ATR eps in {3,4,6,8,12}          T = 45*(eps/2.75)^2
            SD  eps in {1,1.5,2,3}           T = 30*eps^2
            SD requires age>=30 AND sd>=1 ATR          (R19, theory-based)
            trigger re-arms on a crossing OR on an anchor change  (C89)
            AND (2026-09-21, AUD-3/R1) only after the previous setup in the
            cell has CLOSED: primary fill (frozen, wick, tau=0) or T bars
  primary   frozen VWAP level vs matched random level, tau=0 (§2;
            was 0.25 before the switch in turns 338-343)
  adverse   X grid {0.5,1,1.5,2,3,4,6,8,12} ATR (spec.XS; restored 2026-09-21, AUD-30)
  path      post-signal summaries at bars 5, 15, 30       (delayed-entry rules)
  profile   EXACT POC by histogram, plus kurtosis and value-area width  (C97)
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import os
import numpy as np
import pandas as pd

from loader import load, stamp, reopen_mask
from walk_fast import walk
from controls_fast import walk_static, draw_controls
from control_matched import strata

# AUD-29 / AUD-30: every locked constant lives in spec.py; re-exported here so
# existing imports keep working. Nothing is redefined in this file (C3).
from spec import (PRICE_SCALE, ATR_EPS, SD_EPS, NS, ANCHORS, TAUS, XS, PATH_BARS,  # noqa: F401
                  MIN_SIGNALS, SD_MIN_AGE, SD_MIN_BAND, TICK, K_CTRL, t_stop)
OUT = DUMP_DIR


# ------------------------------------------------------------------ anchors
# AUD-9: the ONE rolling_argextreme lives in loader.py (was duplicated here).
from loader import rolling_argextreme  # noqa: E402  (re-exported)


def session_anchor(d):
    reopen = np.array(reopen_mask(d.ts), copy=True)     # AUD-9: one rule
    seg = d.seg.to_numpy()
    reopen |= np.r_[True, seg[1:] != seg[:-1]]
    idx = np.arange(len(d))
    return np.maximum.accumulate(np.where(reopen, idx, -1))


class Cum:
    """The exact integer cumulative sums of an anchored line (see _seg_cumsum).
    Ve/Vs are volume, Pe/Ps are X*volume where X = 12*(typical price - ref)."""

    __slots__ = ("Ve", "Vs", "Pe", "Ps")

    def __init__(self, Ve, Vs, Pe, Ps):
        self.Ve, self.Vs, self.Pe, self.Ps = Ve, Vs, Pe, Ps

    def vol(self, anc, t):
        """Window volume for bars [anc .. t]; anc and t may be arrays."""
        return (self.Ve[t + 1] - self.Vs[anc]).astype(float)

    def line(self, anc, t, W=None):
        """The anchored VWAP at bar t MINUS its segment reference, in points."""
        W = self.vol(anc, t) if W is None else W
        return ((self.Pe[t + 1] - self.Ps[anc]).astype(float)
                / np.maximum(W, 1e-9) / PRICE_SCALE)


def tick_ints(d):
    """(q, v, ok): the typical price as an EXACT whole number and the volume as
    an integer.  q = 4*(high + low + close), so the typical price is q/12
    (spec.PRICE_SCALE).  On a 0.25 price grid q is exact; `ok` is False if any
    bar is off the grid, which must stop the caller rather than pass silently."""
    hlc = (d.high.to_numpy().astype(float) + d.low.to_numpy().astype(float)
           + d.close.to_numpy().astype(float)) * 4.0
    q = np.rint(hlc).astype(np.int64)
    ok = bool(np.all(np.abs(hlc - q) < 1e-6))
    v = np.rint(d.volume.to_numpy().astype(float)).astype(np.int64)
    return q, v, ok


def _seg_cumsum(term, starts, counts):
    """Cumulative sums RESET at each segment, as int64, with an overflow guard.
    Resetting keeps the totals ~1e13 instead of ~1e17, and a window never
    crosses a segment anyway.

    TWO arrays, because one cannot do both jobs at a segment boundary:
      end[i+1]  = sum over [segment start .. i]        (the window's last bar)
      start[j]  = sum over [segment start .. j-1], 0 at a segment start
                                                      (the window's anchor)
      sum over j..i = end[i+1] - start[j]
    With one array, the slot at a segment's first bar had to hold BOTH the
    previous segment's total and zero; it held the total, and window volumes
    came out negative (caught by gate G7 on 2026-09-22)."""
    c = np.zeros(len(term) + 1, dtype=np.int64)
    for s0, k in zip(starts, counts):
        part = term[s0:s0 + k]
        c[s0 + 1:s0 + k + 1] = np.cumsum(part)
        exact = float(np.abs(part).astype(float).sum())
        got = float(c[s0 + k]); want = float(np.asarray(part, dtype=float).sum())
        if exact > 4e18 or abs(got - want) > max(1.0, exact * 1e-9):
            raise OverflowError(
                "the exact integer moments overflowed int64 on one segment "
                "(sum ~%.3g). Stop and report this -- nothing was written." % exact)
    start = c[:-1].copy()
    start[starts] = 0
    return c, start


def vwap_and_sd(d, anchor):
    """Anchored VWAP and volume-weighted SD, computed EXACTLY (AUD-38,
    decision B of 2026-09-22).

    Why exact. The SD is one subtraction of two big numbers (m2 - m1^2): on
    real ES data m2 is ~6e5 while the variance is ~1. With floating sums that
    cost up to 0.0168 points on the author's PC, where numpy's longdouble IS
    float64, and it flipped setups across the eps threshold in three SD cells.

    How. ES prices sit on a 0.25 grid, so q = 4*(high+low+close) is a WHOLE
    number and the typical price is exactly q/12. Whole numbers do not round:
    the anchored sums of X*v and X*X*v (X = q minus q at the segment's first
    bar) are exact int64, reset per segment so they stay ~1e13, with an
    overflow guard. Only the final division is floating point, which leaves
    ~1e-13 points -- 11 orders below one tick.

    A FLAT window (every typical price identical) must give exactly 0, not a
    rounding crumb: those are re-resolved with Python's unlimited integers.
    AUD-37: a one-bar line is flat by definition, so its SD is 0 and
    rearm.eligible drops it.

    Returns vwap (points), sd (points), W (volume), cum (the exact integer
    cumulative sums; the live line at bar t anchored at `anc` is
    ref + cum.line(anc, t)) and ref (the segment's reference price)."""
    q, v, grid_ok = tick_ints(d)
    if not grid_ok:
        raise ValueError("some bar is not on the 0.25 price grid -- the exact "
                         "integer moments do not apply to this data")
    seg = d.seg.to_numpy()
    n = len(d)
    starts = np.r_[0, np.where(seg[1:] != seg[:-1])[0] + 1]
    counts = np.diff(np.r_[starts, n])
    qref = np.repeat(q[starts], counts)
    X = q - qref                                   # exact whole numbers
    Ve, Vs = _seg_cumsum(v, starts, counts)
    Pe, Ps = _seg_cumsum(X * v, starts, counts)
    P2e, P2s = _seg_cumsum(X * X * v, starts, counts)
    cum = Cum(Ve, Vs, Pe, Ps)

    idx = np.arange(n); safe = np.maximum(anchor, 0)
    W = (Ve[idx + 1] - Vs[safe]).astype(float)
    have = (anchor >= 0) & (W > 0)
    S1 = (Pe[idx + 1] - Ps[safe]).astype(float)
    S2 = (P2e[idx + 1] - P2s[safe]).astype(float)
    m1 = np.where(have, S1 / np.maximum(W, 1e-9), np.nan)
    var = np.where(have, S2 / np.maximum(W, 1e-9) - m1 ** 2, np.nan)
    ref = (qref / PRICE_SCALE).astype(float)
    vwap = m1 / PRICE_SCALE + ref
    sd = np.sqrt(np.maximum(var, 0.0)) / PRICE_SCALE

    # flat / near-flat windows: decide with exact integers, so "sd > 0" means
    # "the prices in this window really did differ"
    near = np.where(have & (var <= 1e-4))[0]
    for i in near:
        j = int(safe[i])
        s1 = int(Pe[i + 1] - Ps[j]); s2 = int(P2e[i + 1] - P2s[j]); w = int(Ve[i + 1] - Vs[j])
        num = s2 * w - s1 * s1                     # python ints: unlimited precision
        sd[i] = 0.0 if num <= 0 else np.sqrt(num / float(w) / float(w)) / PRICE_SCALE
    sd[anchor == idx] = 0.0                        # AUD-37: one-bar line, zero width
    return vwap, sd, W, cum, ref


# ------------------------------------------------- exact volume profile (C97)
def profile_at(vp, v, anchor, sig, atr0, vwap_sig, close_sig):
    """Exact POC, kurtosis and value-area width, by histogram per window.
    ~18 us per window (measured)."""
    n = len(sig)
    poc = np.empty(n); kurt = np.empty(n); va = np.empty(n)
    for k in range(n):
        s = slice(anchor[k], sig[k] + 1)
        b = np.round(vp[s] / TICK).astype(np.int64)
        lo = b.min()
        h = np.bincount(b - lo, weights=v[s])
        poc[k] = (np.argmax(h) + lo) * TICK
        tot = h.sum()
        if tot <= 0:
            kurt[k] = np.nan; va[k] = np.nan; continue
        p = h / tot
        px = (np.arange(len(h)) + lo) * TICK
        mu = (p * px).sum(); s2 = (p * (px - mu) ** 2).sum()
        kurt[k] = ((p * (px - mu) ** 4).sum() / max(s2 ** 2, 1e-12)) if s2 > 0 else np.nan
        order = np.argsort(h)[::-1]
        c = np.cumsum(h[order])
        kk = int(np.searchsorted(c, 0.70 * tot)) + 1
        sel = px[order[:kk]]
        va[k] = (sel.max() - sel.min())
    return dict(poc_dist=np.abs(poc - vwap_sig) / atr0,
                poc_side=np.sign(poc - vwap_sig),
                # signed distances -- the minimal complete set. d_VC, "price
                # outside both", "POC between price and line" all derive from
                # these two without another pass.
                d_PC=(close_sig - poc) / atr0,
                profile_kurt=kurt,
                value_area_atr=va / atr0)


def path_features(close, seg, sig, atr0, g0, bars):
    """Post-signal path, measured BEFORE any delayed entry -> still causal."""
    n = len(close); out = {}
    step = np.abs(np.r_[0.0, np.diff(close)])
    cstep = np.concatenate([[0.0], np.cumsum(step)])
    for b in bars:
        e = np.minimum(sig + b, n - 1)
        ok = seg[e] == seg[sig]
        net = np.abs(close[e] - close[sig])
        path = cstep[e + 1] - cstep[sig + 1]
        out[f"er_p{b}"] = np.where(ok, net / np.maximum(path, 1e-9), np.nan)
        out[f"gap_p{b}"] = np.where(ok, (close[e] - (close[sig] - g0)) / atr0, np.nan)
    return out


# ---------------------------------------------------------------- HTF trend
def htf_trend(d, minutes, lookback_bars):
    """Net return over `lookback_bars` of `minutes`-bars, in that timeframe's
    ATR units, evaluated from COMPLETED bars only (no look-ahead), then
    broadcast back to every 1m bar."""
    import pandas as _pd
    g = d.set_index("ts").groupby("seg")
    parts = []
    for seg, sub in g:
        r = sub.resample(f"{minutes}min").agg(
            {"high": "max", "low": "min", "close": "last", "volume": "sum"})
        r = r[r.volume > 0].copy()
        pc = r.close.shift(1)
        tr = _np.maximum(r.high - r.low,
                         _np.maximum((r.high - pc).abs(), (r.low - pc).abs()))
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        ret = (r.close - r.close.shift(lookback_bars)) / atr.replace(0, _np.nan)
        out = ret.shift(1)                      # completed bars only
        out.name = "v"
        parts.append(out.to_frame().assign(seg=seg))
    h = _pd.concat(parts).reset_index().sort_values("ts")
    merged = _pd.merge_asof(d[["ts", "seg"]].sort_values("ts"), h,
                            on="ts", by="seg", direction="backward")
    return merged["v"].to_numpy()


import numpy as _np
