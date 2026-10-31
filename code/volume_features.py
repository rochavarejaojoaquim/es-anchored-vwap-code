"""
volume_features.py -- the four volume covariates (R21).

All are causal: computed from bars up to and including the signal bar.

1. vol_skew / poc_dist_proxy
   The THIRD volume-weighted moment, O(1) per bar from cumulative sums,
   turned into a rough POC offset |skew|*sd. This is a PROXY, labelled as
   such (the EXACT histogram POC is pipeline.profile_at -> poc_dist, AUD-18).
   Under a Pearson type III shape the mode sits ~skew*sd/2 from the mean, so
   the scale is ~2x (AUD-33); the column is used by RANK, so only the +-3
   clip on skew is affected.

2. stickiness
   Fraction of accumulated volume that arrived in the most recent quarter of
   the anchor window. Low = the line is slow to move; high = it will slide.

3. prior_touches -- MOVED to add_covariates.line_features (AUD-21): this
   module counted crossings of whichever line was live at each bar.

4. relvol
   Volume divided by the trailing median for the SAME Chicago clock hour
   (AUD-17), so "quiet 03:00" and "03:00 during a shock" separate (C55).

NUMERICS: same segment-centring as the SD, for the same reason -- raw third
moments of 5,000-level prices overflow float64 precision entirely.
"""
import numpy as np
import pandas as pd


def build(d):
    vp = d.vp.to_numpy()
    v = d.volume.to_numpy().astype(float)
    seg = d.seg.to_numpy()
    LD = np.longdouble

    u = np.unique(seg)
    ref = np.array([dict(zip(u, vp[np.searchsorted(seg, u)]))[s] for s in seg])
    x = np.asarray(vp, dtype=float) - ref      # float64 BEFORE x*x and x**3 (AUD-38)

    cV = np.concatenate([[0.0], np.cumsum(v)])
    c1 = np.concatenate([[LD(0)], np.cumsum((x * v).astype(LD))])
    c2 = np.concatenate([[LD(0)], np.cumsum((x * x * v).astype(LD))])
    c3 = np.concatenate([[LD(0)], np.cumsum((x ** 3 * v).astype(LD))])

    # 4. relative volume vs the trailing median for the same CHICAGO hour
    hour = d.ts.dt.tz_convert("America/Chicago").dt.hour.to_numpy()   # AUD-17
    relvol = np.ones(len(d))
    s = pd.Series(v)
    for h in range(24):
        m = hour == h
        if m.sum() < 200:
            continue
        med = s[m].rolling(1200, min_periods=100).median().shift(1)
        relvol[m] = (s[m] / med.replace(0, np.nan)).to_numpy()
    relvol = np.nan_to_num(relvol, nan=1.0, posinf=1.0)

    return dict(cV=cV, c1=c1, c2=c2, c3=c3, ref=ref, relvol=relvol)


def at_signals(F, sig, anc, atr0, vwap_sig, sd_sig):
    """Evaluate the features at the signal bars only."""
    cV, c1, c2, c3 = F["cV"], F["c1"], F["c2"], F["c3"]
    W = cV[sig + 1] - cV[anc]
    m1 = (c1[sig + 1] - c1[anc]) / np.maximum(W, 1e-9)
    m2 = (c2[sig + 1] - c2[anc]) / np.maximum(W, 1e-9)
    m3 = (c3[sig + 1] - c3[anc]) / np.maximum(W, 1e-9)
    var = np.maximum((m2 - m1 ** 2).astype(float), 1e-12)
    sd = np.sqrt(var)
    skew = ((m3 - 3 * m1 * m2 + 2 * m1 ** 3).astype(float)) / sd ** 3

    # Pearson: mode ~= mean - skew*sd  ->  distance from VWAP to the POC proxy
    poc_off = np.clip(skew, -3, 3) * sd
    poc_dist = np.abs(poc_off) / atr0

    # 2. stickiness: share of volume arriving in the last quarter of the window
    age = sig - anc
    q = anc + (3 * age) // 4
    W_last = cV[sig + 1] - cV[q]
    stickiness = W_last / np.maximum(W, 1e-9)

    return dict(vol_skew=np.clip(skew, -8, 8), poc_dist_proxy=poc_dist,   # AUD-18
                stickiness=stickiness,
                relvol=F["relvol"][sig])
