"""
interactions.py -- the covariate table.

For each cell, each covariate, each training block, fit

    hit ~ is_real + z + is_real:z          z = rank/normal-scores transform

over setups AND their twins. The is_real:z coefficient is the CONDITIONAL
EDGE: how much the setup's advantage over its twin changes per SD of z. It
subtracts the twin automatically, so a covariate that merely makes price
wander more cannot masquerade as an edge.

Reported against BOTH control schemes (M1 hour x vol, M2 + trend quintile).
Day-clustered bootstrap. Per rule C106 the detection floor is 2-4 pp/SD, so a
null here means "no modulation larger than ~3 pp/SD", NOT "no relationship" --
that sentence is printed with the table, not left to the reader.

A covariate counts only if it clears p<0.05 in BOTH blocks with the SAME sign
and agrees across M1/M2.
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
import safeio
from loader import trade_date
import pandas as pd
from scipy.stats import norm

TAU = os.environ.get("VWAP_TAU", "0.0")   # PRIMARY 0.0; set VWAP_TAU=0.25 for the sensitivity run
DUMP = DUMP_DIR
OUT = ROOT_DIR + ("inter0" if TAU == "0.0" else f"inter_tau{TAU}")   # Decision B
os.makedirs(OUT, exist_ok=True)

COVS = ["age", "speed", "W0", "stickiness", "prior_touches", "bars_since_touch",
        "L_all", "L_settled", "slope_15", "slope_60", "sd_in_atr",
        "poc_dist", "d_PC", "profile_kurt", "value_area_atr", "vol_skew",
        "relvol", "rv_close", "atr0", "anchor_spike", "day_pos", "days_to_exp",
        "is_rth", "dow", "ct_hour",   # AUD-17: exchange-local hour
        "trend_15m", "trend_1h",
        # er_p5/15/30 REMOVED 2026-09-21 (AUD-1): built from bars AFTER the
        # signal -> look-ahead here. Entry-time efficiency ratios instead
        # (presignal_er, restored in the rebuild; skipped until present):
        "er_pre15", "er_pre30", "er_pre60", "er_pre120", "er_anchor",
        "dist_atr", "anchor_px_dist",
        "xa_hh_dPV", "xa_ll_dPV", "xa_session_dPV", "xa_hv_dPV",
        "htf_5m_4h_dPV", "htf_15m_4h_dPV", "htf_15m_1d_dPV", "htf_5m_1d_dPV",
        # 2026-09-21 rebuild: R12 linearity, and the "_xs" (x side) versions
        # that let a POOLED table see toward/away effects (AUD-19)
        "lin_maxdev", "lin_meandev", "chord_slope", "lin_maxdev_s",
        "lin_meandev_s", "chord_slope_s", "chord_slope_xs", "chord_slope_s_xs",
        "slope_15_xs", "slope_60_xs", "d_PC_xs", "trend_15m_xs", "trend_1h_xs",
        "xa_hh_dPV_xs", "xa_ll_dPV_xs", "xa_session_dPV_xs", "xa_hv_dPV_xs",
        "htf_5m_4h_dPV_xs", "htf_15m_4h_dPV_xs", "htf_15m_1d_dPV_xs",
        "htf_5m_1d_dPV_xs"]


def ranknorm(z):
    r = pd.Series(z).rank(method="average").to_numpy()
    return norm.ppf((r - 0.375) / (len(r) + 0.25))


def fit_interaction(hit, isr, z, day, B=300, seed=3):
    ok = np.isfinite(z) & np.isfinite(hit)
    if ok.sum() < 400:
        return np.nan, np.nan, 0
    hit, isr, z, day = hit[ok], isr[ok], z[ok], day[ok]
    if np.nanstd(z) == 0:
        return np.nan, np.nan, 0
    z = ranknorm(z)
    X = np.column_stack([np.ones_like(z), isr, z, isr * z])

    def fit(idx):
        try:
            return np.linalg.lstsq(X[idx], hit[idx], rcond=None)[0][3]
        except Exception:
            return np.nan

    beta = fit(np.arange(len(hit)))
    # DAY-CLUSTERED BOOTSTRAP via sufficient statistics: OLS needs only X'X
    # and X'y, and both are additive over days. So accumulate them per day
    # once, then each replicate is a sum of sampled 4x4 matrices instead of a
    # refit over every row. Identical estimator, ~100x faster.
    days, inv = np.unique(day, return_inverse=True)
    D = len(days)
    XtX = np.zeros((D, 4, 4)); Xty = np.zeros((D, 4))
    np.add.at(XtX, inv, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, inv, X * hit[:, None])
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, D, D)
        A = XtX[pick].sum(0); c = Xty[pick].sum(0)
        try:
            draws[b] = np.linalg.solve(A, c)[3]
        except np.linalg.LinAlgError:
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    if len(draws) < 50:
        return float(beta), np.nan, int(len(hit))
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return float(beta), float(p), int(len(hit))


def run_cell(f):
    name = os.path.basename(f).replace(".parquet", "")
    outf = f"{OUT}/{name}.parquet"
    if os.path.exists(outf):
        return None
    x = pd.read_parquet(f)
    T = int(x["T"].iloc[0])
    x["hit"] = ((x[f"w_{TAU}"] > 0) & (x[f"w_{TAU}"] <= T)).astype(float)
    x["day"] = trade_date(x.ts)                 # AUD-13: CME trade date
    real = x[x.is_real == 1]
    if len(real) < 400:
        safeio.write_parquet(pd.DataFrame(), outf); return 0
    # M1b = block-matched twins (C110), read from INSIDE the cell (locally they
    # live there). This is the recipe that produced inter0/ in the container,
    # ported from <cell>.tw2.parquet. M2 twins are VOID (C111) and not used.
    tw = x[x.is_real == 0]
    tw = tw[tw.src_bar.isin(set(real.bar))].reset_index(drop=True)
    twins = {"M1b": tw}

    rows = []
    for c in COVS:
        if c not in real.columns:
            continue
        zr = real[c].to_numpy(float)
        if not np.isfinite(zr).any():
            continue
        srcmap = dict(zip(real.bar.to_numpy(), zr))
        for scheme, tw in twins.items():
            zt = tw.src_bar.map(srcmap).to_numpy(float)
            for B_ in ("A", "B"):
                mr = (real.block == B_).to_numpy()
                mt = (tw.block == B_).to_numpy()
                if mr.sum() < 300 or mt.sum() < 300:
                    continue
                hit = np.r_[real.hit.to_numpy()[mr], tw.hit.to_numpy()[mt]]
                isr = np.r_[np.ones(mr.sum()), np.zeros(mt.sum())]
                z = np.r_[zr[mr], zt[mt]]
                day = np.r_[real.day.to_numpy()[mr], tw.day.to_numpy()[mt]]
                b, p, n = fit_interaction(hit, isr, z, day)
                rows.append(dict(cell=name, tau=float(TAU), cov=c, scheme=scheme, block=B_,
                                 beta=b, p=p, n=n))
    safeio.write_parquet(pd.DataFrame(rows), outf)
    return len(rows)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "N*"
    from loader import cell_files
    files = cell_files(pat, DUMP)
    for f in files:
        r = run_cell(f)
        if r is not None:
            print(f"  {os.path.basename(f):<34} {r} rows", flush=True)
