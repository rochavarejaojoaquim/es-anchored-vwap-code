r"""a5_3_named_check.py -- THE NAMED CHECK of amendment A5(3) (HANDOFF_section_26.md 26.137-26.139), VERSION 2.
DESCRIPTIVE. NOTHING THIS SCRIPT PRINTS OR WRITES ACTS ON S0, K, THE GATE OR ANY THRESHOLD.

    python a5_3_named_check.py --selftest                      toy fixtures only, reads no dump
    python a5_3_named_check.py --k0-only=3 --R=50              pre-flight: known-number checks, writes nothing
    python a5_3_named_check.py --part=3/8                      one interleaved share of the 317 -> a_nc_<date>_part3of8.csv
    python a5_3_named_check.py --summarize                     merge all shares, margin table, print, write csv+json
    python a5_3_named_check.py                                 all 317 in one process, then --summarize

Version 2 REPLACES version 1 (stale\a5_3_named_check_v1_NCV_before_direct_2026-10-12.py, SHA-256 AFDE44D1...3D13A):
v1's NC-V was a MODEL (day-level product of sums) whose errors are derivable, of mixed sign and invisible to its
self-test (reviewer, 26.139). v2 MEASURES the quantity directly for the winning mask.

WHAT IT MEASURES (outcome-blind: no Stage-2 output, no seal bar, no new search)
  STEP 0  per cell+side, the share of ROWS that cannot move under NULL_UNIT = "day_strata" (days whose
          episode count is unique inside their block), by frozen.frozen_fraction, pooled and per block
          (m = 2: production moves every stratum with >= 2 days, floor_v2.py:142). Known number: the 12
          archetype cell+sides measured 0.0-0.3% in HANDOFF_section_24.md:451-453. The derived centre shift
          of the null, share x obs_score / SD(stored nulls) in SDs, for the 31 and the 286; compared with
          the stored-minus-fresh offset of 26.98 (integer sums 3,606 and 2,894 over the 60 ZERO cells).
  DIRECT  per cell+side, for the stored WINNER's mask held FIXED and the setup-minus-twin difference d_i
          (core_v2's edge IS mean over the subset of d_i when every setup has 2 twins), per block, R draws
          of FOUR arms, each moving the mask rows and leaving d where it is:
            perm    production day_strata permutation (floor_v2.shuffle_within_block, the production
                    code): count-CONDITIONAL, whole days, no calendar adjacency
            uperm   unconditional whole-day permutation: no count conditioning, no adjacency
            dshift  whole-day calendar shift: no count conditioning, calendar adjacency KEPT
            roll    bar-order roll (calib_main.shift_covariates, anchor_fracs(j)): ARM 1's dead frame
          (uperm and dshift resample the donor day's mask by relative position, since day counts differ.)
            g (serial) = SD(dshift) / SD(uperm)  the two differ ONLY by calendar adjacency: > 1 means a
                         day-exchangeable null is too NARROW because days have memory
            c_count    = SD(uperm) / SD(perm)    what conditioning on the day's episode count removes (a
                         property of the H0 the production test states, not a defect)
            z_offset   = (mean(perm) - mean(uperm)) / SD(perm)   the covariate-to-count link in the centre
            dilution   = SD(roll) / SD(perm)     arm 1's frame against the production null, same mask
          with a moving-block bootstrap interval for SD(dshift) over the offset-ordered shifts. A self-test
          world shows why the split is needed: a count link alone, with NO memory, gives SD(dshift)/SD(perm)
          far above 1 while g (serial) stays at 1.
          This is the WINNING SUBSET's statistic, not the maximum over ~1,400 candidates (arm 1 is the
          maximum-level instrument and is blind below about a 30% narrowing).
  MARGIN  the installed Romano-Wolf stepdown (loo = True, from rw_check_independent.py by ast) on the 317
          stored 100-null matrix with EVERY null expanded about its mean by a common g (1.00-2.00, step
          0.01): the survivors of the 31 at each g and g* per survivor. A MARGIN DISTRIBUTION (C167), not
          a defect: a threshold-selected list always has its weakest member at the line. g* differences
          below about 0.04 are inside the Monte Carlo resolution of a B = 100 null (adjacent order
          statistics of the maxT distribution differ by 3-4%).

NOTHING STATISTICAL IS REIMPLEMENTED (E24): frozen.frozen_fraction, floor_v2.shuffle_within_block,
calib_main.shift_covariates / anchor_fracs / rw_funcs / fwer_any / stored_floor, core_v2.load_cell / prep /
edge_se / twin_mask, mono_profile.winner_mask / recompute_winner.

KNOWN-NUMBER CHECKS BEFORE ANY RESULT (D3): the self-test (including a simulated world whose true SD of the
statistic across 300 independent worlds is compared with the within-world SDs); per row, the stored winner
reproduced EXACTLY by core_v2.edge_se (one mismatch stops the run, nothing written); mean_S d_i equals
the stored eA to 1e-9 where every setup has 2 twins; floors_v2 is the REAL tree (nperm 100, 100 nulls,
day_strata, joint seed 20260927, VWAP_ROOT unset); at g = 1 the stepdown gives 31 survivors, min adjusted p
0.0099, 17 at the floor (F02); the survivors at g = 1.1 / 1.25 / 1.4 / 1.5 / 1.67 / 2.0 equal 26.134's
independent script's 23 / 13 / 6 / 6 / 3 / 1; the archetype cells' frozen share is under 0.35%; the ZERO-cell
integer sums of k are 3,606 and 2,894 (26.98).
READING RULES (fixed in A5(3) v6 before this runs): no number acts on S0. The paper states the g (serial),
c_count and z_offset distributions for the 31 against the 286 with their intervals, the number of the 31
whose g (serial) interval lies above 1 in both blocks, and the margin table. No verdict word is attached (E2).
WRITES a_nc_<date>.csv / .json (rows carry the stamp, code-set, this script's SHA-256 and the SHA-256 of every
imported root file, C135/C137).
"""
import datetime
import glob
import hashlib
import json
import os
import platform
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "code"))
import core_v2                                                        # noqa: E402
import frozen                                                         # noqa: E402
import mono_profile as MP                                             # noqa: E402
import calib_main as C                                                # noqa: E402

R_DEFAULT = 1000
GRID = np.round(np.arange(1.0, 2.0001, 0.01), 2)
SEED = 20261012
BLOCK_BOOT, NBOOT = 25, 500
ARCHETYPE = ("N60_session_ATR_6.0", "N200_ll_ATR_4.0", "N60_ll_SD_1.0", "N60_hh_ATR_3.0")
KNOWN_MARGIN = {1.0: 31, 1.1: 23, 1.25: 13, 1.4: 6, 1.5: 6, 1.67: 3, 2.0: 1}          # 26.134, independent script
ROOT_FILES = ("frozen.py", "mono_profile.py", "calib_main.py", "calib_main_plan.json",
              "rw_check_independent.py", "rw_check_independent_out.csv", "gate_receipts.py")


# ------------------------------------------------------------------ pieces
def paired_diff(r_hit, c_hit, src_idx, n):
    """d_i = hit_i - mean(hits of the setup's twins); NaN where a setup has no twin."""
    ok = src_idx >= 0
    cnt = np.bincount(src_idx[ok], minlength=n).astype(float)
    s = np.bincount(src_idx[ok], weights=c_hit[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(cnt > 0, s / cnt, np.nan)
    return r_hit - t, cnt


def block_stat(d, ok, mask_new):
    """mean of d over the moved mask inside one block (rows given by `ok`)."""
    mk = mask_new[ok] > 0.5
    return float(d[ok][mk].mean()) if mk.any() else np.nan


def day_layout(day, ix):
    """Rows `ix` of one block grouped by day in calendar order (stable): the pieces a whole-day shift needs."""
    order = ix[np.argsort(day[ix], kind="stable")]
    ds = day[order]
    starts = np.r_[0, np.flatnonzero(ds[1:] != ds[:-1]) + 1]
    cnt = np.diff(np.r_[starts, len(order)])
    didx = np.repeat(np.arange(len(starts)), cnt)
    rel = (np.arange(len(order)) - starts[didx] + 0.5) / cnt[didx]
    return order, starts, cnt, didx, rel


def day_shift(m, lay, s):
    """WHOLE-DAY calendar shift: day t takes the mask of day t+s (circular, in the block's own day
    sequence). Day counts differ, so the donor day's mask is resampled by relative position inside the
    day. Keeps day clusters and calendar adjacency; the mask multiset changes slightly (stated)."""
    order, starts, cnt, didx, rel = lay
    return day_take(m, lay, ((np.arange(len(starts)) + int(s)) % len(starts)))


def day_take(m, lay, donor_of_day):
    """Day t takes the mask of day donor_of_day[t], resampled by relative position inside the day."""
    order, starts, cnt, didx, rel = lay
    donor = np.asarray(donor_of_day)[didx]
    src = starts[donor] + np.minimum((rel * cnt[donor]).astype(np.int64), cnt[donor] - 1)
    return order, m[order][src]


def direct_stats(r2, d, bounds, R, seed_key, mutate=None, floor_v2=None):
    """r2: frame with block, bar, day, m (float mask). Four arms per block, R draws each:
      perm   production day_strata permutation of the mask rows (floor_v2.shuffle_within_block):
             count-CONDITIONAL, whole days, no calendar adjacency
      uperm  UNCONDITIONAL whole-day permutation (day_take with a random day order): no count
             conditioning, no adjacency, the same within-day resampling as dshift
      dshift WHOLE-DAY calendar shift (day_shift): no count conditioning, calendar adjacency KEPT
      roll   bar-order roll (calib_main.shift_covariates), ARM 1's dead-frame construction
    dshift vs uperm differ ONLY by calendar adjacency (serial dependence); uperm vs perm differ ONLY by
    the count conditioning (and the resampling). `mutate` exists only for the self-test."""
    floor_v2 = floor_v2 or C._imports()[1]
    blk = r2.block.to_numpy()
    m = r2.m.to_numpy(float)
    day = np.asarray(floor_v2._as_day_int(r2.day))
    okm = {B: (blk == B) & np.isfinite(d) for B in ("A", "B") if ((blk == B) & np.isfinite(d)).sum() >= 100}
    lay = {B: day_layout(day, np.flatnonzero(blk == B)) for B in okm}
    rng = np.random.default_rng([SEED, zlib.crc32(seed_key.encode())])
    scratch = r2.copy()
    perm = {B: np.empty(R) for B in okm}
    dsh = {B: np.empty(R) for B in okm}
    roll = {B: np.empty(R) for B in okm}
    offs = {B: np.empty(R) for B in okm}
    up = {B: np.empty(R) for B in okm}
    unit_perm = "row" if mutate == "perm_as_row" else "day_strata"
    for j in range(R):
        rp = floor_v2.shuffle_within_block(r2, ["m"], rng, into=scratch, unit=unit_perm, perm_index=None)
        mv = rp["m"].to_numpy()
        for B in okm:
            perm[B][j] = block_stat(d, okm[B], mv)
        mvu = m.copy()
        if mutate == "uperm_as_perm":
            mvu = floor_v2.shuffle_within_block(r2, ["m"], rng, into=scratch, unit="day_strata",
                                                perm_index=None)["m"].to_numpy().copy()
        else:
            for B in okm:
                order, vals = day_take(m, lay[B], rng.permutation(len(lay[B][1])))
                mvu[order] = vals
        for B in okm:
            up[B][j] = block_stat(d, okm[B], mvu)
        fr = C.anchor_fracs(j)
        mv2 = m.copy()
        if mutate == "shift_as_perm":
            mv2 = floor_v2.shuffle_within_block(r2, ["m"], rng, into=scratch, unit="day_strata",
                                                perm_index=None)["m"].to_numpy().copy()
        for B in okm:
            Dn = len(lay[B][1])
            s = min(max(int(round(fr[B] * Dn)), 1), Dn - 1)
            offs[B][j] = s
            if mutate != "shift_as_perm":
                order, vals = day_shift(m, lay[B], s)
                mv2[order] = vals
        for B in okm:
            dsh[B][j] = block_stat(d, okm[B], mv2)
        rd, info = C.shift_covariates(r2, ["m"], fr, bounds, day)
        mv3 = rd["m"].to_numpy()
        for B in okm:
            roll[B][j] = block_stat(d, okm[B], mv3)
    return perm, dsh, roll, offs, up


def sd_ci(x_by_offset, rng):
    x = np.asarray(x_by_offset, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 * BLOCK_BOOT:
        return np.nan, np.nan
    nb = n // BLOCK_BOOT
    st = rng.integers(0, n - BLOCK_BOOT + 1, (NBOOT, nb))
    idx = (st[:, :, None] + np.arange(BLOCK_BOOT)).reshape(NBOOT, -1)
    sds = x[idx].std(axis=1, ddof=1)
    return float(np.percentile(sds, 2.5)), float(np.percentile(sds, 97.5))


BLOCK_KEYS = ("e_obs", "mean_perm", "sd_perm", "mean_uperm", "sd_uperm", "mean_shift", "sd_shift", "g", "g_lo",
              "g_hi", "c_count", "g_total", "z_offset", "sd_roll", "dilution", "z_roll")


def summarize_block(perm, shift, roll, offs, e_obs, rng, uperm):
    """g       = SD(dshift) / SD(uperm): SERIAL DEPENDENCE only (both unconditional, same resampling)
       c_count = SD(uperm)  / SD(perm) : what conditioning on the day's episode count removes
       g_total = SD(dshift) / SD(perm) = g x c_count
       z_offset = (mean(perm) - mean(uperm)) / SD(perm): the covariate-to-count link in the null's CENTRE
       dilution = SD(roll)  / SD(perm) : arm 1's frame against the production null, same mask"""
    pm, ps = float(np.nanmean(perm)), float(np.nanstd(perm, ddof=1))
    um, us = float(np.nanmean(uperm)), float(np.nanstd(uperm, ddof=1))
    sm, ss = float(np.nanmean(shift)), float(np.nanstd(shift, ddof=1))
    rm, rs = float(np.nanmean(roll)), float(np.nanstd(roll, ddof=1))
    order = np.argsort(offs, kind="stable")
    lo, hi = sd_ci(shift[order], rng)
    nz, nu = ps > 0, us > 0
    return dict(e_obs=e_obs, mean_perm=pm, sd_perm=ps, mean_uperm=um, sd_uperm=us, mean_shift=sm, sd_shift=ss,
                g=ss / us if nu else np.nan, g_lo=lo / us if nu else np.nan, g_hi=hi / us if nu else np.nan,
                c_count=us / ps if nz else np.nan, g_total=ss / ps if nz else np.nan,
                z_offset=(pm - um) / ps if nz else np.nan,
                sd_roll=rs, dilution=rs / ps if nz else np.nan, z_roll=(pm - rm) / ps if nz else np.nan)


def widen(NUL, g):
    mu = NUL.mean(1, keepdims=True)
    return mu + np.asarray(g, float)[:, None] * (NUL - mu)


def margin_table(obs, NUL, studentize, romano_wolf, surv):
    first_fail = np.full(len(obs), np.nan)
    n_by_g = {}
    for g in GRID:
        n, mn, adj = C.fwer_any(obs, widen(NUL, np.full(len(obs), g)), studentize, romano_wolf)
        ok = adj <= 0.05
        n_by_g[float(g)] = int((ok & surv).sum())
        fail_now = surv & ~ok & np.isnan(first_fail)
        first_fail[fail_now] = g
    gstar = np.where(surv, np.where(np.isnan(first_fail), GRID[-1], np.round(first_fail - 0.01, 2)), np.nan)
    return gstar, n_by_g


# ------------------------------------------------------------------ self-test
def _ar1(rng, n, phi):
    from scipy.signal import lfilter
    e = rng.standard_normal(n + 200)
    return lfilter([1.0], [1.0, -phi], e)[200:]


def toy_world(rng, D=200, phi=0.6, count_link=0.0, nlo=5, nhi=13):
    """Weekday-gapped days, unequal counts, a day factor in the covariate x AND in d (independent
    processes, so H0 holds). count_link > 0 makes x and d depend on the day's count (a count-link)."""
    days_all = np.array([k for k in range(D * 7 // 5 + 14) if k % 7 not in (5, 6)])[:D]
    a, b = _ar1(rng, D, phi), _ar1(rng, D, phi)
    cnt = rng.integers(nlo, nhi + 1, D)
    zc = (cnt - cnt.mean()) / cnt.std()
    di = np.repeat(np.arange(D), cnt)
    x = a[di] + rng.standard_normal(len(di)) + count_link * 1.5 * zc[di]
    d = b[di] + rng.standard_normal(len(di)) + count_link * 1.0 * zc[di]
    r2 = pd.DataFrame(dict(block="A", bar=np.arange(len(di)), day=days_all[di], m=(x > 1.0).astype(float)))
    return r2, d, (int(days_all.min()), int(days_all.max()))


def selftest():
    print("a5_3_named_check.py v2 SELF-TEST -- detectors before any dump is read.\n")
    res = []

    def chk(name, ok, detail=""):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<76} {detail}")

    core_v2_, floor_v2 = C._imports()
    rng = np.random.default_rng(SEED)
    # --- d_i identity and iid world
    r, c = MP.toy(lambda x: 0.0 * x, n=20000, D=300, seed=11)
    P = core_v2.prep(r, c)
    m = r.x.to_numpy() > 0.7
    cm = core_v2.twin_mask(m, P["src_idx"])
    e = core_v2.edge_se(r, c, P, m, cm, "A", nboot=150)
    d, cnt = paired_diff(r.hit.to_numpy(float), c.hit.to_numpy(float), P["src_idx"], len(r))
    selA = (r.block == "A").to_numpy() & m
    chk("K1 mean over the subset of d_i EQUALS core_v2.edge_se's point estimate (2 twins/setup)",
        abs(d[selA].mean() - e[0]) < 1e-12 and (cnt == 2).all(), f"{d[selA].mean():+.6f} vs {e[0]:+.6f}")
    chk("K2 frozen.py's own self-test (Step 0 arithmetic)", _quiet(frozen.selftest))
    NUL = rng.standard_normal((20, 100))
    chk("K3 per-cell widening with a constant g equals calib_main.shrink_null(g)",
        np.allclose(widen(NUL, np.full(20, 1.3)), C.shrink_null(NUL, 1.3)))

    def one_world(r2, d_, bnd, Rn, key, mutate=None):
        perm, dsh, roll, offs, up = direct_stats(r2, d_, {"A": bnd}, Rn, key, mutate=mutate, floor_v2=floor_v2)
        return summarize_block(perm["A"], dsh["A"], roll["A"], offs["A"], np.nan, rng, up["A"])

    def world_run(phi, link, W=400, W_in=24, Rn=150):
        """TRUE SD across independent worlds of the CENTRED statistic (subset mean of d minus the world's
        overall mean of d: every null here conditions on d, so its grand mean is not part of their spread),
        against the within-world SDs of the arms."""
        true = np.empty(W)
        for w in range(W):
            r2, d_, _ = toy_world(rng, phi=phi, count_link=link)
            true[w] = d_[r2.m.to_numpy() > 0.5].mean() - d_.mean()
        keys = ("g", "c_count", "g_total", "z_offset", "sd_perm", "sd_uperm", "sd_shift", "dilution")
        acc = {k: [] for k in keys}
        for w in range(W_in):
            r2, d_, bnd = toy_world(rng, phi=phi, count_link=link)
            s = one_world(r2, d_, bnd, Rn, f"toy{phi}_{link}_{w}")
            for k in keys:
                acc[k].append(s[k])
        return float(true.std(ddof=1)), {k: float(np.mean(v)) for k, v in acc.items()}

    t_sd, a = world_run(0.6, 0.0)
    g_true = t_sd / a["sd_uperm"]
    chk("K4 memory world: SD(whole-day shift) tracks the TRUE SD across 400 worlds (within 15%)",
        abs(a["sd_shift"] / t_sd - 1) < 0.15, f"SD_shift {a['sd_shift']:.4f} vs true {t_sd:.4f}")
    chk("K5 ... and a day-exchangeable null is NARROWER than the truth (true g > 1.10)",
        g_true > 1.10, f"true g {g_true:.3f}")
    chk("K6 g (serial) = SD(shift)/SD(uperm) reproduces the simulated true g (within 15%)",
        abs(a["g"] / g_true - 1) < 0.15, f"{a['g']:.3f} vs {g_true:.3f}")
    chk("K6b memory world without a count link: c_count about 1 (within 0.10)", abs(a["c_count"] - 1) < 0.10,
        f"{a['c_count']:.3f}")
    t0_, a0 = world_run(0.0, 0.0)
    chk("K7 no-memory world: g about 1 (within 0.07), c_count about 1 (0.07), SD(perm) tracks the truth (10%)",
        abs(a0["g"] - 1) < 0.07 and abs(a0["c_count"] - 1) < 0.07 and abs(a0["sd_perm"] / t0_ - 1) < 0.10,
        f"g {a0['g']:.3f} c {a0['c_count']:.3f}; SD_perm {a0['sd_perm']:.4f} vs true {t0_:.4f}")
    chk("K8 no-memory, no-link world: |z_offset| small (mean within 0.15 SD)", abs(a0["z_offset"]) < 0.15,
        f"{a0['z_offset']:+.3f}")
    chk("K8b the BAR-ORDER ROLL (arm 1's frame) dilutes day clusters: SD(roll)/SD(perm) < 0.97 with no memory",
        a0["dilution"] < 0.97, f"{a0['dilution']:.3f}")
    t1_, a1 = world_run(0.0, 1.0)
    chk("K9 count-link world, no memory: z_offset clearly POSITIVE (> +0.5 SD) and c_count > 1.3",
        a1["z_offset"] > 0.5 and a1["c_count"] > 1.3, f"z {a1['z_offset']:+.3f} c {a1['c_count']:.3f}")
    chk("K9b ... and the count link is NOT read as serial dependence: g about 1 (within 0.10), g_total > 1.3",
        abs(a1["g"] - 1) < 0.10 and a1["g_total"] > 1.3, f"g {a1['g']:.3f} g_total {a1['g_total']:.3f}")
    # --- mutations: each must fail to reproduce a known number
    r2, d_, bnd = toy_world(rng, phi=0.6)
    gs = [one_world(*toy_world(rng, phi=0.6), 150, f"mut1_{w}", mutate="shift_as_perm")["g"] for w in range(12)]
    chk("M1 shift arm replaced by a permutation: g about 1, K6's number is NOT reproduced",
        abs(float(np.mean(gs)) - g_true) > 0.10, f"g {np.mean(gs):.3f} vs {g_true:.3f}")
    cs = [one_world(*toy_world(rng, phi=0.0, count_link=1.0), 100, f"mut2_{w}", mutate="uperm_as_perm")
          for w in range(12)]
    chk("M2 unconditional arm replaced by the production permutation: c_count about 1 and z_offset about 0, "
        "K9 NOT reproduced", abs(float(np.mean([x["c_count"] for x in cs])) - 1) < 0.15
        and abs(float(np.mean([x["z_offset"] for x in cs]))) < 0.3,
        f"c {np.mean([x['c_count'] for x in cs]):.3f} z {np.mean([x['z_offset'] for x in cs]):+.3f}")
    cr = [one_world(*toy_world(rng, phi=0.0), 100, f"mut3_{w}", mutate="perm_as_row")["c_count"] for w in range(12)]
    chk("M3 production null replaced by a ROW shuffle: c_count jumps (rows destroy the day factor), K7 NOT reproduced",
        float(np.mean(cr)) > 1.3, f"c {np.mean(cr):.3f}")
    ok = all(res)
    print(f"\n  {'ALL DETECTORS ALIVE' if ok else 'SELF-TEST FAILED'}  ({sum(res)}/{len(res)})")
    return ok


def _quiet(fn):
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return bool(fn())


# ------------------------------------------------------------------ share and summary
def provenance():
    out = {f: MP.sha256(os.path.join(HERE, f)) for f in ROOT_FILES if os.path.exists(os.path.join(HERE, f))}
    out["a5_3_named_check.py"] = MP.sha256(os.path.abspath(__file__))
    return out


def preconditions():
    core_v2_, floor_v2 = C._imports()
    ok = (os.path.normpath(floor_v2.FLOORS) == os.path.normpath(os.path.join(HERE, "floors_v2"))
          and "VWAP_ROOT" not in os.environ)
    if not ok:
        raise SystemExit("  [FAIL] floors_v2 is not the real tree, or VWAP_ROOT is set. Stage-2 files could be read.")
    return floor_v2


def run_share(opt):
    floor_v2 = preconditions()
    R_ = int(opt.get("--R", R_DEFAULT))
    k0_only = int(opt["--k0-only"]) if "--k0-only" in opt else None
    part, nparts = 1, 1
    if "--part" in opt:
        a, b = opt["--part"].split("/")
        part, nparts = int(a), int(b)
    rwf = opt.get("--rw", os.path.join(HERE, "rw_check_independent_out.csv"))
    R = pd.read_csv(rwf)
    n31, n195 = int((R.rw_adj_p <= 0.05).sum()), int((R.p <= 0.05).sum())
    print(f"\nLIST CHECK (integer counts): rows {len(R)} (317), rw_adj_p<=0.05 {n31} (31), p<=0.05 {n195} (195)")
    if len(R) != 317 or n31 != 31 or n195 != 195:
        raise SystemExit("  LIST CHECK FAILED -- the input is not Stage 1's registered table.")
    R = R.sort_values(["cell", "side"], na_position="first").reset_index(drop=True)
    R = R.iloc[[i for i in range(len(R)) if i % nparts == part - 1]]
    if k0_only:
        R = R.head(k0_only)
    bounds = {B: tuple(v) for B, v in C.load_plan()["bounds"].items()}
    cs, prov = MP.code_set(), provenance()
    stamp = (f"a5_3_named_check v2 | share {part}/{nparts} of the 317 effective cell+sides (list fixed by Stage 1) | "
             f"Step 0 = frozen.frozen_fraction | DIRECT = fixed winner mask, R {R_} draws per arm: production day_strata "
             f"permutation (floor_v2.shuffle_within_block), unconditional whole-day permutation, whole-day calendar "
             f"shift, bar-order roll (calib_main.shift_covariates, anchor_fracs j < R); statistic mean d_i over the "
             f"moved subset per block | margin = installed "
             f"Romano-Wolf loo=True on the stored 100-null matrix | code-set {cs} | "
             f"a5_3_named_check.py {prov['a5_3_named_check.py'][:16]} | {platform.node()} | ES train A+B, sealed "
             f"2018-2022 untouched, no Stage-2 output read | DESCRIPTIVE, acts on nothing")
    print("\nSTAMP: " + stamp + "\n\nKNOWN-NUMBER CHECK FIRST on every row; one mismatch stops the run.\n")
    t0, rows = time.time(), []
    for _, row in R.iterrows():
        cell, side = str(row.cell), MP.side_of(row.side)
        f = os.path.join(core_v2.OUT, f"{cell}_side{side}.parquet")
        st = pd.read_parquet(f).iloc[0]
        if str(st["col"]) != str(row.col) or not MP.agree(st, row, rel=1e-12):
            print(f"  [FAIL] {cell} side {side}: rw file and cores file disagree")
            return None
        sfl = C.stored_floor(cell, C.side_str(row.side))
        if not (sfl["nperm"] == 100 and len(sfl["null"]) == 100 and sfl["null_unit"] == "day_strata"
                and sfl["joint"] and sfl["joint_seed"] == floor_v2.JOINT_SEED):
            print(f"  [FAIL] {cell} side {side}: stored floor is not the Stage-1 100-null day_strata file")
            return None
        r, c, T = core_v2.load_cell(cell, side)
        P = core_v2.prep(r, c)
        got = MP.recompute_winner(r, c, P, st)
        if got is None or not MP.agree(st, got):
            print(f"  [FAIL] K0 {cell} side {side}: stored winner not reproduced. Nothing written.")
            return None
        m = MP.winner_mask(r, st)
        d, cnt = paired_diff(r.hit.to_numpy(float), c.hit.to_numpy(float), P["src_idx"], len(r))
        blk = r.block.to_numpy()
        all2 = bool((cnt == 2).all())
        selA = (blk == "A") & m & np.isfinite(d)
        ident = float(abs(d[selA].mean() - float(st["eA"]))) if selA.any() else np.nan
        if all2 and not (ident < 1e-9):
            print(f"  [FAIL] identity {cell} side {side}: {ident:.3e}. Nothing written.")
            return None
        day = r.day.to_numpy()
        fp, _ = frozen.frozen_fraction(day, blk)
        fA, _ = frozen.frozen_fraction(day[blk == "A"], blk[blk == "A"]) if (blk == "A").any() else (np.nan, 0)
        fB, _ = frozen.frozen_fraction(day[blk == "B"], blk[blk == "B"]) if (blk == "B").any() else (np.nan, 0)
        r2 = pd.DataFrame(dict(block=blk, bar=r.bar.to_numpy(), day=day, m=m.astype(float)))
        perm, dsh, roll, offs, up = direct_stats(r2, d, bounds, R_, f"{cell}_{C.side_str(row.side)}", floor_v2=floor_v2)
        brng = np.random.default_rng([SEED, 7, zlib.crc32(cell.encode())])
        rec = dict(cell=cell, side=C.side_str(row.side), col=str(st["col"]), sense=str(st["sense"]),
                   in31=bool(row.rw_adj_p <= 0.05), rw_adj_p=float(row.rw_adj_p), p=float(row.p),
                   obs_score=float(sfl["obs"]), null_sd=float(np.nanstd(sfl["null"][np.isfinite(sfl["null"])], ddof=1)),
                   frozen_pooled=fp, frozen_A=fA, frozen_B=fB, all_setups_2_twins=all2, identity_err=ident,
                   R=R_)
        for B in ("A", "B"):
            if B in perm:
                e_obs = float(d[(blk == B) & m & np.isfinite(d)].mean())
                s = summarize_block(perm[B], dsh[B], roll[B], offs[B], e_obs, brng, up[B])
            else:
                s = {k: np.nan for k in BLOCK_KEYS}
            for k, v in s.items():
                rec[f"{k}_{B}"] = v
        rec["g_max"] = float(np.nanmax([rec["g_A"], rec["g_B"]]))
        rec["g_lo_min"] = float(np.nanmin([rec["g_lo_A"], rec["g_lo_B"]]))
        rec["z_offset_absmax"] = float(np.nanmax(np.abs([rec["z_offset_A"], rec["z_offset_B"]])))
        rec["dilution_min"] = float(np.nanmin([rec["dilution_A"], rec["dilution_B"]]))
        rec["c_count_max"] = float(np.nanmax([rec["c_count_A"], rec["c_count_B"]]))
        rec["g_total_max"] = float(np.nanmax([rec["g_total_A"], rec["g_total_B"]]))
        rows.append(rec)
        print(f"  [PASS] K0+identity {cell} side {rec['side']}  frozen {100 * fp:.2f}%  g(serial) "
              f"A {rec['g_A']:.3f} B {rec['g_B']:.3f}  c_count A {rec['c_count_A']:.3f} B "
              f"{rec['c_count_B']:.3f}  roll/perm A {rec['dilution_A']:.3f} B {rec['dilution_B']:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    if k0_only:
        print(f"\n  --k0-only: {len(rows)} rows reproduce exactly, {time.time() - t0:.0f}s. Nothing written.")
        return None
    D = pd.DataFrame(rows)
    D["stamp"], D["code_set"], D["producer_sha256"] = stamp, cs, prov["a5_3_named_check.py"]
    D["provenance_json"] = json.dumps(prov, sort_keys=True)
    today = datetime.date.today().isoformat()
    path = opt.get("--out", os.path.join(HERE, f"a_nc_{today}_part{part}of{nparts}.csv"))
    D.to_csv(path, index=False)
    print(f"\n  Wrote {os.path.basename(path)} ({len(D)} rows). {time.time() - t0:.0f}s")
    return path


def summarize(files, opt):
    floor_v2 = preconditions()
    D = pd.concat([pd.read_csv(f, dtype={"side": str}, keep_default_na=False, na_values=[""]) for f in files],
                  ignore_index=True)
    key = D.cell + "|" + D.side
    if len(D) != 317 or key.nunique() != 317:
        print(f"  [FAIL] {len(D)} rows, {key.nunique()} distinct: need exactly 317 once each. Nothing written.")
        return 1
    if D.stamp.nunique() != 1 and D.stamp.str.replace(r"share \d+/\d+", "share", regex=True).nunique() != 1:
        print("  [FAIL] more than one provenance across shares. Nothing written.")
        return 1
    if D.provenance_json.nunique() != 1 or D.R.nunique() != 1:
        print("  [FAIL] shares differ in producer hashes or R. Nothing written.")
        return 1
    D = D.sort_values(["cell", "side"]).reset_index(drop=True)
    studentize, romano_wolf = C.rw_funcs()
    keys = list(zip(D.cell, D.side))
    SF = {k: C.stored_floor(*k) for k in keys}
    NUL = np.vstack([SF[k]["null"] for k in keys])
    obs = np.array([SF[k]["obs"] for k in keys])
    n0, mn0, adj0 = C.fwer_any(obs, NUL, studentize, romano_wolf)
    at_floor = int((np.round(adj0 * 101).astype(int) == 1).sum())
    print(f"\nKNOWN NUMBER: stepdown at g = 1: {n0} survivors (31), min adj p {mn0:.4f} (0.0099), {at_floor} at floor (17)")
    if n0 != 31 or abs(mn0 - 0.0099) > 5e-5 or at_floor != 17:
        print("  [FAIL] the stepdown does not reproduce F02. Nothing written.")
        return 1
    surv = adj0 <= 0.05
    if not np.array_equal(surv, D.in31.to_numpy()):
        print("  [FAIL] survivors differ from the in31 column. Nothing written.")
        return 1
    gstar, n_by_g = margin_table(obs, NUL, studentize, romano_wolf, surv)
    bad = {g: (n_by_g[g], v) for g, v in KNOWN_MARGIN.items() if n_by_g[g] != v}
    print("KNOWN NUMBER: margin survivors vs 26.134's independent script (g: now / then):",
          {g: (n_by_g[g], v) for g, v in KNOWN_MARGIN.items()})
    if bad:
        print(f"  [FAIL] margin counts differ from 26.134: {bad}. Nothing written.")
        return 1
    arche = D[D.cell.isin(ARCHETYPE)]
    amax = float(arche.frozen_pooled.max()) if len(arche) else np.nan
    print(f"KNOWN NUMBER: archetype cell+sides present {len(arche)} (24.11c: 12), max frozen share "
          f"{100 * amax:.3f}% (24.11c: 0.0-0.3%)")
    if len(arche) and not amax < 0.0035:
        print("  [FAIL] frozen share on the archetype cells is not 0.0-0.3%. Nothing written.")
        return 1
    a1 = C.read_parts([os.path.join(HERE, "calib_main_arm1_2026-10-05.csv")])
    a2 = C.read_parts([os.path.join(HERE, "calib_main_arm2_2026-10-05.csv")])
    a1 = a1[(a1.status == "ok") & (a1.j.astype(int) == 0)][["cell", "side", "k"]].rename(columns={"k": "k_stored"})
    a2 = a2[(a2.status == "ok") & (a2.scen == "ZERO")][["cell", "side", "k"]].rename(columns={"k": "k_fresh"})
    Z = a1.merge(a2, on=["cell", "side"])
    ks, kf = int(Z.k_stored.astype(int).sum()), int(Z.k_fresh.astype(int).sum())
    print(f"KNOWN NUMBER: ZERO cells {len(Z)} (60), integer sum k_stored {ks} (3,606), k_fresh {kf} (2,894)")
    if len(Z) != 60 or ks != 3606 or kf != 2894:
        print("  [FAIL] the ZERO-cell sums do not reproduce 26.98. Nothing written.")
        return 1
    D["gstar"] = gstar
    D["shift_sd"] = D.frozen_pooled.to_numpy(float) * D.obs_score.to_numpy(float) / D.null_sd.to_numpy(float)
    a31 = D.in31.to_numpy()

    def q(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return f"min {x.min():.4f}  median {np.median(x):.4f}  p90 {np.percentile(x, 90):.4f}  max {x.max():.4f}"
    off_k = (ks - kf) / 60.0
    print("\nSTEP 0 -- share of rows on days with a unique episode count (cannot move under day_strata)")
    print(f"  the 31  : {q(D.frozen_pooled[a31])}\n  the 286 : {q(D.frozen_pooled[~a31])}")
    print(f"  derived centre shift of the null, share x obs_score / SD (a linear MODEL), in SDs:")
    print(f"      the 31  : {q(D.shift_sd[a31])}\n      the 286 : {q(D.shift_sd[~a31])}")
    print(f"  26.98's stored-minus-fresh offset: {off_k:.1f} k-units on average = about {off_k / 39.9:.2f} SD "
          f"(1 SD is about 39.9 k-units near the median). Derived shift / offset (the 31, median): "
          f"{np.median(D.shift_sd[a31]) / (off_k / 39.9):.3f}")
    print("\nDIRECT -- fixed winner mask, per block, R draws per arm")
    print("  g (serial)  = SD(whole-day calendar shift) / SD(unconditional whole-day permutation): > 1 = a "
          "day-exchangeable null is too narrow because days have memory")
    print("  c_count     = SD(unconditional) / SD(production, count-conditional): what conditioning on the "
          "day's episode count removes (a property of the H0, not a defect)")
    for lab, s in (("the 31 ", a31), ("the 286", ~a31)):
        print(f"  {lab} g (serial), max over blocks: {q(D.g_max[s])}")
        print(f"          g (serial) lower end, min over blocks: {q(D.g_lo_min[s])}")
        print(f"          c_count, max over blocks: {q(D.c_count_max[s])}")
        print(f"          g_total = g x c_count, max over blocks: {q(D.g_total_max[s])}")
        print(f"          |z_offset| (covariate-to-count link in the null's centre, SDs): {q(D.z_offset_absmax[s])}")
        print(f"          SD(bar-order roll) / SD(perm), min over blocks (< 1 = arm 1's frame dilutes day "
              f"clusters): {q(D.dilution_min[s])}")
    inside = D[(D.g_lo_min > 1.0) & a31]
    print(f"  the 31 whose lower end of g (serial) is above 1 in both blocks: {len(inside)}")
    print("\nMARGIN -- survivors of the 31 when EVERY null is expanded about its mean by a common g")
    print("  (the MARGIN distribution of the list against the critical value, C167; not a defect)")
    for g in (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.67, 2.0):
        print(f"  g {g:.2f}: {n_by_g[round(g, 2)]:>2} of 31")
    print(f"  g* of the 31 (largest g survived):  min {np.nanmin(gstar):.2f}  median {np.nanmedian(gstar):.2f}  "
          f"max {np.nanmax(gstar):.2f}   (differences under about 0.04 are inside a B = 100 null's resolution)")
    t31 = D[a31].sort_values("gstar")
    for _, x in t31.head(8).iterrows():
        print(f"    thinnest margins: {x.cell} side {x.side}  g* {x.gstar:.2f}  adj p {x.rw_adj_p:.4f} "
              f"({round(x.rw_adj_p * 101)}/101)  g(serial) {x.g_max:.3f}  c_count {x.c_count_max:.3f}")
    print("\nNO NUMBER ABOVE ACTS ON S0. A5(3) v6 fixes the reading before this was run.")
    prov = json.loads(D.provenance_json.iloc[0])
    today = datetime.date.today().isoformat()
    path = opt.get("--out", os.path.join(HERE, f"a_nc_{today}.csv"))
    D.to_csv(path, index=False)
    summ = dict(stamp=D.stamp.iloc[0], code_set=D.code_set.iloc[0], producers=prov, date=today,
                machine=platform.node(), R=int(D.R.iloc[0]),
                margin_survivors={str(k): v for k, v in n_by_g.items()},
                g_serial_31=dict(median=float(np.median(D.g_max[a31])), max=float(D.g_max[a31].max())),
                c_count_31=dict(median=float(np.median(D.c_count_max[a31])), max=float(D.c_count_max[a31].max())),
                g_serial_286=dict(median=float(np.median(D.g_max[~a31])), max=float(D.g_max[~a31].max())),
                n_31_lower_end_above_1=int(len(inside)),
                csv=os.path.basename(path), csv_sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                shares=[os.path.basename(f) + ":" + MP.sha256(f) for f in files])
    jp = os.path.splitext(path)[0] + ".json"
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1)
    print(f"\n  Wrote {os.path.basename(path)} (317 rows) and {os.path.basename(jp)}.")
    return 0


def main():
    if "--selftest" in sys.argv:
        return 0 if selftest() else 1
    opt = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
    if "--summarize" not in sys.argv:
        if not selftest():
            raise SystemExit("\n  SELF-TEST FAILED -- no data was read.")
    if "--summarize" in sys.argv:
        today = datetime.date.today().isoformat()
        files = sorted(glob.glob(opt.get("--glob", os.path.join(HERE, f"a_nc_{today}_part*of*.csv"))))
        if not files:
            raise SystemExit("  no share files found (use --glob=...)")
        return summarize(files, opt)
    path = run_share(opt)
    if path is None:
        return 0 if "--k0-only" in opt else 1
    if "--part" not in opt:
        return summarize([path], opt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
