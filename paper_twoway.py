r"""
paper_twoway.py -- TRAINING PERIODS ONLY. Two descriptive robustness analyses the outside referee asked for
(HANDOFF_section_26.md 26.214), run on the dump that claim_checks.py read. The held-out period is NOT touched.

    python paper_twoway.py --selftest         toy data only, reads no dump
    python paper_twoway.py --preflight DUMP   two cells; my aggregation must equal claim_checks.py's exactly
    python paper_twoway.py DUMP               the real run (reads only; writes ONE json)

WHAT IT ADDS TO claim_checks.py
  1. OWN-DATE INTERVALS (referee concern 1). claim_checks.py clusters on the SETUP's trade date and carries each
     setup's twins along with it. A twin's outcome, however, belongs to the price path on the twin's OWN date, and
     many setups share twin dates, so twin-date shocks are shared across the setup clusters the bootstrap treats as
     independent. Here every row, setup or twin, is attached to its OWN trade date and dates are resampled
     (iid, and 5-day moving blocks via core_v2.boot_index) with all rows of a resampled date. The point estimate is
     identical to the published one (ratio of pooled means); only the interval changes. Twin sampling noise given
     the dates is not resampled (stated limit).
  2. THE 117 CERTIFIED CELLS (referee concern 2). The registered cells with lookback 60, 100, 160 or 200 (hh/ll/hv x 4 N x 9 eps
     plus session at N=60 x 9 = 117), whose inputs the current code reproduces bit-exactly (paper Section 2.8).
  3. SESSION SPLIT (referee concern 5), DESCRIPTIVE. 'day session' = Chicago clock hours 8-15 (two 4-hour matching
     buckets, 08:00-15:59, an approximation to regular trading hours, which run 08:30-15:00); 'overnight' = the other
     four buckets. Each setup's twins go with it (twins are matched on the 4-hour bucket, so they lie in the same group).

READING RULES, WRITTEN BEFORE ANY REAL RUN (conventions, not derived)
  R1  For each of the 6 headline cells of the paper's Section 3.1 (claims 1 M1b, 1 M2 and 2, in the older and the recent
      period), the claim 'holds under own-date intervals' iff the lower interval end is above zero under BOTH the iid
      and the 5-day-block own-date bootstraps. If any of the six fails, the paper's abstract, Section 3.1 and Section 6.1 are
      changed to say that claim does not hold under own-date intervals in that period; nothing is softened afterwards.
  R2  The ratio of own-date to setup-date half-widths is printed for every cell and is never thresholded.
  R3  The 117-cell estimates are DESCRIPTIVE. Any estimate differing from its 144-cell value by more than 0.5 pp is flagged
      and described in the paper; no rule is changed.
  R4  The session split is DESCRIPTIVE and carries no verdict. A group whose lower end is not above zero is reported as such.
  R5  The held-out period is not run in any of these forms; its intervals are described as carrying the same unresolved
      issue (paper Section 6.5).

KNOWN NUMBERS, checked before any new quantity is printed: counts (4,008,241 setups; 4,008,224 with an M2 twin; 3,260,501 resolved);
the published setup-date estimates and intervals of claim_checks_2026-10-03.json (to 0.0006 pp for estimates, 0.002 pp for interval ends).

Project root; rotates nothing. Writes paper_twoway_<date>.json with its stamp and its own SHA-256 (brain/13 C137).
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claim_checks as CC                                              # noqa: E402
from power_registered_set import registered_cells                      # noqa: E402

MAIN = CC.MAIN_COLS + ["ct_hour"]
M2COLS = ["src_bar", "day", "block", "ct_hour", "w_0.0", "mir_w", "race"]
DAY_HOURS = list(range(8, 16))
SCHEMES = ("M1b", "M2")
PUBLISHED = {   # claim_checks_2026-10-03.json / paper Section 3.1, setup-date iid
    ("c1_M1b", "A"): (4.376, 4.100, 4.659), ("c1_M1b", "B"): (1.310, 0.949, 1.678),
    ("c1_M2", "A"): (4.178, 3.910, 4.463), ("c1_M2", "B"): (1.368, 1.010, 1.729),
    ("c2_M1b", "A"): (4.857, 4.522, 5.214), ("c2_M1b", "B"): (0.780, 0.387, 1.161)}


def day_int(a):
    a = np.asarray(a)
    if a.dtype == object:
        if len(a) and isinstance(a[0], (int, np.integer)):
            return a.astype(np.int64)
        return pd.to_datetime(a).values.astype("datetime64[D]").astype(np.int64)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[D]").astype(np.int64)
    return a.astype(np.int64)


# ------------------------------------------------------------------ aggregation
def frames_from_arrays(S, T1, T2):
    """S: setups (bar, day, block, grp, rf, r_res, r_vf). T1, T2: twin rows (src_bar, day, block, f, res, vf; grp is taken
    from the source setup). Returns date-level sums. role S_<scheme> = setups with >= 1 twin of that scheme;
    role T_<scheme> = twin rows, weight u = 1/k (k = twins of that setup), so a setup's twins average to one;
    axis 'both' for setups, 'own' (twin's own date) and 'setup' (the source setup's date) for twins."""
    out = []
    bar = S["bar"].to_numpy()
    grp_of = pd.Series(S["grp"].to_numpy(), index=bar)
    day_of = pd.Series(S["day"].to_numpy(), index=bar)
    for scheme, tw in (("M1b", T1), ("M2", T2)):
        src = tw["src_bar"].to_numpy()
        has = np.isin(bar, np.unique(src))
        s = S[has]
        g = (s.groupby(["block", "grp", "day"], sort=False)
              .agg(n=("rf", "size"), f=("rf", "sum"), res=("r_res", "sum"), vf=("r_vf", "sum")).reset_index())
        g["role"] = "S_" + scheme; g["axis"] = "both"; out.append(g)
        k = tw.groupby("src_bar")["src_bar"].transform("size").to_numpy(float)
        u = 1.0 / k
        d = pd.DataFrame(dict(block=tw["block"].to_numpy(), grp=grp_of.reindex(src).to_numpy(),
                              day=tw["day"].to_numpy(), sday=day_of.reindex(src).to_numpy(),
                              u=u, uf=u * tw["f"].to_numpy(float), res=tw["res"].to_numpy(float),
                              vf=tw["vf"].to_numpy(float)))
        for axis, col in (("own", "day"), ("setup", "sday")):
            g = (d.groupby(["block", "grp", col], sort=False)
                  .agg(n=("u", "sum"), f=("uf", "sum"), res=("res", "sum"), vf=("vf", "sum")).reset_index()
                  .rename(columns={col: "day"}))
            g["role"] = "T_" + scheme; g["axis"] = axis; out.append(g)
    return pd.concat(out, ignore_index=True)


def cell_frames(dump, c):
    N, anchor, trig, eps = CC.CELL_RE.search(c).groups()
    x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=MAIN)
    T = int(x["T"].iloc[0])
    r = x[x.is_real == 1]
    S = pd.DataFrame(dict(bar=r.bar.to_numpy(), day=day_int(r.day), block=r.block.to_numpy(),
                          grp=np.isin(r.ct_hour.to_numpy(), DAY_HOURS).astype(np.int8),
                          rf=CC.fill(r["w_0.0"], T), r_res=np.isin(r.race, (1, -1)).astype(float),
                          r_vf=(r.race.to_numpy() == 1).astype(float)))
    tws = []
    for tw in (x[x.is_real == 0], pd.read_parquet(os.path.join(dump, c + ".m2.parquet"), columns=M2COLS)):
        tws.append(pd.DataFrame(dict(src_bar=tw.src_bar.to_numpy(), day=day_int(tw.day), block=tw.block.to_numpy(),
                                     f=CC.fill(tw["w_0.0"], T), res=np.isin(tw.race, (1, -1)).astype(float),
                                     vf=(tw.race.to_numpy() == 1).astype(float))))
    g = frames_from_arrays(S, tws[0], tws[1])
    g["N"] = int(N)
    return g, len(S), int((S.r_res == 1).sum()), int(np.isin(S.bar, np.unique(tws[1].src_bar)).sum())


# ------------------------------------------------------------------ estimation
def estimate(agg, scheme, claim, block, grp, axis, kind, cells117=False, S_draws=None):
    """claim 'c1' (touch) or 'c2' (race). axis 'setup' = the published clustering (setup date); 'own' = own date.
    grp 'all' | 0 | 1. kind 'iid' | 'block'."""
    a = agg
    if cells117:
        a = a[a.N != 240]
    a = a[a.block == block]
    if grp != "all":
        a = a[a.grp == grp]
    s = a[a.role == "S_" + scheme]
    t = a[(a.role == "T_" + scheme) & (a.axis == axis)]
    days = np.unique(np.concatenate([s.day.to_numpy(), t.day.to_numpy()]))
    D = len(days)
    cs = np.searchsorted(days, s.day.to_numpy()); ct = np.searchsorted(days, t.day.to_numpy())
    cols = ("f", "n") if claim == "c1" else ("vf", "res")
    sums = {}
    for nm, df, code in (("s", s, cs), ("t", t, ct)):
        sums[nm + "_num"] = np.bincount(code, weights=df[cols[0]].to_numpy(float), minlength=D)
        sums[nm + "_den"] = np.bincount(code, weights=df["n" if claim == "c1" else "res"].to_numpy(float), minlength=D)
    est = CC.ratio_diff(sums, "s_num", "s_den", "t_num", "t_den")
    lo, hi, nd = CC.ci(CC.ratio_diff(sums, "s_num", "s_den", "t_num", "t_den", CC.boot_draws(D, kind)))
    return dict(est_pp=100 * est, lo_pp=100 * lo, hi_pp=100 * hi, D=D, draws=nd,
                n_setups=float(sums["s_den"].sum()) if claim == "c1" else None)


def full_table(agg, cells117, groups=("all", 1, 0)):
    rows = []
    for scheme, claim in (("M1b", "c1"), ("M2", "c1"), ("M1b", "c2")):
        for block in ("A", "B"):
            for grp in groups:
                r = dict(claim=f"{claim}_{scheme}", block=block, grp=grp, cells=117 if cells117 else 144)
                for lab, axis, kind in (("setup_iid", "setup", "iid"), ("setup_blk", "setup", "block"),
                                        ("own_iid", "own", "iid"), ("own_blk", "own", "block")):
                    e = estimate(agg, scheme, claim, block, grp, axis, kind, cells117)
                    r[lab] = (round(e["est_pp"], 4), round(e["lo_pp"], 4), round(e["hi_pp"], 4))
                r["halfwidth_ratio_own_iid"] = round((r["own_iid"][2] - r["own_iid"][1]) /
                                                     (r["setup_iid"][2] - r["setup_iid"][1]), 3)
                r["halfwidth_ratio_own_blk"] = round((r["own_blk"][2] - r["own_blk"][1]) /
                                                     (r["setup_iid"][2] - r["setup_iid"][1]), 3)
                rows.append(r)
    return rows


# ------------------------------------------------------------------ self-test
def _toy_rows(rng, D=500, lam=40, shock_sd=0.05, gap=0.03, p0=0.45, scheme_k=2, disjoint=False):
    """One block 'A'. Setups per date ~ Poisson(lam); each setup has 2 twins on random other dates; outcome probability
    = p0 + date shock (+ gap for setups). Race columns: res / vf drawn the same way (setup vf prob .55+shock)."""
    shock = rng.normal(0, shock_sd, D)
    n_d = rng.poisson(lam, D)
    if disjoint:                      # setups only on the first half of the dates
        n_d[D // 2:] = 0
    day = np.repeat(np.arange(D), n_d)
    n = len(day)
    S = pd.DataFrame(dict(bar=np.arange(n), day=day, block="A", grp=(np.arange(n) % 2).astype(np.int8)))
    S["rf"] = (rng.random(n) < np.clip(p0 + gap + shock[day], 0.01, 0.99)).astype(float)
    S["r_res"] = (rng.random(n) < 0.8).astype(float)
    S["r_vf"] = S.r_res * (rng.random(n) < np.clip(0.55 + shock[day], 0.01, 0.99)).astype(float)
    tsrc = np.repeat(np.arange(n), scheme_k)
    tday = (day[tsrc] + rng.integers(1, D, len(tsrc))) % D
    if disjoint:                      # twins only on the second half
        tday = D // 2 + rng.integers(0, D - D // 2, len(tsrc))
    T = pd.DataFrame(dict(src_bar=tsrc, day=tday, block="A"))
    T["f"] = (rng.random(len(T)) < np.clip(p0 + shock[tday], 0.01, 0.99)).astype(float)
    T["res"] = (rng.random(len(T)) < 0.8).astype(float)
    T["vf"] = T.res * (rng.random(len(T)) < np.clip(0.5 + shock[tday], 0.01, 0.99)).astype(float)
    return S, T, T.copy()


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\npaper_twoway.py SELF-TEST -- toy data only; detectors before any dump is read.\n")
    rng = np.random.default_rng(20261019)

    def world(shock_sd, seed, disjoint=False):
        r = np.random.default_rng(seed)
        S, T1, T2 = _toy_rows(r, shock_sd=shock_sd, disjoint=disjoint)
        return S, T1, T2

    # K1 point estimate identical across clusterings and equal to the direct pooled difference
    S, T1, T2 = world(0.15, 1)
    agg = frames_from_arrays(S, T1, T2)
    e_set = estimate(agg, "M1b", "c1", "A", "all", "setup", "iid")
    e_own = estimate(agg, "M1b", "c1", "A", "all", "own", "iid")
    direct = 100 * (S.rf.mean() - T1.f.mean())
    chk("K1 own-date and setup-date point estimates equal the direct pooled difference",
        abs(e_set["est_pp"] - direct) < 1e-9 and abs(e_own["est_pp"] - direct) < 1e-9,
        f"direct {direct:+.4f}, setup {e_set['est_pp']:+.4f}, own {e_own['est_pp']:+.4f}")
    e2 = estimate(agg, "M1b", "c2", "A", "all", "own", "iid")
    direct2 = 100 * (S.r_vf.sum() / S.r_res.sum() - T1.vf.sum() / T1.res.sum())
    chk("K2 the race estimate equals the pooled resolved-race difference", abs(e2["est_pp"] - direct2) < 1e-9,
        f"direct {direct2:+.4f}, own {e2['est_pp']:+.4f}")
    # K3 groups partition the rows
    n_all = agg[(agg.role == "S_M1b")].n.sum()
    n_g = sum(agg[(agg.role == "S_M1b") & (agg.grp == g)].n.sum() for g in (0, 1))
    chk("K3 the two session groups partition the setups", abs(n_all - len(S)) < 1e-9 and abs(n_g - len(S)) < 1e-9,
        f"{n_g:.0f} of {len(S)}")

    # K4 / K5 / K6 coverage: true SD over worlds against the two bootstrap SEs
    def study(shock_sd, nw_true=300, nw_boot=16, disjoint=False):
        ests = []
        for w in range(nw_true):
            S, T1, T2 = world(shock_sd, 1000 + w, disjoint)
            ests.append(S.rf.mean() - T1.f.mean())
        true_sd = 100 * float(np.std(ests, ddof=1))
        hs, ho = [], []
        for w in range(nw_boot):
            S, T1, T2 = world(shock_sd, 5000 + w, disjoint)
            agg = frames_from_arrays(S, T1, T2)
            a = estimate(agg, "M1b", "c1", "A", "all", "setup", "iid")
            b = estimate(agg, "M1b", "c1", "A", "all", "own", "iid")
            hs.append((a["hi_pp"] - a["lo_pp"]) / 3.92); ho.append((b["hi_pp"] - b["lo_pp"]) / 3.92)
        return true_sd, float(np.mean(hs)), float(np.mean(ho))

    t1, s1, o1 = study(0.15)
    chk("K4a twin dates spread like setup dates (shocks cancel between the arms): the SETUP-date bootstrap OVERSTATES the true SD",
        s1 > 1.3 * t1, f"true SD {t1:.3f} pp, setup-date SE {s1:.3f} ({s1 / t1:.2f}x), own-date SE {o1:.3f} ({o1 / t1:.2f}x)")
    chk("K5a ... and the OWN-date bootstrap is within 20% of the true SD", abs(o1 / t1 - 1) < 0.20, f"{o1 / t1:.2f}x")
    t2, s2, o2 = study(0.15, disjoint=True)
    chk("K4b twin dates disjoint from setup dates (shocks independent): the SETUP-date bootstrap UNDERSTATES the true SD",
        s2 < 0.85 * t2, f"true SD {t2:.3f} pp, setup-date SE {s2:.3f} ({s2 / t2:.2f}x), own-date SE {o2:.3f} ({o2 / t2:.2f}x)")
    chk("K5b ... and the OWN-date bootstrap is within 20% of the true SD", abs(o2 / t2 - 1) < 0.20, f"{o2 / t2:.2f}x")
    t0, s0, o0 = study(0.0)
    chk("K6 with no date shocks the two clusterings agree with each other and with the true SD",
        abs(s0 / t0 - 1) < 0.20 and abs(o0 / t0 - 1) < 0.20 and abs(o0 / s0 - 1) < 0.15,
        f"true {t0:.3f}, setup-date {s0:.3f}, own-date {o0:.3f}")
    # M1 mutation: attach twins to the setup's date and call it 'own' -> must equal the setup-date SE
    S, T1, T2 = world(0.15, 7)
    agg = frames_from_arrays(S, T1, T2)
    agg_m = agg.copy()
    agg_m.loc[agg_m.axis == "own", "axis"] = "dropme"
    agg_m.loc[agg_m.axis == "setup", "axis"] = "own"
    m = estimate(agg_m, "M1b", "c1", "A", "all", "own", "iid")
    a = estimate(agg, "M1b", "c1", "A", "all", "setup", "iid")
    chk("M1 twins attached to the setup date reproduce the setup-date interval (so the own-date arm is not vacuous)",
        abs(m["lo_pp"] - a["lo_pp"]) < 1e-9 and abs(m["hi_pp"] - a["hi_pp"]) < 1e-9,
        f"[{m['lo_pp']:+.3f},{m['hi_pp']:+.3f}] vs [{a['lo_pp']:+.3f},{a['hi_pp']:+.3f}]")
    # K7 the block variant runs and is finite
    b = estimate(agg, "M1b", "c1", "A", "all", "own", "block")
    chk("K7 the 5-day block own-date interval is finite and contains the estimate",
        np.isfinite(b["lo_pp"]) and b["lo_pp"] < b["est_pp"] < b["hi_pp"], f"[{b['lo_pp']:+.3f},{b['hi_pp']:+.3f}]")
    # K8 a setup with a single M2 twin weighs one (u = 1/k)
    S = pd.DataFrame(dict(bar=[0, 1], day=[0, 1], block="A", grp=np.int8(1), rf=[1.0, 0.0], r_res=[1.0, 1.0], r_vf=[1.0, 0.0]))
    T1 = pd.DataFrame(dict(src_bar=[0, 0, 1, 1], day=[5, 6, 5, 6], block="A", f=[1., 0., 0., 0.], res=1.0, vf=[1., 0., 0., 0.]))
    T2 = pd.DataFrame(dict(src_bar=[0, 1, 1], day=[7, 7, 8], block="A", f=[1., 0., 1.], res=1.0, vf=[1., 0., 1.]))
    agg = frames_from_arrays(S, T1, T2)
    t = agg[(agg.role == "T_M2") & (agg.axis == "own")]
    chk("K8 twin weights are 1/k: a lone twin weighs 1, two twins weigh 0.5 each (total 2 for two setups)",
        abs(t.n.sum() - 2.0) < 1e-12 and abs(t.f.sum() - 1.5) < 1e-12, f"weight {t.n.sum():.3f}, weighted fills {t.f.sum():.3f}")
    print("\n" + ("ALL DETECTORS ALIVE" if ok[0] else "SOME DETECTOR FAILED"))
    return ok[0]


# ------------------------------------------------------------------ real run
def load_all(dump, cells):
    fr, ns, nres, nm2 = [], 0, 0, 0
    t0 = time.time()
    for i, c in enumerate(cells):
        g, n, r, m = cell_frames(dump, c)
        fr.append(g); ns += n; nres += r; nm2 += m
        if i % 24 == 0:
            print(f"  loaded {i + 1}/{len(cells)} cells ({time.time() - t0:.0f}s)", flush=True)
    agg = pd.concat(fr, ignore_index=True)
    agg = (agg.groupby(["N", "role", "axis", "block", "grp", "day"], sort=False)[["n", "f", "res", "vf"]].sum().reset_index())
    return agg, ns, nres, nm2


def preflight(dump):
    """Two cells: my aggregation must reproduce claim_checks.py's setup-date numbers EXACTLY on the same cells."""
    cells = ["N60_hh_ATR_3.0", "N240_ll_ATR_8.0"]
    agg, ns, nres, nm2 = load_all(dump, cells)
    df, bad, rb = CC.load(dump, cells)
    chk_ok = True
    for scheme, claim in (("M1b", "c1"), ("M2", "c1"), ("M1b", "c2")):
        for block in ("A", "B"):
            mine = estimate(agg, scheme, claim, block, "all", "setup", "iid")
            d = CC.arm_frame(df[df.block == block], scheme)
            ref = (CC.claim1_block(d, "iid") if claim == "c1" else CC.claim2_block(d, "iid"))
            same = (abs(mine["est_pp"] - ref["est_pp"]) < 1e-9 and abs(mine["lo_pp"] - ref["lo_pp"]) < 1e-9
                    and abs(mine["hi_pp"] - ref["hi_pp"]) < 1e-9)
            chk_ok &= same
            print(f"  [{'PASS' if same else 'FAIL'}] {claim}_{scheme} {block}: setup-date interval equals claim_checks.py"
                  f"   mine {mine['est_pp']:+.4f} [{mine['lo_pp']:+.4f},{mine['hi_pp']:+.4f}]"
                  f"   ref {ref['est_pp']:+.4f} [{ref['lo_pp']:+.4f},{ref['hi_pp']:+.4f}]")
    print("\n" + ("PREFLIGHT PASS" if chk_ok else "PREFLIGHT FAIL"))
    return chk_ok


def main(dump):
    if not selftest():
        return 1
    cells = registered_cells()
    assert len(cells) == 144
    cells117 = [c for c in cells if int(CC.CELL_RE.search(c).group(1)) != 240]
    assert len(cells117) == 117, len(cells117)
    t0 = time.time()
    agg, ns, nres, nm2 = load_all(dump, cells)
    print(f"\nK0 counts: setups {ns:,} (want 4,008,241); with an M2 twin {nm2:,} (want 4,008,224); resolved {nres:,} (want 3,260,501)")
    k0 = ns == 4008241 and nm2 == 4008224 and nres == 3260501
    tab144 = full_table(agg, False)
    bad = []
    for r in tab144:
        if r["grp"] != "all":
            continue
        pub = PUBLISHED[(r["claim"], r["block"])]
        e, lo, hi = r["setup_iid"]
        okk = abs(e - pub[0]) < 0.0006 and abs(lo - pub[1]) < 0.002 and abs(hi - pub[2]) < 0.002
        print(f"K0 {r['claim']} {r['block']}: setup-date {e:+.3f} [{lo:+.3f},{hi:+.3f}] vs published "
              f"{pub[0]:+.3f} [{pub[1]:+.3f},{pub[2]:+.3f}]  {'PASS' if okk else 'FAIL'}")
        if not okk:
            bad.append((r["claim"], r["block"]))
    if not k0 or bad:
        print("KNOWN-NUMBER CHECK FAILED -- nothing below is to be read.", bad)
        return 1
    tab117 = full_table(agg, True)
    print("\n=== OWN-DATE INTERVALS, 144 cells, all sessions (pp; estimate [95% CI]) ===")
    hold = []
    for r in tab144:
        if r["grp"] != "all":
            continue
        holds = r["own_iid"][1] > 0 and r["own_blk"][1] > 0
        hold.append(holds)
        print(f"{r['claim']:7s} {r['block']}: setup-date iid [{r['setup_iid'][1]:+.3f},{r['setup_iid'][2]:+.3f}]  "
              f"own iid [{r['own_iid'][1]:+.3f},{r['own_iid'][2]:+.3f}] (x{r['halfwidth_ratio_own_iid']:.2f})  "
              f"own 5d-block [{r['own_blk'][1]:+.3f},{r['own_blk'][2]:+.3f}] (x{r['halfwidth_ratio_own_blk']:.2f})  "
              f"-> {'HOLDS' if holds else 'DOES NOT HOLD'} (R1)")
    print("R1 VERDICT:", "all six headline cells hold under own-date intervals" if all(hold)
          else "AT LEAST ONE HEADLINE CELL DOES NOT HOLD -- change the paper per R1")
    print("\n=== 117 CERTIFIED CELLS, all sessions (descriptive, R3) ===")
    for r in tab117:
        if r["grp"] != "all":
            continue
        r144 = [x for x in tab144 if x["claim"] == r["claim"] and x["block"] == r["block"] and x["grp"] == "all"][0]
        diff = r["setup_iid"][0] - r144["setup_iid"][0]
        print(f"{r['claim']:7s} {r['block']}: 117 cells {r['setup_iid'][0]:+.3f} [{r['own_iid'][1]:+.3f},{r['own_iid'][2]:+.3f}] own iid; "
              f"144 cells {r144['setup_iid'][0]:+.3f}; difference {diff:+.3f} pp{'  FLAG >0.5' if abs(diff) > 0.5 else ''}")
    print("\n=== SESSION SPLIT, 144 cells (descriptive, R4): 1 = day session (Chicago hours 8-15), 0 = overnight ===")
    for r in tab144:
        if r["grp"] == "all":
            continue
        print(f"{r['claim']:7s} {r['block']} {'day  ' if r['grp'] == 1 else 'night'}: {r['setup_iid'][0]:+.3f}  "
              f"setup-date iid [{r['setup_iid'][1]:+.3f},{r['setup_iid'][2]:+.3f}]  own iid [{r['own_iid'][1]:+.3f},{r['own_iid'][2]:+.3f}]  "
              f"own 5d [{r['own_blk'][1]:+.3f},{r['own_blk'][2]:+.3f}]")
    me = os.path.abspath(__file__)
    stamp = dict(what="paper_twoway.py: own-date intervals, 117 certified cells, session split; TRAINING periods only",
                 date=datetime.date.today().isoformat(), seconds=round(time.time() - t0),
                 script_sha256=hashlib.sha256(open(me, "rb").read()).hexdigest().upper(),
                 claim_checks_sha256=hashlib.sha256(open(CC.__file__.replace(".pyc", ".py"), "rb").read()).hexdigest().upper(),
                 dump=os.path.abspath(dump), machine=platform.node(), python=platform.python_version(),
                 numpy=np.__version__, pandas=pd.__version__, bootstrap="iid dates / 5-day moving block via core_v2.boot_index, B=2000, seed 20261003",
                 R1_all_six_hold=bool(all(hold)))
    out = os.path.join(HERE, f"paper_twoway_{datetime.date.today().isoformat()}.json")
    json.dump(dict(stamp=stamp, table_144=tab144, table_117=tab117), open(out, "w"), indent=1, default=str)
    print("\nwrote", out, "| script SHA-256", stamp["script_sha256"][:16])
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) >= 3 and sys.argv[1] == "--preflight":
        sys.exit(0 if preflight(sys.argv[2]) else 1)
    if len(sys.argv) >= 2:
        sys.exit(main(sys.argv[1]))
    print(__doc__)
