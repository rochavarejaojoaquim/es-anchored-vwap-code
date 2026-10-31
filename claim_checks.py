r"""
claim_checks.py -- CLAIM 1 and CLAIM 2 exactly as the author decided them (HANDOFF.md 21.17,
21.19, 21.20; 2026-09-21), plus the two robustness checks the 2026-10-02/03 audit asked for
(HANDOFF 26.62 F4, 26.63 G1, G2).

    python claim_checks.py --selftest      toy fixtures only, reads no dump
    python claim_checks.py data\dump        the real run (reads only; writes ONE json)

THE RULES, QUOTED FROM 21.19 / 21.17, NOT RE-DECIDED HERE
  CLAIM 1 "The VWAP level is special."
    edge_M = P(setup: wick fill of the frozen VWAP within T, tau=0)
           - P(twin_M: wick fill of its frozen level within T),   M = M1b and M = M2.
    PROVEN only if BOTH blocks' 95% day-bootstrap CIs lie above 0 for M1b AND for M2
    (Berger 1982 intersection-union: no multiplicity penalty).
  CLAIM 2 "Fading the stretch beats following it."
    race per setup: VWAP vs MIRROR, both frozen, wick-touched, within T.
    rate_setup = P(VWAP first | resolved); ties (same bar) and opens (neither) EXCLUDED
    and COUNTED. baseline = the same race on the M1b twins (EMPIRICAL). edge_2 = rate_setup
    - baseline. PROVEN only if both blocks' 95% CIs lie above 0.
  THE AUTHOR'S BLIND PREDICTION (21.19): edge_2 INCREASES with eps. Rule as written there,
    "(proposed; confirm)" -- its confirmation is NOT recorded, so it is reported as the
    proposed rule: slope of edge_2 on eps, ATR and SD separately, per block, day-bootstrap
    CI. CONFIRMED if both blocks' slope CIs > 0; REFUTED if either < 0; else INCONCLUSIVE.
  CAVEAT 21.19 attaches: eps and T are locked (T ~ eps^2), so a slope cannot separate the
    bigger stretch from the longer race.

TWO CHOICES 21.19 DOES NOT PIN DOWN, fixed here before the run and printed with it:
  cells      the 144 REGISTERED cells (hh/ll/hv x 5 N x full eps + session at N=60),
             the set sections 25 and 27 used, so stamps stay comparable (A2).
  estimand   P(...) read as a POOLED proportion over setups (ratio of sums), not a mean
             of day means. The day-mean figure is printed beside it.

THE TWO ROBUSTNESS CHECKS (they report; they do not change the verdict rule)
  BLOCK   the same CIs from the 5-day moving block bootstrap (core_v2.boot_index, the
          draws every Stage 1 score pays for). 26.62 F4: DX1's "blocking does not widen
          the SE" was measured with both arms at the SAME moment and does not transfer.
  DEDUP   each signal event once. An event = (anchor, trigger, eps, bar, side); the same
          event sits in up to five N-cells (24-74% overlap measured, 26.63 G2). Outcomes
          are averaged across its copies, so every event weighs 1.

NOTHING HERE WAS TUNED ON ITS OUTPUT. The day bootstrap is iid over trade dates, B = 2000,
seed 20261003; the block bootstrap uses core_v2.boot_index(D, 2000, 5, seed=1). Clusters
are the SETUP's trade date (twins follow their setup, as in section 27).
DISCLOSED: TEST B (26.53) already showed DIFFERENT but related numbers (separate fill
rates within T, not the race). Nothing of the race itself has been computed before.

Project root; rotates nothing. Writes claim_checks_<date>.json with its stamp, code-set
and its own SHA-256 (brain/13 C137).
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

import core_v2                                                          # noqa: E402
from power_registered_set import registered_cells                       # noqa: E402
from pseudo_null_pair import CELL_RE                                    # noqa: E402

B_BOOT, SEED, BLOCKLEN = 2000, 20261003, 5
PUB = dict(real=51.297, twin=48.174, pooled_gap=3.1234, m2_daymean=2.8633,
           mirror=47.235, twin_mirror=48.232)          # 26.51 / 26.53, known numbers
MAIN_COLS = ["is_real", "bar", "src_bar", "day", "block", "side", "T",
             "w_0.0", "mir_w", "race"]


# ------------------------------------------------------------------ the statistics
def _sums(df, cols, codes, D):
    return {c: np.bincount(codes, weights=df[c].to_numpy(float), minlength=D) for c in cols}


_DRAWS = {}


def boot_draws(D, kind, rng=None):
    """Rows of day indices, one row per bootstrap draw. Deterministic in (D, kind), so
    they are built once and reused -- every claim on the same days shares the draws."""
    if (D, kind) in _DRAWS:
        return _DRAWS[(D, kind)]
    if kind == "iid":
        S = np.random.default_rng(SEED).integers(0, D, size=(B_BOOT, D))
        _DRAWS[(D, kind)] = S
        return S
    S = core_v2.boot_index(D, B_BOOT, BLOCKLEN, 1)
    if not isinstance(S, np.ndarray):
        raise ValueError("ragged block draws (D < blocklen)")
    _DRAWS[(D, kind)] = S
    return S


def ratio_diff(sums, num_a, den_a, num_b, den_b, S=None):
    """(sum num_a / sum den_a) - (sum num_b / sum den_b); over all days, or per draw."""
    if S is None:
        return sums[num_a].sum() / sums[den_a].sum() - sums[num_b].sum() / sums[den_b].sum()
    a = sums[num_a][S].sum(1) / sums[den_a][S].sum(1)
    b = sums[num_b][S].sum(1) / sums[den_b][S].sum(1)
    return a - b


def ci(draws):
    d = draws[np.isfinite(draws)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), int(len(d))


def day_codes(day):
    u, codes = np.unique(np.asarray(day), return_inverse=True)   # sorted = chronological
    return codes, len(u)


def claim1_block(df, kind, rng=None):
    """edge = P(setup fill) - P(twin fill), pooled, with a CI. df: one row per unit,
    columns rf (setup fill), tf (twin fill, mean of its twins), w (weight 1)."""
    codes, D = day_codes(df.day)
    sm = _sums(df, ["rf", "tf", "w"], codes, D)
    est = ratio_diff(sm, "rf", "w", "tf", "w")
    lo, hi, n = ci(ratio_diff(sm, "rf", "w", "tf", "w", boot_draws(D, kind, rng)))
    return dict(est_pp=100 * est, lo_pp=100 * lo, hi_pp=100 * hi, D=D, n=len(df), draws=n)


def claim2_block(df, kind, rng=None):
    """edge_2 = P(VWAP first | resolved)_setups - same on the twins (pooled)."""
    codes, D = day_codes(df.day)
    sm = _sums(df, ["r_vf", "r_res", "t_vf", "t_res"], codes, D)
    est = ratio_diff(sm, "r_vf", "r_res", "t_vf", "t_res")
    lo, hi, n = ci(ratio_diff(sm, "r_vf", "r_res", "t_vf", "t_res", boot_draws(D, kind, rng)))
    return dict(est_pp=100 * est, lo_pp=100 * lo, hi_pp=100 * hi, D=D, n=len(df), draws=n,
                rate_setup=float(sm["r_vf"].sum() / sm["r_res"].sum()),
                rate_twin=float(sm["t_vf"].sum() / sm["t_res"].sum()))


def slope_block(df, kind, rng=None):
    """Slope of edge_2 on eps across the eps levels present, with a bootstrap CI.
    ONE set of day draws is shared by every eps level, so the levels stay paired."""
    codes, D = day_codes(df.day)
    levels = sorted(df.eps.unique())
    per = {e: _sums(df[df.eps == e], ["r_vf", "r_res", "t_vf", "t_res"],
                    codes[(df.eps == e).to_numpy()], D) for e in levels}
    x = np.asarray(levels, float)
    pt = np.array([ratio_diff(per[e], "r_vf", "r_res", "t_vf", "t_res") for e in levels])
    S = boot_draws(D, kind, rng)
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.column_stack([ratio_diff(per[e], "r_vf", "r_res", "t_vf", "t_res", S)
                             for e in levels])
    ok = np.isfinite(Y).all(1)
    xc = x - x.mean()
    slopes = (Y[ok] - Y[ok].mean(1, keepdims=True)) @ xc / (xc @ xc)
    lo, hi, n = ci(slopes)
    return dict(slope_pp=float(100 * (pt - pt.mean()) @ xc / (xc @ xc)),
                lo_pp=100 * lo, hi_pp=100 * hi, levels=[float(v) for v in x],
                edge_by_eps_pp=[float(100 * v) for v in pt], draws=n)


def verdict_proven(blocks):
    """blocks: list of (label, result). PROVEN iff every CI lies entirely above 0."""
    fails = [lab for lab, r in blocks if not r["lo_pp"] > 0]
    return ("PROVEN" if not fails else "NOT PROVEN"), fails


def verdict_slope(rA, rB):
    if rA["lo_pp"] > 0 and rB["lo_pp"] > 0:
        return "CONFIRMED"
    if rA["hi_pp"] < 0 or rB["hi_pp"] < 0:
        return "REFUTED"
    return "INCONCLUSIVE"


def dedup(df, cols):
    """One row per signal event (anchor, trigger, eps, bar, side); outcomes averaged
    over the event's copies across N. Day and block are properties of the bar."""
    key = ["anchor", "trigger", "eps", "bar", "side"]
    g = df.groupby(key, sort=False)
    out = g[cols].mean()
    out["day"] = g["day"].first(); out["block"] = g["block"].first()
    nd = int((g["day"].nunique() > 1).sum()) + int((g["block"].nunique() > 1).sum())
    out = out.reset_index()
    out["w"] = 1.0
    return out, nd


# ------------------------------------------------------------------ loading
def load(dump, cells):
    parts, bad_k, race_bad = [], {"M1b": 0, "M2": 0}, 0
    for c in cells:
        N, anchor, trig, eps = CELL_RE.search(c).groups()
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=MAIN_COLS)
        T = int(x["T"].iloc[0])
        r = x[x.is_real == 1]
        race_bad += race_violations(r, T)
        rows = pd.DataFrame(dict(
            cell=c, N=int(N), anchor=anchor, trigger=trig, eps=float(eps),
            bar=r.bar.to_numpy(), side=np.sign(r.side.to_numpy()), day=r.day.to_numpy(),
            block=r.block.to_numpy(),
            rf=fill(r["w_0.0"], T), r_res=np.isin(r.race, (1, -1)).astype(float),
            r_vf=(r.race.to_numpy() == 1).astype(float),
            r_tie=(r.race.to_numpy() == 0).astype(float),
            r_open=(r.race.to_numpy() == 9).astype(float),
            r_mf=fill(r["mir_w"], T)))
        tw1 = x[x.is_real == 0][["src_bar", "w_0.0", "mir_w", "race"]]
        tw2 = pd.read_parquet(os.path.join(dump, c + ".m2.parquet"),
                              columns=["src_bar", "w_0.0", "mir_w", "race"])
        for tag, tw in (("M1b", tw1), ("M2", tw2)):
            race_bad += race_violations(tw, T)
            t = pd.DataFrame(dict(src_bar=tw.src_bar.to_numpy(), f=fill(tw["w_0.0"], T),
                                  mf=fill(tw["mir_w"], T),
                                  res=np.isin(tw.race, (1, -1)).astype(float),
                                  vf=(tw.race.to_numpy() == 1).astype(float)))
            a = t.groupby("src_bar").agg(f=("f", "mean"), mf=("mf", "mean"), res=("res", "sum"),
                                         vf=("vf", "sum"), k=("f", "size"))
            a.columns = [f"{tag}_{k}" for k in a.columns]
            rows = rows.join(a, on="bar")
            bad_k[tag] += int(rows[f"{tag}_k"].isna().sum())       # setups with NO twin
        parts.append(rows)
    return pd.concat(parts, ignore_index=True), bad_k, race_bad


def fill(w, T):
    w = np.asarray(w, float)
    return ((w >= 1) & (w <= T)).astype(float)


def race_violations(x, T):
    """race must agree with the two fill columns (_walkextra.py:56-61)."""
    v = fill(x["w_0.0"], T).astype(bool); m = fill(x["mir_w"], T).astype(bool)
    w = np.asarray(x["w_0.0"], float); mw = np.asarray(x["mir_w"], float)
    want = np.full(len(x), 9)
    want[v & (~m | (w < mw))] = 1
    want[m & (~v | (mw < w))] = -1
    want[v & m & (w == mw)] = 0
    return int((want != np.asarray(x["race"])).sum())


def arm_frame(df, tag):
    """Claim-1 / claim-2 columns for one twin scheme. A setup needs >= 1 twin of that
    scheme; with one twin its mean is that twin. This is level_control.py's population
    (26.51: M2 leaves 17 setups with no twin at all), so the known numbers apply."""
    s = df[df[f"{tag}_k"].fillna(0) >= 1].copy()
    s["tf"] = s[f"{tag}_f"]; s["t_res"] = s[f"{tag}_res"]; s["t_vf"] = s[f"{tag}_vf"]
    s["w"] = 1.0
    return s


# ------------------------------------------------------------------ self-test
def _toy(n_days=800, per_day=60, gap1=0.03, gap1_B=None, gap_m2=None, p_vf=0.55,
         p_vf_twin=0.50, eps_levels=(3.0, 4.0, 6.0, 8.0, 12.0), slope=0.0, seed=11):
    """Setups with 2 twins per scheme. Block A = first half of the days."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    block = np.where(day < n_days // 2, "A", "B")
    eps = rng.choice(eps_levels, n)
    g1 = np.full(n, gap1) if gap1_B is None else np.where(block == "A", gap1, gap1_B)
    rf = (rng.random(n) < 0.45 + g1).astype(float)
    m1 = (rng.random((n, 2)) < 0.45).mean(1)
    gm2 = g1 if gap_m2 is None else np.full(n, gap_m2)
    m2 = (rng.random((n, 2)) < (0.45 + g1 - gm2)[:, None]).mean(1)
    pv = p_vf + slope * (eps - np.mean(eps_levels))
    res = rng.random(n) < 0.8
    vf = res & (rng.random(n) < pv)
    tres = (rng.random((n, 2)) < 0.8)
    tvf = tres & (rng.random((n, 2)) < p_vf_twin)
    return pd.DataFrame(dict(day=day, block=block, eps=eps, rf=rf, w=1.0,
                             M1b_f=m1, M2_f=m2, r_res=res.astype(float), r_vf=vf.astype(float),
                             t_res=tres.sum(1).astype(float), t_vf=tvf.sum(1).astype(float)))


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\nclaim_checks.py SELF-TEST -- detectors before any dump is read.\n")

    def c1(t, tag, kind="iid"):
        f = t.copy(); f["tf"] = f[f"{tag}_f"]
        return [(f"{tag} {b}", claim1_block(f[f.block == b], kind)) for b in "AB"]

    t = _toy()
    blocks = c1(t, "M1b") + c1(t, "M2")
    v, fails = verdict_proven(blocks)
    chk("K1 CLAIM 1: +3 pp against BOTH schemes in BOTH blocks -> PROVEN", v == "PROVEN",
        " ".join(f"{lab} {r['est_pp']:+.2f}[{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}]"
                 for lab, r in blocks))
    tb = _toy(gap1_B=0.0, seed=12)
    v, fails = verdict_proven(c1(tb, "M1b") + c1(tb, "M2"))
    chk("M1 an effect only in block A -> NOT PROVEN, block B named", v == "NOT PROVEN"
        and all(" B" in f for f in fails), f"fails {fails}")
    tm = _toy(gap_m2=0.0, seed=13)
    v, fails = verdict_proven(c1(tm, "M1b") + c1(tm, "M2"))
    chk("M2 beats M1b but NOT M2 -> NOT PROVEN (intersection-union needs both)",
        v == "NOT PROVEN" and all(f.startswith("M2") for f in fails), f"fails {fails}")

    r2 = [(f"race {b}", claim2_block(t[t.block == b], "iid")) for b in "AB"]
    v, _ = verdict_proven(r2)
    chk("K2 CLAIM 2: VWAP-first 55% vs twin 50% -> edge_2 ~ +5 pp, PROVEN",
        v == "PROVEN" and all(abs(r["est_pp"] - 5) < 1.5 for _, r in r2),
        " ".join(f"{lab} {r['est_pp']:+.2f}" for lab, r in r2))
    t0 = _toy(p_vf=0.50, seed=14)
    r0 = [(f"race {b}", claim2_block(t0[t0.block == b], "iid")) for b in "AB"]
    v, _ = verdict_proven(r0)
    chk("M3 no race edge -> NOT PROVEN", v == "NOT PROVEN",
        " ".join(f"{lab} {r['est_pp']:+.2f}[{r['lo_pp']:+.2f},{r['hi_pp']:+.2f}]" for lab, r in r0))

    # K3 the conditional rate EXCLUDES ties and opens from its denominator, exactly
    k = pd.DataFrame(dict(day=[0, 0, 1, 1, 2, 2], block="A", r_res=[1, 1, 1, 0, 0, 1],
                          r_vf=[1, 0, 1, 0, 0, 1], t_res=[2, 2, 2, 2, 2, 2],
                          t_vf=[1, 1, 1, 1, 1, 1]))
    r = claim2_block(k, "iid")
    chk("K3 ties/opens leave the denominator: 3 VWAP-first of 4 resolved = 75% vs 50%",
        abs(r["rate_setup"] - 0.75) < 1e-12 and abs(r["est_pp"] - 25.0) < 1e-9,
        f"rate {r['rate_setup']:.4f}, edge {r['est_pp']:+.4f} pp")

    # K4 / M4 / M5 the eps-slope rule
    ts = _toy(slope=0.01, seed=15)
    sA, sB = (slope_block(ts[ts.block == b], "iid") for b in "AB")
    chk("K4 a planted RISING edge_2 with eps -> CONFIRMED", verdict_slope(sA, sB) == "CONFIRMED",
        f"A {sA['slope_pp']:+.3f}[{sA['lo_pp']:+.3f},{sA['hi_pp']:+.3f}] pp/eps")
    tf_ = _toy(slope=0.0, seed=16)
    fA, fB = (slope_block(tf_[tf_.block == b], "iid") for b in "AB")
    chk("M4 a FLAT edge_2 -> not CONFIRMED", verdict_slope(fA, fB) != "CONFIRMED",
        verdict_slope(fA, fB))
    td = _toy(slope=-0.01, seed=17)
    dA, dB = (slope_block(td[td.block == b], "iid") for b in "AB")
    chk("M5 a FALLING edge_2 -> REFUTED", verdict_slope(dA, dB) == "REFUTED", verdict_slope(dA, dB))

    # K5 dedup: copies of an event must count ONCE. Build events, copy the strong ones
    # 3x; occurrence-weighting is pulled toward them, event-weighting is not.
    rng = np.random.default_rng(3)
    ev = pd.DataFrame(dict(anchor="ll", trigger="ATR", eps=3.0, bar=np.arange(4000),
                           side=1.0, day=np.arange(4000) // 20, block="A",
                           rf=(rng.random(4000) < 0.5).astype(float)))
    ev["tf"] = 0.5
    strong = ev[ev.bar < 1000].assign(rf=1.0)
    ev.loc[ev.bar < 1000, "rf"] = 1.0
    copies = pd.concat([ev, strong, strong], ignore_index=True)
    dd, nd = dedup(copies, ["rf", "tf"])
    occ = copies.rf.mean() - 0.5
    evw = ev.rf.mean() - 0.5
    chk("K5 dedup restores EVENT weighting (each event once) and nothing else",
        len(dd) == 4000 and nd == 0 and abs((dd.rf.mean() - 0.5) - evw) < 1e-12
        and abs(occ - evw) > 0.05, f"events {len(dd)}, occurrence {occ:+.4f} vs event {evw:+.4f}")

    # K6 the iid day bootstrap is calibrated against the analytic day-cluster SE
    t6 = _toy(gap1=0.0, seed=18)
    f6 = t6[t6.block == "A"].copy(); f6["tf"] = f6.M1b_f
    r6 = claim1_block(f6, "iid")
    codes, D = day_codes(f6.day)
    dm = (np.bincount(codes, f6.rf - f6.tf, D) / np.bincount(codes, minlength=D))
    se_a = 100 * dm.std(ddof=1) / np.sqrt(D)
    se_b = (r6["hi_pp"] - r6["lo_pp"]) / (2 * 1.96)
    chk("K6 iid day-bootstrap SE within 15% of the analytic day-cluster SE",
        abs(se_b / se_a - 1) < 0.15, f"{se_b:.4f} vs {se_a:.4f} pp")

    # K7 the block bootstrap IS core_v2.boot_index, not a copy of it
    chk("K7 block draws == core_v2.boot_index(D, 2000, 5, seed=1)",
        np.array_equal(boot_draws(300, "block"), core_v2.boot_index(300, B_BOOT, 5, 1)))
    # M6 and the block bootstrap is not the iid one under day-level serial dependence
    rng = np.random.default_rng(19)
    Dd = 800
    shock = np.repeat(rng.normal(0, 0.08, Dd // 10), 10)        # 10-day regimes
    t7 = pd.DataFrame(dict(day=np.repeat(np.arange(Dd), 40), block="A"))
    p = np.clip(0.45 + np.repeat(shock, 40), 0, 1)
    t7["rf"] = (rng.random(len(t7)) < p).astype(float); t7["tf"] = 0.45; t7["w"] = 1.0
    wi = claim1_block(t7, "iid"); wb = claim1_block(t7, "block")
    chk("M6 under multi-day regimes the BLOCK CI is wider than the iid one",
        (wb["hi_pp"] - wb["lo_pp"]) > 1.2 * (wi["hi_pp"] - wi["lo_pp"]),
        f"width {wb['hi_pp']-wb['lo_pp']:.3f} vs {wi['hi_pp']-wi['lo_pp']:.3f} pp")

    # K8 race consistency detector: a planted swap is caught
    x = pd.DataFrame({"w_0.0": [3, -1, 5, 2], "mir_w": [5, 2, -1, 2], "race": [1, -1, 1, 0]})
    chk("K8 race agrees with the fill columns on a correct frame", race_violations(x, 10) == 0)
    x.loc[0, "race"] = -1
    chk("M7 a planted race error is caught", race_violations(x, 10) == 1)

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


# ------------------------------------------------------------------ main
def sha256(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fmt(r):
    return f"{r['est_pp']:+7.3f} pp  95% CI [{r['lo_pp']:+7.3f}, {r['hi_pp']:+7.3f}]  ({r['D']:,} days, {r['n']:,})"


def run_claims(df, tag_note, out):
    """Every verdict and robustness line for one population (primary or dedup)."""
    res = {}
    for kind in ("iid", "block"):
        c1 = []
        for tag in ("M1b", "M2"):
            s = arm_frame(df, tag)
            for b in "AB":
                c1.append((f"{tag} {b}", claim1_block(s[s.block == b], kind)))
        s1 = arm_frame(df, "M1b")
        c2 = [(f"race {b}", claim2_block(s1[s1.block == b], kind)) for b in "AB"]
        sl = {}
        for trig in ("ATR", "SD"):
            q = s1[s1.trigger == trig]
            sl[trig] = {b: slope_block(q[q.block == b], kind) for b in "AB"}
        res[kind] = dict(c1=c1, c2=c2, slope=sl)

    p = res["iid"]
    v1, f1 = verdict_proven(p["c1"]); v2, f2 = verdict_proven(p["c2"])
    print(f"\n  ===== {tag_note} =====")
    print("  CLAIM 1  edge = P(setup fill) - P(twin fill), pooled, day-bootstrap 95% CI")
    for lab, r in p["c1"]:
        print(f"    {lab:<7} {fmt(r)}")
    print(f"    VERDICT (21.19, M1b AND M2, both blocks): {v1}" + (f"  -- fails: {f1}" if f1 else ""))
    print("  CLAIM 2  edge_2 = P(VWAP first | resolved) - same on the M1b twins")
    for lab, r in p["c2"]:
        print(f"    {lab:<7} {fmt(r)}   setup {100*r['rate_setup']:.3f}% vs twin "
              f"{100*r['rate_twin']:.3f}%")
    print(f"    VERDICT (21.19, both blocks): {v2}" + (f"  -- fails: {f2}" if f2 else ""))
    print("  BLIND PREDICTION  edge_2 rises with eps (rule 'proposed; confirm' in 21.19)")
    vs = {}
    for trig in ("ATR", "SD"):
        a, b = p["slope"][trig]["A"], p["slope"][trig]["B"]
        vs[trig] = verdict_slope(a, b)
        for blk, r in (("A", a), ("B", b)):
            print(f"    {trig} {blk}: slope {r['slope_pp']:+.4f} pp/eps  CI [{r['lo_pp']:+.4f}, "
                  f"{r['hi_pp']:+.4f}]   edge_2 by eps "
                  + " ".join(f"{e:g}:{v:+.2f}" for e, v in zip(r["levels"], r["edge_by_eps_pp"])))
        print(f"    {trig}: {vs[trig]}")

    q = res["block"]
    vb1, fb1 = verdict_proven(q["c1"]); vb2, fb2 = verdict_proven(q["c2"])
    vbs = {t: verdict_slope(q["slope"][t]["A"], q["slope"][t]["B"]) for t in ("ATR", "SD")}
    print("  ROBUSTNESS -- 5-day moving block bootstrap (26.62 F4), same estimates:")
    for lab, r in q["c1"] + q["c2"]:
        print(f"    {lab:<7} CI [{r['lo_pp']:+7.3f}, {r['hi_pp']:+7.3f}]")
    print(f"    claim 1 {vb1}, claim 2 {vb2}, slope ATR {vbs['ATR']}, SD {vbs['SD']}"
          + ("   (SAME as the registered day bootstrap)"
             if (vb1, vb2, vbs) == (v1, v2, vs) else "   ** DIFFERS from the day bootstrap **"))

    def pack(lst):
        return {lab: r for lab, r in lst}
    out[tag_note] = dict(claim1=v1, claim1_fails=f1, claim2=v2, claim2_fails=f2, slope=vs,
                         block_claim1=vb1, block_claim2=vb2, block_slope=vbs,
                         iid=dict(c1=pack(p["c1"]), c2=pack(p["c2"]), slope=p["slope"]),
                         block=dict(c1=pack(q["c1"]), c2=pack(q["c2"]), slope=q["slope"]))


def main(dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    t0 = time.time()
    cells = registered_cells()
    df, bad_k, race_bad = load(dump, cells)
    print(f"CLAIM CHECKS -- {len(cells)} registered cells from {dump}, {time.time()-t0:.0f}s load\n")

    m1 = arm_frame(df, "M1b")
    real, twin = 100 * df.rf.mean(), 100 * m1.tf.mean()
    gap = 100 * (m1.rf.mean() - m1.tf.mean())
    mir, tmir = 100 * df.r_mf.mean(), 100 * df["M1b_mf"].mean()
    m2 = arm_frame(df, "M2")
    codes, D = day_codes(m2.day)
    dm2 = 100 * (np.bincount(codes, m2.rf - m2.tf, D) / np.bincount(codes, minlength=D)).mean()
    checks = [
        ("K0a every setup has exactly 2 M1b twins",
         bad_k["M1b"] == 0 and set(df["M1b_k"].unique()) == {2},
         f"{bad_k['M1b']} without; k values {sorted(df['M1b_k'].dropna().unique())}"),
        ("K0b M2: exactly 26.51's 17 setups have no M2 twin", bad_k["M2"] == 17,
         f"{bad_k['M2']} without; k counts {df['M2_k'].value_counts(dropna=False).to_dict()}"),
        ("K0c setup fill reproduces 26.51", abs(real - PUB["real"]) < 0.0015, f"{real:.4f}%"),
        ("K0d M1b twin fill reproduces 26.51", abs(twin - PUB["twin"]) < 0.0015, f"{twin:.4f}%"),
        ("K0e pooled gap reproduces 26.51's +3.1234", abs(gap - PUB["pooled_gap"]) < 0.0015,
         f"{gap:+.4f} pp"),
        ("K0f M2 day-mean gap reproduces 26.51's +2.8633", abs(dm2 - PUB["m2_daymean"]) < 0.0015,
         f"{dm2:+.4f} pp"),
        ("K0g mirror fill reproduces 26.53", abs(mir - PUB["mirror"]) < 0.0015, f"{mir:.4f}%"),
        ("K0h twin-mirror fill reproduces 26.53", abs(tmir - PUB["twin_mirror"]) < 0.0015,
         f"{tmir:.4f}%"),
        ("K0i race agrees with w_0.0 and mir_w on every row", race_bad == 0,
         f"{race_bad} violations"),
    ]
    for lab, okk, det in checks:
        print(f"  [{'PASS' if okk else 'FAIL'}] {lab}   {det}")
    if not all(c[1] for c in checks):
        raise SystemExit("A known-number check FAILED -- no claim was evaluated.")

    ties, opens = int(df.r_tie.sum()), int(df.r_open.sum())
    cs = __import__("gate_receipts").code_set()
    me = sha256(os.path.abspath(__file__))
    stamp = (f"claims 1-2 as decided in HANDOFF.md 21.17/21.19 | {len(cells)} registered cells | "
             f"{len(df):,} setups, {df.day.nunique():,} trade dates | clusters = setup trade "
             f"date | day bootstrap iid B={B_BOOT} seed {SEED}; block = core_v2.boot_index b=5 | "
             f"tau 0, wick fill within T | dump {dump} | code-set {cs} | claim_checks.py "
             f"{me[:16]} | {platform.node()} | ES, train A+B, sealed 2018-2022 untouched | raw, "
             f"no cost")
    print(f"\nSTAMP: {stamp}")
    print(f"  race on setups: {int(df.r_res.sum()):,} resolved, {ties:,} ties and {opens:,} "
          f"open -- excluded from claim 2's rate and COUNTED here (21.19)")

    out = dict(stamp=stamp, code_set=cs, producer_sha256=me, known_numbers=[c[0] for c in checks],
               n_setups=len(df), ties=ties, opens=opens)
    run_claims(df, "PRIMARY: all setups of the 144 registered cells", out)

    ddf_parts = []
    for tag in ("M1b", "M2"):
        s = arm_frame(df, tag)
        s = s.rename(columns={})
        dd, nd = dedup(s.assign(t_res=s["t_res"], t_vf=s["t_vf"]),
                       ["rf", "tf", "r_res", "r_vf", "t_res", "t_vf"])
        if nd:
            raise SystemExit(f"DEDUP: {nd} events have more than one day or block -- stop.")
        dd[f"{tag}_k"] = 1                       # every surviving event has >= 1 twin
        dd[f"{tag}_f"] = dd["tf"]; dd[f"{tag}_res"] = dd["t_res"]; dd[f"{tag}_vf"] = dd["t_vf"]
        ddf_parts.append((tag, dd))
    print(f"\n  DEDUP: {len(df):,} setup rows -> {len(ddf_parts[0][1]):,} distinct events (M1b) "
          f"/ {len(ddf_parts[1][1]):,} (M2); every event weighs 1")
    # build a frame carrying both arms, keyed by event
    key = ["anchor", "trigger", "eps", "bar", "side"]
    a = ddf_parts[0][1][key + ["day", "block", "rf", "r_res", "r_vf", "M1b_k", "M1b_f",
                                "M1b_res", "M1b_vf"]]
    b = ddf_parts[1][1][key + ["M2_k", "M2_f", "M2_res", "M2_vf"]]
    dd = a.merge(b, on=key, how="left")
    dd["M2_k"] = dd["M2_k"].fillna(0)
    run_claims(dd, "DEDUP SENSITIVITY: each signal event once (26.63 G2)", out)

    path = os.path.join(HERE, f"claim_checks_{datetime.date.today().isoformat()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n  wrote {os.path.basename(path)}   {time.time()-t0:.0f}s")
    print("  The verdict is the PRIMARY population under the DAY bootstrap, as 21.19 decided. The "
          "block\n  bootstrap and dedup lines are robustness checks; if either DIFFERS, that is "
          "reported, not\n  used to overturn the registered verdict.")
    return 0


def preflight(dump):
    """Two real cells through the WHOLE path -- load, race consistency, both arms, dedup,
    every claim function under both bootstraps -- printing PASS/FAIL ONLY. No claim value
    is printed, so the real result is not read before the registered run."""
    cells = ["N60_ll_ATR_3.0", "N100_ll_ATR_3.0", "N60_ll_ATR_4.0"]   # dup across N; 2 eps
    df, bad_k, race_bad = load(dump, cells)
    res = []

    def chk(lab, cond):
        res.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {lab}")
    chk("loads 3 real cells, M1b twins 2 per setup", bad_k["M1b"] == 0
        and set(df["M1b_k"].unique()) == {2})
    chk("race agrees with w_0.0 and mir_w on every real row (setups, M1b, M2)", race_bad == 0)
    dd, nd = dedup(arm_frame(df, "M1b"), ["rf", "tf", "r_res", "r_vf", "t_res", "t_vf"])
    chk(f"dedup: every event has one day and one block ({len(df):,} rows -> {len(dd):,} events)",
        nd == 0 and len(dd) < len(df))
    finite = True
    for kind in ("iid", "block"):
        for tag in ("M1b", "M2"):
            s = arm_frame(df, tag)
            for b in "AB":
                r = claim1_block(s[s.block == b], kind)
                finite &= all(np.isfinite([r["est_pp"], r["lo_pp"], r["hi_pp"]]))
        s = arm_frame(df, "M1b")
        for b in "AB":
            r = claim2_block(s[s.block == b], kind)
            finite &= all(np.isfinite([r["est_pp"], r["lo_pp"], r["hi_pp"]]))
            sl = slope_block(s[(s.block == b)], kind)       # eps 3 and 4
            finite &= all(np.isfinite([sl["slope_pp"], sl["lo_pp"], sl["hi_pp"]]))
    chk("every claim function returns finite estimates and CIs under both bootstraps", finite)
    print(f"\n  PREFLIGHT {'PASS' if all(res) else 'FAIL'} -- no claim value was printed.")
    return 0 if all(res) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--preflight" in sys.argv:
        a = [x for x in sys.argv[1:] if not x.startswith("-")]
        sys.exit(preflight(a[0] if a else os.path.join(HERE, "data", "dump")))
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args[0]))
