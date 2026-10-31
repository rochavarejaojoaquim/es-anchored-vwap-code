r"""
pseudo_power.py -- §25.4 procedure validation (F5) for the pseudo-VWAP test.

    python pseudo_power.py --selftest
    python pseudo_power.py DUMP_DIR CELL [CELL ...]

WHAT IT CHECKS (criteria written BEFORE any run):
  Inject a known +2.0 pp real-minus-pseudo effect into a world where the true
  effect is exactly zero, and confirm the §25.3 test detects it (one-sided
  p <= 0.05) in >= 80% of 200 draws.  If not, the sample cannot answer the
  question and a null result may NOT be reported as evidence for H0.
  PASS requires ALL of:
    P1  detection >= 80% of 200 draws at +2.0 pp
    P2  null calibration: with NO injection the rejection rate is in [1%, 10%]
        (the test is not secretly liberal or conservative on this sample)
    P3  the realized injected gap is 2.0 pp (to within rounding of 1 setup)

THE ZERO-EFFECT WORLD.  Per trade date, real and pseudo fills are SWAPPED with
probability 1/2 -- whole days together, never single setups.  Swapping makes the
two lines exchangeable (true gap 0, whatever the real gap is) while keeping each
day's setups together, so the intra-day dependence of the paired differences is
preserved.  (Swapping setups independently would destroy that dependence and
overstate power.)  The INJECTION then flips a random 2.0% of the setups' (new)
pseudo fills from hit to miss: a shift of exactly +2.0 pp, setup-weighted.
THE TEST is §25.2-25.3's: d_i = real fill - pseudo fill, mean within trade date,
one-sample t across the day means, one-sided.

THIS SCRIPT NEVER COMPUTES OR PRINTS THE REAL GAP.  The day signs are randomized
before anything is aggregated, so the output is the same whether the true gap is
+5 pp or -5 pp.  Self-test runs first, always; a failure aborts before the data.

NOTE: §25.3 says "the pooled setup set at the primary (eps, m) configuration" and
does not say which cells that is.  Pass exactly the cells it denotes.
"""
import os
import sys

import numpy as np
from scipy import stats

POWER_SEED = 20260930        # first use in this project
ALPHA, DELTA, NDRAW, NEED = 0.05, 0.02, 200, 0.80
FPR_LO, FPR_HI = 0.01, 0.10


def one_sided_p(d, g, D, test="day"):
    if test == "day":                       # §25.2: cluster on trade date, F1 exact
        cnt = np.bincount(g, minlength=D)
        m = np.bincount(g, weights=d, minlength=D) / cnt
        t = m.mean() / (m.std(ddof=1) / np.sqrt(D))
        return float(stats.t.sf(t, D - 1))
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))          # MUTATION ONLY
    return float(stats.t.sf(t, len(d) - 1))


def run(r, p, g, inject, delta, ndraw, seed, test="day"):
    """-> (rejection rate, worst |realized gap - 2.0pp| over draws)."""
    rng = np.random.default_rng(seed)
    n, D = len(r), int(g.max()) + 1
    rej, worst = 0, 0.0
    for _ in range(ndraw):
        swap = rng.random(D) < 0.5
        sg = swap[g]
        real = np.where(sg, p, r).astype(float)
        pse = np.where(sg, r, p).astype(float)
        d0 = real - pse
        if inject:
            k = int(round(delta * n))
            hits = np.flatnonzero(pse == 1)
            if len(hits) < k:
                raise SystemExit("CANNOT INJECT: fewer pseudo hits than setups to flip")
            pse[rng.choice(hits, k, replace=False)] = 0
        d = real - pse
        worst = max(worst, abs((d.mean() - d0.mean()) - DELTA))
        rej += one_sided_p(d, g, D, test) <= ALPHA
    # the INTEGER count is returned, not the ratio: the receipt records counts so
    # P1 and P2 can be re-checked in integer arithmetic (brain/02; E02, where float
    # equality on p-values produced two different "exact" counts from one dataset).
    return int(rej), int(ndraw), worst


def validate(r, p, g, test="day", delta=DELTA, ndraw=NDRAW, seed=POWER_SEED):
    k_det, n_draws, gap_err = run(r, p, g, True, delta, ndraw, seed, test)
    k_fpr, _, _ = run(r, p, g, False, delta, ndraw, seed + 1, test)
    power, fpr = k_det / n_draws, k_fpr / n_draws
    n = len(r)
    p1, p2 = power >= NEED, FPR_LO <= fpr <= FPR_HI
    p3 = gap_err <= 0.5 / n + 1e-9
    return dict(power=power, fpr=fpr, gap_err=gap_err, n=n, D=int(g.max()) + 1,
                n_draws=int(n_draws), k_detected=int(k_det), k_null_rejected=int(k_fpr),
                P1=bool(p1), P2=bool(p2), P3=bool(p3), ok=bool(p1 and p2 and p3))


def report(v, label=""):
    print(f"  {label}setups {v['n']:,}, trade dates {v['D']:,}")
    print(f"    [{'PASS' if v['P1'] else 'FAIL'}] P1 detection of +2.0 pp : {100*v['power']:5.1f}% of {NDRAW} draws (need >= 80%)")
    print(f"    [{'PASS' if v['P2'] else 'FAIL'}] P2 null calibration    : {100*v['fpr']:5.1f}% rejected with no effect (need 1-10%)")
    print(f"    [{'PASS' if v['P3'] else 'FAIL'}] P3 injected gap exact    : worst |error| {1e4*v['gap_err']:.3f} bp (need <= 1 setup)")
    print(f"    VERDICT: {'PASS' if v['ok'] else 'FAIL'}")


# ------------------------------------------------------------------ self-test
def synth(D, m, seed, day_sd=0.0):
    """D days x m setups, fills Bernoulli(0.5) for both arms, independent.
    day_sd > 0 adds a day-level shock to the REAL arm's fill probability."""
    rr = np.random.default_rng(seed)
    g = np.repeat(np.arange(D), m)
    pr = np.clip(0.5 + rr.normal(0, day_sd, D), 0, 1)[g] if day_sd else 0.5
    return (rr.random(D * m) < pr).astype(int), (rr.random(D * m) < 0.5).astype(int), g


def selftest():
    """Known answers.  Fills iid p=0.5, m=10 setups/day: var(d)=0.5, so the day
    mean has variance 0.05 and SE = sqrt(0.05/D).  Shift 0.02:
        D=400  -> z = 1.789 -> power = Phi(1.789-1.645) = 0.557
        D=1200 -> z = 3.098 -> power = Phi(3.098-1.645) = 0.927"""
    print("SELF-TEST -- known answers first, then each mutation must FAIL\n")
    res = []

    def check(name, ok):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    r4, p4, g4 = synth(400, 10, 1)
    r12, p12, g12 = synth(1200, 10, 2)
    v4 = validate(r4, p4, g4)
    v12 = validate(r12, p12, g12)
    check(f"K1 D=400 : power {100*v4['power']:.1f}% vs analytic 55.7% (within 10 pts); verdict FAIL as it must be",
          abs(v4["power"] - 0.557) < 0.10 and not v4["P1"])
    check(f"K2 D=1200: power {100*v12['power']:.1f}% vs analytic 92.7% (within 10 pts); verdict PASS",
          abs(v12["power"] - 0.927) < 0.10 and v12["ok"])
    check(f"K3 null calibration, no effect: {100*v4['fpr']:.1f}% (D=400) and {100*v12['fpr']:.1f}% (D=1200) in 1-10%",
          v4["P2"] and v12["P2"])
    check(f"K4 injected gap exact: worst error {1e4*max(v4['gap_err'], v12['gap_err']):.3f} bp",
          v4["P3"] and v12["P3"])
    m1 = validate(r12, p12, g12, delta=0.0)
    check(f"M1 inject NOTHING -> detection {100*m1['power']:.1f}% -> verdict FAILS", not m1["P1"] and not m1["ok"])
    rd, pd_, gd = synth(400, 30, 3, day_sd=0.6)
    m2 = validate(rd, pd_, gd, test="setup")
    m2ok = validate(rd, pd_, gd, test="day")
    check(f"M2 ignore day clustering (test on setups) -> null rejects {100*m2['fpr']:.1f}% "
          f"(correct test {100*m2ok['fpr']:.1f}%) -> P2 FAILS",
          not m2["P2"] and m2ok["P2"])
    m3 = validate(r12, p12, g12, delta=0.01)
    check(f"M3 inject 1.0 pp but claim 2.0 -> realized gap error {1e4*m3['gap_err']:.0f} bp -> P3 FAILS",
          not m3["P3"])
    r1, p1, g1 = synth(100, 10, 4)
    m4 = validate(r1, p1, g1)
    check(f"M4 too few days (100) -> detection {100*m4['power']:.1f}% -> verdict FAILS", not m4["ok"])
    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


# ---------------------------------------------------------------------- data
def load(dump, cells):
    import pandas as pd
    parts, unm, tot = [], 0, 0
    for c in cells:
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"),
                            columns=["is_real", "day", "w_0.0", "pseudo_w_0.0", "pseudo_matched", "T"])
        x = x[x.is_real == 1]
        tot += len(x)
        unm += int((x.pseudo_matched != 1).sum())
        x = x[x.pseudo_matched == 1]
        T = x["T"].to_numpy()
        w, pw = x["w_0.0"].to_numpy(), x["pseudo_w_0.0"].to_numpy()
        parts.append(pd.DataFrame(dict(day=x["day"].to_numpy(), r=((w >= 1) & (w <= T)).astype(int),
                                       p=((pw >= 1) & (pw <= T)).astype(int))))
    a = pd.concat(parts, ignore_index=True)
    _, g = np.unique(a["day"].to_numpy(), return_inverse=True)
    return a["r"].to_numpy(), a["p"].to_numpy(), g, tot, unm


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) < 2:
        raise SystemExit(__doc__)
    if not selftest():
        raise SystemExit("\n  SELF-TEST FAILED -- no data was read.")
    print("\nPOWER VALIDATION ON DATA (no real gap is computed or printed)\n")
    r, p, g, tot, unm = load(args[0], args[1:])
    print(f"  stamp: {len(args) - 1} cells from {args[0]} | {tot:,} setups, {unm:,} unmatched "
          f"({100*unm/max(tot,1):.1f}%) excluded | null: day-level swap of real/pseudo | "
          f"test: one-sided, cluster = trade date | seed {POWER_SEED} | ES, train blocks only")
    v = validate(r, p, g)
    report(v)

    # ---- machine-readable receipt (26.28). This script used to serialize NOTHING,
    # so 26.21's PASS existed only as terminal output. gate_receipts.require_pass()
    # re-derives P1/P2 from the integer counts below and P3 from gap_err and n.
    try:
        import hashlib
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gate_receipts
        cells = sorted(args[1:])
        stats = dict(n_draws=v["n_draws"], k_detected=v["k_detected"],
                     k_null_rejected=v["k_null_rejected"], gap_err=float(v["gap_err"]),
                     n=int(v["n"]), D=int(v["D"]), need=float(NEED),
                     fpr_lo=float(FPR_LO), fpr_hi=float(FPR_HI), delta=float(DELTA),
                     n_cells=len(cells),
                     cells_sha=hashlib.sha256("\n".join(cells).encode()).hexdigest()[:16])
        path = gate_receipts.write(
            "power", args[0], "PASS" if v["ok"] else "FAIL", stats,
            extra=dict(seed=POWER_SEED, alpha=ALPHA, setups_total=int(tot),
                       unmatched=int(unm), P1=bool(v["P1"]), P2=bool(v["P2"]),
                       P3=bool(v["P3"])))
        print(f"\n  wrote {path}")
    except Exception as e:
        print(f"\n  [WARN] receipt NOT written: {type(e).__name__}: {e}\n"
              f"         pseudo_gap.py will refuse to run until it exists.")
    sys.exit(0 if v["ok"] else 1)
