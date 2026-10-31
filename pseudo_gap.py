r"""
pseudo_gap.py -- the REAL minus PSEUDO fill-rate gap. Section 25 amendment 3,
clauses (b) to (f). 26.4 step 7.

THIS IS THE QUANTITY THE WHOLE STUDY IS ABOUT, and section 26.3 forbids reading it
until the exchangeability gate (26.19) and the power validation (26.21) both PASS.
That order is enforced mechanically, in ONE place (research rule C3), by the first
statement of main(): gate_receipts.require_pass(dump). There is no override flag.

NOTHING HERE IS A NEW METHOD. Every statistical component is IMPORTED from the
script that already carries it, never reimplemented (E24):
  one_sided_p        <- pseudo_power.  THIS MATTERS: 26.21's power validation
                        certified THIS function at THIS sample size. A different
                        test, however reasonable, would void that certification.
  day_clustered_mean <- pseudo_null_pair (F1 exact: never a divided t).
  registered_cells   <- power_registered_set (the 144 cells section 25's second
                        amendment registers; 4,008,241 setups, 4.650% unmatched).
  CELL_RE            <- pseudo_null_pair, to parse trigger and eps from a cell name.
This file contributes the stratification, the block replication check, the H2 fits
and the verdict ladder -- and nothing else.

THE FILL RULE is pseudo_power.load's, and K1 below PROVES agreement rather than
assuming it: a setup fills when 1 <= w_0.0 <= T (real) or 1 <= pseudo_w_0.0 <= T
(pseudo), on rows with is_real == 1 and pseudo_matched == 1.

ONE REGISTRATION AMBIGUITY, FLAGGED NOT BURIED. Section 25.2 defines the statistic
as "groupby(trade_date).diff.mean() and then a one-sample test across day means",
which is the MEAN OF DAY MEANS. Section 25.3 and amendment 3(d) set the 1.0pp and
5.0pp thresholds against "raw gap", without saying whether that is the mean of day
means or the setup-pooled mean. They differ whenever trade dates carry unequal
setup counts. THIS SCRIPT APPLIES THE VERDICT TO THE MEAN OF DAY MEANS, because
that is the quantity the registered test is performed on and an estimate should
match its own test. The setup-pooled mean is printed beside it on every line, with
their difference, so the choice is visible and reversible. Recorded in
brain/paper_dof_log.md as a degree of freedom, not a finding.

    python pseudo_gap.py --selftest        toy fixtures, reads no dump
    python pseudo_gap.py data\dump         the real thing

Written 2026-10-01. Project root, so outside the code-set hash; it rotates nothing
(26.8, 26.24). Nothing under code\ is edited or needed for this.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gate_receipts import require_pass                                  # noqa: E402
from power_registered_set import registered_cells                       # noqa: E402
from pseudo_power import load as power_load, one_sided_p                # noqa: E402
from pseudo_null_pair import CELL_RE, day_clustered_mean               # noqa: E402

FLOOR_PP = 1.0        # section 25.3; a CONVENTION (amendment 3), not derived
LARGE_PP = 5.0        # annotate only; the verdict does not change
ALPHA = 0.05
NQ = 5                # distance quintiles within cell, amendment 3(b)
VOID_UNMATCHED = 0.10 # gate 3: pooled over the registered set
COLS = ["is_real", "day", "block", "T", "dist_atr", "w_0.0", "pseudo_w_0.0",
        "pseudo_matched"]


# ------------------------------------------------------------------ statistics
def day_codes(day):
    """Contiguous 0..D-1 codes. one_sided_p divides by per-group counts, so every
    group must be non-empty; np.unique guarantees that."""
    _, g = np.unique(np.asarray(day), return_inverse=True)
    return g.astype(np.int64), int(g.max()) + 1 if len(g) else 0


def gap_stats(r, p, day):
    """The registered statistic, both conventions, plus the two one-sided tests.

    Returns pp units throughout. 'daymean' is section 25.2's mean of day means and
    is what the verdict uses; 'pooled' is the setup-weighted mean (F3's raw rate
    view). reverse_p tests pseudo > real and is evaluated FIRST by amendment 3(c).
    """
    d = (np.asarray(r, dtype=np.int64) - np.asarray(p, dtype=np.int64))
    g, D = day_codes(day)
    cnt = np.bincount(g, minlength=D)
    m = np.bincount(g, weights=d, minlength=D) / cnt          # per-day mean diff
    res = day_clustered_mean(100.0 * d, g, D)                 # pooled + cluster SE
    return dict(
        daymean_pp=float(100.0 * m.mean()),
        daymean_se_pp=float(100.0 * m.std(ddof=1) / np.sqrt(D)) if D > 1 else float("nan"),
        pooled_pp=float(res["mean"]), pooled_se_pp=float(res["se"]),
        pooled_lo_pp=float(res["lo"]), pooled_hi_pp=float(res["hi"]),
        fill_real=float(100.0 * np.mean(r)), fill_pseudo=float(100.0 * np.mean(p)),
        n_setups=int(len(d)), n_days=int(D),
        claim_p=float(one_sided_p(d, g, D, "day")),            # real > pseudo
        reverse_p=float(one_sided_p(-d, g, D, "day")),         # pseudo > real
    )


def stratified_gap(df, nq=NQ):
    """Amendment 3(b): setup-weighted mean, over distance quintiles WITHIN cell,
    of the real-minus-pseudo gap. Descriptive; the test stays unstratified."""
    num = den = 0.0
    strata = 0
    for _, sub in df.groupby("cell", sort=False):
        dist = sub["dist_atr"].to_numpy()
        # rank-based quantile cut: ties cannot create empty or overlapping strata
        q = np.floor(stats.rankdata(dist, method="ordinal") * nq / (len(dist) + 1e-9))
        q = np.clip(q, 0, nq - 1).astype(int)
        d = sub["r"].to_numpy(np.int64) - sub["p"].to_numpy(np.int64)
        for k in range(nq):
            msk = q == k
            w = int(msk.sum())
            if w:
                num += w * float(d[msk].mean())
                den += w
                strata += 1
    return (100.0 * num / den if den else float("nan")), strata


def quintile_profile(df, nq=NQ):
    """Section 25 amendment 5(2): DESCRIPTIVE ONLY, no verdict reads it.

    Quintiles are formed WITHIN cell (as 3(b) specifies) and then pooled across
    cells by quintile index, so quintile 0 is "the nearest fifth of each cell's
    setups". Reported per quintile: setups, mean dist_atr, the day-clustered gap
    and its SE. These five numbers CAN move with a distance tilt; the single
    collapsed figure 3(b) registers provably cannot.
    """
    # Indexed assignment, never positional: a positional fill would silently
    # mis-assign quintiles if the rows were ever not contiguous by cell.
    d = df.reset_index(drop=True)
    q_all = np.full(len(d), -1, dtype=int)
    for _, sub in d.groupby("cell", sort=False):
        dist = sub["dist_atr"].to_numpy()
        q = np.floor(stats.rankdata(dist, method="ordinal") * nq / (len(dist) + 1e-9))
        q_all[sub.index.to_numpy()] = np.clip(q, 0, nq - 1).astype(int)
    assert (q_all >= 0).all(), "a row was never assigned a quintile"
    out = []
    for k in range(nq):
        m = q_all == k
        s = d[m]
        st = gap_stats(s["r"].to_numpy(), s["p"].to_numpy(), s["day"].to_numpy())
        out.append((k + 1, int(m.sum()), float(s["dist_atr"].mean()),
                    st["daymean_pp"], st["daymean_se_pp"], st["n_days"]))
    return out, q_all


def h2_fit(df, trigger):
    """Amendment 3(f): fit the gap against eps separately within ATR and within SD.

    Per eps level the gap and its day-clustered SE; then a weighted least squares
    of gap on eps with weights 1/se^2. H2 needs a positive slope at p <= 0.05 AND a
    positive gap at EVERY eps level (section 25.3, unchanged by amendment 3).
    """
    sub = df[df.trigger == trigger]
    rows = []
    for eps, s in sub.groupby("eps", sort=True):
        st = gap_stats(s["r"].to_numpy(), s["p"].to_numpy(), s["day"].to_numpy())
        rows.append((float(eps), st["daymean_pp"], st["daymean_se_pp"],
                     st["n_setups"], st["n_days"]))
    if len(rows) < 3:
        return dict(levels=rows, slope=float("nan"), se=float("nan"),
                    p=float("nan"), all_positive=False, supported=False,
                    note=f"only {len(rows)} eps levels; no slope fitted")
    x = np.array([a for a, _, _, _, _ in rows])
    y = np.array([b for _, b, _, _, _ in rows])
    se = np.array([c for _, _, c, _, _ in rows])
    w = 1.0 / np.maximum(se, 1e-12) ** 2
    xb = np.sum(w * x) / np.sum(w)
    yb = np.sum(w * y) / np.sum(w)
    sxx = np.sum(w * (x - xb) ** 2)
    slope = float(np.sum(w * (x - xb) * (y - yb)) / sxx)
    se_b = float(np.sqrt(1.0 / sxx))
    pval = float(stats.t.sf(slope / se_b, len(rows) - 2))
    allpos = bool(np.all(y > 0))
    return dict(levels=rows, slope=slope, se=se_b, p=pval, all_positive=allpos,
                supported=bool(pval <= ALPHA and slope > 0 and allpos),
                note=f"WLS on {len(rows)} eps levels, weights 1/se^2")


def verdict(st, gA, gB, strat_pp):
    """Amendment 3(c) then 3(d), evaluated in the registered order."""
    gap = st["daymean_pp"]
    if st["reverse_p"] <= ALPHA:
        return ("PSEUDO BEATS REAL -- STOP AND DIAGNOSE",
                f"reverse test one-sided p = {st['reverse_p']:.4f} <= {ALPHA}")
    if st["claim_p"] > ALPHA:
        return ("H0 NOT REJECTED",
                f"claim test one-sided p = {st['claim_p']:.4f} > {ALPHA}")
    if gap < FLOOR_PP:
        return ("REAL BUT NEGLIGIBLE",
                f"p = {st['claim_p']:.4f} but the gap {gap:+.4f} pp is under the "
                f"{FLOOR_PP} pp floor")
    if not (gA > 0 and gB > 0):
        return ("NOT REPLICATED",
                f"gap is {gA:+.4f} pp in block A and {gB:+.4f} pp in block B; "
                f"amendment 3(d) requires both positive")
    if strat_pp < FLOOR_PP:
        return ("UNRESOLVED (distance confound)",
                f"distance-stratified gap {strat_pp:+.4f} pp is under the "
                f"{FLOOR_PP} pp floor while the raw gap is {gap:+.4f} pp")
    tag = "  [LARGE: above the 5.0 pp flag; verdict unchanged]" if gap > LARGE_PP else ""
    return ("H1 SUPPORTED" + tag,
            f"p = {st['claim_p']:.4f}, gap {gap:+.4f} pp, positive in both blocks, "
            f"stratified {strat_pp:+.4f} pp")


# ------------------------------------------------------------------------ data
def load_rich(dump, cells):
    """Like pseudo_power.load but keeping block, distance, trigger and eps.
    K1 proves it agrees with pseudo_power.load on r, p and the day grouping."""
    parts, tot, unm = [], 0, 0
    per_cell = []
    for c in cells:
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=COLS)
        x = x[x.is_real == 1]
        n_all = len(x)
        n_unm = int((x.pseudo_matched != 1).sum())
        tot += n_all
        unm += n_unm
        per_cell.append((c, n_all, n_unm))
        x = x[x.pseudo_matched == 1]
        T = x["T"].to_numpy()
        w, pw = x["w_0.0"].to_numpy(), x["pseudo_w_0.0"].to_numpy()
        m = CELL_RE.search(c)
        parts.append(pd.DataFrame(dict(
            cell=c, trigger=m.group(3), eps=float(m.group(4)),
            day=x["day"].to_numpy(), block=x["block"].to_numpy(),
            dist_atr=x["dist_atr"].to_numpy(),
            r=((w >= 1) & (w <= T)).astype(np.int64),
            p=((pw >= 1) & (pw <= T)).astype(np.int64))))
    return pd.concat(parts, ignore_index=True), tot, unm, per_cell


# -------------------------------------------------------------------- selftest
def _synth(D, per_day, gap_pp, seed, base=0.50, dist_effect=0.0, blockB_gap=None,
           day_sd=0.0, ncell=1):
    """D days x per_day setups, arms drawn INDEPENDENTLY at base -+ gap/2 so the
    planted gap is realised EXACTLY in expectation.

    The first version of this helper flipped r to 1 only where p was already 0,
    which halved every planted gap, and at gap 0 made r and p identical so the
    t statistic was 0/0. Both were defects in the FIXTURE, not in the code under
    test -- the self-test caught them, which is the point of having one.

    day_sd adds a per-day shift so the day-clustered test has something to protect
    against; dist_effect tilts the gap with distance; ncell splits the rows across
    cells so stratification has more than one group to work with.
    """
    rng = np.random.default_rng(seed)
    n = D * per_day
    day = np.repeat(np.arange(D), per_day)
    block = np.where(day < D // 2, "A", "B")
    dist = rng.uniform(1.0, 5.0, n)
    g = np.full(n, gap_pp / 100.0)
    if blockB_gap is not None:
        g = np.where(block == "B", blockB_gap / 100.0, g)
    if dist_effect:
        g = g + dist_effect / 100.0 * (dist - dist.mean())
    if day_sd:
        g = g + (rng.normal(0.0, day_sd / 100.0, D))[day]
    pr_r = np.clip(base + g / 2.0, 0.0, 1.0)
    pr_p = np.clip(base - g / 2.0, 0.0, 1.0)
    r = (rng.random(n) < pr_r).astype(np.int64)
    p = (rng.random(n) < pr_p).astype(np.int64)
    # UNEQUAL cell sizes on purpose. With equal-sized cells every (cell, quintile)
    # stratum holds the same count, so an equal-weight mean over strata coincides
    # with the setup-weighted one and M9b cannot fail for the wrong reason.
    edges = np.linspace(0.0, 1.0, ncell + 1)[1:-1] ** 2 if ncell > 1 else []
    cell = np.array([f"N60_toy{k}_ATR_3.0"
                     for k in np.digitize(np.arange(n) / n, edges)])
    return pd.DataFrame(dict(cell=cell, trigger="ATR", eps=3.0,
                             day=day, block=block, dist_atr=dist, r=r, p=p))


def stratified_gap_equalweight(df, nq=NQ):
    """NOT REGISTERED. A diagnostic, printed only so the identity below is visible.

    Amendment 3(b)'s stratified gap is SETUP-WEIGHTED, and a setup-weighted mean
    over an exhaustive partition is algebraically the pooled mean -- so it cannot
    differ from it, ever. This variant weights every stratum EQUALLY, which can
    differ, and is therefore the thing that would actually reveal a distance tilt.
    It has no standing under section 25 and no verdict depends on it.
    """
    vals = []
    for _, sub in df.groupby("cell", sort=False):
        dist = sub["dist_atr"].to_numpy()
        q = np.floor(stats.rankdata(dist, method="ordinal") * nq / (len(dist) + 1e-9))
        q = np.clip(q, 0, nq - 1).astype(int)
        d = sub["r"].to_numpy(np.int64) - sub["p"].to_numpy(np.int64)
        for k in range(nq):
            msk = q == k
            if msk.sum():
                vals.append(float(d[msk].mean()))
    return 100.0 * float(np.mean(vals)) if vals else float("nan")


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\npseudo_gap.py SELF-TEST -- every detector must be shown ALIVE before any\n"
          "real gap is read (26.0; the ordering rule that slipped at 26.16 and 26.18).\n")

    # K1 -- the statistic agrees with the function 26.21 certified.
    # n is large here on purpose: the arms are drawn INDEPENDENTLY, so Var(r - p)
    # is about 0.50 against roughly 0.0005 in the real paired data. At 12,000 rows
    # the fixture's own SE is 0.65 pp and a 0.5 pp tolerance fails on noise alone,
    # which is what the first run of this self-test did.
    df = _synth(500, 400, 2.0, seed=11)
    st = gap_stats(df.r.to_numpy(), df.p.to_numpy(), df.day.to_numpy())
    g, D = day_codes(df.day.to_numpy())
    ref = one_sided_p(df.r.to_numpy() - df.p.to_numpy(), g, D, "day")
    chk("K1 claim_p is exactly pseudo_power.one_sided_p (the certified test)",
        abs(st["claim_p"] - ref) < 1e-15, f"{st['claim_p']:.6g}")

    chk("K2 a planted +2.0 pp gap is recovered within 0.5 pp",
        abs(st["daymean_pp"] - 2.0) < 0.5, f"estimated {st['daymean_pp']:+.4f} pp")
    chk("K3 and it is detected", st["claim_p"] <= ALPHA, f"p = {st['claim_p']:.5f}")
    chk("K4 the reverse test does NOT fire on a positive gap",
        st["reverse_p"] > ALPHA, f"reverse p = {st['reverse_p']:.4f}")

    z = _synth(400, 30, 0.0, seed=12)
    sz = gap_stats(z.r.to_numpy(), z.p.to_numpy(), z.day.to_numpy())
    chk("K5 a true-zero gap is NOT detected", sz["claim_p"] > ALPHA,
        f"p = {sz['claim_p']:.4f}, gap {sz['daymean_pp']:+.4f} pp")

    chk("K6 both gap conventions agree when days are balanced",
        abs(st["daymean_pp"] - st["pooled_pp"]) < 1e-9,
        f"daymean {st['daymean_pp']:+.4f} vs pooled {st['pooled_pp']:+.4f}")

    sg, nstr = stratified_gap(df)
    chk("K7 THE REGISTERED STRATIFIED GAP IS ALGEBRAICALLY THE POOLED GAP "
        "(see the flag)", abs(sg - st["pooled_pp"]) < 1e-9,
        f"stratified {sg:+.10f} vs pooled {st['pooled_pp']:+.10f} pp -- identical")
    chk("K8 quintiles partition every cell exactly", nstr == NQ, f"{nstr} strata")

    prof, qa = quintile_profile(df)
    chk("K9 the quintile profile covers every row exactly once, balanced",
        sum(p[1] for p in prof) == len(df) and (qa >= 0).all()
        and max(p[1] for p in prof) - min(p[1] for p in prof) <= len(df) // NQ,
        f"counts {[p[1] for p in prof]} sum to {sum(p[1] for p in prof):,} of {len(df):,}")
    chk("K10 the profile is ordered by distance, near quintile to far",
        all(prof[i][2] < prof[i + 1][2] for i in range(NQ - 1)),
        "mean dist_atr " + " < ".join(f"{p[2]:.2f}" for p in prof))

    # M -- planted faults, each must change the verdict
    rev = _synth(500, 400, -2.0, seed=13)
    sr = gap_stats(rev.r.to_numpy(), rev.p.to_numpy(), rev.day.to_numpy())
    v, _ = verdict(sr, -2.0, -2.0, -2.0)
    chk("M1 pseudo beating real triggers STOP AND DIAGNOSE, before anything else",
        v.startswith("PSEUDO BEATS REAL"), f"reverse p = {sr['reverse_p']:.2g}")

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=0.4), 0.4, 0.4, 0.4)
    chk("M2 a significant but sub-floor gap is REAL BUT NEGLIGIBLE",
        v == "REAL BUT NEGLIGIBLE", v)

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=2.0), 3.0, -0.5, 2.0)
    chk("M3 a gap negative in block B is NOT REPLICATED", v == "NOT REPLICATED", v)

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=2.0), 2.0, 2.0, 0.3)
    chk("M4 a raw gap killed by distance stratification is UNRESOLVED",
        v.startswith("UNRESOLVED"), v)

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=2.0), 2.0, 2.0, 2.0)
    chk("M5 and only the clean case is H1 SUPPORTED", v == "H1 SUPPORTED", v)

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=6.0), 6.0, 6.0, 6.0)
    chk("M6 a gap over 5.0 pp is annotated LARGE, verdict unchanged",
        v.startswith("H1 SUPPORTED") and "LARGE" in v, v)

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.20, daymean_pp=2.0), 2.0, 2.0, 2.0)
    chk("M7 a large gap with p > 0.05 is still H0 NOT REJECTED",
        v == "H0 NOT REJECTED", v)

    # M8 -- with a real day-level effect, ignoring the clustering must be more
    # liberal, or F1/F2 are decorative and the clustered test is doing nothing
    # Compared as STANDARD ERRORS, not p-values: a p-value comparison flips sign
    # when the realized mean happens to be negative, which made this check fail
    # for a reason that had nothing to do with clustering.
    cl = _synth(120, 400, 0.0, seed=14, day_sd=3.0)
    dd = cl.r.to_numpy() - cl.p.to_numpy()
    gg, DD = day_codes(cl.day.to_numpy())
    cnt = np.bincount(gg, minlength=DD)
    mday = np.bincount(gg, weights=dd, minlength=DD) / cnt
    se_day = mday.std(ddof=1) / np.sqrt(DD)
    se_row = dd.std(ddof=1) / np.sqrt(len(dd))
    chk("M8 under a day-level effect the naive setup-level SE UNDERSTATES the "
        "day-clustered one", se_row < se_day,
        f"setup-level {100*se_row:.4f} pp vs day-clustered {100*se_day:.4f} pp "
        f"= {se_day/se_row:.2f}x  (HANDOFF.md:319 measured 1.7-3x)")

    # M9 -- a planted distance confound. The REGISTERED statistic cannot see it;
    # the equal-weight variant can. This is the flagged flaw, as a live detector.
    cf = _synth(400, 30, 0.0, seed=15, dist_effect=6.0, ncell=4)
    sc = gap_stats(cf.r.to_numpy(), cf.p.to_numpy(), cf.day.to_numpy())
    sgc, _ = stratified_gap(cf)
    sge = stratified_gap_equalweight(cf)
    chk("M9a the REGISTERED stratified gap is blind to a planted distance tilt "
        "(it equals the pooled gap by construction)",
        abs(sgc - sc["pooled_pp"]) < 1e-9,
        f"pooled {sc['pooled_pp']:+.6f} == stratified {sgc:+.6f} pp")
    chk("M9b the unregistered equal-weight variant is NOT blind to it",
        abs(sge - sc["pooled_pp"]) > 1e-6,
        f"equal-weight {sge:+.6f} vs pooled {sc['pooled_pp']:+.6f} pp")

    # M10 -- H2 refuses to be supported when a level is negative
    lv = pd.concat([_synth(200, 20, gp, seed=20 + i).assign(eps=e, trigger="ATR")
                    for i, (e, gp) in enumerate([(3.0, -1.0), (4.0, 1.0),
                                                 (6.0, 2.0), (8.0, 3.0)])],
                   ignore_index=True)
    h = h2_fit(lv, "ATR")
    chk("M10 H2 is NOT supported when any eps level has a negative gap",
        not h["supported"] and not h["all_positive"],
        f"slope {h['slope']:+.4f}, p {h['p']:.4f}, all_positive {h['all_positive']}")

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


def preflight(dump, cells, k=6):
    """Known-number check against REAL data, before any gap is computed: this
    file's loader must reproduce pseudo_power.load's r, p and day grouping
    exactly on the first k registered cells. Compares as integers (E02)."""
    sub = cells[:k]
    df, _, _, _ = load_rich(dump, sub)
    r2, p2, g2, tot2, unm2 = power_load(dump, sub)
    g1, _ = day_codes(df["day"].to_numpy())
    ok = (len(df) == len(r2)
          and int(np.sum(df["r"].to_numpy() != r2)) == 0
          and int(np.sum(df["p"].to_numpy() != p2)) == 0
          and int(np.sum(g1 != g2)) == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] K0 this loader reproduces "
          f"pseudo_power.load on {len(sub)} real cells   "
          f"{len(df):,} rows, {int(np.sum(df['r'].to_numpy() != r2))} r-diffs, "
          f"{int(np.sum(df['p'].to_numpy() != p2))} p-diffs, "
          f"{int(np.sum(g1 != g2))} day-code diffs")
    return ok


# ------------------------------------------------------------------------ main
def main(dump):
    # 1. THE ENFORCEMENT POINT. Nothing below runs unless both receipts verify.
    stamp = require_pass(dump)

    # 2. Detectors before data (26.0).
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no real data was read.")

    cells = registered_cells()
    print(f"REGISTERED SET: {len(cells)} cells\n")
    if not preflight(dump, cells):
        raise SystemExit("K0 FAILED -- this loader disagrees with the certified "
                         "one. No gap was computed.")

    # 3. Only now is real data read for a result.
    df, tot, unm, per_cell = load_rich(dump, cells)
    frac = unm / max(tot, 1)
    worst = max(per_cell, key=lambda t: t[2] / max(t[1], 1))
    print(f"\n  (e) unmatched, POOLED over the registered set: {unm:,} of {tot:,} "
          f"= {100 * frac:.3f}%   (void at {100 * VOID_UNMATCHED:.0f}%)")
    print(f"      worst cell, REPORTED not gated: {worst[0]} at "
          f"{100 * worst[2] / max(worst[1], 1):.2f}%")
    if frac > VOID_UNMATCHED:
        raise SystemExit("VOID: pooled unmatched exceeds 10% (gate 3).")

    st = gap_stats(df.r.to_numpy(), df.p.to_numpy(), df.day.to_numpy())
    sgap, nstr = stratified_gap(df)
    prof, _ = quintile_profile(df)
    blk = {}
    for b in ("A", "B"):
        s = df[df.block == b]
        blk[b] = gap_stats(s.r.to_numpy(), s.p.to_numpy(), s.day.to_numpy())["daymean_pp"] \
            if len(s) else float("nan")

    print("\nSTAMP: real-minus-pseudo fill-rate gap (section 25 amendment 3 b-f) | "
          f"{len(cells)} cells | {st['n_setups']:,} matched setups of {tot:,} | "
          f"independent units: {st['n_days']:,} trade dates | clustered on trade "
          f"date | tau 0.0, wick fill within T | ON THE SETUPS MATCHED IN THE "
          f"STORED DRAW | code-set {stamp['gate']['code_set']} | dump {dump} | "
          f"ES, train blocks A+B, sealed 2018-2022 untouched | raw, no cost modelled")

    print(f"\n  fill rate   real {st['fill_real']:7.3f}%   pseudo {st['fill_pseudo']:7.3f}%")
    print(f"  GAP (mean of day means, the registered statistic and the one the "
          f"verdict uses)\n      {st['daymean_pp']:+.4f} pp   SE {st['daymean_se_pp']:.4f}")
    print(f"  gap (setup-pooled, F3's raw-rate view)\n"
          f"      {st['pooled_pp']:+.4f} pp   SE {st['pooled_se_pp']:.4f}   "
          f"95% CI [{st['pooled_lo_pp']:+.4f}, {st['pooled_hi_pp']:+.4f}]")
    print(f"      difference between the two conventions: "
          f"{st['daymean_pp'] - st['pooled_pp']:+.4f} pp")
    print(f"  (b) distance-stratified gap, {NQ} quintiles within each of "
          f"{len(cells)} cells ({nstr} strata): {sgap:+.4f} pp")
    print(f"      AMENDMENT 5(1): this is ALGEBRAICALLY the pooled gap "
          f"({st['pooled_pp']:+.4f} pp, difference {sgap - st['pooled_pp']:+.2e}).")
    print(f"      A setup-weighted mean over an exhaustive partition IS the pooled")
    print(f"      mean, so it cannot differ. Reported as registered, not relied on.")
    print(f"      block A {blk['A']:+.4f} pp | block B {blk['B']:+.4f} pp")
    print(f"  (2) QUINTILE PROFILE (amendment 5(2), DESCRIPTIVE -- no verdict reads it)")
    print(f"      {'q':>2}  {'setups':>10}  {'mean dist_atr':>13}  {'gap pp':>9}  "
          f"{'SE':>7}  {'dates':>6}")
    for k, n, md, g_, se_, nd in prof:
        print(f"      {k:>2}  {n:>10,}  {md:>13.3f}  {g_:>+9.4f}  {se_:>7.4f}  {nd:>6,}")
    print(f"  (c) reverse test (pseudo > real), evaluated FIRST: p = {st['reverse_p']:.4f}")
    print(f"      claim  test (real > pseudo)                  : p = {st['claim_p']:.4f}")

    v, why = verdict(st, blk["A"], blk["B"], sgap)
    print(f"\n  (d) VERDICT: {v}\n      {why}")

    if v.startswith("PSEUDO BEATS REAL"):
        print("\n  Amendment 3(c): STOP. Do not interpret. Something is inverted "
              "in the build.")
        return v

    print("\n  (f) H2, fitted SEPARATELY within ATR and within SD (eps has "
          "different units).\n      Section 25 amendment 4: a positive slope is "
          "NOT independent support -- Grant-Wolf-Yu\n      (2005) document "
          "reversion growing with excursion size in this market, no VWAP involved.")
    for trig in ("ATR", "SD"):
        h = h2_fit(df, trig)
        print(f"    {trig}: slope {h['slope']:+.5f} pp per eps unit, "
              f"SE {h['se']:.5f}, one-sided p {h['p']:.4f} | {h['note']}")
        for e, y, se, n, D in h["levels"]:
            print(f"        eps {e:<5} gap {y:+8.4f} pp  SE {se:.4f}  "
                  f"n {n:,} over {D:,} dates")
        print(f"      all levels positive: {h['all_positive']}   -> H2 "
              f"{'SUPPORTED' if h['supported'] else 'NOT SUPPORTED'}")
    return v


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    main(args[0])
