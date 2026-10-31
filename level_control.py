r"""
level_control.py -- section 27 TEST A: is a VWAP LEVEL special, or is it the moment?

THE QUESTION. Section 25 settled that the ANCHOR is decoration: a VWAP anchored at
an arbitrary bar in the same window fills as often as one anchored at the extreme.
It never tested VWAP-NESS, because both arms were VWAPs (26.48). This does: a frozen
anchored-VWAP level against a MATCHED RANDOM LEVEL at the same ATR distance on the
same side, placed at a matched other moment (control_matched.py's M1b twins).

WHY IT MATTERS NOW. HANDOFF.md:242 puts the level effect at +3.8 pp -- "the level
itself" -- and HANDOFF.md:154 flags that whole section as OLD-RULE, measured when
the same episode was counted 3.26 times over. IT HAS NEVER BEEN RE-MEASURED. The
project's central claim rests on it.

WHAT THIS TEST CANNOT SUFFER FROM, and it is worth stating because the last one did:
dist_atr is IDENTICAL between the arms by construction (measured: 3.4041 both on
N60_hh_ATR_3.0). There is no redraw cap, no acceptance band and no first-hit bias,
so the asymmetry artefact that produced section 25's inversion CANNOT arise here.
ITS OWN CONFOUND IS DIFFERENT AND IS STATED, NOT ARGUED AWAY: the twin sits at a
DIFFERENT MOMENT, matched on hour bucket x ATR quintile (M1b). Section 25 held the
moment fixed and moved the anchor; this holds the line fixed in one arm and moves
both line and moment. The two are reported SIDE BY SIDE, never one instead of the
other.

THE PAIRING IS REAL, not assumed: every twin's src_bar matches a setup's bar
(63,186 of 63,186 on the test cell) and every setup has EXACTLY 2 twins.

NOTHING STATISTICAL IS REIMPLEMENTED (E24): one_sided_p from pseudo_power -- the
function 26.21's power validation certified at this sample size -- and
day_clustered_mean from pseudo_null_pair. gap_stats is NOT reused, because it casts
both arms to int64 and the paired twin mean is FRACTIONAL (0, 0.5 or 1 with two
twins); that cast would silently floor every 0.5 to 0.

    python level_control.py --selftest      toy fixtures, reads no dump
    python level_control.py data\dump       TEST A

Written 2026-10-02. Project root; rotates nothing. Reads only.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from power_registered_set import registered_cells                      # noqa: E402
from pseudo_null_pair import day_clustered_mean                        # noqa: E402
from pseudo_power import one_sided_p                                   # noqa: E402
from pseudo_gap import day_codes                                       # noqa: E402

FLOOR_PP, ALPHA = 1.0, 0.05            # section 27.1, section 25's convention
PUB_REAL_FILL = 51.573                 # 26.34, for the K0 known-number check
COLS = ["is_real", "bar", "src_bar", "day", "block", "T", "w_0.0", "pseudo_matched"]

# POPULATION. TEST A runs on ALL setups of the 144 registered cells, not on the
# matched subset section 25 was restricted to. The M1b twins exist for EVERY setup
# and do not depend on the pseudo draw at all, so TEST A is FREE of the matchability
# selection F09 flagged (unmatched setups fill 4.0 pp less than matched ones).
# That is an advantage, but it means the populations differ, so the matched-only
# subset is ALSO reported for a like-for-like comparison with section 25 (A2).
# The first version of K0b compared the ALL-setups rate against 26.34's
# matched-only 51.573% and correctly FAILED at 51.297% -- the check caught a
# population mismatch before any result was read, which is what it is for.


def paired_stats(d, day):
    """Day-clustered statistics for a PAIRED, possibly FRACTIONAL difference.

    Mirrors pseudo_gap.gap_stats but without its int64 cast on the two arms: here
    the twin arm is the MEAN of two binary outcomes, so it takes the values 0, 0.5
    and 1, and casting would floor every 0.5 to 0 and bias the whole test.
    """
    d = np.asarray(d, dtype=float)
    g, D = day_codes(day)
    cnt = np.bincount(g, minlength=D)
    m = np.bincount(g, weights=d, minlength=D) / cnt
    res = day_clustered_mean(100.0 * d, g, D)
    return dict(
        daymean_pp=float(100.0 * m.mean()),
        daymean_se_pp=float(100.0 * m.std(ddof=1) / np.sqrt(D)) if D > 1 else float("nan"),
        pooled_pp=float(res["mean"]), pooled_se_pp=float(res["se"]),
        lo_pp=float(res["lo"]), hi_pp=float(res["hi"]),
        n=int(len(d)), D=int(D),
        claim_p=float(one_sided_p(d, g, D, "day")),      # real > twin
        reverse_p=float(one_sided_p(-d, g, D, "day")),   # twin > real
    )


def load_pairs(dump, cells, m2=False):
    """One row per SETUP: its fill, the mean fill of its twins, its day and block."""
    parts, n_bad = [], 0
    for c in cells:
        main = os.path.join(dump, c + ".parquet")
        x = pd.read_parquet(main, columns=COLS)
        if m2:
            tw = pd.read_parquet(os.path.join(dump, c + ".m2.parquet"),
                                 columns=["src_bar", "T", "w_0.0"])
            r = x[x.is_real == 1]
        else:
            r = x[x.is_real == 1]
            tw = x[x.is_real == 0][["src_bar", "T", "w_0.0"]]
        rf = ((r["w_0.0"].to_numpy() >= 1) & (r["w_0.0"].to_numpy() <= r["T"].to_numpy())
              ).astype(float)
        tf = ((tw["w_0.0"].to_numpy() >= 1) & (tw["w_0.0"].to_numpy() <= tw["T"].to_numpy())
              ).astype(float)
        tm = pd.DataFrame(dict(src_bar=tw["src_bar"].to_numpy(), f=tf)) \
               .groupby("src_bar")["f"].agg(["mean", "size"])
        j = pd.DataFrame(dict(cell=c, bar=r["bar"].to_numpy(), day=r["day"].to_numpy(),
                              block=r["block"].to_numpy(), real=rf,
                              matched=r["pseudo_matched"].to_numpy())).join(
            tm, on="bar")
        n_bad += int(j["mean"].isna().sum())
        j = j[j["mean"].notna()]
        parts.append(j.rename(columns={"mean": "twin", "size": "k"}))
    out = pd.concat(parts, ignore_index=True)
    return out, n_bad


def grid_tables(df, d, title):
    """DESCRIPTIVE breakdown by anchor type, lookback N, and eps x BLOCK.

    No verdict reads any of this and nothing here is registered -- it is a cut of
    a result already registered and read (section 27.3). It exists because the
    pooled figure is NOT a typical cell: section 25 amendment 2 records the
    registered set as dominated by small-eps cells (ATR 3.0/4.0 and SD 1.0/1.5 hold
    83.5% of setups), and the effect is largest exactly there, so the setup-weighted
    pooled number leans toward the strongest corner.
    eps x BLOCK is included because the eps table and the block line are ORTHOGONAL
    cuts -- pooling A and B inside every eps row hides the recent-window profile,
    which is the one a strategy inherits.
    """
    from pseudo_null_pair import CELL_RE
    g = df.copy()
    g["d"] = np.asarray(d, dtype=float)
    meta = g["cell"].map(lambda c: CELL_RE.search(c).groups())
    g["N"] = [int(m[0]) for m in meta]
    g["anchor"] = [m[1] for m in meta]
    g["trigger"] = [m[2] for m in meta]
    g["eps"] = [float(m[3]) for m in meta]

    def line(sub):
        if not len(sub):
            return None
        s = paired_stats(sub["d"].to_numpy(), sub["day"].to_numpy())
        out = {"all": s["daymean_pp"], "se": s["daymean_se_pp"], "n": s["n"]}
        for b in ("A", "B"):
            q = sub[sub.block == b]
            out[b] = (paired_stats(q["d"].to_numpy(), q["day"].to_numpy())["daymean_pp"]
                      if len(q) else float("nan"))
        return out

    print(f"\n  {title}")
    for key, name in (("anchor", "ANCHOR TYPE"), ("N", "LOOKBACK N")):
        print(f"\n    by {name:<12} {'pooled':>9} {'SE':>7} {'block A':>9} "
              f"{'block B':>9} {'setups':>12}")
        for v, sub in g.groupby(key, sort=True):
            r = line(sub)
            print(f"      {str(v):<20} {r['all']:>+9.4f} {r['se']:>7.4f} "
                  f"{r['A']:>+9.4f} {r['B']:>+9.4f} {r['n']:>12,}")
    print(f"\n    by eps x BLOCK (the cut a strategy inherits)")
    print(f"      {'trigger eps':<14} {'pooled':>9} {'block A':>9} {'block B':>9} "
          f"{'setups':>12}")
    for (tg, e), sub in g.groupby(["trigger", "eps"], sort=True):
        r = line(sub)
        print(f"      {tg + ' ' + str(e):<14} {r['all']:>+9.4f} {r['A']:>+9.4f} "
              f"{r['B']:>+9.4f} {r['n']:>12,}")


def verdict(st, gA, gB):
    """Section 27.1's ladder, reverse test FIRST."""
    gap = st["daymean_pp"]
    if st["reverse_p"] <= ALPHA:
        return ("TWIN BEATS VWAP -- STOP AND DIAGNOSE",
                f"reverse one-sided p = {st['reverse_p']:.4f} <= {ALPHA}")
    if st["claim_p"] > ALPHA:
        return ("H0a NOT REJECTED -- the level is not special",
                f"claim one-sided p = {st['claim_p']:.4f} > {ALPHA}")
    if gap < FLOOR_PP:
        return ("REAL BUT NEGLIGIBLE",
                f"p = {st['claim_p']:.4g} but the gap {gap:+.4f} pp is under the "
                f"{FLOOR_PP} pp floor")
    if not (gA > 0 and gB > 0):
        return ("NOT REPLICATED",
                f"gap is {gA:+.4f} pp in block A and {gB:+.4f} pp in block B; "
                f"section 27.1 requires both positive")
    return ("H1a SUPPORTED",
            f"p = {st['claim_p']:.4g}, gap {gap:+.4f} pp, positive in both blocks")


# -------------------------------------------------------------------- selftest
def _toy(D=600, per_day=120, gap_pp=3.0, seed=4, base=0.42, blockB_gap=None):
    """D days x per_day setups, each with TWO twins. The twin arm is the mean of
    two Bernoullis, so it takes 0, 0.5 and 1 -- the fractional case the int64 cast
    in gap_stats would destroy."""
    rng = np.random.default_rng(seed)
    n = D * per_day
    day = np.repeat(np.arange(D), per_day)
    block = np.where(day < D // 2, "A", "B")
    g = np.full(n, gap_pp / 100.0)
    if blockB_gap is not None:
        g = np.where(block == "B", blockB_gap / 100.0, g)
    real = (rng.random(n) < np.clip(base + g, 0, 1)).astype(float)
    t1 = (rng.random(n) < base).astype(float)
    t2 = (rng.random(n) < base).astype(float)
    return pd.DataFrame(dict(day=day, block=block, real=real, twin=(t1 + t2) / 2.0))


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\nlevel_control.py SELF-TEST -- detectors before any dump is read.\n")

    df = _toy(gap_pp=3.0)
    st = paired_stats(df.real - df.twin, df.day.to_numpy())
    chk("K1 a planted +3.0 pp level effect is recovered within 0.5 pp",
        abs(st["daymean_pp"] - 3.0) < 0.5, f"{st['daymean_pp']:+.4f} pp")
    chk("K2 and it is detected", st["claim_p"] <= ALPHA, f"p = {st['claim_p']:.3g}")
    chk("K3 the reverse test does NOT fire on a positive gap",
        st["reverse_p"] > ALPHA, f"reverse p = {st['reverse_p']:.4f}")

    z = _toy(gap_pp=0.0, seed=5)
    sz = paired_stats(z.real - z.twin, z.day.to_numpy())
    chk("K4 a true-zero effect is NOT detected", sz["claim_p"] > ALPHA,
        f"p = {sz['claim_p']:.4f}, gap {sz['daymean_pp']:+.4f} pp")

    # THE CAST GUARD -- the reason gap_stats is not reused
    frac = np.array([0.5, 0.5, 0.5, 0.5])
    chk("K5 THE CAST GUARD: a fractional twin arm survives; int64 would floor it",
        abs(float(np.mean(1.0 - frac)) - 0.5) < 1e-12
        and abs(float(np.mean(1 - frac.astype(np.int64))) - 1.0) < 1e-12,
        "mean(1 - 0.5) = 0.5 correctly, but 1 - int64(0.5) = 1.0 would double it")

    n = _toy(gap_pp=-3.0, seed=6)
    sn = paired_stats(n.real - n.twin, n.day.to_numpy())
    v, _ = verdict(sn, -3.0, -3.0)
    chk("M1 twins beating the VWAP triggers STOP AND DIAGNOSE, before anything else",
        v.startswith("TWIN BEATS VWAP"), f"reverse p = {sn['reverse_p']:.3g}")

    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=0.4), 0.4, 0.4)
    chk("M2 a significant but sub-floor effect is REAL BUT NEGLIGIBLE",
        v == "REAL BUT NEGLIGIBLE", v)
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=3.0), 4.0, -0.5)
    chk("M3 an effect negative in block B is NOT REPLICATED", v == "NOT REPLICATED", v)
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=3.0), 3.0, 3.0)
    chk("M4 and only the clean case is H1a SUPPORTED", v == "H1a SUPPORTED", v)
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.30, daymean_pp=3.0), 3.0, 3.0)
    chk("M5 a large effect with p > 0.05 is still H0a NOT REJECTED",
        v.startswith("H0a NOT REJECTED"), v)

    # M6 -- the pairing must matter: shuffling the twin arm across setups should
    # not change the MEAN (it is the same numbers) but must change the day pattern
    rng = np.random.default_rng(1)
    sh = paired_stats(df.real.to_numpy() - rng.permutation(df.twin.to_numpy()),
                      df.day.to_numpy())
    chk("M6 shuffling the twin arm preserves the mean but changes its SE, so the "
        "pairing is doing something",
        abs(sh["daymean_pp"] - st["daymean_pp"]) < 0.25
        and abs(sh["daymean_se_pp"] - st["daymean_se_pp"]) > 1e-6,
        f"gap {sh['daymean_pp']:+.4f} vs {st['daymean_pp']:+.4f}, "
        f"SE {sh['daymean_se_pp']:.4f} vs {st['daymean_se_pp']:.4f}")

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


# ------------------------------------------------------------------------ main
def main(dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    cells = registered_cells()
    print(f"SECTION 27 TEST A -- the level control. {len(cells)} registered cells "
          f"from {dump}\n")

    df, n_bad = load_pairs(dump, cells, m2=False)
    ks = df["k"].value_counts().to_dict()
    print(f"  [{'PASS' if set(ks) == {2} else 'FAIL'}] K0a every setup has EXACTLY "
          f"2 M1b twins   {ks}, {n_bad} setups unpaired")
    # K0b is checked on the MATCHED SUBSET, because that is the population 26.34
    # measured. The primary test below runs on ALL setups -- see the POPULATION
    # note at the top.
    mm = df[df.matched == 1]
    real_rate = 100.0 * float(mm.real.mean())
    k0b = abs(real_rate - PUB_REAL_FILL) < 0.05
    print(f"  [{'PASS' if k0b else 'FAIL'}] K0b the real fill rate on the MATCHED "
          f"subset reproduces 26.34   {real_rate:.3f}% vs {PUB_REAL_FILL}%   "
          f"({len(mm):,} of {len(df):,} setups)")
    print(f"         (all setups: {100*df.real.mean():.3f}% -- a DIFFERENT "
          f"population, lower as F09 predicts, and the one TEST A uses)")
    if not (set(ks) == {2} and k0b):
        raise SystemExit("A known-number check FAILED -- no result was read.")

    d = df.real.to_numpy() - df.twin.to_numpy()
    st = paired_stats(d, df.day.to_numpy())
    blk = {}
    for b in ("A", "B"):
        s = df[df.block == b]
        blk[b] = paired_stats(s.real.to_numpy() - s.twin.to_numpy(),
                              s.day.to_numpy())["daymean_pp"] if len(s) else float("nan")

    print(f"\nSTAMP: section 27 TEST A, VWAP level vs matched random level (M1b "
          f"twins) | {len(cells)} cells | {st['n']:,} setups, {2*st['n']:,} twins | "
          f"independent units: {st['D']:,} trade dates | paired via src_bar, "
          f"clustered on the SETUP's trade date | tau 0.0, wick fill within T | "
          f"distance matched BY CONSTRUCTION | dump {dump} | ES, train blocks A+B, "
          f"sealed 2018-2022 untouched | raw, no cost modelled")
    print(f"\n  fill rate   VWAP {100*df.real.mean():7.3f}%   "
          f"matched random level {100*df.twin.mean():7.3f}%")
    print(f"  GAP (mean of day means, the registered statistic)")
    print(f"      {st['daymean_pp']:+.4f} pp   SE {st['daymean_se_pp']:.4f}")
    print(f"  gap (setup-pooled)  {st['pooled_pp']:+.4f} pp   "
          f"95% CI [{st['lo_pp']:+.4f}, {st['hi_pp']:+.4f}]")
    print(f"      block A {blk['A']:+.4f} pp | block B {blk['B']:+.4f} pp")
    print(f"  reverse test (twin > VWAP), evaluated FIRST: p = {st['reverse_p']:.4g}")
    print(f"  claim   test (VWAP > twin)                 : p = {st['claim_p']:.4g}")

    v, why = verdict(st, blk["A"], blk["B"])
    print(f"\n  VERDICT: {v}\n      {why}")

    sm = paired_stats(mm.real.to_numpy() - mm.twin.to_numpy(), mm.day.to_numpy())
    print(f"\n  LIKE-FOR-LIKE WITH SECTION 25 (matched subset only, {sm['n']:,} "
          f"setups):\n      {sm['daymean_pp']:+.4f} pp   SE {sm['daymean_se_pp']:.4f}"
          f"   p {sm['claim_p']:.4g}")
    print(f"      Reported so this and section 25's null differ in ONE field (the "
          f"control),\n      not two (A2). The PRIMARY above uses all setups, which "
          f"carries no\n      matchability selection at all.")

    print(f"\n  OLD-RULE COMPARISON, NOT APPLES TO APPLES: HANDOFF.md:242 put this "
          f"at\n  +3.8 pp on a dump that counted the same episode 3.26 times over. "
          f"That figure is\n  SUPERSEDED by the number above, not confirmed or "
          f"contradicted by it.")

    try:
        d2, nb2 = load_pairs(dump, cells, m2=True)
        s2 = paired_stats(d2.real.to_numpy() - d2.twin.to_numpy(), d2.day.to_numpy())
        print(f"\n  SENSITIVITY, M2 twins (also matched on the 240-bar return "
              f"quintile):\n      {s2['daymean_pp']:+.4f} pp   SE "
              f"{s2['daymean_se_pp']:.4f}   p {s2['claim_p']:.4g}   "
              f"({s2['n']:,} setups, {nb2} unpaired)")
        print("      Reported as a sensitivity, NOT the primary (26.18: 30 setups "
              "have no M2 pair).")
    except Exception as e:
        print(f"\n  [WARN] M2 sensitivity not computed: {type(e).__name__}: {e}")

    grid_tables(df, df.real.to_numpy() - df.twin.to_numpy(),
                "GRID BREAKDOWN -- descriptive, no verdict reads it")

    print("\n  Section 25's null (the ANCHOR BAR within a window is decoration) and "
          "this result\n  answer DIFFERENT questions and are reported SIDE BY SIDE. "
          "Section 25 says nothing\n  about anchor TYPE or N -- see the tables above "
          "for those (HANDOFF 26.54).")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args[0]))
