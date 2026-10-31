r"""
mirror_did.py -- section 27 TEST B: the MIRROR, as a difference in differences.

THE ORIGINAL PROPOSAL is the author's, 23.11 (2026-09-26): compare the VWAP with the
MIRROR, the level the same distance on the OTHER side of price. It matches same bar,
same moment, same volatility, same distance -- no draw, no cap. mir_w and race are
already stored (HANDOFF.md:2538-2540).

23.11 STATES ITS OWN CONFOUND AND IT IS FATAL TO THE NAIVE FORM. Price arrived here
by moving AWAY from the VWAP, so reaching the VWAP is a REVERSAL and reaching the
mirror is a CONTINUATION. If large excursions mean-revert at all -- regardless of
what line is behind them -- the mirror loses systematically and it READS AS "the
VWAP pulls". Section 25 has now MEASURED that the anchor is decoration, and section
27 TEST A has measured a +2.96 pp level effect whose own confound is the MOMENT.
So a positive naive mirror result is EXPECTED whether or not the VWAP is special,
and it cannot count as evidence.

THE FIX, FREE, AND IT IS WHAT MAKES THIS TEST WORTH RUNNING. mir_w is populated on
the TWINS as well (measured: 63,186 of 63,186 on M1b, and present in .m2.parquet).
The twin is a random level at the same distance WITH ITS OWN MIRROR, so the
reversal/continuation structure exists in BOTH arms and differences out:

    A_real = fill(VWAP)       - fill(its mirror)            per setup
    A_twin = fill(twin level) - fill(twin's mirror)         per twin -> setup mean
    DiD    = A_real - A_twin                                <- THE PRIMARY TEST

THIS IS THE ONLY REGISTERED TEST THAT HOLDS THE MOMENT FIXED WHILE CHANGING THE
LINE. Section 25 held the moment fixed but had VWAPs in both arms; TEST A changed
the line but also the moment. TEST B changes the line with the moment pinned.

K2 and K4 in the self-test are the heart of it: on a fixture containing PURE generic
reversion and no VWAP effect at all, the DiD correctly returns ~0 while the NAIVE
A_real returns a large positive -- i.e. the test demonstrates the exact mistake it
exists to prevent.

NOTHING STATISTICAL IS REIMPLEMENTED (E24): paired_stats from level_control (which
is one_sided_p from pseudo_power, the function 26.21 certified, plus
day_clustered_mean, without gap_stats' int64 cast -- A_twin is fractional).

    python mirror_did.py --selftest      toy fixtures, reads no dump
    python mirror_did.py data\dump       TEST B

Written 2026-10-02. Project root; rotates nothing. Reads only.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from power_registered_set import registered_cells                     # noqa: E402
from pseudo_null_pair import CELL_RE                                  # noqa: E402
from level_control import FLOOR_PP, ALPHA, grid_tables, paired_stats  # noqa: E402

PUB_REAL_FILL = 51.297        # 26.51, TEST A on ALL setups -- the K0b reference
COLS = ["is_real", "bar", "src_bar", "day", "block", "T", "w_0.0", "mir_w"]


def load_four_arms(dump, cells):
    """One row per SETUP: A_real, A_twin, and all four raw fill indicators."""
    parts = []
    for c in cells:
        m = CELL_RE.search(c)
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=COLS)
        r = x[x.is_real == 1]
        t = x[x.is_real == 0]

        def fills(f):
            T = f["T"].to_numpy()
            ln = ((f["w_0.0"].to_numpy() >= 1) & (f["w_0.0"].to_numpy() <= T)).astype(float)
            mi = ((f["mir_w"].to_numpy() >= 1) & (f["mir_w"].to_numpy() <= T)).astype(float)
            return ln, mi

        rl, rm = fills(r)
        tl, tm_ = fills(t)
        tw = pd.DataFrame(dict(src_bar=t["src_bar"].to_numpy(),
                               tl=tl, tm=tm_, a=tl - tm_)) \
               .groupby("src_bar").agg(tl=("tl", "mean"), tm=("tm", "mean"),
                                       A_twin=("a", "mean"), k=("a", "size"))
        j = pd.DataFrame(dict(cell=c, eps=float(m.group(4)), trigger=m.group(3),
                              bar=r["bar"].to_numpy(), day=r["day"].to_numpy(),
                              block=r["block"].to_numpy(),
                              rl=rl, rm=rm, A_real=rl - rm)).join(tw, on="bar")
        parts.append(j[j["A_twin"].notna()])
    return pd.concat(parts, ignore_index=True)


def eps_slope(df, trigger):
    """23.11's eps hypothesis, secondary. Section 25 amendment 4's caution applies
    unchanged: reversion-versus-continuation asymmetry grows with excursion size
    anyway (Grant-Wolf-Yu 2005), so a positive slope is NOT independent support."""
    sub = df[df.trigger == trigger]
    rows = []
    for e, s in sub.groupby("eps", sort=True):
        st = paired_stats(s.A_real.to_numpy() - s.A_twin.to_numpy(), s.day.to_numpy())
        rows.append((float(e), st["daymean_pp"], st["daymean_se_pp"], st["n"]))
    if len(rows) < 3:
        return dict(levels=rows, slope=float("nan"), p=float("nan"),
                    all_positive=False, supported=False)
    x = np.array([a for a, _, _, _ in rows]); y = np.array([b for _, b, _, _ in rows])
    se = np.array([c for _, _, c, _ in rows]); w = 1.0 / np.maximum(se, 1e-12) ** 2
    xb = np.sum(w * x) / np.sum(w); yb = np.sum(w * y) / np.sum(w)
    sxx = np.sum(w * (x - xb) ** 2)
    slope = float(np.sum(w * (x - xb) * (y - yb)) / sxx)
    se_b = float(np.sqrt(1.0 / sxx))
    p = float(stats.t.sf(slope / se_b, len(rows) - 2))
    allpos = bool(np.all(y > 0))
    return dict(levels=rows, slope=slope, se=se_b, p=p, all_positive=allpos,
                supported=bool(p <= ALPHA and slope > 0 and allpos))


def verdict(st, gA, gB):
    """Section 27.2's ladder, reverse test FIRST."""
    d = st["daymean_pp"]
    if st["reverse_p"] <= ALPHA:
        return ("MIRROR DiD NEGATIVE -- STOP AND DIAGNOSE",
                f"reverse one-sided p = {st['reverse_p']:.4g} <= {ALPHA}")
    if st["claim_p"] > ALPHA:
        return ("H0b NOT REJECTED -- the mirror asymmetry is generic reversion",
                f"claim one-sided p = {st['claim_p']:.4g} > {ALPHA}")
    if d < FLOOR_PP:
        return ("REAL BUT NEGLIGIBLE",
                f"p = {st['claim_p']:.4g} but the DiD {d:+.4f} pp is under the "
                f"{FLOOR_PP} pp floor")
    if not (gA > 0 and gB > 0):
        return ("NOT REPLICATED",
                f"DiD is {gA:+.4f} pp in block A and {gB:+.4f} pp in block B; "
                f"section 27.2 requires both positive")
    return ("H1b SUPPORTED",
            f"p = {st['claim_p']:.4g}, DiD {d:+.4f} pp, positive in both blocks")


# -------------------------------------------------------------------- selftest
def _toy(D=600, per_day=120, reversion_pp=6.0, vwap_extra_pp=0.0, seed=3, base=0.48):
    """Both arms carry the SAME generic reversion asymmetry; only vwap_extra_pp is
    VWAP-specific. So A_real ~ reversion + extra, A_twin ~ reversion, DiD ~ extra."""
    rng = np.random.default_rng(seed)
    n = D * per_day
    day = np.repeat(np.arange(D), per_day)
    block = np.where(day < D // 2, "A", "B")
    R, E = reversion_pp / 100.0, vwap_extra_pp / 100.0
    rl = (rng.random(n) < np.clip(base + R / 2 + E / 2, 0, 1)).astype(float)
    rm = (rng.random(n) < np.clip(base - R / 2 - E / 2, 0, 1)).astype(float)
    tl = ((rng.random(n) < np.clip(base + R / 2, 0, 1)).astype(float)
          + (rng.random(n) < np.clip(base + R / 2, 0, 1)).astype(float)) / 2
    tm = ((rng.random(n) < np.clip(base - R / 2, 0, 1)).astype(float)
          + (rng.random(n) < np.clip(base - R / 2, 0, 1)).astype(float)) / 2
    return pd.DataFrame(dict(day=day, block=block, rl=rl, rm=rm, tl=tl, tm=tm,
                             A_real=rl - rm, A_twin=tl - tm))


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\nmirror_did.py SELF-TEST -- detectors before any dump is read.\n")

    # K1-K2: PURE generic reversion, NO VWAP effect. The whole point of the test.
    g = _toy(reversion_pp=6.0, vwap_extra_pp=0.0)
    sg = paired_stats(g.A_real - g.A_twin, g.day.to_numpy())
    naive = paired_stats(g.A_real, g.day.to_numpy())
    chk("K1 under PURE generic reversion the DiD is ~0 (|.| < 0.5 pp)",
        abs(sg["daymean_pp"]) < 0.5, f"DiD {sg['daymean_pp']:+.4f} pp")
    chk("K2 and it is NOT detected", sg["claim_p"] > ALPHA, f"p = {sg['claim_p']:.4f}")
    chk("K3 while the NAIVE A_real is LARGE and overwhelmingly significant -- the "
        "exact mistake this test exists to prevent",
        naive["daymean_pp"] > 4.0 and naive["claim_p"] < 1e-20,
        f"naive {naive['daymean_pp']:+.4f} pp, p = {naive['claim_p']:.3g}")

    # K4: a genuine VWAP-specific effect on top of the same reversion
    e = _toy(reversion_pp=6.0, vwap_extra_pp=3.0, seed=11)
    se_ = paired_stats(e.A_real - e.A_twin, e.day.to_numpy())
    chk("K4 a planted +3.0 pp VWAP-SPECIFIC effect is recovered within 0.8 pp",
        abs(se_["daymean_pp"] - 3.0) < 0.8, f"DiD {se_['daymean_pp']:+.4f} pp")
    chk("K5 and it IS detected", se_["claim_p"] <= ALPHA, f"p = {se_['claim_p']:.3g}")
    chk("K6 THE CAST GUARD: A_twin is fractional (0, +-0.5, +-1) and must not be "
        "floored", set(np.unique(np.round(g.A_twin.to_numpy(), 6))) <= {-1.0, -0.5, 0.0, 0.5, 1.0}
        and 0.5 in set(np.unique(np.round(g.A_twin.to_numpy(), 6))),
        f"values {sorted(set(np.unique(np.round(g.A_twin.to_numpy(), 6))))}")

    n = _toy(reversion_pp=6.0, vwap_extra_pp=-3.0, seed=12)
    sn = paired_stats(n.A_real - n.A_twin, n.day.to_numpy())
    v, _ = verdict(sn, -3.0, -3.0)
    chk("M1 a NEGATIVE DiD triggers STOP AND DIAGNOSE, before anything else",
        v.startswith("MIRROR DiD NEGATIVE"), f"reverse p = {sn['reverse_p']:.3g}")
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=0.4), 0.4, 0.4)
    chk("M2 a significant but sub-floor DiD is REAL BUT NEGLIGIBLE",
        v == "REAL BUT NEGLIGIBLE", v)
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=3.0), 4.0, -0.5)
    chk("M3 a DiD negative in block B is NOT REPLICATED", v == "NOT REPLICATED", v)
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.01, daymean_pp=3.0), 3.0, 3.0)
    chk("M4 and only the clean case is H1b SUPPORTED", v == "H1b SUPPORTED", v)
    v, _ = verdict(dict(reverse_p=1.0, claim_p=0.30, daymean_pp=3.0), 3.0, 3.0)
    chk("M5 a large DiD with p > 0.05 is still H0b NOT REJECTED",
        v.startswith("H0b NOT REJECTED"), v)

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


# ------------------------------------------------------------------------ main
def main(dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    cells = registered_cells()
    print(f"SECTION 27 TEST B -- the mirror, as a difference in differences. "
          f"{len(cells)} cells from {dump}\n")
    df = load_four_arms(dump, cells)

    ks = df["k"].value_counts().to_dict()
    print(f"  [{'PASS' if set(ks) == {2} else 'FAIL'}] K0a every setup has EXACTLY "
          f"2 M1b twins   {ks}")
    rr = 100.0 * float(df.rl.mean())
    k0b = abs(rr - PUB_REAL_FILL) < 0.05
    print(f"  [{'PASS' if k0b else 'FAIL'}] K0b the VWAP fill rate reproduces "
          f"26.51's TEST A   {rr:.3f}% vs {PUB_REAL_FILL}%")
    if not (set(ks) == {2} and k0b):
        raise SystemExit("A known-number check FAILED -- no result was read.")

    st = paired_stats(df.A_real.to_numpy() - df.A_twin.to_numpy(), df.day.to_numpy())
    nv = paired_stats(df.A_real.to_numpy(), df.day.to_numpy())
    tw = paired_stats(df.A_twin.to_numpy(), df.day.to_numpy())
    blk = {}
    for b in ("A", "B"):
        s = df[df.block == b]
        blk[b] = paired_stats(s.A_real.to_numpy() - s.A_twin.to_numpy(),
                              s.day.to_numpy())["daymean_pp"] if len(s) else float("nan")

    print(f"\nSTAMP: section 27 TEST B, mirror difference-in-differences | "
          f"{len(cells)} cells | {st['n']:,} setups, {2*st['n']:,} twins | "
          f"independent units: {st['D']:,} trade dates | paired via src_bar, "
          f"clustered on the SETUP's trade date | tau 0.0, wick fill within T | "
          f"MOMENT HELD FIXED, line changed | dump {dump} | ES, train blocks A+B, "
          f"sealed 2018-2022 untouched | raw, no cost modelled")
    print(f"\n  the four arms")
    print(f"    VWAP        {100*df.rl.mean():7.3f}%      its mirror   {100*df.rm.mean():7.3f}%")
    print(f"    twin level  {100*df.tl.mean():7.3f}%      twin mirror  {100*df.tm.mean():7.3f}%")
    print(f"  A_real  (VWAP - its mirror) {nv['daymean_pp']:+8.4f} pp   SE {nv['daymean_se_pp']:.4f}")
    print(f"  A_twin  (twin - its mirror) {tw['daymean_pp']:+8.4f} pp   SE {tw['daymean_se_pp']:.4f}")
    print(f"\n  DiD = A_real - A_twin  (THE PRIMARY TEST)")
    print(f"      {st['daymean_pp']:+.4f} pp   SE {st['daymean_se_pp']:.4f}")
    print(f"      pooled {st['pooled_pp']:+.4f} pp   95% CI "
          f"[{st['lo_pp']:+.4f}, {st['hi_pp']:+.4f}]")
    print(f"      block A {blk['A']:+.4f} pp | block B {blk['B']:+.4f} pp")
    print(f"  reverse test (DiD < 0), evaluated FIRST: p = {st['reverse_p']:.4g}")
    print(f"  claim   test (DiD > 0)                 : p = {st['claim_p']:.4g}")

    v, why = verdict(st, blk["A"], blk["B"])
    print(f"\n  VERDICT: {v}\n      {why}")

    print(f"\n  SECONDARY, REPORTED AND EXPLICITLY NOT EVIDENCE (section 27.2): the "
          f"NAIVE\n  mirror comparison A_real = {nv['daymean_pp']:+.4f} pp, "
          f"p = {nv['claim_p']:.3g}. 23.11 predicted this would\n  be positive from "
          f"generic reversion ALONE. A_twin = {tw['daymean_pp']:+.4f} pp is the same\n"
          f"  asymmetry measured on random levels, and it is what the DiD subtracts.")

    print(f"\n  23.11's eps HYPOTHESIS, secondary. Section 25 amendment 4's caution "
          f"applies\n  unchanged: a positive slope is NOT independent support "
          f"(Grant-Wolf-Yu 2005).")
    for trig in ("ATR", "SD"):
        h = eps_slope(df, trig)
        print(f"    {trig}: slope {h['slope']:+.5f} pp per eps unit, one-sided "
              f"p {h['p']:.4g}, all levels positive {h['all_positive']} -> "
              f"{'SUPPORTED' if h['supported'] else 'NOT SUPPORTED'}")
        for e, y, se, n in h["levels"]:
            print(f"        eps {e:<5} DiD {y:+8.4f} pp  SE {se:.4f}  n {n:,}")

    grid_tables(df, df.A_real.to_numpy() - df.A_twin.to_numpy(),
                "GRID BREAKDOWN of the DiD -- descriptive, no verdict reads it")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args[0]))
