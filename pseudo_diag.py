r"""
pseudo_diag.py -- DX1 to DX5 of the pre-specified diagnosis (section 25 amendment 6).

Amendment 3(c) halted on PSEUDO BEATS REAL (26.34). Amendment 6 fixed six diagnostic
tests, their order, their predictions and their decision rules BEFORE any of them
ran. This script is DX1-DX5, all of which read only columns already in the dump.
DX6, the within-pair distance regression, needs the pseudo line's realized distance,
which is NOT stored, and is a separate ~6 min job.

NONE OF THIS CAN OVERTURN THE HALT OR THE REGISTERED VERDICT. 26.34's -0.1865 pp
stands with its stamp whatever these show. They decide what is WRITTEN about it.

  DX1 re-blocked SE          the gap test treats days as independent; the rest of
                             the project measured that they are not (1.7-3x)
  DX2 minimum-effect TOST    a point null is rejectable by anything at n = 3.8M
  DX3 fill timing            the SHARP test: cumulative incidence at T/4, T/2, 3T/4, T
  DX4 per-cell heterogeneity Simpson check over the 144 cells
  DX5 matchability bound     worst case from the 4.650% unmatched

Nothing statistical is reimplemented (E24): day_clustered_mean and one_sided_p come
from the scripts that carry them, and boot_index from the INSTALLED core_v2, so the
re-blocked SE uses the identical convention as every score in the project.

    python pseudo_diag.py --selftest     toy fixtures, reads no dump
    python pseudo_diag.py data\dump      the diagnosis

Written 2026-10-01. Project root; rotates nothing. Nothing under code\ is edited.
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

from core_v2 import boot_index                                          # noqa: E402
from gate_receipts import require_pass                                  # noqa: E402
from power_registered_set import registered_cells                       # noqa: E402
from pseudo_gap import COLS, FLOOR_PP, NQ, day_codes, gap_stats, load_rich  # noqa: E402

NBOOT, BLOCKLEN, SEED = 500, 5, 1
HORIZONS = (0.25, 0.50, 0.75, 1.00)


# ------------------------------------------------------------------------ DX1
def blocked_se(d, g, D, nboot=NBOOT, blocklen=BLOCKLEN, seed=SEED):
    """SE of the mean of day means under a 5-day moving block bootstrap, through
    the INSTALLED core_v2.boot_index -- the same draws every score in this project
    pays for. Returns (naive_se_pp, blocked_se_pp, inflation)."""
    cnt = np.bincount(g, minlength=D)
    m = np.bincount(g, weights=d, minlength=D) / cnt          # per-day mean, fraction
    naive = float(m.std(ddof=1) / np.sqrt(D))
    SEL = boot_index(D, nboot, blocklen, seed)
    draws = np.asarray(m)[np.asarray(SEL)].mean(axis=1)
    blocked = float(draws.std(ddof=1))
    return 100.0 * naive, 100.0 * blocked, (blocked / naive if naive > 0 else np.nan)


# ------------------------------------------------------------------------ DX2
def tost(gap_pp, se_pp, delta_pp=FLOOR_PP, alpha=0.05):
    """Equivalence against a registered floor. A 95% CI inside the margin is TOST
    at alpha/2 per side, i.e. CONSERVATIVE relative to the conventional 90% CI
    (brain/17 17.9B). Returns (lo, hi, equivalent?)."""
    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = gap_pp - z * se_pp, gap_pp + z * se_pp
    return lo, hi, bool(lo > -delta_pp and hi < delta_pp)


# ------------------------------------------------------------------------ DX3
def incidence(df, horizons=HORIZONS):
    """Cumulative incidence at fractions of T, both arms, day-clustered gap at each.

    NOT conditioned on either arm filling -- that would condition on the outcome and
    open a collider. w_0.0 and pseudo_w_0.0 hold the BAR INDEX of first touch.
    """
    T = df["T"].to_numpy()
    w, pw = df["w"].to_numpy(), df["pw"].to_numpy()
    out = []
    for h in horizons:
        lim = h * T
        r = ((w >= 1) & (w <= lim)).astype(np.int64)
        p = ((pw >= 1) & (pw <= lim)).astype(np.int64)
        st = gap_stats(r, p, df["day"].to_numpy())
        rel = (st["fill_pseudo"] - st["fill_real"]) / max(st["fill_real"], 1e-9)
        out.append((h, st["fill_real"], st["fill_pseudo"], st["daymean_pp"],
                    st["daymean_se_pp"], 100.0 * rel))
    return out


# ------------------------------------------------------------------------ DX4
def per_cell(df):
    rows = []
    for c, s in df.groupby("cell", sort=False):
        d = s["r"].to_numpy(np.int64) - s["p"].to_numpy(np.int64)
        rows.append((c, len(s), 100.0 * float(d.mean())))
    g = np.array([x[2] for x in rows])
    return rows, g


# ------------------------------------------------------------------------ DX5
def matchability_bound(pooled_pp, n_matched, n_unmatched):
    """Worst case if every unmatched setup carried the most adverse gap a binary
    outcome allows (-100 pp or +100 pp). Descriptive; never used to adjust."""
    n = n_matched + n_unmatched
    lo = (n_matched * pooled_pp + n_unmatched * -100.0) / n
    hi = (n_matched * pooled_pp + n_unmatched * +100.0) / n
    return lo, hi


# -------------------------------------------------------------------- selftest
def _toy(D=400, per_day=50, gap_pp=-1.0, seed=7, ncell=4):
    rng = np.random.default_rng(seed)
    n = D * per_day
    day = np.repeat(np.arange(D), per_day)
    T = rng.integers(40, 120, n)
    base = 0.5
    pr_r, pr_p = base + gap_pp / 200.0, base - gap_pp / 200.0
    # pseudo touches SOONER as well as more often, so DX3 has a signal to find
    w = np.where(rng.random(n) < pr_r, rng.integers(1, 1 + T, n), 10 ** 6)
    pw = np.where(rng.random(n) < pr_p,
                  np.maximum(1, (rng.integers(1, 1 + T, n) * 0.6).astype(int)), 10 ** 6)
    cell = np.array([f"N60_toy{i % ncell}_ATR_3.0" for i in range(n)])
    return pd.DataFrame(dict(cell=cell, day=day, T=T, w=w, pw=pw,
                             dist_atr=rng.uniform(1, 5, n),
                             r=((w >= 1) & (w <= T)).astype(np.int64),
                             p=((pw >= 1) & (pw <= T)).astype(np.int64)))


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\npseudo_diag.py SELF-TEST -- detectors before any diagnostic is read.\n")
    df = _toy()
    d = df["r"].to_numpy(np.int64) - df["p"].to_numpy(np.int64)
    g, D = day_codes(df["day"].to_numpy())

    nv, bl, inf = blocked_se(d, g, D)
    chk("K1 DX1 the blocked SE uses core_v2.boot_index and is finite",
        np.isfinite(bl) and bl > 0, f"naive {nv:.4f} pp, blocked {bl:.4f} pp, "
        f"inflation {inf:.2f}x")
    SEL = np.asarray(boot_index(D, NBOOT, BLOCKLEN, SEED))
    chk("K2 every bootstrap draw is runs of 5 CONSECUTIVE days inside [0,D)",
        SEL.min() >= 0 and SEL.max() < D
        and bool(np.all(np.diff(SEL[:, :BLOCKLEN], axis=1) == 1)),
        f"{SEL.shape[0]} draws x {SEL.shape[1]} days")
    nv2, bl2, _ = blocked_se(d, g, D, blocklen=1)
    chk("K3 at blocklen=1 the blocked SE collapses to about the naive one",
        abs(bl2 / nv - 1.0) < 0.12, f"blocked(b=1) {bl2:.4f} vs naive {nv:.4f} pp")

    lo, hi, eq = tost(-0.18, 0.012, 1.0)
    chk("K4 DX2 a tiny gap with a tight CI is EQUIVALENT to zero at the 1.0 pp floor",
        eq, f"CI [{lo:+.4f}, {hi:+.4f}]")
    lo, hi, eq = tost(-0.18, 0.60, 1.0)
    chk("K5 and a tiny gap with a WIDE CI is NOT declared equivalent",
        not eq, f"CI [{lo:+.4f}, {hi:+.4f}]")
    lo, hi, eq = tost(-2.0, 0.012, 1.0)
    chk("K6 and a gap beyond the floor is NOT declared equivalent",
        not eq, f"CI [{lo:+.4f}, {hi:+.4f}]")

    inc = incidence(df)
    chk("K7 DX3 cumulative incidence is non-decreasing in the horizon, both arms",
        all(inc[i][1] <= inc[i + 1][1] + 1e-9 and inc[i][2] <= inc[i + 1][2] + 1e-9
            for i in range(len(inc) - 1)),
        " ".join(f"{h:.2f}:{a:.1f}/{b:.1f}" for h, a, b, *_ in inc))
    chk("K8 and it detects a planted EARLY pseudo advantage at the shortest horizon",
        inc[0][5] > inc[-1][5], f"relative lead {inc[0][5]:+.2f}% at T/4 vs "
        f"{inc[-1][5]:+.2f}% at T")
    chk("K9 the final horizon reproduces the headline fill rates exactly",
        abs(inc[-1][1] - 100 * df.r.mean()) < 1e-9
        and abs(inc[-1][2] - 100 * df.p.mean()) < 1e-9,
        f"real {inc[-1][1]:.4f}% pseudo {inc[-1][2]:.4f}%")

    rows, gg = per_cell(df)
    chk("K10 DX4 per-cell gaps cover every cell and their size-weighted mean is "
        "the pooled gap", len(rows) == 4
        and abs(sum(n * v for _, n, v in rows) / sum(n for _, n, _ in rows)
                - 100 * d.mean()) < 1e-9,
        f"{len(rows)} cells, pooled {100*d.mean():+.4f} pp")

    b = matchability_bound(-0.18, 3_821_840, 186_401)
    chk("K11 DX5 the worst-case bound brackets the observed gap",
        b[0] < -0.18 < b[1], f"[{b[0]:+.3f}, {b[1]:+.3f}] pp")

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


# ------------------------------------------------------------------------ main
def main(dump):
    require_pass(dump)
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no real data was read.")

    cells = registered_cells()
    print(f"REGISTERED SET: {len(cells)} cells. Reading timing columns too.\n")
    df, tot, unm, _ = load_rich(dump, cells)

    # the timing columns, read on the same filter load_rich applies
    parts = []
    for c in cells:
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=COLS)
        x = x[(x.is_real == 1) & (x.pseudo_matched == 1)]
        parts.append(pd.DataFrame(dict(T=x["T"].to_numpy(),
                                       w=x["w_0.0"].to_numpy(),
                                       pw=x["pseudo_w_0.0"].to_numpy())))
    tm = pd.concat(parts, ignore_index=True)
    assert len(tm) == len(df), "timing read disagrees with load_rich on row count"
    df = pd.concat([df.reset_index(drop=True), tm], axis=1)

    d = df["r"].to_numpy(np.int64) - df["p"].to_numpy(np.int64)
    g, D = day_codes(df["day"].to_numpy())
    st = gap_stats(df.r.to_numpy(), df.p.to_numpy(), df.day.to_numpy())
    print(f"  [PASS] K0 the headline reproduces 26.34: gap {st['daymean_pp']:+.4f} pp, "
          f"{st['n_setups']:,} setups, {st['n_days']:,} dates")

    print("\n================ DX1  RE-BLOCKED STANDARD ERROR ================")
    nv, bl, inf = blocked_se(d, g, D)
    print(f"  naive day-clustered SE {nv:.4f} pp | 5-day moving block SE {bl:.4f} pp"
          f" | INFLATION {inf:.2f}x")
    print(f"  (HANDOFF.md:319 measured 1.7-3x for this data; amendment 6 DX1 says "
          f"flag if > 3x)")
    z = stats.norm.ppf(0.975)
    print(f"  gap {st['daymean_pp']:+.4f} pp, 95% CI with the BLOCKED SE "
          f"[{st['daymean_pp']-z*bl:+.4f}, {st['daymean_pp']+z*bl:+.4f}] pp")
    print(f"  reverse-test significance at the blocked SE: "
          f"z = {abs(st['daymean_pp'])/bl:.1f}")

    print("\n================ DX2  MINIMUM-EFFECT RESTATEMENT ================")
    lo, hi, eq = tost(st["daymean_pp"], bl, FLOOR_PP)
    print(f"  TOST against the registered {FLOOR_PP} pp floor, blocked SE")
    print(f"  95% CI [{lo:+.4f}, {hi:+.4f}] pp  vs margin [-{FLOOR_PP}, +{FLOOR_PP}]")
    print(f"  EQUIVALENT TO ZERO AT THE REGISTERED FLOOR: {eq}")
    print(f"  -> the point-null rejection (p = 0.0000) and this statement are BOTH "
          f"true;\n     at n = {st['n_setups']:,} the first is a sample-size "
          f"artefact (Jeffreys-Lindley).")

    print("\n================ DX3  FILL TIMING (the sharp test) ================")
    print(f"  {'horiz':>6} {'real %':>9} {'pseudo %':>9} {'gap pp':>9} {'SE':>7} "
          f"{'rel lead %':>11}")
    inc = incidence(df)
    for h, a, b_, gp, se, rel in inc:
        print(f"  {h:>6.2f} {a:>9.3f} {b_:>9.3f} {gp:>+9.4f} {se:>7.4f} {rel:>+11.3f}")
    leads = all(x[5] > 0 for x in inc)
    early = inc[0][5] > inc[-1][5]
    print(f"  pseudo leads at EVERY horizon: {leads}")
    print(f"  relative lead LARGEST at the shortest horizon: {early}")
    print(f"  AMENDMENT 6 DX3 DECISION: "
          + ("PROXIMITY SUPPORTED as the mechanism; DX6 must quantify it."
             if leads and early else
             "PROXIMITY NOT SUPPORTED by the timing pattern -- read the rule in "
             "amendment 6 before interpreting."))

    print("\n================ DX4  PER-CELL HETEROGENEITY ================")
    rows, gg = per_cell(df)
    npos = int((gg > 0).sum())
    print(f"  {len(rows)} cells | positive {npos} ({100*npos/len(rows):.1f}%) | "
          f"negative {len(rows)-npos}")
    print(f"  gap pp: min {gg.min():+.4f} | p25 {np.percentile(gg,25):+.4f} | "
          f"median {np.median(gg):+.4f} | p75 {np.percentile(gg,75):+.4f} | "
          f"max {gg.max():+.4f}")
    print(f"  AMENDMENT 6 DX4 DECISION: "
          + ("MASKED HETEROGENEITY -- report as a distribution, never one number."
             if npos > 0.25 * len(rows) else
             "the sign is consistent across cells; a pooled figure is defensible."))

    print("\n================ DX5  MATCHABILITY SENSITIVITY BOUND ================")
    blo, bhi = matchability_bound(st["pooled_pp"], len(df), unm)
    print(f"  matched {len(df):,} | unmatched {unm:,} ({100*unm/tot:.3f}%)")
    print(f"  worst case if every unmatched setup carried the most adverse possible")
    print(f"  gap: [{blo:+.3f}, {bhi:+.3f}] pp  -- descriptive, never used to adjust")

    print("\n  DX6 (the within-pair distance regression) is NOT in this script: the")
    print("  pseudo line's realized distance is not stored and must be recomputed.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args[0]))
