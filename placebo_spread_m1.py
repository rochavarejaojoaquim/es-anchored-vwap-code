r"""
placebo_spread_m1.py -- the placebo hit-rate SPREAD under the locked (m = 1) horizon,
exactly as registered in HANDOFF_section_25.md, AMENDMENT 2026-09-30 (second):

    SPREAD  = max minus min, across the nine (trigger, eps) groups of the pooled
              set, of the M1b twin fill rate (w_0.0 within T), pooled over N,
              anchors and blocks.
    CEILING = 2 x SPREAD.        FLOOR = 1.0 pp, fixed, NOT rescaled.
    If the new spread makes 1.0 pp clearly wrong as an absolute floor (e.g. above
    the new ceiling) this script FLAGS it; it does not decide.

POOLED SET (same amendment): hh, ll, hv cells at all five N, plus session cells at
N = 60 ONLY.  144 cells; every (trigger, eps) group must hold 16 cells or the run
aborts.  Twins are the is_real == 0 rows of the cell file, restricted to twins whose
src_bar is one of the cell's own setups (as core_v2.load_cell does).

    python placebo_spread_m1.py --selftest          toy fixtures only, reads no real data
    python placebo_spread_m1.py DUMP_DIR            the real run (read-only); runs the
                                                    self-test first and aborts if it fails
DUMP_DIR may be data\dump or rebuild_pseudo_2026-09-30\data\dump: the twin columns are
bit-identical in both (verify_columns, 2026-09-30).

It reads only twin columns.  It never touches a pseudo fill or the real-minus-pseudo gap.
Uncertainty: each group rate gets a trade-date-clustered 95% CI; the SPREAD gets a
trade-date bootstrap (percentile) interval.  The max-minus-min of noisy rates is biased
UPWARD by sampling noise, so the point estimate is the registered statistic and the
interval is descriptive of its noise, not a test.  Section 16's 2.63 pp is printed
for reference only and is NOT used: its statistic was never defined.
"""
import glob
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

BOOT_SEED = 20261001                    # first use in this project
NBOOT = 2000
FLOOR_PP = 1.0
LEGACY_PP = 2.63                        # HANDOFF 16, reference only
GROUPS = [("ATR", 3.0), ("ATR", 4.0), ("ATR", 6.0), ("ATR", 8.0), ("ATR", 12.0),
          ("SD", 1.0), ("SD", 1.5), ("SD", 2.0), ("SD", 3.0)]
POOLED_ANCHORS = ("hh", "ll", "hv")
CELL_RE = re.compile(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)$")
N_LOOKBACKS = 5
CELLS_PER_GROUP = len(POOLED_ANCHORS) * N_LOOKBACKS + 1           # 16


def pooled_cells(dump, session_n60_only=True):
    out = []
    for f in sorted(glob.glob(os.path.join(dump, "N*.parquet"))):
        if ".m2." in f or ".tw2." in f:
            continue
        m = CELL_RE.match(os.path.basename(f)[:-8])
        N, ak, trig, eps = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        if ak == "session" and session_n60_only and N != 60:
            continue
        out.append((f, N, ak, trig, eps))
    return out


def collect(cells, filter_src=True):
    """-> H, C (groups x days: twin hits, twin rows), cells per group, D."""
    days = set()
    for f, *_ in cells:
        days.update(pd.read_parquet(f, columns=["day"])["day"].unique().tolist())
    days = pd.Index(sorted(days))
    D = len(days)
    gi = {g: i for i, g in enumerate(GROUPS)}
    H, C = np.zeros((len(GROUPS), D)), np.zeros((len(GROUPS), D))
    ncell = np.zeros(len(GROUPS), dtype=int)
    for f, N, ak, trig, eps in cells:
        x = pd.read_parquet(f, columns=["is_real", "bar", "src_bar", "day", "w_0.0", "T"])
        tw = x[x.is_real == 0]
        if filter_src:
            tw = tw[tw.src_bar.isin(set(x.loc[x.is_real == 1, "bar"]))]
        w, T = tw["w_0.0"].to_numpy(), tw["T"].to_numpy()
        hit = ((w >= 1) & (w <= T)).astype(float)
        g = days.get_indexer(tw["day"])
        i = gi[(trig, eps)]
        H[i] += np.bincount(g, weights=hit, minlength=D)
        C[i] += np.bincount(g, minlength=D)
        ncell[i] += 1
    return H, C, ncell, D


def group_rates(H, C):
    return H.sum(1) / C.sum(1)


def spread_of(H, C):
    r = group_rates(H, C)
    return float(r.max() - r.min())


def group_ci(H, C):
    """Trade-date-clustered 95% CI of each group rate (ratio estimator)."""
    r = group_rates(H, C)
    D = H.shape[1]
    out = []
    for i in range(len(r)):
        d = C[i] > 0
        n = C[i].sum()
        var = (d.sum() / max(d.sum() - 1, 1)) * np.sum((H[i, d] - r[i] * C[i, d]) ** 2) / n ** 2
        se = np.sqrt(var)
        out.append((r[i] - 1.96 * se, r[i] + 1.96 * se))
    return out


def boot_spread(H, C, nboot=NBOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    D = H.shape[1]
    out = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, D, D)
        r = H[:, idx].sum(1) / C[:, idx].sum(1)
        out[b] = r.max() - r.min()
    return out


def assess(spread_pp):
    ceiling = 2 * spread_pp
    flag = ceiling <= FLOOR_PP
    return ceiling, flag


# ------------------------------------------------------------------ self-test
def toy_dump(root, rates, contaminate=True, rng=None, noisy=False, lookbacks=(60, 100)):
    """A fake dump: 9 groups x (hh, ll, hv) x every lookback + session at every
    lookback, 10 trade dates x 10 twins per cell, hit counts EXACT (rates are
    multiples of 0.1).  Contamination: session cells at N != 60 with hit rate 0.9
    (must be excluded) and ORPHAN twins (src_bar not among the setups) with
    hit rate 1.0 (must be filtered)."""
    os.makedirs(root, exist_ok=True)
    for (trig, eps), rate in zip(GROUPS, rates):
        for ak, N in [(a, n) for a in POOLED_ANCHORS for n in lookbacks] + [("session", n) for n in lookbacks]:
            r = 0.9 if (ak == "session" and N != 60 and contaminate) else rate
            rows = []
            for d in range(10):
                for k in range(10):
                    bar = d * 10 + k
                    rows.append(dict(is_real=1, bar=bar, src_bar=-1, day=d, **{"w_0.0": 5}, T=30))
                    if noisy:
                        hit = rng.random() < r + 0.15 * np.sin(d)
                    else:
                        hit = k < int(round(r * 10))
                    rows.append(dict(is_real=0, bar=10_000 + bar, src_bar=bar, day=d,
                                     **{"w_0.0": 7 if hit else -1}, T=30))
            if contaminate:
                for k in range(20):
                    rows.append(dict(is_real=0, bar=90_000 + k, src_bar=-999, day=k % 10,
                                     **{"w_0.0": 3}, T=30))
            pd.DataFrame(rows).to_parquet(os.path.join(root, f"N{N}_{ak}_{trig}_{eps}.parquet"))


def selftest():
    print("SELF-TEST (toy fixtures; known answers first, then each mutation must FAIL)\n")
    res = []

    def check(name, ok):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    rates = [0.4, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.5]        # planted; spread = 0.6 - 0.4 = 0.2
    with tempfile.TemporaryDirectory() as tmp:
        toy_dump(tmp, rates)
        cells = pooled_cells(tmp)
        H, C, ncell, D = collect(cells)
        r = group_rates(H, C)
        check(f"K1 the registered statistic recovers the planted spread exactly: {spread_of(H, C):.6f} (planted 0.200000); "
              f"every group rate equals its planted rate", abs(spread_of(H, C) - 0.2) < 1e-12 and np.allclose(r, rates))
        check(f"K2 pooled set = pooled anchors at both toy lookbacks + session N=60 only: {sorted(set(ncell))} cells per group (expect 7), "
              f"{len(cells)} cells", set(ncell) == {7} and len(cells) == 63)
        b = boot_spread(H, C, 200)
        check(f"K3 noise-free data: the bootstrap collapses onto the point estimate (width {b.max()-b.min():.2e})",
              b.max() - b.min() < 1e-12 and abs(b[0] - 0.2) < 1e-12)
        tmp2 = os.path.join(tmp, "noisy")
        toy_dump(tmp2, rates, contaminate=False, rng=np.random.default_rng(3), noisy=True)
        Hn, Cn, _, _ = collect(pooled_cells(tmp2))
        bn = boot_spread(Hn, Cn, 300)
        check(f"K4 noisy data: the bootstrap has real width ({100*(np.percentile(bn, 97.5)-np.percentile(bn, 2.5)):.2f} pp)",
              np.percentile(bn, 97.5) - np.percentile(bn, 2.5) > 0.005)
        c1, f1 = assess(0.4)
        c2, f2 = assess(2.6)
        check(f"K5 floor flag: spread 0.4pp -> ceiling {c1:.1f}pp <= 1.0pp floor -> FLAGGED; spread 2.6pp -> ceiling {c2:.1f}pp -> not flagged",
              f1 and not f2)
        # ---- mutations: each wrong version must give a WRONG answer on the same toy data
        Hm, Cm, _, _ = collect(pooled_cells(tmp, session_n60_only=False))
        check(f"M1 including session at every N (the thing the amendment forbids) moves the spread to {spread_of(Hm, Cm):.4f} (!= 0.2) -> detected",
              abs(spread_of(Hm, Cm) - 0.2) > 1e-6)
        Hf, Cf, _, _ = collect(cells, filter_src=False)
        check(f"M2 not restricting twins to the cell's own setups moves the spread to {spread_of(Hf, Cf):.4f} (!= 0.2) -> detected",
              abs(spread_of(Hf, Cf) - 0.2) > 1e-6)
        wrong = float(np.std(r))
        check(f"M3 using the standard deviation instead of max-min gives {wrong:.4f} (!= 0.2) -> detected", abs(wrong - 0.2) > 1e-6)
        check("M4 a pooled set with a missing group-cell is REFUSED by the 16-cells-per-group rule",
              not cells_ok(np.array([7, 7, 7, 7, 7, 7, 7, 7, 6]), expect=7))
        # ---- the REAL-RUN path end to end, on a toy dump with the real 180-cell layout
        import contextlib
        import io
        full = os.path.join(tmp, "full")
        toy_dump(full, rates, lookbacks=(60, 100, 160, 200, 240))
        old = os.getcwd()
        os.chdir(tmp)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                main(full)
        finally:
            os.chdir(old)
        txt = buf.getvalue()
        csv = pd.read_csv(os.path.join(tmp, "placebo_spread_m1_groups.csv"))
        check("K6 main() on a toy dump shaped like the real one (180 cells -> 144 pooled, 16 per group): prints SPREAD = 20.000 pp, "
              "CEILING 40.000 pp, no flag, and writes a 9-row CSV with the planted rates",
              "144 cells" in txt and "SPREAD (max - min over the nine groups) = 20.000 pp" in txt and "CEILING = 2 x SPREAD = 40.000 pp" in txt
              and "FLAG" not in txt and len(csv) == 9 and np.allclose(csv.fill_rate, rates) and set(csv.cells) == {16})
        shutil_path = os.path.join(full, "N240_hv_SD_3.0.parquet")
        os.remove(shutil_path)                      # a real-shaped dump with ONE pooled cell missing
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main(full)
            refused = False
        except SystemExit as e:
            refused = "ABORT" in str(e)
        check("M5 a real-shaped dump with one pooled cell missing -> main() ABORTS (143 cells)", refused)
    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


def cells_ok(ncell, expect=CELLS_PER_GROUP):
    return bool(np.all(np.asarray(ncell) == expect))


# ------------------------------------------------------------------------ run
def main(dump):
    cells = pooled_cells(dump)
    print(f"pooled set: {len(cells)} cells from {dump}", flush=True)
    H, C, ncell, D = collect(cells)
    if not cells_ok(ncell) or len(cells) != 144:
        raise SystemExit(f"ABORT: expected 144 cells with {CELLS_PER_GROUP} in each of the nine groups; "
                         f"got {len(cells)} cells, per group {ncell.tolist()}")
    r = group_rates(H, C)
    ci = group_ci(H, C)
    sp = spread_of(H, C)
    b = boot_spread(H, C)
    days_used = int((C.sum(0) > 0).sum())
    ceiling, flag = assess(100 * sp)
    print("\nSTAMP: M1b twin fill rate (w_0.0 within T), LOCKED horizon (m=1; spec.t_stop) | pooled set = hh/ll/hv x 5 N + "
          f"session N=60 | {len(cells)} cells | twin rows {int(C.sum()):,} | independent units: {days_used:,} trade dates | "
          "pooled over N, anchors and blocks A+B | raw rates (no null, no edge) | clustered on trade date | "
          f"bootstrap {NBOOT} draws, seed {BOOT_SEED} | ES, train blocks only, sealed 2018-2022 untouched")
    print(f"\n  {'group':<10}{'cells':>6}{'twin rows':>13}{'trade dates':>13}{'fill rate':>11}{'95% CI (clustered)':>24}")
    rows = []
    for i, (trig, eps) in enumerate(GROUPS):
        print(f"  {trig + ' ' + str(eps):<10}{ncell[i]:>6}{int(C[i].sum()):>13,}{int((C[i] > 0).sum()):>13,}"
              f"{100*r[i]:>10.2f}%   [{100*ci[i][0]:.2f}, {100*ci[i][1]:.2f}]")
        rows.append(dict(trigger=trig, eps=eps, cells=int(ncell[i]), twin_rows=int(C[i].sum()),
                         trade_dates=int((C[i] > 0).sum()), fill_rate=r[i], ci_lo=ci[i][0], ci_hi=ci[i][1]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"\n  SPREAD (max - min over the nine groups) = {100*sp:.3f} pp   "
          f"[bootstrap over trade dates: median {100*np.median(b):.3f}, 95% interval {100*lo:.3f} - {100*hi:.3f}]")
    print(f"  highest group: {GROUPS[int(np.argmax(r))]}   lowest group: {GROUPS[int(np.argmin(r))]}")
    print(f"\n  REGISTERED RULE:  CEILING = 2 x SPREAD = {ceiling:.3f} pp    FLOOR = {FLOOR_PP:.1f} pp (fixed)")
    print(f"  for reference only, NOT used: HANDOFF 16's 2.63 pp (statistic undefined) -> old ceiling {2*LEGACY_PP:.2f} pp; "
          f"the registered 5.0 pp ceiling")
    if flag:
        print(f"\n  *** FLAG: the ceiling ({ceiling:.3f} pp) is not above the {FLOOR_PP:.1f} pp floor. The amendment says this is "
              f"flagged and DECIDED THEN, by the author -- this script decides nothing. ***")
    else:
        print(f"\n  floor check: {FLOOR_PP:.1f} pp is below the ceiling {ceiling:.3f} pp -- not flagged.")
    out = os.path.join(os.getcwd(), "placebo_spread_m1_groups.csv")
    pd.DataFrame(rows).assign(spread_pp=100 * sp, ceiling_pp=ceiling).to_csv(out, index=False)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if not a:
        raise SystemExit(__doc__)
    if not selftest():
        raise SystemExit("\nSELF-TEST FAILED -- no real data was read.")
    print()
    main(a[0])
