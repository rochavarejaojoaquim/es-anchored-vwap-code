r"""a5_2_shape_power_toy.py -- TOY ONLY, reads no data. For the A5(2) design question (HANDOFF 26.148):
which ONE statistic should the covariate-discovery test use, and do the pinned 'AUC' (26.114) and
'equals / approximately = band' (26.118) ideas belong inside it?

Three candidate statistics for "does covariate X modulate the paired outcome d?", each given the SAME
permutation null (d permuted against a fixed covariate order), on iid rows:
  CUT   the Stage-1 family: best of 18 one-sided cuts (10th..90th percentile, 'lo' and 'hi'), |z| maximised
  RANK  |correlation of d with the covariate's rank| (monotone only; the AUC / Spearman family)
  BAND  the RANGE (max minus min) of the centred cumulative sum of d in covariate-rank order (Kuiper type).
        Identity checked below: the range equals the largest centred sum over any CIRCULAR interval of ranks,
        i.e. it IS the best 'band' search over every pair of cut points, including one-sided cuts (a band
        touching an end), a single value of a discrete covariate (one tie block), and a U-shape or a wrap
        around midnight (a band that wraps), with no bin count and no cut grid.
Planted shapes, effect delta in the mean of d: null; monotone; step in the top fifth; step at an OFF-GRID
quantile (top 15%); middle band (40-60%); narrow band (45-55%); U (both outer tenths); cyclic bump that
wraps (ranks 90%-100% and 0%-10% as one band = the same as U here, shown via a rotated order).
Prints the rejection rate at 5% per statistic. A design aid, NOT a result about the data.

    python a5_2_shape_power_toy.py            (about a minute)
"""
import sys
import time
import numpy as np

N, REPS, B, DELTA, SEED = 8000, 200, 200, 0.05, 20261013
CUTS = np.arange(0.1, 0.91, 0.1)


def stat_cut(dc, n):
    """dc: d in covariate order, centred. Best |z| over 18 one-sided cuts (lo: ranks <= q, hi: ranks > q)."""
    cs = np.cumsum(dc)
    k = (CUTS * n).astype(int)
    lo = cs[k - 1]                                   # sum of the lowest k (the 'hi' sum is its negative)
    z = np.abs(lo) / np.sqrt(k * (n - k) / n)
    return float(z.max())


def stat_rank(dc, n, r=None):
    r = np.arange(n) - (n - 1) / 2.0
    return float(abs(np.dot(dc, r)) / np.sqrt((r * r).sum()))


def stat_band(dc, n):
    cs = np.cumsum(dc)
    return float((max(cs.max(), 0.0) - min(cs.min(), 0.0)) / np.sqrt(n))


STATS = dict(CUT=stat_cut, RANK=stat_rank, BAND=stat_band)


def shapes(n):
    u = (np.arange(n) + 0.5) / n
    out = {
        "null": np.zeros(n),
        "monotone (linear, same total as a fifth-step)": (u - 0.5) * 2 * 0.8,
        "step, top fifth (on the cut grid)": (u > 0.8) * 1.0,
        "step, top 15% (off the cut grid)": (u > 0.85) * 1.0,
        "middle band 40-60%": ((u > 0.4) & (u < 0.6)) * 1.0,
        "narrow band 45-55%": ((u > 0.45) & (u < 0.55)) * 1.0,
        "U: both outer tenths": ((u < 0.1) | (u > 0.9)) * 1.0,
        "one value of a 10-valued covariate (tie block 3 of 10)": ((u > 0.2) & (u < 0.3)) * 1.0,
    }
    return out


def identity_check(rng):
    """range of the centred cumulative sum == max over circular intervals of the centred sum (brute force)."""
    n = 60
    d = rng.standard_normal(n); d -= d.mean()
    cs = np.r_[0.0, np.cumsum(d)]
    rng_stat = cs.max() - cs.min()
    best = -np.inf
    for a in range(n):
        for L in range(0, n + 1):
            idx = (a + np.arange(L)) % n
            best = max(best, d[idx].sum())
    return abs(rng_stat - best) < 1e-9, rng_stat, best


def main():
    rng = np.random.default_rng(SEED)
    ok, a, b = identity_check(rng)
    print(f"[{'PASS' if ok else 'FAIL'}] identity: range of centred cumulative sum {a:.6f} == best circular-interval sum {b:.6f}")
    if not ok:
        return 1
    t0 = time.time()
    sh = shapes(N)
    print(f"\nn = {N} rows, noise SD 0.6 (d takes values like -1..1), delta = {DELTA}, {REPS} replicates, {B} permutations each, 5% level")
    print("COMBO = ONE statistic: the smaller of CUT's and BAND's permutation p-values, itself referred to the same "
          "permutations (min-p).")
    print(f"{'shape':<58}" + "".join(f"{k:>8}" for k in STATS) + f"{'COMBO':>8}")
    names = list(STATS)
    for name, f in sh.items():
        rej = {k: 0 for k in STATS}
        rej_combo = 0
        for _ in range(REPS):
            d = 0.6 * rng.standard_normal(N) + DELTA * f
            dc = d - d.mean()
            V = np.empty((B + 1, len(names)))                    # row 0 = observed
            V[0] = [STATS[k](dc, N) for k in names]
            for j in range(1, B + 1):
                p = rng.permutation(dc)
                V[j] = [STATS[k](p, N) for k in names]
            # p-value of every row against all rows (observed counted as one draw)
            P = (V[None, :, :] >= V[:, None, :]).sum(1) / (B + 1.0)          # (B+1) x 3
            for i, k in enumerate(names):
                rej[k] += P[0, i] <= 0.05
            mp = np.minimum(P[:, names.index("CUT")], P[:, names.index("BAND")])
            rej_combo += (mp <= mp[0]).mean() <= 0.05
        print(f"{name:<58}" + "".join(f"{100 * rej[k] / REPS:7.1f}%" for k in STATS) + f"{100 * rej_combo / REPS:7.1f}%")
    # rotation invariance: BAND is unchanged if the rank order is rotated (a cyclic covariate's origin is arbitrary)
    d = 0.6 * rng.standard_normal(N) + DELTA * sh["U: both outer tenths"]; dc = d - d.mean()
    rot = np.roll(dc, N // 3)
    print(f"\nrotation of the rank origin by a third: BAND {stat_band(dc, N):.4f} -> {stat_band(rot, N):.4f} (unchanged); "
          f"CUT {stat_cut(dc, N):.3f} -> {stat_cut(rot, N):.3f}; RANK {stat_rank(dc, N):.3f} -> {stat_rank(rot, N):.3f}")
    print(f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
