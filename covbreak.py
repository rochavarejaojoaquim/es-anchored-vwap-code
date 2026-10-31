"""
covbreak.py -- which covariates does NULL_UNIT="day_strata" actually break?

The day-level null moves WHOLE DAYS. A covariate whose value depends only on
where a row sits inside its day (ct_hour is the obvious candidate) is almost
unchanged when days swap, so the null's search sees nearly the real covariate
and the link the null is meant to destroy survives.

For each cell+side and each search column this reports the mean Spearman
correlation between the covariate BEFORE and AFTER a day_strata shuffle,
averaged over nperm draws.

    ~0.0  the null breaks this covariate completely  (what we want)
    ~1.0  the null cannot break it at all            (its floor means nothing)

Read-only: loads cells and prints. Writes covbreak.csv only.

    python covbreak.py N60_session_ATR_6.0 N240_hv_ATR_6.0 --nperm 20
    python covbreak.py --selftest

Self-test first, against three synthetic covariates with known answers (D3).
"""
import sys

import numpy as np
import pandas as pd


def spearman_cols(A, B):
    """Column-wise Spearman between two equal-shaped arrays."""
    out = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        a, b = A[:, j], B[:, j]
        if np.all(a == a[0]) or np.all(b == b[0]):
            out[j] = np.nan
            continue
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        out[j] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def measure(r, cols, shuffle, nperm, rng_maker):
    before = r[cols].to_numpy().astype(float)
    acc = np.zeros(len(cols))
    n = 0
    for k in range(nperm):
        rp = shuffle(r, cols, rng_maker(k), perm_index=k)
        c = spearman_cols(before, rp[cols].to_numpy().astype(float))
        acc += np.nan_to_num(c)
        n += 1
    return acc / max(n, 1)


# ------------------------------------------------------------------ selftest
def selftest():
    sys.path.insert(0, "/home/sandbox")
    import day_perm_fix as REF

    rr = np.random.default_rng(4)
    D = 600
    counts = np.maximum(1, rr.poisson(4, D))
    day = np.repeat(np.arange(REF.JOINT_DAY_ORIGIN + 50,
                              REF.JOINT_DAY_ORIGIN + 50 + D), counts)
    n = len(day)
    pos = np.concatenate([np.arange(c) for c in counts])      # index inside day
    daylevel = np.repeat(rr.standard_normal(D), counts)       # one value per day
    r = pd.DataFrame({
        "day": day,
        "block": (day >= day[n // 2]).astype(int),
        "noise": rr.standard_normal(n),        # iid per row      -> expect ~0
        "day_level": daylevel,                 # constant in day  -> expect ~0
        "pos_in_day": pos.astype(float),       # position in day  -> expect ~1
    })
    cols = ["noise", "day_level", "pos_in_day"]

    def shuffle(r, cols, rng, perm_index=None):
        bl, dy = r.block.to_numpy(), r.day.to_numpy()
        va = r[cols].to_numpy().astype(float)
        new = REF.corrected(bl, dy, va, rng, mode="strata")
        out = r.copy(); out[cols] = new; return out

    c = measure(r, cols, shuffle, 20, lambda k: np.random.default_rng([3, k]))
    exp = {"noise": 0.0, "day_level": 0.0, "pos_in_day": 1.0}
    ok = True
    print("SELF-TEST against known numbers (D3)")
    for name, got in zip(cols, c):
        e = exp[name]
        good = abs(got - e) < 0.12
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<12} corr {got:+.3f}  "
              f"(expected ~{e:+.1f})")
    print(f"\n  {'the metric works' if ok else 'THE METRIC IS BROKEN -- do not use'}")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    nperm = int(a[a.index("--nperm") + 1]) if "--nperm" in a else 20
    cells = [x for i, x in enumerate(a)
             if not x.startswith("--") and (i == 0 or a[i - 1] != "--nperm")]
    if not cells:
        raise SystemExit("usage: python covbreak.py <cell> [<cell> ...] [--nperm 20]")

    sys.path.insert(0, "code")
    import core_v2
    import floor_v2
    print(f"  NULL_UNIT={floor_v2.NULL_UNIT}  |  {len(cells)} cells, "
          f"{nperm} shuffles each\n")

    rows = []
    for cell in cells:
        for side in (None, 1.0, -1.0):
            try:
                r, c, T = core_v2.load_cell(cell, side)
            except Exception as e:
                print(f"  {cell} side {side}: load failed {type(e).__name__}: {e}")
                continue
            if r is None or len(r) == 0:
                continue
            cols = core_v2.search_cols(r, side)
            corr = measure(
                r, cols,
                lambda rr_, cc_, rng_, perm_index=None: floor_v2.shuffle_within_block(
                    rr_, cc_, rng_, unit=floor_v2.NULL_UNIT, perm_index=perm_index),
                nperm, lambda k: np.random.default_rng([7, k]))
            for col, v in zip(cols, corr):
                rows.append(dict(cell=cell, side=str(side), col=col, corr=float(v)))

    t = pd.DataFrame(rows)
    if not len(t):
        raise SystemExit("nothing measured")
    t.to_csv("covbreak.csv", index=False)
    print("  mean correlation before-vs-after, by covariate "
          "(0 = null breaks it, 1 = null cannot touch it):")
    g = t.groupby("col").corr.agg(["mean", "max", "size"]).sort_values(
        "mean", ascending=False)
    print("  " + g.round(3).to_string().replace("\n", "\n  "))
    bad = g[g["mean"] > 0.5]
    print(f"\n  covariates the null barely breaks (mean corr > 0.5): "
          f"{len(bad)} of {len(g)}")
    if len(bad):
        print("  " + ", ".join(bad.index.tolist()))
    print("\n  written: covbreak.csv")
