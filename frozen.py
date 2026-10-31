"""
frozen.py -- how many rows CANNOT move under NULL_UNIT="day_strata"?

"day_strata" permutes whole days only among days with the SAME episode count.
A day whose count is unique inside its block has no partner, so it stays put
and its covariate-outcome link is never broken.  This measures that fraction
on the real cells.  Read-only: it loads cells and prints; it writes nothing.

    python frozen.py N60_session_ATR_6.0 N200_ll_ATR_4.0 N60_ll_SD_1.0 N60_hh_ATR_3.0

Runs its own arithmetic self-test against known numbers first (D3).
"""
import sys

import numpy as np


def frozen_fraction(day, block):
    """Fraction of ROWS on days whose episode count is unique in their block,
    plus the mean number of days per count-stratum."""
    day, block = np.asarray(day), np.asarray(block)
    stuck = tot = 0
    strata = []
    for B in np.unique(block):
        m = block == B
        _, cnt = np.unique(day[m], return_counts=True)
        lens, how_many = np.unique(cnt, return_counts=True)
        lone = np.isin(cnt, lens[how_many < 2])
        stuck += int(cnt[lone].sum())
        tot += int(cnt.sum())
        strata.append(len(cnt) / max(len(lens), 1))
    return stuck / max(tot, 1), float(np.mean(strata))


def selftest():
    print("SELF-TEST against known numbers")
    # 3 days of 2 episodes, 1 day of 5 -> the 5-episode day is alone: 5 of 11
    day = np.array([1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4])
    blk = np.zeros(11, int)
    f, s = frozen_fraction(day, blk)
    ok = abs(f - 5 / 11) < 1e-12
    print(f"  [{'PASS' if ok else 'FAIL'}] one lone 5-episode day of 11 rows: "
          f"{f:.4f}, expected {5/11:.4f}")
    # every day the same length -> nothing frozen
    day2 = np.repeat(np.arange(10), 3)
    f2, _ = frozen_fraction(day2, np.zeros(30, int))
    ok2 = f2 == 0.0
    print(f"  [{'PASS' if ok2 else 'FAIL'}] all days equal length: {f2:.4f}, "
          f"expected 0.0000")
    # two blocks, the lone day in block 1 only
    day3 = np.r_[np.repeat(np.arange(5), 2), np.repeat(np.arange(10, 13), 2), [13]]
    blk3 = np.r_[np.zeros(10, int), np.ones(7, int)]
    f3, _ = frozen_fraction(day3, blk3)
    ok3 = abs(f3 - 1 / 17) < 1e-12
    print(f"  [{'PASS' if ok3 else 'FAIL'}] lone day inside block 1 only: "
          f"{f3:.4f}, expected {1/17:.4f}")
    return ok and ok2 and ok3


if __name__ == "__main__":
    if not selftest():
        raise SystemExit("self-test failed -- do not trust the numbers below")
    cells = sys.argv[1:]
    if not cells:
        raise SystemExit("usage: python frozen.py <cell> [<cell> ...]")

    sys.path.insert(0, "code")
    from core_v2 import load_cell          # the one line I could not test here

    print(f"\n{'cell':<24}{'side':>6}{'rows':>8}{'days':>7}"
          f"{'rows frozen':>13}{'days/stratum':>14}")
    for cell in cells:
        for side in (None, 1.0, -1.0):
            try:
                r, c, T = load_cell(cell, side)
            except Exception as e:
                print(f"{cell:<24}{str(side):>6}   -- load failed: "
                      f"{type(e).__name__}: {e}")
                continue
            if r is None or len(r) == 0:
                continue
            f, s = frozen_fraction(r.day.to_numpy(), r.block.to_numpy())
            print(f"{cell:<24}{str(side):>6}{len(r):>8}{r.day.nunique():>7}"
                  f"{f:>12.1%}{s:>14.1f}")
    print("\nREAD IT AS: a small frozen fraction means 'day_strata' is usable.")
    print("A large one means most days have no partner to swap with and the")
    print("corrected null is weak exactly where it matters. No threshold is")
    print("set here -- report the number and decide with it in front of you.")
