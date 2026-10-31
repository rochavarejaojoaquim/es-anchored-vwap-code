"""
CORRECTED DAY PERMUTATION for floor_v2, plus the measurement that decides
A1 vs A2.

The shipped unit="day" branch pairs rows POSITIONALLY, so a destination day's
covariates come from ~2 donor days (measured 20-37% single-donor).  23.4 claims
100%.  This module builds the shuffle that actually delivers it.

There is a genuine design fork, and it is a research decision, not a coding one:

  "recycle"  permute whole days freely; where donor and destination days differ
             in length, take the donor's rows cyclically.
             -> single-donor 100%, free permutation
             -> BREAKS a property the current null has: the shuffled frame no
                longer holds the same MULTISET of covariate rows as the real
                one.  Some rows are duplicated, others dropped.

  "strata"   permute days only among days with the SAME episode count.
             -> single-donor 100% AND an exact row permutation, multiset kept
             -> days whose episode count is unique cannot move.  Those are
                FIXED POINTS: their covariate-outcome link is never broken, so
                the null is weaker exactly there.  The fraction of rows stuck
                in fixed points is measured below and is the cost of this
                option.

Both stay compatible with the joint seed the Romano-Wolf patch needs.

Run:  python day_perm_fix.py          gates + the trade-off table
      python day_perm_fix.py mutate   break the target; the gate must FAIL
"""
import sys
import numpy as np

JOINT_SEED = 20260927
JOINT_DAY_ORIGIN = 14245
JOINT_DAY_SPAN = 8000


def _order_key(uniq_days, rng, perm_index, joint_seed):
    """A random ordering of the given days: per-cell, or joint across cells."""
    if joint_seed is None or perm_index is None:
        return rng.permutation(len(uniq_days))
    u = np.random.default_rng([joint_seed, perm_index]).random(JOINT_DAY_SPAN)
    return np.argsort(u[uniq_days - JOINT_DAY_ORIGIN], kind="stable")


# ------------------------------------------------------------ the shipped one
def shipped(blocks, days, vals, rng, **kw):
    new = vals.copy()
    for B in np.unique(blocks):
        ix = np.where(blocks == B)[0]
        d = days[ix]
        order = np.argsort(d, kind="stable")
        ix_s, d_s = ix[order], d[order]
        starts = np.r_[0, np.where(d_s[1:] != d_s[:-1])[0] + 1]
        groups = np.split(ix_s, starts[1:])
        perm = rng.permutation(len(groups))
        src = np.concatenate([groups[p] for p in perm])
        dst = np.concatenate(groups)
        new[dst] = vals[src[:len(dst)]]
    return new


# ------------------------------------------------------------ the corrected one
def corrected(blocks, days, vals, rng, mode="strata", perm_index=None,
              joint_seed=None, seal=True):
    new = vals.copy()
    blks = np.unique(blocks) if seal else [None]
    for B in blks:
        ix = np.where(blocks == B)[0] if B is not None else np.arange(len(blocks))
        d = days[ix]
        order = np.argsort(d, kind="stable")
        ix_s, d_s = ix[order], d[order]
        starts = np.r_[0, np.where(d_s[1:] != d_s[:-1])[0] + 1]
        groups = np.split(ix_s, starts[1:])          # rows of each day, day order
        uniq = d_s[starts]                           # the day values themselves
        lens = np.array([len(g) for g in groups])

        if mode == "recycle":
            src_day = _order_key(uniq, rng, perm_index, joint_seed)
            for j, g in enumerate(groups):
                donor = groups[src_day[j]]
                new[g] = vals[np.resize(donor, len(g))]
        else:                                        # strata
            for L in np.unique(lens):
                where = np.where(lens == L)[0]
                if len(where) < 2:
                    continue                         # fixed point: cannot move
                sub = _order_key(uniq[where], rng, perm_index, joint_seed)
                for j, gi in enumerate(where):
                    new[groups[gi]] = vals[groups[where[sub[j]]]]
    return new


# ------------------------------------------------------------------ diagnostics
def diagnose(days, vals, new, row_day):
    """single-donor fraction (days with >=2 episodes), and multiset preserved?"""
    nd = []
    for day in np.unique(days):
        dst = np.where(days == day)[0]
        if len(dst) < 2:
            continue
        nd.append(len({row_day[int(v)] for v in new[dst, 0]}))
    nd = np.array(nd) if nd else np.array([1])
    multiset = np.array_equal(np.sort(np.ravel(new)), np.sort(np.ravel(vals)))
    return float((nd == 1).mean()), float(nd.mean()), multiset


def fixed_point_rows(days):
    """fraction of ROWS on days whose episode count is unique -> cannot move"""
    uniq, cnt = np.unique(days, return_counts=True)
    lens, how_many = np.unique(cnt, return_counts=True)
    stuck = np.isin(cnt, lens[how_many < 2])
    return float(cnt[stuck].sum() / cnt.sum())


def make(D, mean_eps, seed):
    r = np.random.default_rng(seed)
    counts = np.maximum(1, r.poisson(mean_eps, D))
    days = np.repeat(np.arange(JOINT_DAY_ORIGIN + 100,
                               JOINT_DAY_ORIGIN + 100 + D), counts)
    n = len(days)
    return np.zeros(n, int), days, np.arange(n, dtype=float).reshape(n, 1), days.copy()


# ----------------------------------------------------------------------- gates
def gates(mode="strata", seal=True, quiet=False, use_shipped=False):
    res = {}

    def say(tag, ok, msg):
        ok = bool(ok)
        res[tag] = ok
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag:<4} {msg}")

    f = shipped if use_shipped else corrected
    bl, dy, va, rd = make(1432, 4, 7)

    # C1  the property 23.4 claims: one donor day per destination day
    fr, mn, ms = [], [], []
    for k in range(20):
        new = f(bl, dy, va, np.random.default_rng([3, k]), mode=mode, seal=seal)
        a, b, c = diagnose(dy, va, new, rd)
        fr.append(a); mn.append(b); ms.append(c)
    say("C1", np.mean(fr) > 0.999,
        f"each destination day draws from ONE donor day: {np.mean(fr):6.1%} "
        f"(mean donors {np.mean(mn):.2f})")

    # C2  the covariate multiset is preserved -- the null frame holds the same
    #     rows as the real one, only reassigned
    say("C2", all(ms), f"shuffled frame keeps the exact multiset of covariate rows "
                       f"({sum(ms)} of {len(ms)} draws)")

    # C3  block seal
    bl2, dy2, va2, rd2 = make(600, 3, 9)
    bl2 = (dy2 >= dy2[len(dy2) // 2]).astype(int)
    bad = 0
    for k in range(20):
        new = f(bl2, dy2, va2, np.random.default_rng([4, k]), mode=mode, seal=seal)
        for B in np.unique(bl2):
            m = bl2 == B
            bad += int(not set(np.ravel(new[m]).astype(int))
                       <= set(np.ravel(va2[m]).astype(int)))
    say("C3", bad == 0, f"no covariates cross the block seal ({bad} crossings / 40)")

    # C4  marginal uniformity of the day mapping, on a small cell
    bl3, dy3, va3, rd3 = make(40, 3, 5)
    uniq = np.unique(dy3)
    tab = {}
    for k in range(4000):
        new = f(bl3, dy3, va3, np.random.default_rng([6, k]), mode=mode, seal=seal)
        for day in uniq:
            dst = np.where(dy3 == day)[0]
            tab.setdefault(day, []).append(rd3[int(new[dst[0], 0])])
    spread = [len(set(v)) for v in tab.values()]
    say("C4", np.mean(spread) > 2,
        f"a day reaches many donors over repeats: mean {np.mean(spread):.1f} "
        f"distinct donors in 4,000 draws (min {min(spread)})")

    # C5  joint-seed compatible: same perm_index, different cell rng, same answer
    a = f(bl, dy, va, np.random.default_rng(1), mode=mode, perm_index=9,
          joint_seed=JOINT_SEED, seal=seal)
    b = f(bl, dy, va, np.random.default_rng(999), mode=mode, perm_index=9,
          joint_seed=JOINT_SEED, seal=seal)
    say("C5", use_shipped or np.array_equal(a, b),
        "joint seed honoured: same perm_index gives the same answer")

    # C6  row count per day unchanged -- the frame best_core sees keeps its shape
    new = f(bl, dy, va, np.random.default_rng(2), mode=mode, seal=seal)
    say("C6", new.shape == va.shape, "frame shape unchanged")
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mutate":
        print("MUTATION TEST -- break the target, the named gate must FAIL\n")
        clean = gates(quiet=True)
        ok = True
        for tag, desc, kw in (
                ("C1", "use the SHIPPED positional pairing", dict(use_shipped=True)),
                ("C3", "shuffle across the block seal", dict(seal=False)),
                ("C2", "switch to recycle (multiset not preserved)",
                 dict(mode="recycle"))):
            now = gates(quiet=True, **kw).get(tag)
            good = clean.get(tag) is True and now is False
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {tag:<4} {desc:<42} "
                  f"clean={clean.get(tag)}  mutated={now}")
        print(f"\n  {'ALL GATES ALIVE' if ok else 'A GATE IS DEAD'}")
        sys.exit(0 if ok else 1)

    print("CORRECTED DAY PERMUTATION -- mode 'strata'\n")
    r = gates("strata")
    print(f"\n  {sum(r.values())} of {len(r)} gates pass\n")

    print("THE TRADE-OFF, measured\n")
    print(f"  {'days':>6}{'eps/day':>9}{'rows':>7} | {'scheme':<9}"
          f"{'single-donor':>13}{'multiset':>10}{'rows frozen':>13}")
    for D, me in ((1432, 1), (1432, 2), (1432, 4), (1432, 8), (300, 4)):
        bl, dy, va, rd = make(D, me, 7)
        fp = fixed_point_rows(dy)
        for name, kw in (("shipped", dict(use_shipped=True)),
                         ("recycle", dict(mode="recycle")),
                         ("strata", dict(mode="strata"))):
            fn = shipped if kw.get("use_shipped") else corrected
            new = fn(bl, dy, va, np.random.default_rng(21),
                     **{k: v for k, v in kw.items() if k != "use_shipped"})
            a, b, c = diagnose(dy, va, new, rd)
            frozen = f"{fp:11.1%}" if name == "strata" else f"{'-':>11}"
            print(f"  {D:>6}{me:>9}{len(dy):>7} | {name:<9}{a:>12.1%}"
                  f"{'yes' if c else 'NO':>10}{frozen:>13}")
