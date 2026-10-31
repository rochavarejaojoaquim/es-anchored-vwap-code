"""
PATCH UNDER TEST: the joint-seed day ordering that Romano-Wolf needs, added to
floor_v2.shuffle_within_block WITHOUT changing anything else about the null.

The shipped body is reproduced verbatim as `shipped`.  `patched` is the same
code with ONE substitution at the line

    perm = rng.permutation(len(groups))

and nothing else touched.  The positional pairing that smears days
(measured separately: 20-37% single-donor) is DELIBERATELY LEFT AS IS, so
tonight's null is exactly the one the 0-of-12 calibration validated.

Run:  python rw_patch.py          the gates
      python rw_patch.py mutate   break each target; the named gate must FAIL
"""
import sys
import time

import numpy as np
import pandas as pd

# locked constants -> spec.py
JOINT_SEED = 20260927
JOINT_DAY_ORIGIN = 14245        # 2009-01-01 as days since epoch
JOINT_DAY_SPAN = 8000           # covers to 2030; one uniform per calendar day


def _as_day_int(d):
    """Day values -> int64 day numbers, whatever r.day holds."""
    a = np.asarray(d)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int64)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[D]").astype(np.int64)
    return pd.to_datetime(a).values.astype("datetime64[D]").astype(np.int64)


# ------------------------------------------------ shipped body, verbatim
def shipped(blocks, days, vals, rng):
    new = vals.copy()
    for B in np.unique(blocks):
        ix = np.where(blocks == B)[0]
        d = days[ix]
        order = np.argsort(d, kind="stable")
        ix_s, d_s = ix[order], d[order]
        starts = np.r_[0, np.where(d_s[1:] != d_s[:-1])[0] + 1]
        groups = np.split(ix_s, starts[1:])
        perm = rng.permutation(len(groups))
        src_rows = np.concatenate([groups[p] for p in perm])
        dst_rows = np.concatenate(groups)
        if len(src_rows) < len(dst_rows):
            reps = int(np.ceil(len(dst_rows) / max(len(src_rows), 1)))
            src_rows = np.tile(src_rows, reps)
        new[dst_rows] = vals[src_rows[:len(dst_rows)]]
    return new


# ------------------------------------------------ the patched body
def patched(blocks, days, vals, rng, perm_index=None, joint_seed=None, seal=True):
    new = vals.copy()
    blks = np.unique(blocks) if seal else [None]
    for B in blks:
        ix = np.where(blocks == B)[0] if B is not None else np.arange(len(blocks))
        d = days[ix]
        order = np.argsort(d, kind="stable")
        ix_s, d_s = ix[order], d[order]
        starts = np.r_[0, np.where(d_s[1:] != d_s[:-1])[0] + 1]
        groups = np.split(ix_s, starts[1:])

        # ---- THE ONLY CHANGED LINES -------------------------------------
        if joint_seed is None:
            perm = rng.permutation(len(groups))            # unchanged default
        else:
            # One uniform per CALENDAR DAY, identical in every cell for this
            # permutation index, so the 409 nulls are drawn jointly and a
            # Romano-Wolf stepdown can use the cross-cell dependence.
            # Keyed by the day itself, not by position, so two machines need
            # no shared day list -- only the seed and the fixed origin.
            u = np.random.default_rng([joint_seed, perm_index]).random(JOINT_DAY_SPAN)
            gday = _as_day_int(d_s[starts])
            perm = np.argsort(u[gday - JOINT_DAY_ORIGIN], kind="stable")
        # -----------------------------------------------------------------

        src_rows = np.concatenate([groups[p] for p in perm])
        dst_rows = np.concatenate(groups)
        if len(src_rows) < len(dst_rows):
            reps = int(np.ceil(len(dst_rows) / max(len(src_rows), 1)))
            src_rows = np.tile(src_rows, reps)
        new[dst_rows] = vals[src_rows[:len(dst_rows)]]
    return new


# ------------------------------------------------ fixtures
def make(day_lo, day_hi, mean_eps, seed, two_blocks=True):
    r = np.random.default_rng(seed)
    d = np.arange(day_lo, day_hi)
    counts = np.maximum(1, r.poisson(mean_eps, len(d)))
    days = np.repeat(d, counts)
    n = len(days)
    blocks = (days >= (day_lo + day_hi) // 2).astype(int) if two_blocks else np.zeros(n, int)
    vals = np.arange(n, dtype=float).reshape(n, 1)
    return blocks, days, vals


def gates(joint=True, seal=True, quiet=False):
    res = {}

    def say(tag, ok, msg):
        res[tag] = ok
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag:<4} {msg}")

    O = JOINT_DAY_ORIGIN
    bl, dy, va = make(O + 100, O + 900, 4, 1)
    js = JOINT_SEED if joint else None

    # P1  BACK-COMPAT: with the joint seed off, byte-for-byte the shipped result
    same = all(np.array_equal(
        shipped(bl, dy, va, np.random.default_rng([7, k])),
        patched(bl, dy, va, np.random.default_rng([7, k]), joint_seed=None))
        for k in range(50))
    say("P1", same, "joint_seed=None reproduces the shipped null EXACTLY (50 draws)")

    # P2  still a rearrangement of the block's own rows, seal intact
    ok = True
    for k in range(50):
        new = patched(bl, dy, va, np.random.default_rng([7, k]),
                      perm_index=k, joint_seed=js, seal=seal)
        for B in np.unique(bl):
            m = bl == B
            ok &= set(np.ravel(new[m]).astype(int)) == set(np.ravel(va[m]).astype(int))
    say("P2", ok, "every block's rows stay inside their own block (no seal crossing)")

    # P3  JOINTNESS: two cells with overlapping days must order the shared
    #     days the same way, or the nulls are still independent
    blA, dyA, vaA = make(O + 100, O + 600, 3, 11, two_blocks=False)
    blB, dyB, vaB = make(O + 350, O + 900, 5, 22, two_blocks=False)
    shared = np.intersect1d(np.unique(dyA), np.unique(dyB))
    agree = []
    for k in range(100):
        oA = _order_of_days(dyA, k, js)
        oB = _order_of_days(dyB, k, js)
        rA = np.array([oA[d] for d in shared])
        rB = np.array([oB[d] for d in shared])
        agree.append(float(np.corrcoef(np.argsort(np.argsort(rA)),
                                       np.argsort(np.argsort(rB)))[0, 1]))
    m = float(np.mean(agree))
    say("P3", m > 0.99, f"shared days ordered identically in both cells: "
                        f"mean Spearman {m:+.4f} (per-cell seeds give ~0)")

    # P4  determinism: the result depends on (joint_seed, perm_index) only
    a = patched(bl, dy, va, np.random.default_rng(1), perm_index=5, joint_seed=js)
    b = patched(bl, dy, va, np.random.default_rng(999), perm_index=5, joint_seed=js)
    say("P4", (not joint) or np.array_equal(a, b),
        "identical output for the same perm_index whatever the cell's own rng")

    # P5  the day dtype does not matter
    dates = (dy.astype("datetime64[D]") - np.datetime64(0, "D")) + np.datetime64(0, "D")
    a = patched(bl, dy, va, np.random.default_rng(1), perm_index=3, joint_seed=js)
    b = patched(bl, np.asarray(dy, dtype="datetime64[D]"), va,
                np.random.default_rng(1), perm_index=3, joint_seed=js)
    say("P5", (not joint) or np.array_equal(a, b),
        "int days and datetime64 days give the same answer")

    return res


def _order_of_days(days, k, js):
    """rank each distinct day gets, under the patched ordering"""
    d = np.unique(days)
    if js is None:
        perm = np.random.default_rng([7, k]).permutation(len(d))
    else:
        u = np.random.default_rng([js, k]).random(JOINT_DAY_SPAN)
        perm = np.argsort(u[_as_day_int(d) - JOINT_DAY_ORIGIN], kind="stable")
    rank = np.empty(len(d), int)
    rank[perm] = np.arange(len(d))
    return dict(zip(d.tolist(), rank.tolist()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mutate":
        print("MUTATION TEST -- break the target, the named gate must FAIL\n")
        clean = gates(quiet=True)
        muts = {"P3": ("per-cell seeds instead of the joint one",
                       lambda: gates(joint=False, quiet=True)),
                "P2": ("shuffle across the block seal",
                       lambda: gates(seal=False, quiet=True))}
        allok = True
        for tag, (desc, f) in muts.items():
            now = f().get(tag)
            good = clean.get(tag) is True and now is False
            allok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {tag:<4} {desc:<40} "
                  f"clean={clean.get(tag)}  mutated={now}")
        print(f"\n  {'ALL GATES ALIVE' if allok else 'A GATE IS DEAD'}")
        sys.exit(0 if allok else 1)

    print("JOINT-SEED PATCH TO floor_v2.shuffle_within_block\n")
    r = gates()
    print(f"\n  {sum(r.values())} of {len(r)} gates pass")

    bl, dy, va = make(JOINT_DAY_ORIGIN + 100, JOINT_DAY_ORIGIN + 1532, 4, 3)
    for label, js in (("shipped (joint off)", None), ("patched (joint on)", JOINT_SEED)):
        t = time.perf_counter()
        for k in range(100):
            patched(bl, dy, va, np.random.default_rng([7, k]), perm_index=k, joint_seed=js)
        print(f"  {label:<22} {(time.perf_counter()-t)*1000:7.1f} ms per 100 permutations "
              f"({len(dy):,} rows, {len(np.unique(dy)):,} days)")
