r"""power_registered_set.py -- run section 25.4's power validation on the REGISTERED
pooled set, without typing 144 cell names by hand.

    python power_registered_set.py --dry-run          build + check the list, run nothing
    python power_registered_set.py data\dump          the real validation

WHY THIS EXISTS.  pseudo_power.py takes literal cell names (`DUMP_DIR CELL [CELL ...]`)
and its own docstring warns that section 25.3 "does not say which cells that is. Pass
exactly the cells it denotes."  The set WAS registered, in HANDOFF_section_25.md
amendment "2026-09-30, second":

    POOLING: hh, ll and hv cells pooled across all five N and the full eps grid
    (ATR and SD ...).  Session cells at N=60 ONLY (22.4 decision 14) ...
    The pooled set is 144 cells, 4,008,241 setups

so this script builds exactly that and refuses to run if the arithmetic disagrees.
It adds NOTHING to the method: it shells out to pseudo_power.py unchanged, so
pseudo_power runs its own self-test first and prints its own stamp.

CHECKS BEFORE IT WILL RUN (all must pass):
  C1  3 anchors x 5 N x 9 eps + session at N=60 x 9 eps  ==  144 cells
  C2  no session cell at N != 60 is in the list   (22.4 decision 14)
  C3  every one of the 144 .parquet files exists in the dump
  C4  the cells NOT in the list are exactly the 36 session cells at N in
      {100,160,200,240}, and 144 + 36 == 180
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

NS = [60, 100, 160, 200, 240]
ANCHORS = ["hh", "ll", "hv"]
EPS = ["ATR_3.0", "ATR_4.0", "ATR_6.0", "ATR_8.0", "ATR_12.0",
       "SD_1.0", "SD_1.5", "SD_2.0", "SD_3.0"]
SESSION_PRIMARY_N = 60


def registered_cells():
    """The 144-cell pooled set of amendment '2026-09-30, second'."""
    out = [f"N{n}_{a}_{e}" for n in NS for a in ANCHORS for e in EPS]
    out += [f"N{SESSION_PRIMARY_N}_session_{e}" for e in EPS]
    return out


def all_cells():
    return [f"N{n}_{a}_{e}" for n in NS for a in ANCHORS + ["session"] for e in EPS]


def main():
    dry = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not dry and not args:
        raise SystemExit(__doc__)
    dump = args[0] if args else os.path.join(HERE, "data", "dump")

    cells = registered_cells()
    every = all_cells()
    ok = []

    def chk(name, good, detail=""):
        ok.append(bool(good))
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<62} {detail}")

    print("REGISTERED POOLED SET -- checks before anything is read\n")
    chk("C1 3 anchors x 5 N x 9 eps + session at N=60 x 9 == 144",
        len(cells) == 144, f"{len(cells)} cells, {len(set(cells))} distinct")
    stray = [c for c in cells if "_session_" in c and not c.startswith(f"N{SESSION_PRIMARY_N}_")]
    chk("C2 no session cell at N != 60 (22.4 decision 14)", not stray, f"{len(stray)} stray")
    missing = [c for c in cells if not os.path.exists(os.path.join(dump, c + ".parquet"))]
    chk("C3 every registered cell exists in the dump", not missing,
        f"dump={dump}; missing {len(missing)}" + (f" e.g. {missing[:2]}" if missing else ""))
    excluded = sorted(set(every) - set(cells))
    want_excl = sorted(f"N{n}_session_{e}" for n in NS if n != SESSION_PRIMARY_N for e in EPS)
    chk("C4 the 36 excluded cells are exactly session at N!=60",
        excluded == want_excl and len(every) == 180,
        f"{len(cells)} + {len(excluded)} = {len(cells) + len(excluded)} of {len(every)}")

    if not all(ok):
        raise SystemExit("\n  A CHECK FAILED -- nothing was run.")
    print("\n  all checks pass.")

    cmd = [sys.executable, os.path.join(HERE, "pseudo_power.py"), dump] + cells
    if dry:
        print(f"\n  DRY RUN. Would invoke pseudo_power.py with {len(cells)} cells.")
        print(f"  command length {sum(len(a) + 1 for a in cmd):,} chars "
              f"(Windows CreateProcess limit is 32,767)")
        print(f"  first 3 cells: {cells[:3]}")
        print(f"  last 3 cells : {cells[-3:]}")
        return 0
    print(f"\n  invoking pseudo_power.py on {len(cells)} cells "
          f"(it runs its OWN self-test first and aborts if that fails)\n")
    sys.stdout.flush()
    return subprocess.call(cmd, cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
