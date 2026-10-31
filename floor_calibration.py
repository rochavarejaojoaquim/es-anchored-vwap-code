r"""
floor_calibration.py -- is the permutation floor calibrated? (rule F5: validate
the procedure before trusting what it says)

Stage 1 reported 234 of 317 cell+sides above their own 95th-percentile floor,
where chance alone should give ~16. Before believing that, the machinery has to
be tested on data where the answer is KNOWN to be nothing.

The test. Take a real cell and shuffle the OUTCOME within each block: the
setup's hit/miss is moved to another setup of the same block. Everything else
is untouched -- the covariates, the twins, the days, the block structure, the
cut points, the sample sizes. Any relationship between covariate and outcome is
destroyed, so this cell is DEAD by construction.

Then run the whole procedure on it: search for the best cut, build its floor
from permutations, ask whether the observed score beats the floor.

    A calibrated floor is beaten ~5% of the time.
    A floor that is beaten most of the time is BROKEN, and every stage-1
    survivor is meaningless.

    python floor_calibration.py                 6 SMALLEST cells x 3, nperm 30
    python floor_calibration.py 10 5 50         cells, shuffles, permutations
    python floor_calibration.py --largest 4 3 30        the 4 BIGGEST cells
    python floor_calibration.py --cells N60_session_ATR_6.0:1.0,N200_ll_ATR_4.0:1.0 3 30
        (with --cells, the two numbers are SHUFFLES and PERMUTATIONS)

The default picks the smallest cells because they are quick -- but a floor can
be calibrated on small cells and not on large ones: the concern (a null that
destroys day-level structure the real covariates have) grows with sample size
and clustering. Test the cells your survivors actually live in, by name or with
--largest.

Nothing here writes to cores_v2 or floors_v2; it writes calibration.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
import core_v2                                                    # noqa: E402
import floor_v2                                                   # noqa: E402


def _shuffle_hits(x, rng):
    hit = x["hit"].to_numpy().copy()
    blocks = x.block.to_numpy()
    for B in np.unique(blocks):
        ix = np.where(blocks == B)[0]
        hit[ix] = hit[rng.permutation(ix)]
    out = x.copy()
    out["hit"] = hit
    return out


def dead_cell(r, c, rng):
    """Move the OUTCOME to another row of the same block -- for the setups AND
    for the twins. The cell keeps every covariate, every twin link, every day,
    and has no covariate-outcome relationship left on either side.

    Both sides matter: the edge is setup hit MINUS twin hit, so shuffling only
    the setups leaves the search able to hunt for subsets whose TWINS did badly,
    which scores positive on a cell that is supposed to be dead (found while
    building this, 2026-09-25)."""
    return _shuffle_hits(r, rng), _shuffle_hits(c, rng)


if __name__ == "__main__":
    a = sys.argv[1:]
    largest = "--largest" in a
    named = None
    if a and a[0] == "--cells":
        named = [x.split(":") for x in a[1].split(",")]
        a = ["0"] + a[2:]          # the cell count is fixed by the list itself
    elif largest:
        a = [x for x in a if x != "--largest"]
    ncell = int(a[0]) if len(a) > 0 else 6
    nrep = int(a[1]) if len(a) > 1 else 3
    nperm = int(a[2]) if len(a) > 2 else 30
    import glob
    fl = []
    for f in glob.glob(os.path.join(floor_v2.FLOORS, "*.parquet")):
        t = pd.read_parquet(f)
        if len(t):
            fl.append((int(t.nA.iloc[0]) + int(t.nB.iloc[0]), str(t.cell.iloc[0]),
                       None if pd.isna(t.side.iloc[0]) else float(t.side.iloc[0])))
    if not fl:
        raise SystemExit("no floors yet")
    fl.sort(reverse=largest)        # smallest first (quick), or largest (the real ones)
    picks = fl[:ncell]
    if named:
        picks = [(0, c, None if s_ in ("None", "none") else float(s_)) for c, s_ in named]
    unit = getattr(floor_v2, "NULL_UNIT", "row")
    print(f"  {len(picks)} cells x {nrep} outcome-shuffles x {nperm} permutations"
          f"  |  null unit: {unit.upper()}"
          + ("  (whole days move together)" if unit == "day" else "  (single episodes move)"))
    print(f"  a calibrated floor is beaten ~5% of the time\n")
    rows = []
    for _, cell, side in picks:
        r0, c0, T = core_v2.load_cell(cell, side)
        cols = core_v2.search_cols(r0, side)
        for rep in range(nrep):
            rng = np.random.default_rng([12345, rep, abs(hash(cell)) % 10 ** 6])
            r, cc = dead_cell(r0, c0, rng)
            P = core_v2.prep(r, cc)          # day statistics follow the shuffled hits
            t0 = time.time()
            obs = core_v2.best_core(r, cc, P, cols)
            if obs is None:
                print(f"  {cell} side {side} rep {rep}: no candidate at all")
                continue
            null = np.empty(nperm)
            # `into=` only exists in the fast-floor patch; work with either
            # version of floor_v2, since this check must run as installed.
            import inspect
            fast = "into" in inspect.signature(floor_v2.shuffle_within_block).parameters
            scratch = r.copy() if fast else None
            for k in range(nperm):
                # test exactly the null production will use (VWAP_NULL_UNIT)
                kw = {"unit": floor_v2.NULL_UNIT} if hasattr(floor_v2, "NULL_UNIT") else {}
                rp = (floor_v2.shuffle_within_block(r, cols, rng, into=scratch, **kw) if fast
                      else floor_v2.shuffle_within_block(r, cols, rng, **kw))
                b = core_v2.best_core(rp, cc, P, cols)
                null[k] = b["score"] if b is not None else -np.inf
            fin = null[np.isfinite(null)]
            f95 = float(np.percentile(fin, 95)) if len(fin) else -np.inf
            beat = obs["score"] > f95
            rows.append(dict(cell=cell, side=str(side), rep=rep, col=obs["col"],
                             obs_score=obs["score"], floor95=f95,
                             edge_over_floor=obs["score"] - f95, beat=bool(beat),
                             null_mean=float(fin.mean()) if len(fin) else np.nan))
            print(f"  {cell:<22} side {str(side):<5} rep {rep}  "
                  f"obs {obs['score']:+7.3f}  floor95 {f95:+7.3f}  "
                  f"null mean {rows[-1]['null_mean']:+7.3f}  "
                  f"{'BEATS THE FLOOR' if beat else 'below floor (expected)'}"
                  f"   [{time.time() - t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows)
    if not len(t):
        raise SystemExit("nothing ran")
    t.to_csv(os.path.join(HERE, "calibration.csv"), index=False)
    rate = t.beat.mean()
    print(f"\n  dead cells that beat their floor: {int(t.beat.sum())} of {len(t)} "
          f"= {100 * rate:.0f}%   (calibrated = ~5%)")
    if rate > 0.20:
        print("  [FAIL]  the floor is TOO LOW: the procedure finds 'signal' in data")
        print("          with no effect in it. Stage-1 survivors cannot be trusted.")
    elif rate < 0.005 and len(t) > 10:
        print("  [WARN]  the floor may be TOO HIGH -- real effects would be hidden.")
    else:
        print("  [PASS]  the floor behaves as it should on data with no effect.")
    print("  written: calibration.csv")
