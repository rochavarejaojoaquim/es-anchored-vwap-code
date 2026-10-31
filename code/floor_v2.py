"""
floor_v2.py -- the permutation selection floor for core_v2 (approved by
The author 2026-09-21; replaces rescore_all.py phase 2, which ran a DIFFERENT
search -- terciles, simple SE, phase-1 candidates: §21.12).

    python code\\floor_v2.py N60           one lookback (run one N per window)
    python code\\floor_v2.py N60 200       ... with 200 permutations (default 100)

QUESTION  Is a core's score higher than the best score the SAME search finds
          when the covariates carry no information about the outcome?

HOW
  1. Load the cell exactly as core_v2 does (load_cell, same side filter).
  2. Re-run core_v2.best_core on the real data. D3 CHECK: this must reproduce
     the score stored in cores_v2 EXACTLY, printed PASS/FAIL. On FAIL the
     cell is skipped and nothing is written -- the floor would be measuring
     a different thing than the candidate (C107).
  3. NPERM times: shuffle the covariate ROWS among the setups, WITHIN each
     block. Whole rows move together, so the correlations between covariates
     -- which set how many effectively different candidates the search tries
     -- are kept; only the link between covariates and outcome is broken.
     Within block, so each regime keeps its own covariate distribution.
     Twins are NOT touched: they inherit covariate values through src_bar
     inside the search, so they follow the shuffle automatically.
     Hits, days and blocks are never shuffled.
  4. Run core_v2.best_core -- the IDENTICAL two-stage procedure and the same
     score function -- on each shuffle. A shuffle with no admissible candidate
     scores -inf (the search found nothing).

OUTPUT  floors_v2\\<cell>_side<s>.parquet (floors_v2_tau<t> if tau != 0):
  obs_score   the candidate's score (must equal cores_v2)
  floor95     95th percentile of the null best scores -- the UPPER tail, since
              a higher score is a stronger candidate (D4)
  p           (1 + #null >= obs) / (1 + NPERM): the permutation p-value,
              never zero; its smallest possible value is 1/(1+NPERM)
  edge_over_floor = obs_score - floor95
  plus eA, eB (raw edges, F3), n, nperm, the null scores, tau, seed

READING IT: failing the floor does NOT mean "no edge". The cell may still
carry a real UNCONDITIONAL edge over its twins. It means the covariate filter
adds nothing beyond what picking the best of many random subsets gives.
(Tested 2026-09-21: a cell with a +5pp unconditional edge and NO covariate
effect gave core scores +1.8..+3.1 -- floor p 0.36 / 0.48 / 0.87.)

Cells whose cores_v2 file is empty (no candidate) need no floor: skipped.
Resume: a cell with an existing output file is skipped.
"""
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
import safeio
import pandas as pd

from core_v2 import (DUMP, OUT as CORES, TAU, best_core, load_cell, prep,
                     search_cols)

NPERM_DEFAULT = 100
SEED = 20260921
# Default "day_strata" (24.3a): whole days swap among days with the same episode
# count. "day" and "row" are kept only to reproduce earlier floors (see 24.3).
NULL_UNIT = os.environ.get("VWAP_NULL_UNIT", "day_strata")
# "day"        the shipped shuffle: positional pairing, ~2 donor days per
#              destination day (24.3). Kept to reproduce the 0-of-12 calibration.
# "day_strata" the corrected one: whole days swap among days with the same
#              episode count. 100% single-donor, exact row permutation.
# "row"        the original, kept to reproduce the 2026-09-25 floors.
JOINT_NULL = True
JOINT_SEED = 20260927
JOINT_DAY_ORIGIN = 14245     # 2009-01-01 in days since epoch
JOINT_DAY_SPAN = 8000        # covers to 2030
FLOORS = CORES.replace("cores_v2", "floors_v2")

def _as_day_int(d):
    a = np.asarray(d)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int64)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[D]").astype(np.int64)
    return pd.to_datetime(a).values.astype("datetime64[D]").astype(np.int64)


def _day_order(uniq, rng, perm_index):
    """Random ordering of `uniq` days: per-cell, or joint across cells."""
    if not JOINT_NULL or perm_index is None:
        return rng.permutation(len(uniq))
    u = np.random.default_rng([JOINT_SEED, perm_index]).random(JOINT_DAY_SPAN)
    return np.argsort(u[_as_day_int(uniq) - JOINT_DAY_ORIGIN], kind="stable")
def shuffle_within_block(r, cols, rng, into=None, unit="day", perm_index=None):
    """Move covariate rows among setups, separately inside each block.

    unit="row"         each episode's covariate row moves independently. The
        original. Kept so the 2026-09-25 floors reproduce exactly.
    unit="day"         the 2026-09-26 shuffle. Rows are grouped by day, the day
        groups are reordered, and the two streams are paired off BY POSITION.
        Days differ in length, so the group boundaries do not line up and a
        destination day's covariates come from about two donor days -- measured
        20-37% single-donor (24.3). This is NOT the whole-day permutation the
        handoff described, but it IS what the 0-of-12 calibration validated,
        so it is kept unchanged and is the default.
    unit="day_strata"  the corrected one (24.3a). Whole days swap, and only
        among days with the SAME episode count, so every destination day's
        covariates come from ONE donor day and the frame keeps the exact
        multiset of covariate rows. A day whose episode count is unique inside
        its block has no partner and cannot move; `frozen.py` measures how many
        rows that strands.

    perm_index is the permutation number k. With JOINT_NULL it selects the
    shared day ordering, so cell A and cell B move the same days the same way
    on permutation k and the nulls are drawn jointly (24.6).

    `into` reuses a frame allocated once instead of copying the whole cell per
    permutation (2026-09-24). A cell can be 87 columns x 90,000 rows, so the
    old `r.copy()` moved ~60 MB per permutation, 100 times per side -- memory
    traffic, which is what saturates when every core is busy.
    """
    blocks = r.block.to_numpy()
    vals = r[cols].to_numpy()
    new = vals.copy()
    if unit == "row":
        for B in np.unique(blocks):
            ix = np.where(blocks == B)[0]
            new[ix] = vals[rng.permutation(ix)]
    else:
        days = r.day.to_numpy()
        for B in np.unique(blocks):
            ix = np.where(blocks == B)[0]
            d = days[ix]
            order = np.argsort(d, kind="stable")          # group the block's days
            ix_s, d_s = ix[order], d[order]
            starts = np.r_[0, np.where(d_s[1:] != d_s[:-1])[0] + 1]
            groups = np.split(ix_s, starts[1:])           # one array of rows per day
            uniq = d_s[starts]                            # the day values
            if unit == "day_strata":
                lens = np.array([len(g) for g in groups])
                for L in np.unique(lens):
                    w = np.where(lens == L)[0]
                    if len(w) < 2:
                        continue                          # no partner: cannot move
                    sub = _day_order(uniq[w], rng, perm_index)
                    for j, gi in enumerate(w):
                        new[groups[gi]] = vals[groups[w[sub[j]]]]
            else:
                perm = _day_order(uniq, rng, perm_index)
                src_rows = np.concatenate([groups[p] for p in perm])
                dst_rows = np.concatenate(groups)
                new[dst_rows] = vals[src_rows[:len(dst_rows)]]
    rp = r.copy() if into is None else into
    rp[cols] = new
    return rp


def floor_cell(cell, side, stored, nperm, rng):
    r, c, T = load_cell(cell, side)
    cols = search_cols(r, side)        # AUD-39: identical column set to the search
    P = prep(r, c)              # uses only days, blocks, hits -> shuffle-proof
    obs = best_core(r, c, P, cols)
    same = obs is not None and np.isclose(obs["score"], stored, rtol=0, atol=1e-12)
    print(f"  [{'PASS' if same else 'FAIL'}]  {cell} side {side}: re-run score "
          f"{100 * obs['score'] if obs else float('nan'):+.4f} vs stored "
          f"{100 * stored:+.4f}", flush=True)
    if not same:
        return None
    null = np.empty(nperm)
    scratch = r.copy()                 # allocated ONCE, overwritten per permutation
    t0 = time.time()
    for k in range(nperm):
        b = best_core(shuffle_within_block(r, cols, rng, into=scratch,
                                           unit=NULL_UNIT, perm_index=k),
                      c, P, cols)
        null[k] = b["score"] if b is not None else -np.inf
        if k == 0:
            print(f"      one permutation took {time.time() - t0:.1f}s -> "
                  f"~{(time.time() - t0) * nperm / 60:.0f} min for this cell",
                  flush=True)
    fin = null[np.isfinite(null)]
    floor95 = float(np.percentile(fin, 95)) if len(fin) else -np.inf
    p = (1 + int((null >= obs["score"]).sum())) / (1 + nperm)
    return dict(cell=cell, side=side, tau=float(TAU), T=T, col=obs["col"],
                obs_score=obs["score"], floor95=floor95,
                edge_over_floor=obs["score"] - floor95, p=p,
                eA=obs["eA"], eB=obs["eB"], seA=obs["seA"], seB=obs["seB"],
                nA=obs["nA"], nB=obs["nB"], nperm=nperm,
                n_null_empty=int((~np.isfinite(null)).sum()),
                null_scores=json.dumps([float(v) for v in null]), seed=SEED,
                null_unit=NULL_UNIT, joint=JOINT_NULL, joint_seed=JOINT_SEED,
                seed_rule=("default_rng([joint_seed, k]) per permutation, by day"
                           if JOINT_NULL else "default_rng([seed, crc32(tag)])"))

if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "N*"
    nperm = int(sys.argv[2]) if len(sys.argv) > 2 else NPERM_DEFAULT
    os.makedirs(FLOORS, exist_ok=True)
    # AUD-44 (2026-09-24): the pattern used to be glued to "_*_side*", so a FULL
    # cell name ("N60_hh_ATR_3.0") matched nothing and the script exited in
    # silence. Accept a prefix at any depth, and say so when nothing matches.
    files = sorted(set(glob.glob(f"{CORES}/{pat}_*_side*.parquet"))
                   | set(glob.glob(f"{CORES}/{pat}*_side*.parquet")))
    if not files:
        raise SystemExit(f"no result files in cores_v2\\ match '{pat}'. Give a cell "
                         f"name (N60_hh_ATR_3.0), a prefix (N60_hh) or N*.")
    print(f"  {len(files)} cell+side floors to consider for '{pat}' "
          f"({nperm} permutations each)", flush=True)
    for f in files:
        tag = os.path.basename(f)[:-8]
        outf = f"{FLOORS}/{tag}.parquet"
        if os.path.exists(outf):
            continue
        core = pd.read_parquet(f)
        if core.empty:
            continue                      # no candidate -> nothing to gate
        cell, s = tag.rsplit("_side", 1)
        side = None if s == "None" else float(s)
        # per-cell seed from the cell's NAME: results never depend on run
        # order or on which cells a resumed run had already finished
        rng = np.random.default_rng([SEED, zlib.crc32(tag.encode())])
        row = floor_cell(cell, side, float(core.score.iloc[0]), nperm, rng)
        if row is None:
            print(f"      NOT WRITTEN -- the floor would not match the candidate")
            continue
        safeio.write_parquet(pd.DataFrame([row]), outf)   # atomic (C112)
        print(f"      score {100*row['obs_score']:+.2f}  floor95 "
              f"{100*row['floor95']:+.2f}  p {row['p']:.3f}  "
              f"(raw eA {100*row['eA']:+.2f} eB {100*row['eB']:+.2f})", flush=True)
