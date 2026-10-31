"""
compare_old_new.py -- run ONCE after the rebuild. Two jobs:

1. KNOWN-NUMBER CHECK (D3). The re-arm rule starts from the OLD trigger, so the
   new dump's CANDIDATES are the old dump's SETUPS -- except where decision B
   (2026-09-22) changed the arithmetic: the old line used float32 prices and a
   floating SD, the new one is exact. A bar may therefore cross the eps
   threshold in one build and not the other.

   So this prints, per cell: how many bars differ, and for each differing bar
   how far its distance sits from eps. A PRECISION difference lands within a
   whisker of the threshold (|dist - eps| tiny). A RULE difference would not.
   The verdict is on that distance, not on "zero differences".

2. THE BURST RATE (R1: "record before overwriting"). Per cell: old setups,
   new setups, ratio old/new, per block. Written to rearm_burst.csv.

    python code\\compare_old_new.py                        (old = data\\dump_old_rule)
    python code\\compare_old_new.py D:\\some\\other\\old_dump
"""
import json
import os
import sys
import numpy as np
import pandas as pd

from loader import cell_files, DUMP_DIR, load, rolling_argextreme
from pipeline import session_anchor, vwap_and_sd
from rearm import bars_hash, eligible, candidates
from spec import t_stop, H_REC_MULT
import safeio

TOL = 1e-3          # |dist - eps|, in eps units: "a whisker" -- for REPORTING,
#                     no verdict is taken from it (E2); the numbers are printed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "dump_old_rule")

if __name__ == "__main__":
    if not os.path.isdir(OLD):
        raise SystemExit(f"old dump folder not found: {OLD}")
    d = load("train")
    seg = d.seg.to_numpy(); atr = d.atr.to_numpy(); close = d.close.to_numpy()
    cache = {}

    def cand_bars(N, akey, trig, eps):
        """The new build's candidates, recomputed (the dump stores only their
        fingerprint), plus each bar's distance in eps units."""
        if (N, akey) not in cache:
            cache.clear()
            a = session_anchor(d) if akey == "session" else rolling_argextreme(
                d["high" if akey == "hh" else ("low" if akey == "ll" else "volume")]
                .to_numpy().astype(float), N, seg, akey != "ll")
            vw, sd, *_ = vwap_and_sd(d, a)
            cache[(N, akey)] = (a, vw, sd)
        a, vw, sd = cache[(N, akey)]
        el = eligible(d, a, vw, sd, atr, trig)
        scale = sd if trig == "SD" else atr
        dist = np.where(el, np.abs(close - vw) / np.maximum(scale, 1e-9), np.nan)
        h = H_REC_MULT * t_stop(trig, eps)
        return candidates(dist, a, el, eps, len(d), h), dist

    rows, bad = [], []
    for f in cell_files("N*", DUMP_DIR):
        cell = os.path.basename(f)[:-8]
        cj = os.path.join(DUMP_DIR, cell + ".counts.json")
        of = os.path.join(OLD, cell + ".parquet")
        c = json.load(open(cj))
        if not os.path.exists(of):
            bad.append(f"{cell}: no old file"); continue
        o = pd.read_parquet(of, columns=["bar", "is_real", "block"])
        o = o[o.is_real == 1]
        same = bars_hash(o.bar.to_numpy()) == c["candidates_sha"]
        ndiff = 0; worst = 0.0
        if not same:
            N, akey, trig, eps = (int(cell.split("_")[0][1:]), cell.split("_")[1],
                                  cell.split("_")[2], float(cell.split("_")[3]))
            cb, dist = cand_bars(N, akey, trig, eps)
            diff = np.array(sorted(set(cb.tolist()) ^ set(o.bar.tolist())), dtype=np.int64)
            ndiff = len(diff)
            if ndiff:
                gap = np.abs(dist[diff] - eps) / eps          # NaN = not eligible now
                worst = float(np.nanmax(gap)) if np.isfinite(gap).any() else float("nan")
            bad.append(f"{cell}: {ndiff} bars differ, worst |dist-eps|/eps = {worst:.2e}")
        rows.append(dict(cell=cell, old_setups=len(o), candidates=c["n_candidates"],
                         bars_differ=ndiff, worst_gap=worst,
                         new_setups=c["n_setups"], same_bars=same,
                         burst=len(o) / max(c["n_setups"], 1),
                         old_A=int((o.block == "A").sum()), new_A=c["n_setups_by_block"]["A"],
                         old_B=int((o.block == "B").sum()), new_B=c["n_setups_by_block"]["B"]))
    t = pd.DataFrame(rows)
    safeio.write_parquet(t, os.path.join(HERE, "rearm_burst.parquet"))
    tmp = os.path.join(HERE, "rearm_burst.csv.tmp")
    t.to_csv(tmp, index=False); os.replace(tmp, os.path.join(HERE, "rearm_burst.csv"))
    n = len(t); ok = int(t.same_bars.sum()) if n else 0
    print(f"  {ok}/{n} cells reproduce the old setups bar for bar (ref 180 cells)")
    if n and not t.same_bars.all():
        dd = t[~t.same_bars]
        print(f"  {len(dd)} cells differ: {int(dd.bars_differ.sum())} bars in total, "
              f"{100 * dd.bars_differ.sum() / max(t.old_setups.sum(), 1):.4f}% of old setups")
        print(f"  worst |dist - eps| / eps among them: {np.nanmax(dd.worst_gap):.2e}  "
              f"(precision differences sit at the threshold; a rule change would not)")
    for b in bad[:12]:
        print("        ", b)
    if n:
        print(f"  old setups {t.old_setups.sum():,}  ->  new setups {t.new_setups.sum():,}  "
              f"(overall burst ratio {t.old_setups.sum() / max(t.new_setups.sum(), 1):.2f})")
        print("  per-cell burst ratio: median %.2f, min %.2f, max %.2f" %
              (t.burst.median(), t.burst.min(), t.burst.max()))
    print("  written: rearm_burst.csv")
