r"""blocklen_sensitivity.py -- is the 5-day block bootstrap's block length load-bearing?

    python blocklen_sensitivity.py --selftest        toy + invariants, reads no dump
    python blocklen_sensitivity.py 12                the 12 largest stored searches
    python blocklen_sensitivity.py 12 --bl 1,2,5,10,20
    python blocklen_sensitivity.py 12 --part=1/4     interleaved share (as runner.py)

WHY. HANDOFF.md:319 records a MEASURED reason to block at all ("naive SE understates
by 1.7-3x"), but nothing in the tree derives the number 5. brain/17 17.10 logs this as
an A5-class item: a parameter every published score depends on, carrying no
justification. The question is not whether 5 is "right" in the abstract -- it is
whether any reported result MOVES with it.

WHAT IT DOES. For each stored cores_v2 winner it rebuilds that winner's mask exactly
as core_v2.best_core does (core_v2.py:242-244) and re-runs ONLY the error bar, at
each block length, through the installed core_v2.edge_se. It does NOT re-run the
search, so it answers "does the pessimistic SCORE move with b?" -- the question that
matters for the 195 and the 31. It does NOT answer "does the WINNER change with b?";
that needs a full re-search per b and is a separate, much more expensive job.

CELLS ARE CHOSEN BY FILE SIZE, largest first, exactly as verify_speedup.py does.
That is selection on SAMPLE SIZE, never on the outcome, so it is not selection on the
result (E4) and it exercises the bootstrap hardest.

NO EDIT TO code\ IS NEEDED. edge_se already takes blocklen and rebuilds its selection
table for any value other than the default (core_v2.py:166, :176-177).

READS data\dump and cores_v2. WRITES NOTHING -- print only; redirect if you want a file.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
import core_v2                                                        # noqa: E402

DEFAULT_BL = [1, 2, 5, 10, 20]
NBOOT = 500                      # best_core re-estimates the winner at 500 (core_v2.py:245)
FIELDS = ["eA", "eB", "seA", "seB", "nA", "nB"]


# ----------------------------------------------------------------- self-test
def selftest():
    """Known answers and invariants first, then every mutation must FAIL."""
    print("SELF-TEST -- known answers and invariants, then each mutation must FAIL\n")
    res = []

    def chk(name, ok, detail=""):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<64} {detail}")

    # K1 boot_index is a prefix in nboot. core_v2.py:119-120 claims exactly this:
    # "a 150-draw run is the first 150 rows of a 300-draw one".
    a = core_v2.boot_index(400, 300, 5, 1)
    b = core_v2.boot_index(400, 150, 5, 1)
    chk("K1 boot_index(.,300)[:150] == boot_index(.,150)  (docstring :119-120)",
        isinstance(a, np.ndarray) and np.array_equal(a[:150], b), f"{a.shape} vs {b.shape}")

    # K2 structure: nblk = D//b runs of exactly b consecutive days, all inside [0,D).
    D, bl = 400, 5
    T = core_v2.boot_index(D, 50, bl, 1)
    nblk = D // bl
    width_ok = T.shape[1] == nblk * bl
    runs_ok = True
    for row in T:
        for j in range(0, len(row), bl):
            blk = row[j:j + bl]
            if not (np.array_equal(blk, np.arange(blk[0], blk[0] + bl)) and 0 <= blk[0] <= D - bl):
                runs_ok = False
    chk("K2 every draw is D//b runs of b consecutive days inside [0,D)",
        width_ok and runs_ok and T.min() >= 0 and T.max() < D,
        f"width {T.shape[1]} == {nblk}x{bl}; range {T.min()}..{T.max()}")

    # K3 a longer block truncates the resample: width == (D//b)*b, which is <= D.
    w = {bb: core_v2.boot_index(D, 5, bb, 1).shape[1] for bb in (1, 2, 5, 10, 20, 30)}
    chk("K3 resample width == (D//b)*b for every b",
        all(w[bb] == (D // bb) * bb for bb in w), str(w))

    # K4 score() must FALL when an SE grows. Catches a sign error in the one
    # function every reported number passes through.
    s_small = core_v2.score((0.05, 0.010, 5000), (0.04, 0.010, 5000))
    s_big = core_v2.score((0.05, 0.020, 5000), (0.04, 0.020, 5000))
    chk("K4 score falls when SE grows (core_v2.score, Z99=2.6)",
        s_big < s_small, f"{s_small:.6f} -> {s_big:.6f}")

    # K5 and the hand-checkable value: score = min(|e|-2.6*se) over the two blocks.
    want = min(0.05 - 2.6 * 0.010, 0.04 - 2.6 * 0.010)
    chk("K5 score == min(|e| - 2.6*se) by hand", abs(s_small - want) < 1e-12,
        f"{s_small:.6f} vs {want:.6f}")

    # --- mutations -------------------------------------------------------
    m1 = core_v2.boot_index(400, 150, 5, 2)
    chk("M1 a DIFFERENT seed must break the prefix invariant of K1",
        not np.array_equal(a[:150], m1))
    m2 = core_v2.boot_index(400, 150, 6, 1)
    chk("M2 blocklen 6 must not satisfy K2's b=5 run structure",
        m2.shape[1] != nblk * bl, f"width {m2.shape[1]} != {nblk * bl}")
    chk("M3 the stored-vs-recomputed comparator catches a 1% change in one field",
        not agree({"seA": 0.01}, {"seA": 0.0101}), "1% on seA -> flagged")
    chk("M4 the comparator does NOT flag an exact match", agree({"seA": 0.01}, {"seA": 0.01}))

    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


def agree(stored, now, keys=None):
    """Exact float comparison, as verify_speedup.py does. No tolerance."""
    for k in (keys or now.keys()):
        if k not in stored:
            continue
        a, bb = stored[k], now[k]
        if (a != bb) if isinstance(bb, str) else (float(a) != float(bb)):
            return False
    return True


# ----------------------------------------------------------------- the run
def recompute(r, c, P, row, bl):
    """The stored winner's error bars at block length bl. Mask rebuilt exactly as
    core_v2.best_core does (core_v2.py:242-244).

    Also returns the two per-block terms and WHICH BLOCK BINDS, because
    score = min(|eA| - Z99*seA, |eB| - Z99*seB) is set by the binding block and a
    summary that reports only seA cannot explain a score move (brain/13 E31).
    """
    v = r[str(row["col"])].to_numpy(float)
    m = v <= float(row["cut"]) if str(row["sense"]) == "lo" else v > float(row["cut"])
    cm = core_v2.twin_mask(m, P["src_idx"])
    eA = core_v2.edge_se(r, c, P, m, cm, "A", blocklen=bl, nboot=NBOOT)
    eB = core_v2.edge_se(r, c, P, m, cm, "B", blocklen=bl, nboot=NBOOT)
    if eA is None or eB is None:
        return None
    Z = core_v2.Z99
    tA, tB = abs(eA[0]) - Z * eA[1], abs(eB[0]) - Z * eB[1]
    return dict(eA=eA[0], eB=eB[0], seA=eA[1], seB=eB[1], nA=eA[2], nB=eB[2],
                score=core_v2.score(eA, eB), tA=tA, tB=tB,
                bind="A" if tA <= tB else "B", gap=abs(tA - tB))


def main():
    if "--selftest" in sys.argv:
        return 0 if selftest() else 1
    sm = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--summarize=")), None)
    if sm:
        return summarize(sm)
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    want = int(pos[0]) if pos else 12
    bls = DEFAULT_BL
    if "--bl" in sys.argv:
        bls = [int(x) for x in sys.argv[sys.argv.index("--bl") + 1].split(",")]
    if 5 not in bls:
        raise SystemExit("ABORT: 5 must be in --bl; it is the known-number check.")

    if not selftest():
        raise SystemExit("\n  SELF-TEST FAILED -- no data was read.")

    files = [(os.path.getsize(f), f) for f in glob.glob(os.path.join(core_v2.OUT, "*_side*.parquet"))
             if os.path.getsize(f) > 1200]
    files = [f for _, f in sorted(files, reverse=True)][:want]
    part = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--part=")), None)
    if part:
        sel, k = part.split("/")
        keep, k = {int(x) for x in sel.split(",")}, int(k)
        files = [f for i, f in enumerate(files) if (i % k) + 1 in keep]
        print(f"  --part {part}: {len(files)} of the {want} largest\n", flush=True)
    if not files:
        raise SystemExit("no finished searches in cores_v2\\")

    print("\nBLOCK-LENGTH SENSITIVITY OF THE PESSIMISTIC SCORE")
    print("known-number check at b=5 FIRST, per cell, before any sensitivity number\n")
    t0 = time.time()
    rows, k1_bad = [], 0
    for f in files:
        t = pd.read_parquet(f)
        row = t.iloc[0]
        cell, side = str(row["cell"]), (None if pd.isna(row["side"]) else float(row["side"]))
        r, c, T = core_v2.load_cell(cell, side)
        P = core_v2.prep(r, c)
        got = {}
        for bl in bls:
            got[bl] = recompute(r, c, P, row, bl)
        stored = {k: row[k] for k in FIELDS}
        ok5 = got[5] is not None and agree(stored, got[5], FIELDS)
        k1_bad += not ok5
        tag = os.path.basename(f).replace("_side", " side").replace(".parquet", "")
        print(f"  [{'PASS' if ok5 else 'FAIL'}] b=5 reproduces stored  {tag}", flush=True)
        if not ok5:
            if got[5] is None:
                print("        edge_se returned None at b=5 -- cell too small; skipped")
            else:
                print("        " + "; ".join(f"{k}: stored {stored[k]} vs now {got[5][k]}"
                                             for k in FIELDS
                                             if float(stored[k]) != float(got[5][k])))
            continue
        base = got[5]
        for bl in bls:
            g = got[bl]
            if g is None:
                continue
            rows.append(dict(cell=cell, side=side, col=str(row["col"]), bl=bl,
                             seA=g["seA"], seB=g["seB"], score=g["score"],
                             d_score_pp=100.0 * (g["score"] - base["score"]),
                             se_ratio_A=g["seA"] / base["seA"] if base["seA"] else np.nan,
                             se_ratio_B=g["seB"] / base["seB"] if base["seB"] else np.nan,
                             bind=g["bind"], bind5=base["bind"],
                             switched=g["bind"] != base["bind"],
                             gap5_pp=100.0 * base["gap"],
                             # the ratio of the SE that actually SETS the score at b=5
                             se_ratio_bind=((g["seA"] / base["seA"]) if base["bind"] == "A"
                                            else (g["seB"] / base["seB"]))))
    print()
    print(f"  known-number check at b=5: {len(files) - k1_bad} of {len(files)} reproduce "
          f"the stored row exactly")
    if k1_bad:
        print("  A b=5 MISMATCH IS A STOP CONDITION. Nothing below may be read.")
        return 1
    if not rows:
        print("  no usable cells."); return 1

    df = pd.DataFrame(rows)
    print("\n  STAMP: %d cell+sides, chosen by stored-file SIZE (never by outcome) | "
          "winner's mask rebuilt from the stored (col, cut, sense) | nboot=%d, seed=1 | "
          "Kuensch moving-block, non-circular | code-set per verify_pc | scores are "
          "FRACTIONS here, x100 for pp | raw, no null | ES, blocks A+B | PC"
          % (df.cell.nunique(), NBOOT))
    print("\n  per block length. se_bind is the ratio of the SE that actually SETS")
    print("  the score at b=5; 'switch' counts cells where the BINDING BLOCK changed.")
    g = df.groupby("bl")
    print("    %-4s %9s %9s %9s %11s %7s %7s" %
          ("b", "seA/seA5", "seB/seB5", "se_bind", "max|dscore|", "switch", "n"))
    for bl in bls:
        if bl not in g.groups:
            continue
        s = g.get_group(bl)
        print("    %-4d %9.3f %9.3f %9.3f %11.4f %7d %7d"
              % (bl, s.se_ratio_A.mean(), s.se_ratio_B.mean(), s.se_ratio_bind.mean(),
                 s.d_score_pp.abs().max(), int(s.switched.sum()), len(s)))

    print("\n  PER CELL -- max |dscore| over the block lengths, and whether the")
    print("  binding block ever switched. gap5 is |tA - tB| at b=5 in pp: a small")
    print("  gap is what makes a switch possible at all.")
    print("    %-30s %5s %8s %9s %7s %-22s" %
          ("cell", "side", "gap5 pp", "max|dsc|", "switch", "col"))
    for (cell, side, col), s in df.groupby(["cell", "side", "col"], dropna=False):
        print("    %-30s %5s %8.4f %9.4f %7s %-22s"
              % (cell, side, s.gap5_pp.iloc[0], s.d_score_pp.abs().max(),
                 "YES" if s.switched.any() else "-", col[:22]))

    worst = df.loc[df.d_score_pp.abs().idxmax()]
    print("\n  LARGEST SCORE MOVE: %.4f pp at b=%d, %s side %s (%s)"
          % (worst.d_score_pp, worst.bl, worst.cell, worst.side, worst.col))
    print("    binds %s at b=5, binds %s at b=%d, switched=%s | gap5 %.4f pp"
          % (worst.bind5, worst.bind, worst.bl, worst.switched, worst.gap5_pp))
    print("    seA/seA5 %.4f  seB/seB5 %.4f  se_bind %.4f"
          % (worst.se_ratio_A, worst.se_ratio_B, worst.se_ratio_bind))
    print("    CHECK: a score move with no switch must equal Z99 x the binding SE's")
    print("    absolute change. Z99=%.1f, so expected %.4f pp from se_bind alone."
          % (core_v2.Z99,
             100.0 * core_v2.Z99 * abs(worst.se_ratio_bind - 1.0)
             * (worst.seA if worst.bind5 == "A" else worst.seB) / worst.se_ratio_bind))
    print("  (b=5 is the reference, so its row is 0 by construction.)")
    print("\n  INDICATIVE ONLY, not a PASS/FAIL: HANDOFF.md:319 records that the naive SE "
          "understates by 1.7-3x.")
    print("  b=1 here is a DAY-LEVEL iid resample. What :319 means by 'naive' is "
          "NOT DETERMINED, so")
    print("  the seA/seA5 column at b=1 is reported for comparison, not gated against "
          "1.7-3x.")
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    if out:
        df.to_csv(out, index=False)
        print("\n  wrote %s  (%d rows). Aggregate the shares with:" % (out, len(df)))
        print('    python blocklen_sensitivity.py --summarize="blocklen_part*.csv"')
    print(f"\n  {time.time() - t0:.0f}s")
    return 0


FLIP_31 = 1.3718     # pp: thinnest p-margin among the 31 (26.25, fixed BEFORE the sweep)
FLIP_195 = 0.0028    # pp: thinnest p-margin among the 195


def summarize(pattern):
    """Aggregate sharded --out files into ONE verdict against 26.25's criterion.

    The criterion was fixed in writing before the sweep ran. This only applies it.
    """
    import glob as _glob
    fs = sorted(_glob.glob(pattern))
    if not fs:
        raise SystemExit(f"no files matched {pattern!r}")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    cells = df.groupby(["cell", "side", "col"], dropna=False)
    print("AGGREGATE over %d share file(s): %s" % (len(fs), ", ".join(os.path.basename(f) for f in fs)))
    print("  %d cell+sides, %d rows, block lengths %s"
          % (cells.ngroups, len(df), sorted(df.bl.unique())))
    dup = int(cells.size().gt(len(df.bl.unique())).sum())
    print("  [%s] no cell appears in two shares (shares must be disjoint)"
          % ("PASS" if dup == 0 else "FAIL"))
    b5 = df[df.bl == 5]
    print("  [%s] every cell has its b=5 reference row  (%d of %d)"
          % ("PASS" if len(b5) == cells.ngroups else "FAIL", len(b5), cells.ngroups))

    mx = df.d_score_pp.abs().max()
    w = df.loc[df.d_score_pp.abs().idxmax()]
    print("\n  MAX |dscore| over all shares = %.4f pp" % mx)
    print("    at b=%d, %s side %s (%s) | binds %s->%s switched=%s | gap5 %.4f pp"
          % (w.bl, w.cell, w.side, w.col, w.bind5, w.bind, w.switched, w.gap5_pp))
    print("  binding-block switches: %d of %d rows, on %d distinct cells"
          % (int(df.switched.sum()), len(df),
             df[df.switched].groupby(["cell", "side", "col"], dropna=False).ngroups))
    sw = df[df.switched]
    if len(sw):
        print("    switches occur at gap5 up to %.4f pp; their |dscore| max is %.4f pp"
              % (sw.gap5_pp.max(), sw.d_score_pp.abs().max()))

    print("\n  VERDICT against 26.25's criterion, fixed BEFORE this sweep:")
    print("    threshold = %.4f pp (thinnest p-margin among the 31)" % FLIP_31)
    if mx < FLIP_31:
        print("    %.4f < %.4f  ->  NO Romano-Wolf survivor can be flipped by the"
              % (mx, FLIP_31))
        print("    block-length choice. Headroom %.2fx. Report the bound and that the"
              % (FLIP_31 / mx))
        print("    5-day convention is not load-bearing FOR THE 31.")
    else:
        print("    %.4f >= %.4f  ->  AT LEAST ONE of the 31 lies inside block-length"
              % (mx, FLIP_31))
        print("    noise. This is a FINDING. Report it; the 31 need a robustness")
        print("    qualifier. Do NOT re-run to make it go away.")
    print("\n  LIMIT, unchanged and still NOT MEASURED: the p-margins this is compared")
    print("  against come from nulls generated at b=5. floor_v2 re-runs the IDENTICAL")
    print("  procedure, so at another b the nulls move too. One cell's floor at b=10")
    print("  (nperm=100 x ~7.3 s, about 12 min) is the spot check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
