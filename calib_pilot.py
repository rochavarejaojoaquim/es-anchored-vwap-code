r"""
calib_pilot.py -- STAGE 1 of the redesigned null calibration (HANDOFF 26.71, audit G3).
A PILOT: it measures what the main run's registration needs. It draws NO verdict.

WHAT IT MEASURES, on 5 cell+sides chosen by SIZE (10/30/50/70/90th percentile of the 317
effective cell+sides' nA+nB; never by outcome, E4), 2 reps each, B = 100:
  C1  SHIFT dead cell + the PRODUCTION null (day_strata, joint perm_index=k)   the new test
  C2  SHIFT dead cell + the ROW null                                            POSITIVE CONTROL:
      23.4 showed the row null liberal (4 of 12); a calibration that cannot flag it is useless
  C3  the OLD row-shuffle dead cell (floor_calibration.dead_cell, unchanged) + production null
      -- one factor at a time (A2): C1 vs C3 isolates the dead-cell change
  per test: integer k = #{null >= obs} (floor_v2's rule), p = (1+k)/(1+B), runtime
  per cell: persistence -- integrated autocorrelation time (in trade days) of the day-level
      setup hit rate per block, the input the main run's shift rule will be derived from

THE SHIFT DEAD CELL. Within each block, setups are put in time order (bar) and their hit
vector is CIRCULARLY SHIFTED by an offset drawn uniformly from the middle third of the block
([n/3, 2n/3] setups); twins likewise, with their own offset. Covariates, days, blocks and twin
links are untouched; the outcome series keeps its persistence; its alignment with the
covariates is broken. The middle third is the PILOT's rule only; the main run's offset rule is
registered from the persistence measured here.

    python calib_pilot.py --selftest
    python calib_pilot.py --plan                      list the 5 cells and 30 jobs, run nothing
    python calib_pilot.py --workers 9                 the pilot (resumable; parts in calib_pilot_parts\)
    python calib_pilot.py --merge                     summary + calib_pilot_<date>.csv

Seeds: default_rng([20261003, condition, crc32(cell|side), rep]) -- no hash(), reproducible.
Never writes calibration.csv. Project root; rotates nothing.
"""
import datetime
import glob
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "calib_pilot_parts")
SEED, NPERM, NREP = 20261003, 100, 2
CONDS = {"C1": ("shift", "day_strata"), "C2": ("shift", "row"), "C3": ("rowshuf", "day_strata")}
QUANTS = (0.10, 0.30, 0.50, 0.70, 0.90)

# --set=archetype (HANDOFF 26.73, brain/13 C144): THE POSITIVE CONTROL DONE PROPERLY. 23.4's
# row-null failure (4 of 12 dead tests beat the floor) was measured on FOUR SPECIFIC
# cell+sides with the ROW-SHUFFLE dead cell. This mode reproduces it on those cells, adds
# C4 = row-shuffle + row null (the one combination known to fail, missing from the pilot),
# runs 3 reps as 23.4 did, and records BOTH rules per test: the registered k <= 4 at B = 100
# and the old "obs > 95th percentile of the nulls", also on the FIRST 30 nulls (23.4 used
# nperm 30). Its own parts folder and output file; the pilot's are untouched.
ARCHETYPE = [("N60_session_ATR_6.0", "1.0"), ("N200_ll_ATR_4.0", "1.0"),
             ("N60_ll_SD_1.0", "1.0"), ("N60_hh_ATR_3.0", "None")]        # HANDOFF.md:2453-2456
SET = "archetype" if "--set=archetype" in sys.argv else "pilot"
if SET == "archetype":
    PARTS = os.path.join(HERE, "calib_arch_parts")
    NREP = 3
    CONDS = dict(CONDS, C4=("rowshuf", "row"))


# ------------------------------------------------------------------ dead cells
def _shift_side(x, rng):
    """Circular shift of the hit vector within each block, in time (bar) order."""
    hit = x["hit"].to_numpy().copy()
    out = hit.copy()
    blocks = x["block"].to_numpy()
    bars = x["bar"].to_numpy()
    offsets = {}
    for B in np.unique(blocks):
        ix = np.where(blocks == B)[0]
        ix = ix[np.argsort(bars[ix], kind="stable")]
        n = len(ix)
        if n < 3:
            continue
        off = int(rng.integers(n // 3, 2 * n // 3 + 1))
        out[ix] = np.roll(hit[ix], off)
        offsets[B] = (off, n)
    y = x.copy()
    y["hit"] = out
    return y, offsets


def dead_cell_shift(r, c, rng):
    r2, offr = _shift_side(r, rng)
    c2, offc = _shift_side(c, rng)
    return r2, c2, dict(setup=offr, twin=offc)


def seeds(cond, cell, side_s, rep):
    return [SEED, int(cond[1]), zlib.crc32(f"{cell}|{side_s}".encode()), rep]


def k_count(obs, null):
    null = np.asarray(null, float)
    return int((null >= obs).sum())


def tau_int(series, c=5.0):
    """Integrated autocorrelation time, tau = 1 + 2 sum_{t=1..M} rho(t), with Sokal's
    adaptive window: the smallest M with M >= c * tau(M). Stopping at the first
    non-positive lag (the first version) truncates on noisy series and under-reads
    persistence -- self-test K5 caught it."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    if len(x) < 10 or x.std() == 0:
        return float("nan")
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau = 1.0
    for M in range(1, n // 2):
        tau = 1.0 + 2.0 * acf[1:M + 1].sum()
        if M >= c * tau:
            break
    return float(tau)


# ------------------------------------------------------------------ one job
def run_one(cond, cell, side_s, rep, nperm=NPERM, parts=None):
    parts = parts or PARTS
    os.makedirs(parts, exist_ok=True)
    tag = f"{cond}__{cell}__side{side_s}__rep{rep}"
    outf = os.path.join(parts, tag + ".csv")
    if os.path.exists(outf):
        return 0
    sys.path[:0] = [HERE, os.path.join(HERE, "code")]
    import core_v2
    import floor_v2
    from floor_calibration import dead_cell
    dead, unit = CONDS[cond]
    if unit == "day_strata" and floor_v2.NULL_UNIT != "day_strata":
        raise SystemExit(f"production NULL_UNIT is {floor_v2.NULL_UNIT}, expected day_strata")
    side = None if side_s in ("None", "none") else float(side_s)
    r0, c0, T = core_v2.load_cell(cell, side)
    cols = core_v2.search_cols(r0, side)
    rng = np.random.default_rng(seeds(cond, cell, side_s, rep))
    if dead == "shift":
        r, cc, offs = dead_cell_shift(r0, c0, rng)
    else:
        r, cc = dead_cell(r0, c0, rng)
        offs = {}
    P = core_v2.prep(r, cc)
    t0 = time.time()
    obs = core_v2.best_core(r, cc, P, cols)
    if obs is None:
        pd.DataFrame([dict(cond=cond, cell=cell, side=side_s, rep=rep, status="no candidate")]
                     ).to_csv(outf, index=False)
        return 0
    null, scratch = np.empty(nperm), r.copy()
    for k in range(nperm):
        rp = floor_v2.shuffle_within_block(r, cols, rng, into=scratch, unit=unit, perm_index=k)
        b = core_v2.best_core(rp, cc, P, cols)
        null[k] = b["score"] if b is not None else -np.inf
    k_ = k_count(obs["score"], null)
    b95, f95 = beat95(obs["score"], null)
    b30, f30 = beat95(obs["score"], null[:30])
    import platform
    pd.DataFrame([dict(set=SET, cond=cond, dead=dead, null_unit=unit, cell=cell, side=side_s,
                       rep=rep, status="ok", col=obs["col"], obs_score=obs["score"], k=k_,
                       p=(1 + k_) / (1 + nperm), nperm=nperm, floor95=f95, beat95=b95,
                       floor95_first30=f30, beat95_first30=b30, secs=time.time() - t0,
                       offsets=str(offs), seed=str(seeds(cond, cell, side_s, rep)),
                       machine=platform.node())]
                 ).to_csv(outf, index=False)
    return 0


def beat95(obs, null):
    """The OLD rule (floor_calibration.py:128-130): obs > 95th percentile of the finite nulls."""
    fin = np.asarray(null, float)
    fin = fin[np.isfinite(fin)]
    f95 = float(np.percentile(fin, 95)) if len(fin) else float("-inf")
    return bool(obs > f95), f95


# ------------------------------------------------------------------ plan / pool / merge
def pick_cells():
    if SET == "archetype":
        u = pd.read_csv(os.path.join(HERE, "rebuild_pseudo_2026-09-30", "unmatched_by_cell.csv"))
        size = dict(zip(u.cell, u.setups))
        return [(c, s, int(size[c])) for c, s in ARCHETYPE]
    t = pd.read_csv(os.path.join(HERE, "rw_check_independent_out.csv"))
    if len(t) != 317:
        raise SystemExit(f"expected 317 effective rows, found {len(t)}")
    # SIZE = the cell's real setups (NOT nA+nB, which is the size of the WINNING SUBSET --
    # an outcome-dependent quantity, caught by --plan on the first version)
    u = pd.read_csv(os.path.join(HERE, "rebuild_pseudo_2026-09-30", "unmatched_by_cell.csv"))
    size = dict(zip(u.cell, u.setups))
    t["size"] = t.cell.map(size)
    if t["size"].isna().any():
        raise SystemExit("a cell has no setup count in unmatched_by_cell.csv")
    t = t.sort_values(["size", "cell", "side"]).reset_index(drop=True)
    picks = []
    for q in QUANTS:
        row = t.iloc[int(round(q * (len(t) - 1)))]
        picks.append((row.cell, "None" if pd.isna(row.side) else str(float(row.side)), int(row["size"])))
    return picks


def jobs():
    return [(cond, c, s, rep) for cond in CONDS for c, s, _ in pick_cells() for rep in range(NREP)]


def run_pool(workers):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    queue, live, done, failed = list(jobs()), {}, 0, 0
    t0 = last = time.time()
    while queue or live:
        while queue and len(live) < workers:
            cond, cell, side_s, rep = queue.pop(0)
            tag = f"{cond}__{cell}__side{side_s}__rep{rep}"
            if os.path.exists(os.path.join(PARTS, tag + ".csv")):
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--one", f"{cond}:{cell}:{side_s}:{rep}"]
            if SET == "archetype":
                cmd.append("--set=archetype")
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
            log = open(os.path.join(HERE, "logs", f"calibpilot_{tag}.log"), "w")
            live[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)] = (tag, log)
        for p in list(live):
            if p.poll() is None:
                continue
            tag, log = live.pop(p); log.close()
            if p.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"  [FAIL] {tag} rc={p.returncode} -- logs\\calibpilot_{tag}.log", flush=True)
        if time.time() - last > 60:
            last = time.time()
            print(f"  ... {done} done, {len(live)} running, {len(queue)} queued, "
                  f"{(time.time()-t0)/60:.0f} min", flush=True)
        time.sleep(0.5)
    print(f"\n  {done} done, {failed} failed, {(time.time()-t0)/60:.1f} min wall")
    return failed


def persistence(picks):
    sys.path[:0] = [HERE, os.path.join(HERE, "code")]
    import core_v2
    rows = []
    for cell, side_s, _ in picks:
        r, c, T = core_v2.load_cell(cell, None if side_s == "None" else float(side_s))
        for B in ("A", "B"):
            d = r[r.block == B].groupby("day").hit.mean().sort_index()
            rows.append(dict(cell=cell, side=side_s, block=B, days=len(d), tau_int_days=tau_int(d)))
    return pd.DataFrame(rows)


def merge():
    fs = sorted(glob.glob(os.path.join(PARTS, "*.csv")))
    if not fs:
        raise SystemExit("no parts yet")
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    ok = t[t.status == "ok"]
    print(f"\nCALIBRATION {SET.upper()} -- {len(t)} parts ({len(ok)} ok), B = {NPERM}. "
          f"DESCRIPTIVE, NO VERDICT.")
    old = "beat95" in ok.columns
    print("  cond  dead     null        tests  k<=4  k==0" + ("  beat95  beat95@30" if old else "")
          + "  mean secs/test   k sorted")
    for cond, g in ok.groupby("cond"):
        extra = (f"  {int(g.beat95.sum()):>6}  {int(g.beat95_first30.sum()):>9}" if old else "")
        print(f"  {cond}    {g.dead.iloc[0]:<8} {g.null_unit.iloc[0]:<11} {len(g):>5} "
              f"{int((g.k <= 4).sum()):>5} {int((g.k == 0).sum()):>5}{extra}  {g.secs.mean():>8.0f}"
              f"         {sorted(int(v) for v in g.k)}")
    if SET == "archetype":
        print("  REFERENCE (HANDOFF 23.4, nperm 30, 95th-percentile rule, row-shuffle dead cell):")
        print("    row null 4 of 12 beat the floor; day null 0 of 12. C4's beat95@30 is the")
        print("    like-for-like column; seeds differ, so the count, not the cells, is compared.")
    print("  (nominal: k<=4 in 4.95%, k==0 in 0.99% of tests; with 10 tests per condition these")
    print("   counts are too few to judge size -- the pilot measures cost, persistence and the")
    print("   positive control's direction, which feed the main run's registration)")
    per = persistence(pick_cells())
    print("\n  PERSISTENCE: integrated autocorrelation time of the day-level setup hit rate")
    for _, r in per.iterrows():
        print(f"    {r.cell:<22} side {r.side:<5} block {r.block}: {r.days:>5} days, "
              f"tau_int {r.tau_int_days:6.2f} days")
    path = os.path.join(HERE, f"calib_{SET}_{datetime.date.today().isoformat()}.csv")
    t.to_csv(path, index=False)
    per.to_csv(path.replace(".csv", "_persistence.csv"), index=False)
    print(f"\n  wrote {os.path.basename(path)} and the _persistence file")


# ------------------------------------------------------------------ self-test
def selftest():
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else ""))
        ok[0] &= bool(cond)
    print("\ncalib_pilot.py SELF-TEST -- toy data only.\n")
    rng = np.random.default_rng(1)
    n = 6000
    blk = np.where(np.arange(n) < n // 2, "A", "B")
    regime = np.repeat(rng.normal(0, 1, n // 50), 50)              # 50-setup regimes
    cov = regime + rng.normal(0, 0.3, n)
    hit = (rng.random(n) < 0.5 + 0.15 * np.tanh(regime)).astype(float)
    r = pd.DataFrame(dict(bar=np.arange(n), block=blk, day=np.arange(n) // 10, hit=hit, cov=cov))
    c = r.copy()
    r2, c2, offs = dead_cell_shift(r, c, np.random.default_rng(2))
    chk("K1 the shift keeps each block's hit multiset (same counts)",
        all(r2.hit[r2.block == B].sum() == r.hit[r.block == B].sum() for B in "AB"))
    off_ok = all(n_ // 3 <= o <= 2 * n_ // 3 for o, n_ in offs["setup"].values())
    chk("K2 offsets lie in the middle third of each block", off_ok, str(offs["setup"]))
    lag1 = lambda h: np.corrcoef(h[:-1], h[1:])[0, 1]
    chk("K3 persistence is kept: lag-1 autocorrelation of hits unchanged within 0.02",
        abs(lag1(r2.hit[r2.block == "A"].to_numpy()) - lag1(r.hit[r.block == "A"].to_numpy())) < 0.02,
        f"{lag1(r.hit[r.block=='A'].to_numpy()):.3f} -> {lag1(r2.hit[r2.block=='A'].to_numpy()):.3f}")
    before = np.corrcoef(r["cov"], r["hit"])[0, 1]; after = np.corrcoef(r2["cov"], r2["hit"])[0, 1]
    chk("K4 the covariate-outcome link is broken: |corr| drops below a third of its planted size",
        abs(after) < abs(before) / 3, f"{before:+.3f} -> {after:+.3f}")
    ar = np.zeros(20000)
    e = rng.normal(0, 1, 20000)
    for i in range(1, 20000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    t_ar, t_wn = tau_int(ar), tau_int(rng.normal(0, 1, 20000))
    chk("K5 tau_int recovers a KNOWN answer: AR(1) phi 0.8 has tau = 9; white noise ~1",
        abs(t_ar - 9.0) < 1.5 and abs(t_wn - 1.0) < 0.3, f"AR {t_ar:.2f} (true 9), noise {t_wn:.2f}")
    chk("K6 k counts null >= obs as an integer (floor_v2's rule); p = (1+k)/101",
        k_count(0.5, [0.1, 0.5, 0.7, 0.2]) == 2)
    # M1 a zero offset would NOT break the link -- the offset is load-bearing
    z = r.copy()
    chk("M1 a zero shift leaves the link intact (so K4 is not vacuous)",
        abs(np.corrcoef(z["cov"], z["hit"])[0, 1]) > 0.1, f"{np.corrcoef(z["cov"], z["hit"])[0,1]:+.3f}")
    s1 = seeds("C1", "N60_ll_ATR_3.0", "1.0", 0)
    chk("K7 seeds are deterministic and differ by condition and rep",
        s1 == seeds("C1", "N60_ll_ATR_3.0", "1.0", 0) and s1 != seeds("C2", "N60_ll_ATR_3.0", "1.0", 0)
        and s1 != seeds("C1", "N60_ll_ATR_3.0", "1.0", 1))
    nn = np.arange(100) / 100.0                       # 95th percentile = 0.9405
    chk("K8 the old rule: obs beats the 95th percentile of the nulls, hand-checked",
        beat95(0.95, nn) == (True, float(np.percentile(nn, 95))) and beat95(0.94, nn)[0] is False,
        f"floor95 {beat95(0.95, nn)[1]:.4f}")
    chk("M2 a non-finite null is dropped from the percentile, not counted as a low score",
        beat95(0.5, np.array([-np.inf, 0.4, 0.6]))[1] == float(np.percentile([0.4, 0.6], 95)))
    if SET == "archetype":
        j = jobs()
        chk("K9 archetype: 23.4's 4 cell+sides x 4 conditions x 3 reps = 48 jobs, C4 = rowshuf+row",
            len(j) == 48 and CONDS["C4"] == ("rowshuf", "row") and len({(c, s) for _, c, s, _ in j}) == 4,
            f"{len(j)} jobs")
    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    if "--one" in a:
        cond, cell, side_s, rep = a[a.index("--one") + 1].split(":")
        sys.exit(run_one(cond, cell, side_s, int(rep)))
    if "--plan" in a:
        for c, s, n in pick_cells():
            print(f"  {c:<22} side {s:<5} size {n:,}")
        print(f"  {len(jobs())} jobs = {len(CONDS)} conditions x {len(pick_cells())} cells x "
              f"{NREP} reps, B = {NPERM}")
        sys.exit(0)
    if "--merge" in a:
        merge(); sys.exit(0)
    if "--preflight" in a:
        # all three conditions on the SMALLEST pilot cell, 2 permutations, into a temp dir:
        # proves the real-data path runs; prints only PASS/FAIL and the time per search
        import tempfile
        tmp = tempfile.mkdtemp()
        cell, side_s, _ = pick_cells()[0]
        res = []
        for cond in CONDS:
            t0 = time.time()
            run_one(cond, cell, side_s, 0, nperm=2, parts=tmp)
            g = pd.read_csv(os.path.join(tmp, f"{cond}__{cell}__side{side_s}__rep0.csv"))
            ok_ = g.status.iloc[0] == "ok" and int(g.nperm.iloc[0]) == 2 and 0 <= int(g.k.iloc[0]) <= 2
            res.append(ok_)
            print(f"  [{'PASS' if ok_ else 'FAIL'}] {cond} {CONDS[cond]} ran on {cell} side {side_s}: "
                  f"3 searches in {time.time()-t0:.0f}s (no calibration value shown)")
        print(f"\n  PREFLIGHT {'PASS' if all(res) else 'FAIL'}  (temp dir {tmp}; nothing in the project)")
        sys.exit(0 if all(res) else 1)
    if not selftest():
        raise SystemExit("SELF-TEST FAILED")
    w = int(a[a.index("--workers") + 1]) if "--workers" in a else 9
    failed = run_pool(w)
    merge()
    sys.exit(1 if failed else 0)
