r"""
calib.py -- run floor_calibration's work with ONE PROCESS PER (cell, rep).

floor_calibration.py loops 4 cells x 3 reps in a single process: 12 jobs in
series, one core, ~10% CPU. Each (cell, rep) is independent -- line 107 builds
a fresh rng per rep -- so they can all run at once.

    python calib.py --cells A:1.0,B:-1.0,C:None --nrep 3 --nperm 30 --workers 12

REPRODUCIBILITY: floor_calibration line 107 seeds from abs(hash(cell)), and
Python randomizes string hashing per process. This wrapper pins
PYTHONHASHSEED=0 in every child so the same command gives the same answer
twice. That CHANGES the seeds relative to any earlier calibration run. The
value used is recorded in the output.

Writes calib_parts\<tag>.csv per job, then merges to calibration.csv with the
same PASS/FAIL rule as the original (>20% FAIL, <0.5% WARN, else PASS).

    python calib.py --selftest      scheduler test, no real work
"""
import glob
import os
import subprocess
import sys
import time

PARTS = "calib_parts"


def run_one(cell, side_s, rep, nperm, fake=False):
    import numpy as np
    import pandas as pd
    os.makedirs(PARTS, exist_ok=True)
    tag = f"{cell}__side{side_s}__rep{rep}"
    outf = os.path.join(PARTS, f"{tag}.csv")
    if os.path.exists(outf):
        return 0
    if fake:
        pd.DataFrame([dict(cell=cell, side=side_s, rep=rep, col="c0",
                           obs_score=0.01, floor95=0.02, edge_over_floor=-0.01,
                           beat=False, null_mean=0.0)]).to_csv(outf, index=False)
        return 0
    sys.path.insert(0, ".")
    sys.path.insert(0, "code")
    import core_v2
    import floor_v2
    from floor_calibration import dead_cell

    side = None if side_s in ("None", "none") else float(side_s)
    r0, c0, T = core_v2.load_cell(cell, side)
    cols = core_v2.search_cols(r0, side)
    rng = np.random.default_rng([12345, rep, abs(hash(cell)) % 10 ** 6])
    r, cc = dead_cell(r0, c0, rng)
    P = core_v2.prep(r, cc)
    t0 = time.time()
    obs = core_v2.best_core(r, cc, P, cols)
    if obs is None:
        print(f"  {cell} side {side} rep {rep}: no candidate at all")
        return 0
    null = np.empty(nperm)
    scratch = r.copy()
    for k in range(nperm):
        rp = floor_v2.shuffle_within_block(r, cols, rng, into=scratch,
                                           unit=floor_v2.NULL_UNIT)
        b = core_v2.best_core(rp, cc, P, cols)
        null[k] = b["score"] if b is not None else -np.inf
    fin = null[np.isfinite(null)]
    f95 = float(np.percentile(fin, 95)) if len(fin) else -np.inf
    beat = bool(obs["score"] > f95)
    pd.DataFrame([dict(cell=cell, side=str(side), rep=rep, col=obs["col"],
                       obs_score=obs["score"], floor95=f95,
                       edge_over_floor=obs["score"] - f95, beat=beat,
                       null_mean=float(fin.mean()) if len(fin) else np.nan,
                       null_unit=floor_v2.NULL_UNIT, nperm=nperm,
                       pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"))
                  ]).to_csv(outf, index=False)
    print(f"  {cell:<22} side {str(side):<5} rep {rep}  obs {obs['score']:+7.3f}  "
          f"floor95 {f95:+7.3f}  "
          f"{'BEATS THE FLOOR' if beat else 'below floor (expected)'}"
          f"   [{time.time() - t0:.0f}s]", flush=True)
    return 0


def run_pool(jobs, workers, nperm, fake=False):
    os.makedirs("logs", exist_ok=True)
    queue, live, done, failed = list(jobs), {}, 0, 0
    t0 = last = time.time()
    while queue or live:
        while queue and len(live) < workers:
            cell, side_s, rep = queue.pop(0)
            tag = f"{cell}__side{side_s}__rep{rep}"
            cmd = [sys.executable, os.path.abspath(__file__), "--one",
                   f"{cell}:{side_s}:{rep}", "--nperm", str(nperm)] + \
                  (["--fake"] if fake else [])
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
            log = open(os.path.join("logs", f"calib_{tag}.log"), "w")
            live[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  env=env)] = (tag, log)
            print(f"  [start] {tag:<44} ({len(live)} running, {len(queue)} queued)",
                  flush=True)
        for p in list(live):
            if p.poll() is None:
                continue
            tag, log = live.pop(p)
            log.close()
            if p.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"  [FAIL ] {tag} rc={p.returncode} -- logs\\calib_{tag}.log",
                      flush=True)
        if time.time() - last > 30:
            last = time.time()
            print(f"  ... {done}/{len(jobs)} jobs, "
                  f"{(time.time()-t0)/60:.0f} min elapsed, {len(live)} running, "
                  f"{len(queue)} queued", flush=True)
        time.sleep(0.5)
    print(f"\n  {done} done, {failed} failed, {(time.time()-t0)/60:.1f} min wall")
    return failed


def merge():
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(PARTS, "*.csv")))
    if not fs:
        raise SystemExit("no parts to merge")
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    t.to_csv("calibration.csv", index=False)
    rate = t.beat.mean()
    print(f"\n  dead cells that beat their floor: {int(t.beat.sum())} of {len(t)} "
          f"= {100*rate:.0f}%   (calibrated = ~5%)")
    if rate > 0.20:
        print("  [FAIL]  the floor is TOO LOW.")
    elif rate < 0.005 and len(t) > 10:
        print("  [WARN]  the floor may be TOO HIGH -- real effects would be hidden.")
    else:
        print("  [PASS]  the floor behaves as it should on data with no effect.")
    print(f"  PRE-REGISTERED CRITERION: pass is <= 2 of {len(t)} beats  -> "
          f"{'PASS' if int(t.beat.sum()) <= 2 else 'FAIL'}")
    print("  written: calibration.csv")
    return t


def selftest():
    import tempfile
    import pandas as pd
    os.chdir(tempfile.mkdtemp())
    cells = [("A_hh_ATR_3.0", "1.0"), ("B_ll_SD_1.0", "None"),
             ("C_hv_ATR_4.0", "-1.0"), ("D_session_SD_2.0", "1.0")]
    jobs = [(c, s, r) for c, s in cells for r in range(3)]
    ok1 = len(jobs) == 12
    print(f"  [{'PASS' if ok1 else 'FAIL'}] T1  4 cells x 3 reps = {len(jobs)} jobs")
    failed = run_pool(jobs, workers=12, nperm=1, fake=True)
    n = len(glob.glob(os.path.join(PARTS, "*.csv")))
    ok2 = failed == 0 and n == 12
    print(f"  [{'PASS' if ok2 else 'FAIL'}] T2  12 workers ran all 12 "
          f"({n} parts, {failed} failed)")
    t = merge()
    ok3 = len(t) == 12 and os.path.exists("calibration.csv")
    print(f"  [{'PASS' if ok3 else 'FAIL'}] T3  merged to calibration.csv, {len(t)} rows")
    failed2 = run_pool(jobs, workers=12, nperm=1, fake=True)
    ok4 = failed2 == 0 and len(glob.glob(os.path.join(PARTS, "*.csv"))) == 12
    print(f"  [{'PASS' if ok4 else 'FAIL'}] T4  re-run skips finished parts, "
          f"still {len(glob.glob(os.path.join(PARTS,'*.csv')))} parts")
    return all([ok1, ok2, ok3, ok4])


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    get = lambda f, d: (type(d)(a[a.index(f) + 1]) if f in a else d)
    nperm = get("--nperm", 30)
    if "--one" in a:
        cell, side_s, rep = a[a.index("--one") + 1].rsplit(":", 2)
        sys.exit(run_one(cell, side_s, int(rep), nperm, "--fake" in a))
    if "--merge" in a:
        merge(); sys.exit(0)
    spec = get("--cells", "")
    if not spec:
        raise SystemExit("usage: python calib.py --cells C1:side,C2:side "
                         "[--nrep 3] [--nperm 30] [--workers 12]")
    nrep, workers = get("--nrep", 3), get("--workers", 12)
    cells = [x.rsplit(":", 1) for x in spec.split(",")]
    jobs = [(c, s, r) for c, s in cells for r in range(nrep)]
    sys.path.insert(0, "code")
    from floor_v2 import NULL_UNIT
    print(f"  NULL_UNIT={NULL_UNIT}  |  {len(cells)} cells x {nrep} reps = "
          f"{len(jobs)} jobs, {workers} at a time, {nperm} permutations each")
    print(f"  PYTHONHASHSEED=0 pinned in children (see the module docstring)\n")
    if run_pool(jobs, workers, nperm):
        print("\n  some jobs failed -- merging what finished")
    merge()
