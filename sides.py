"""
sides.py -- run floors with ONE PROCESS PER CELL+SIDE, largest first.

floor_v2.py's own loop runs a cell's three sides one after another inside one
process, so a 4-cell job can only ever use 4 cores and finishes when the
biggest cell's three sides have run in series.  This splits at cell+side
instead: 4 cells = 12 independent jobs, and the big sides start first so no
worker is left holding a monster at the end.

SAFE BY CONSTRUCTION: floor_v2 seeds each job from the TAG, which already
includes the side --  np.random.default_rng([SEED, crc32(tag)]) -- so a
cell+side's result does not depend on what else ran, in what order, or in
which process.  Splitting per side reproduces the per-cell result exactly.

    python sides.py N60_session_ATR_6.0 N200_ll_ATR_4.0 --workers 8
    python sides.py N60 --workers 8 --nperm 100
    python sides.py --selftest                 scheduler test, no real work

Honours VWAP_NULL_UNIT from the environment; children inherit it.
"""
import glob
import os
import subprocess
import sys
import time


def enumerate_jobs(cores_dir, floors_dir, pats):
    """(tag, weight) for every cell+side still to do, heaviest first."""
    files = set()
    for pat in pats:
        files |= set(glob.glob(f"{cores_dir}/{pat}_*_side*.parquet"))
        files |= set(glob.glob(f"{cores_dir}/{pat}*_side*.parquet"))
    jobs = []
    import pandas as pd
    for f in sorted(files):
        tag = os.path.basename(f)[:-8]
        if os.path.exists(f"{floors_dir}/{tag}.parquet"):
            continue
        try:
            core = pd.read_parquet(f)
        except Exception:
            continue
        if core.empty:
            continue
        w = 0.0
        for col in ("nA", "nB"):
            if col in core.columns:
                w += float(core[col].iloc[0])
        if w == 0.0:
            w = float(os.path.getsize(f))
        jobs.append((tag, w))
    jobs.sort(key=lambda t: -t[1])
    return jobs


def run_pool(jobs, workers, nperm, fake=False):
    queue = list(jobs)
    live = {}
    done = failed = 0
    t0 = time.time()
    last = 0.0
    while queue or live:
        while queue and len(live) < workers:
            tag, w = queue.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--one", tag,
                   "--nperm", str(nperm)] + (["--fake"] if fake else [])
            env = dict(os.environ, OMP_NUM_THREADS="1")
            log = open(os.path.join("logs", f"side_{tag}.log"), "w")
            live[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  env=env)] = (tag, log)
            print(f"  [start] {tag:<34} ({len(live)} running, {len(queue)} queued)",
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
                print(f"  [FAIL ] {tag} rc={p.returncode} -- see logs\\side_{tag}.log",
                      flush=True)
        if time.time() - last > 30:
            last = time.time()
            el = (time.time() - t0) / 60
            eta = (el / done * len(queue and queue or [])) if done else None
            print(f"  ... {done}/{len(jobs)} sides, {el:.0f} min elapsed, "
                  f"{len(live)} running, {len(queue)} queued"
                  + (f", ETA ~{eta:.0f} min" if eta else ""), flush=True)
        time.sleep(0.5)
    print(f"\n  {done} done, {failed} failed, "
          f"{(time.time()-t0)/60:.1f} min wall", flush=True)
    return failed


# --------------------------------------------------------------- child mode
def run_one(tag, nperm, fake=False):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "code")
    if fake:
        os.makedirs("floors_v2", exist_ok=True)
        pd.DataFrame([dict(cell=tag, fake=True)]).to_parquet(f"floors_v2/{tag}.parquet")
        return 0
    import zlib
    import safeio
    from floor_v2 import CORES, FLOORS, SEED, floor_cell
    outf = f"{FLOORS}/{tag}.parquet"
    if os.path.exists(outf):
        return 0
    core = pd.read_parquet(f"{CORES}/{tag}.parquet")
    if core.empty:
        return 0
    cell, s = tag.rsplit("_side", 1)
    side = None if s == "None" else float(s)
    rng = np.random.default_rng([SEED, zlib.crc32(tag.encode())])
    row = floor_cell(cell, side, float(core.score.iloc[0]), nperm, rng)
    if row is None:
        print("      NOT WRITTEN -- the floor would not match the candidate")
        return 3
    safeio.write_parquet(pd.DataFrame([row]), outf)
    print(f"      score {100*row['obs_score']:+.2f}  floor95 "
          f"{100*row['floor95']:+.2f}  p {row['p']:.3f}")
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    import tempfile
    import pandas as pd
    d = tempfile.mkdtemp()
    os.chdir(d)
    os.makedirs("cores_v2"); os.makedirs("floors_v2"); os.makedirs("logs")
    sizes = {"A_hh_ATR_3.0_side1.0": 900, "A_hh_ATR_3.0_sideNone": 300,
             "A_hh_ATR_3.0_side-1.0": 600, "B_ll_SD_1.0_side1.0": 5000,
             "B_ll_SD_1.0_sideNone": 8000, "B_ll_SD_1.0_side-1.0": 2000}
    for tag, n in sizes.items():
        pd.DataFrame([dict(score=0.03, nA=n, nB=n // 2)]).to_parquet(
            f"cores_v2/{tag}.parquet")
    jobs = enumerate_jobs("cores_v2", "floors_v2", ["A", "B"])
    order = [t for t, _ in jobs]
    exp = sorted(sizes, key=lambda t: -(sizes[t] * 1.5))
    ok1 = order == exp
    print(f"  [{'PASS' if ok1 else 'FAIL'}] S1  6 cell+sides found, heaviest first")
    print(f"           {order}")
    failed = run_pool(jobs, workers=4, nperm=1, fake=True)
    written = len(glob.glob("floors_v2/*.parquet"))
    ok2 = failed == 0 and written == 6
    print(f"  [{'PASS' if ok2 else 'FAIL'}] S2  4 workers ran all 6 jobs "
          f"({written} files written, {failed} failed)")
    jobs2 = enumerate_jobs("cores_v2", "floors_v2", ["A", "B"])
    ok3 = len(jobs2) == 0
    print(f"  [{'PASS' if ok3 else 'FAIL'}] S3  finished jobs are skipped on "
          f"re-run ({len(jobs2)} left)")
    os.remove("floors_v2/B_ll_SD_1.0_sideNone.parquet")
    jobs3 = enumerate_jobs("cores_v2", "floors_v2", ["A", "B"])
    ok4 = [t for t, _ in jobs3] == ["B_ll_SD_1.0_sideNone"]
    print(f"  [{'PASS' if ok4 else 'FAIL'}] S4  deleting one output re-queues "
          f"exactly that one ({[t for t,_ in jobs3]})")
    return all([ok1, ok2, ok3, ok4])


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    nperm = int(a[a.index("--nperm") + 1]) if "--nperm" in a else 100
    fake = "--fake" in a
    if "--one" in a:
        sys.exit(run_one(a[a.index("--one") + 1], nperm, fake))
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 8
    pats = [x for i, x in enumerate(a)
            if not x.startswith("--")
            and (i == 0 or a[i - 1] not in ("--workers", "--nperm", "--one"))]
    if not pats:
        raise SystemExit("usage: python sides.py <cell-or-prefix> [...] "
                         "[--workers 8] [--nperm 100]")
    sys.path.insert(0, "code")
    from floor_v2 import CORES, FLOORS, NULL_UNIT
    os.makedirs(FLOORS, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    jobs = enumerate_jobs(CORES, FLOORS, pats)
    if not jobs:
        raise SystemExit(f"nothing to do for {pats} -- all outputs already exist?")
    print(f"  NULL_UNIT={NULL_UNIT}  |  {len(jobs)} cell+sides, {workers} at a "
          f"time, {nperm} permutations each", flush=True)
    sys.exit(1 if run_pool(jobs, workers, nperm) else 0)
