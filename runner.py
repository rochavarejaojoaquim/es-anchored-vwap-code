r"""
runner.py -- run a whole stage unattended: it works out what is left, keeps N
jobs going at once, starts the next one the moment a worker frees up, and never
gives two workers overlapping work.

    python runner.py search                 the core searches (cores_v2\)
    python runner.py floors                 the permutation floors (floors_v2\)
    python runner.py floors --nperm 50      fewer permutations
    python runner.py search --workers 4     override the worker count

Two machines: give each a disjoint SHARE of the cells, sized to how fast it is.
Neither machine needs to know about the other.
    PC (12 threads):   python runner.py floors --part 1,2,3/4     (three quarters)
    laptop (4 cores):  python runner.py floors --part 4/4         (one quarter)
"i/k" takes every k-th cell starting at i; "1,2,3/4" takes three of every four.
Cells are interleaved, not blocked, so each share is a fair mix of big and
small cells.

  * A job is one CELL (its three sides). Jobs whose outputs all exist are
    skipped, so stopping with Ctrl+C and starting again loses nothing.
  * Each worker gets OMP_NUM_THREADS=1, so the processes do not fight over
    threads; the parallelism comes from running several of them.
  * Default workers = CPU count - 2 (- 1 on a 4-thread machine), and never
    more than free RAM / 0.8 GB. A search worker was MEASURED at ~0.5 GB after
    the column-subset change, so 0.8 leaves headroom without throttling the
    CPU, which is what actually limits this work. Override with --workers.
  * Every job's output is written to logs\<stage>_<cell>.log, and a summary
    line prints every 30 seconds.
  * Ctrl+C stops the workers; finished cells stay finished.
"""
import os
import subprocess
import sys
import zlib
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
LOGS = os.path.join(HERE, "logs")
sys.path.insert(0, CODE)


def free_gb():
    try:
        import ctypes
        b = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GlobalMemoryStatusEx  # noqa: B018  (exists on Windows)

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1e9
    except Exception:
        return 8.0                      # not Windows / unknown: assume a modest amount


def jobs_for(stage):
    """One job per CELL; a cell is skipped when its three sides are done."""
    from loader import cell_files, DUMP_DIR
    import glob
    out = os.path.join(HERE, "cores_v2" if stage == "search" else "floors_v2")
    os.makedirs(out, exist_ok=True)
    todo = []
    for f in cell_files("N*", DUMP_DIR):
        cell = os.path.basename(f)[:-8]
        if stage == "search":
            want = [f"{cell}_side{s}.parquet" for s in ("None", "1.0", "-1.0")]
        else:                                    # only sides that HAVE a result
            cores = os.path.join(HERE, "cores_v2")
            want = [os.path.basename(p) for p in glob.glob(f"{cores}/{cell}_side*.parquet")]
            want = [w for w in want if os.path.getsize(os.path.join(cores, w)) > 1200]
        if want and not all(os.path.exists(os.path.join(out, w)) for w in want):
            todo.append(cell)
    return todo


def part(jobs, spec):
    """--part 2/4 -> a quarter of the cells; --part 1,2,3/4 -> three quarters.
    Splitting unevenly is the point: a fast machine takes more.

    A cell's share is decided by its NAME, not by its position in the list
    (2026-09-24). Position-based splitting re-partitioned the work on every
    restart, because the list of unfinished cells shrinks: the two machines
    would then compute their shares from different lists and could take the
    same cell twice, or leave one for neither."""
    if not spec:
        return jobs
    idx, k = spec.split("/")
    want = {int(x) - 1 for x in idx.split(",")}
    k = int(k)
    if not want or max(want) >= k:
        raise SystemExit(f"--part {spec}: the indices must be 1..{k}")
    return [j for j in jobs if zlib.crc32(j.encode()) % k in want]


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("search", "floors"):
        raise SystemExit(__doc__)
    stage = a[0]
    opt = {a[i]: a[i + 1] for i in range(1, len(a) - 1, 2)}
    nperm = opt.get("--nperm", "100")
    jobs = part(jobs_for(stage), opt.get("--part"))
    if not jobs:
        raise SystemExit(f"  nothing left to do for '{stage}' -- every cell is finished")
    # one worker per thread, less a couple so the machine stays usable; on a
    # 4-core laptop that reservation is too expensive, so keep one back, not two.
    cpu = os.cpu_count() or 4
    nw = int(opt.get("--workers", 0)) or max(1, min(cpu - (1 if cpu <= 4 else 2),
                                                    int(free_gb() / 0.8)))
    os.makedirs(LOGS, exist_ok=True)
    print(f"  {stage}: {len(jobs)} cells to do, {nw} at a time"
          + (f", {nperm} permutations each" if stage == "floors" else "")
          + f"  ({free_gb():.1f} GB free, {os.cpu_count()} threads)", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONPATH=CODE)
    running = {}          # proc -> (cell, log handle, started)
    queue = list(jobs)
    done = 0
    t0 = last = time.time()
    try:
        while queue or running:
            while queue and len(running) < nw:
                cell = queue.pop(0)
                script = "core_v2.py" if stage == "search" else "floor_v2.py"
                args = [sys.executable, os.path.join(CODE, script), cell]
                if stage == "floors":
                    args.append(nperm)
                fh = open(os.path.join(LOGS, f"{stage}_{cell}.log"), "w")
                p = subprocess.Popen(args, cwd=CODE, env=env, stdout=fh,
                                     stderr=subprocess.STDOUT)
                running[p] = (cell, fh, time.time())
                print(f"  [start] {cell:<26} ({len(running)} running, "
                      f"{len(queue)} queued)", flush=True)
            time.sleep(2)
            for p in [p for p in running if p.poll() is not None]:
                cell, fh, st = running.pop(p)
                fh.close()
                done += 1
                # a zero exit code is NOT proof of work: a pattern that matches
                # nothing used to exit 0 having done nothing (AUD-44). Check the
                # files the job was supposed to produce.
                left = [c for c in jobs_for(stage) if c == cell]
                if p.returncode != 0:
                    tag = f"FAILED rc={p.returncode} -- see logs\\{stage}_{cell}.log"
                elif left:
                    tag = f"NO OUTPUT -- see logs\\{stage}_{cell}.log"
                else:
                    tag = "done"
                print(f"  [{tag}] {cell:<26} {time.time() - st:6.0f}s   "
                      f"{done}/{len(jobs)} cells", flush=True)
            if time.time() - last > 30 and running:
                el = time.time() - t0
                rate = done / el if done else 0
                if rate:
                    eta = f"ETA {(len(jobs) - done) / rate / 60:.0f} min"
                else:                       # nothing finished yet: say so plainly
                    eta = "ETA unknown until the first cell finishes"
                print(f"  ... {done}/{len(jobs)} cells, {el/60:.0f} min elapsed, "
                      f"{eta}, {len(running)} running", flush=True)
                last = time.time()
    except KeyboardInterrupt:
        print("\n  stopping -- finished cells stay finished, the rest resume next time")
        for p in running:
            p.terminate()
    print(f"\n  {done} cells finished in {(time.time() - t0)/60:.1f} min. "
          f"Logs in logs\\. Run the same command again to pick up anything left.")
