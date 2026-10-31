r"""
launch_rebuild.py -- the pseudo-VWAP dump rebuild, built into its OWN root so that
data\dump is never touched.  Run it in a separate terminal.

    python launch_rebuild.py --plan-only      settle, measure memory, print the caps
                                              and the plan, launch NOTHING
    python launch_rebuild.py                  the real run

SHARD COUNT, computed ONCE at launch from a live memory reading:
    live_free   = median of the last 5 of 15 one-second samples of AVAILABLE
                  physical memory (close other apps first; it settles for 15 s)
    cpu_cap     = 9      (the project's recorded figure for this PC)
    mem_cap     = floor((live_free - 2 GiB) / 2.8 GiB)
                  2.8 GiB = the measured peak of ONE dump process, N=60 only.
                  A working number, not a guarantee for every N.
    unit_cap    = the number of independently launchable units of work.
                  The dump and attach_context take ONE lookback N per process
                  (dump_episodes.py and attach_context.py accept only N values),
                  so there are len(NS) = 5 units.  cov_phase splits finer
                  (N x anchor = 20) but runs in seconds.  Two processes must
                  never converge on the same unit (HANDOFF 19, "the one trap").
    shard_count = min(cpu_cap, mem_cap, unit_cap)      all three are printed.

Each shard runs ONE lookback's chain, in order:
    dump_episodes.py N  ->  cov_phase.py A/B N <anchor> (x4)  ->  attach_context.py N
(the order run_all.py uses).  Shards pull lookbacks from a queue, largest first.
Every child process gets VWAP_CHUNK=1000000, OMP_NUM_THREADS=1 (and MKL/OPENBLAS)
in ITS OWN environment only.  monitor_mem.py is started as a separate standalone
process and suspends/resumes the children if available memory drops under 2 GiB.

It stops at the first failed step (AUD-28) and is resumable: run the same command
again and finished cells are skipped.  It does NOT swap anything into data\dump.
"""
import argparse
import glob
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, CODE)
sys.path.insert(1, HERE)
from spec import ANCHORS, ATR_EPS, NS, SD_EPS                          # noqa: E402
import monitor_mem as MM                                               # noqa: E402

CPU_CAP = 9
PER_PROC_GIB = 2.8
RESERVE_GIB = 2.0
SETTLE_S = 15
AKEYS = [a[0] for a in ANCHORS]
DEFAULT_ROOT = os.path.join(HERE, "rebuild_pseudo_2026-09-30")
EXPECT_COLS = 149


def settle_free_gib(seconds, assume):
    if assume is not None:
        return assume, dict(note="ASSUMED via --assume-free-gib (TEST HOOK, not a reading)")
    s = []
    print(f"  settling {seconds}s -- sampling available memory once a second ...", flush=True)
    for _ in range(seconds):
        s.append(MM.avail_gib())
        time.sleep(1)
    return statistics.median(s[-5:]), dict(min=min(s), max=max(s), last=s[-1], n=len(s))


def compute_caps(live, n_units, cpu_cap=CPU_CAP):
    mem_cap = int(math.floor((live - RESERVE_GIB) / PER_PROC_GIB))
    caps = {"cpu_cap": cpu_cap, "mem_cap": mem_cap, "unit_cap": n_units}
    shard = min(caps.values())
    binding = [k for k, v in caps.items() if v == shard]
    return caps, shard, binding


def print_caps(live, info, caps, shard, binding):
    print("\nMEMORY AT LAUNCH")
    if "note" in info:
        print(f"  {info['note']}")
    else:
        print(f"  {info['n']} samples: min {info['min']:.2f}, max {info['max']:.2f}, "
              f"last {info['last']:.2f} GiB available")
    print(f"  live_free_at_launch = {live:.2f} GiB   (median of the last 5 samples)")
    print("\nSHARD COUNT")
    print(f"  cpu_cap  = {caps['cpu_cap']}")
    print(f"  mem_cap  = floor(({live:.2f} - {RESERVE_GIB:.1f}) / {PER_PROC_GIB:.1f}) = {caps['mem_cap']}")
    print(f"  unit_cap = {caps['unit_cap']}   (independently launchable units: one per lookback N in {sorted(NS)})")
    print(f"  shard_count = min({caps['cpu_cap']}, {caps['mem_cap']}, {caps['unit_cap']}) = {shard}"
          f"   binding: {' + '.join(binding)}")


def stage_list(N):
    s = [("dump", "dump_episodes.py", [str(N)])]
    for ak in AKEYS:
        s.append((f"covA_{ak}", "cov_phase.py", ["A", str(N), ak]))
        s.append((f"covB_{ak}", "cov_phase.py", ["B", str(N), ak]))
    s.append(("attach", "attach_context.py", [str(N)]))
    return s


def link_sources():
    """(source, destination-relative-to-root) for everything the rebuild READS."""
    out = [(os.path.join(HERE, "data", "outright.parquet"), os.path.join("data", "outright.parquet"))]
    for n in NS:
        for ak in AKEYS:
            out.append((os.path.join(HERE, "covcache", f"N{n}_{ak}.npz"),
                        os.path.join("covcache", f"N{n}_{ak}.npz")))
    for rule in AKEYS:
        for m in (5, 15):
            for dur in ("4h", "1d"):
                fn = f"htf_{rule}_{m}m_{dur}.npz"
                out.append((os.path.join(HERE, fn), fn))
    return out


def preflight(root, create):
    root = os.path.abspath(root)
    if os.path.realpath(root) == os.path.realpath(HERE) or \
            os.path.realpath(os.path.join(root, "data", "dump")) == os.path.realpath(os.path.join(HERE, "data", "dump")):
        raise SystemExit("REFUSED: the rebuild root must not be the project root (data\\dump would be touched).")
    pairs = link_sources()
    missing = [s for s, _ in pairs if not os.path.exists(s)]
    if missing:
        raise SystemExit("REFUSED: inputs missing (cov_phase B / attach_context would refuse too):\n  "
                         + "\n  ".join(missing))
    todo = [(s, os.path.join(root, d)) for s, d in pairs if not os.path.exists(os.path.join(root, d))]
    dump = os.path.join(root, "data", "dump")
    state = os.path.join(root, "rebuild", "launch_state.json")
    existing = glob.glob(os.path.join(dump, "*")) if os.path.isdir(dump) else []
    if existing and not os.path.exists(state):
        raise SystemExit(f"REFUSED: {dump} is not empty and was not made by this launcher. "
                         f"Rename it aside; nothing is deleted.")
    print(f"\nPREFLIGHT  root {root}")
    print(f"  inputs present: {len(pairs)} ({len(pairs) - len(todo)} already linked, {len(todo)} to hard-link)")
    print(f"  dump folder: {'resuming (' + str(len(existing)) + ' files)' if existing else 'empty / new'}")
    if create:
        for s, d in todo:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            os.link(s, d)                      # hard link: a write elsewhere cannot alter the original
        os.makedirs(dump, exist_ok=True)
        os.makedirs(os.path.join(root, "rebuild", "logs"), exist_ok=True)
    return root


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def child_env(root):
    e = dict(os.environ)
    e.update(PYTHONPATH=CODE, VWAP_ROOT=root, VWAP_CHUNK="1000000", OMP_NUM_THREADS="1",
             MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return e


def fmt(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m{int(sec % 60):02d}s"


def final_checks(root):
    import pyarrow.parquet as pq
    d = os.path.join(root, "data", "dump")
    cells = [f for f in glob.glob(os.path.join(d, "N*.parquet")) if ".m2." not in f and ".tw2." not in f]
    cols = {}
    for f in cells:
        cols.setdefault(len(pq.read_schema(f).names), 0)
        cols[len(pq.read_schema(f).names)] += 1
    n_cells = len(NS) * len(AKEYS) * (len(ATR_EPS) + len(SD_EPS))
    res = [("cells", len(cells) == n_cells, f"{len(cells)} of {n_cells}"),
           ("m2 twin files", len(glob.glob(os.path.join(d, "*.m2.parquet"))) == n_cells, ""),
           ("counts files", len(glob.glob(os.path.join(d, "*.counts.json"))) == n_cells, ""),
           (".done markers", len(glob.glob(os.path.join(d, "*.done"))) == len(NS) * len(AKEYS), ""),
           (f"every cell has {EXPECT_COLS} columns", cols == {EXPECT_COLS: n_cells}, str(cols)),
           ("pseudo_manifest.json", os.path.exists(os.path.join(d, "pseudo_manifest.json")), ""),
           ("no .tmp files", not glob.glob(os.path.join(d, "*.tmp")), "")]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--settle", type=int, default=SETTLE_S)
    ap.add_argument("--heartbeat", type=float, default=60.0)
    ap.add_argument("--ns", type=int, nargs="*")
    ap.add_argument("--skip-verify", action="store_true", help="TEST ONLY")
    ap.add_argument("--assume-free-gib", type=float, help="TEST ONLY: use this instead of a reading")
    ap.add_argument("--stage-script", help="TEST ONLY: run this script instead of the real stages")
    ap.add_argument("--monitor-suspend-below", type=float, default=2.0)
    ap.add_argument("--monitor-resume-above", type=float, default=3.0)
    a = ap.parse_args()
    test = bool(a.stage_script)
    units = sorted(a.ns if a.ns else NS, reverse=True)           # largest lookback first

    print("PSEUDO-VWAP REBUILD LAUNCHER" + ("   [TEST MODE: fake stages]" if test else ""))
    if not test and not a.skip_verify:
        print("  verify_pc.py ... ", end="", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, "verify_pc.py")], capture_output=True, text=True, cwd=HERE)
        if r.returncode != 0 or "ALL CHECKS PASS" not in r.stdout:
            print("FAILED\n" + "\n".join(l for l in r.stdout.splitlines() if "FAIL" in l))
            raise SystemExit("REFUSED: verify_pc.py does not pass, so the run would be unattributable.")
        print("ALL CHECKS PASS")

    live, info = settle_free_gib(a.settle, a.assume_free_gib)
    caps, shard, binding = compute_caps(live, len(units))
    print_caps(live, info, caps, shard, binding)
    if shard < 1:
        raise SystemExit(f"\nREFUSED: not enough free memory for one worker (need > "
                         f"{RESERVE_GIB + PER_PROC_GIB:.1f} GiB available, have {live:.2f}). Close apps and retry.")

    root = a.root if not test else os.path.abspath(a.root)
    if not test:
        root = preflight(root, create=not a.plan_only)
    else:
        os.makedirs(os.path.join(root, "rebuild", "logs"), exist_ok=True)
        os.makedirs(os.path.join(root, "data", "dump"), exist_ok=True)
    rb = os.path.join(root, "rebuild")
    logdir = os.path.join(rb, "logs")
    pidfile, stopfile = os.path.join(rb, "active_pids.json"), os.path.join(rb, "monitor.stop")

    print("\nPLAN")
    print(f"  {shard} shard(s); lookbacks queued largest-first: {units}")
    print(f"  per chain: {' -> '.join(s[0] for s in stage_list(units[0]))}")
    print(f"  child env (each process only): VWAP_ROOT={root}  VWAP_CHUNK=1000000  OMP_NUM_THREADS=1")
    print(f"  monitor: suspend < {a.monitor_suspend_below} GiB, resume >= {a.monitor_resume_above} GiB, "
          f"log {os.path.join(logdir, 'monitor_mem.log')}")
    if a.plan_only:
        print("\n--plan-only: nothing launched.")
        return

    for f in (stopfile,):
        if os.path.exists(f):
            os.remove(f)
    write_json_atomic(os.path.join(rb, "launch_state.json"),
                      dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), shard_count=shard, caps=caps,
                           live_free_gib=live, units=units, test=test))
    write_json_atomic(pidfile, dict(pids=[], done=False))
    mon = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "monitor_mem.py"), "--pids-file", pidfile,
         "--log", os.path.join(logdir, "monitor_mem.log"), "--stop-file", stopfile,
         "--watch-pid", str(os.getpid()), "--suspend-below-gib", str(a.monitor_suspend_below),
         "--resume-above-gib", str(a.monitor_resume_above), "--interval", "5"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    print(f"\nmonitor_mem.py started as its own process (pid {mon.pid})")

    lock, active, q = threading.Lock(), {}, queue.Queue()
    abort, failures, current, timings = threading.Event(), [], {}, {}
    for n in units:
        q.put(n)
    env = child_env(root)
    t0 = time.time()

    def publish():
        write_json_atomic(pidfile, dict(pids=sorted(active.values()), done=False))

    def run_step(wid, N, name, script, args):
        if a.stage_script:
            cmd = [sys.executable, a.stage_script, name, str(N)] + args
        else:
            cmd = [sys.executable, os.path.join(CODE, script)] + args
        with open(os.path.join(logdir, f"N{N}_{name}.log"), "a") as lf:
            s0 = time.time()
            p = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=lf, stderr=subprocess.STDOUT)
            with lock:
                active[(wid, name)] = p.pid
                current[wid] = f"N={N} {name} (pid {p.pid})"
                publish()
            rc = p.wait()
            with lock:
                active.pop((wid, name), None)
                publish()
        timings[(N, name)] = time.time() - s0
        print(f"  [{fmt(time.time() - t0)}] shard {wid}  N={N:<4} {name:<14} {'ok' if rc == 0 else 'FAILED rc=' + str(rc)}"
              f"  {fmt(time.time() - s0)}", flush=True)
        return rc

    def worker(wid):
        while not abort.is_set():
            try:
                N = q.get_nowait()
            except queue.Empty:
                return
            for name, script, args in stage_list(N):
                if abort.is_set():
                    return
                if run_step(wid, N, name, script, args) != 0:
                    failures.append((N, name))
                    abort.set()
                    print(f"\n  STOPPED: N={N} {name} failed -- see {os.path.join(logdir, f'N{N}_{name}.log')}."
                          f" No new step will start; running steps finish. Re-run the same command to resume.",
                          flush=True)
                    return
        with lock:
            current.pop(wid, None)

    threads = [threading.Thread(target=worker, args=(i + 1,), daemon=True) for i in range(shard)]
    for t in threads:
        t.start()
    try:
        last = time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            if time.time() - last >= a.heartbeat:
                last = time.time()
                with lock:
                    pids = list(active.values())
                ws = sum(MM.working_set_gib(p) for p in pids)
                print(f"  ... {fmt(time.time() - t0)} | available {MM.avail_gib():.2f} GiB | workers hold "
                      f"{ws:.2f} GiB | {'; '.join(current.values()) or 'idle'} | queue {q.qsize()}", flush=True)
    except KeyboardInterrupt:
        print("\n  Ctrl+C: stopping workers (finished cells stay finished; run again to resume)", flush=True)
        abort.set()
        with lock:
            for pid in list(active.values()):
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    for t in threads:
        t.join(timeout=30)

    write_json_atomic(pidfile, dict(pids=[], done=True))
    open(stopfile, "w").close()
    try:
        mon.wait(timeout=20)
    except subprocess.TimeoutExpired:
        mon.kill()
    wall = time.time() - t0
    print(f"\nWALL CLOCK {fmt(wall)}")
    if failures or abort.is_set():
        raise SystemExit(f"NOT COMPLETE: {failures or 'interrupted'}")
    if test:
        print("TEST MODE: done.")
        return
    print("\nFINAL CHECKS")
    res = final_checks(root)
    for n, ok, msg in res:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:<34} {msg}")
    allok = all(r[1] for r in res)
    write_json_atomic(os.path.join(rb, "rebuild_done.json"),
                      dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=wall, checks_ok=allok,
                           timings={f"N{k[0]}_{k[1]}": v for k, v in timings.items()}))
    print(f"\n{'BUILD COMPLETE' if allok else 'BUILD INCOMPLETE'} -- NOT swapped into data\\dump.")
    print("Next, in this order (each must pass before the next):")
    print(f"  python verify_columns.py data\\dump {root}\\data\\dump ALL")
    print("  python selfcheck.py")
    print("  only then the swap (renames, nothing is deleted):")
    print("    Rename-Item data\\dump dump_pre_pseudo_2026-09-30")
    print(f"    Move-Item {root}\\data\\dump data\\dump")
    print("    python verify_pc.py")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
