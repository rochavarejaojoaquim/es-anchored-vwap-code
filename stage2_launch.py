r"""
stage2_launch.py -- run Stage 2 floors at a higher nperm WITHOUT touching floors_v2\.

WHY runner.py CANNOT DO THIS, read from the files, not assumed:
  * runner.py:66,77  a floors job is SKIPPED when its outputs already exist in
    floors_v2\. All 409 exist, so `runner.py floors --nperm 500` prints "nothing
    left to do" and does nothing at all.
  * floor_v2.py:76   FLOORS = CORES.replace("cores_v2","floors_v2"), and :30 shows
    the folder changes only with TAU, never with nperm. A forced re-run would mix
    nperm=100 and nperm=500 results in ONE folder with nothing to tell them apart --
    in the folder brain/02 lists as never-touch, which the 195 and the 31 rest on.
  * runner.py:66,74  compute `out` and `cores` from runner's OWN directory, NOT from
    VWAP_ROOT, so setting VWAP_ROOT does not redirect runner's bookkeeping even
    though it does redirect the child floor_v2.py. That is why this file exists
    rather than a flag on runner.py.

HOW THIS IS SAFE. loader.py:30 and core_v2.py:23,45 both resolve their paths from
VWAP_ROOT when it is set. A SCRATCH ROOT is built holding directory JUNCTIONS to the
real data\ and cores_v2\ (read-only use, no copying, no duplication of 5.6 GB) and a
REAL, EMPTY floors_v2\ that is the only thing written. The real floors_v2\ is never
opened. C3 below asserts that before a single worker starts.

    python stage2_launch.py --dry-run --cells cells.txt
    python stage2_launch.py --cells cells.txt --nperm 500 --workers 9

--cells is a text file of one cell name per line. IT IS NOT DERIVED HERE, on purpose:
the Stage 2 shortlist is the top 20 by Romano-Wolf adjusted p, and whether that is 20
cell+SIDES or 20 CELLS is NOT RESOLVED in the files this script can see (26.4 step 8
says "top 20 by Romano-Wolf adjusted p"; floor_v2 runs per CELL, all its sides). That
ambiguity is the author's to settle, and guessing it inside a launcher would bury the
choice. Section 26's header also still restricts interpreting the survivor list.

Written 2026-10-01. Project root; rotates nothing. Nothing under code\ is edited.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
SCRATCH = os.path.join(HERE, "stage2_root")
LOGS = os.path.join(HERE, "logs_stage2")
REAL_FLOORS = os.path.join(HERE, "floors_v2")


def junction(link, target):
    """A Windows directory junction: no copy, no admin, read-through."""
    if os.path.exists(link):
        return "exists"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                       capture_output=True, text=True)
    return "made" if r.returncode == 0 else f"FAILED: {r.stderr.strip() or r.stdout.strip()}"


def preflight(cells, nperm, make=False):
    ok = True

    def chk(tag, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}   {detail}")
        ok &= bool(cond)

    print("\nSTAGE 2 PRE-FLIGHT -- nothing is written until every check passes\n")
    chk("C1 a cell list was supplied and has no duplicates",
        bool(cells) and len(cells) == len(set(cells)),
        f"{len(cells)} cells, {len(set(cells))} distinct")

    dump = os.path.join(HERE, "data", "dump")
    cores = os.path.join(HERE, "cores_v2")
    chk("C2 every named cell exists in the dump and has a stored search",
        all(os.path.exists(os.path.join(dump, c + ".parquet")) for c in cells)
        and all(any(f.startswith(c + "_side") for f in os.listdir(cores))
                for c in cells) if cells else False,
        f"checked against {dump} and {cores}")

    if make:
        os.makedirs(SCRATCH, exist_ok=True)
        print(f"  [INFO] scratch root {SCRATCH}")
        print(f"         data\\     -> {junction(os.path.join(SCRATCH, 'data'), os.path.join(HERE, 'data'))}")
        print(f"         cores_v2\\ -> {junction(os.path.join(SCRATCH, 'cores_v2'), cores)}")
        os.makedirs(os.path.join(SCRATCH, "floors_v2"), exist_ok=True)

    out = os.path.join(SCRATCH, "floors_v2")
    chk("C3 THE WRITE TARGET IS UNDER THE SCRATCH ROOT, NOT THE REAL floors_v2\\",
        os.path.normcase(os.path.abspath(out)) !=
        os.path.normcase(os.path.abspath(REAL_FLOORS)),
        f"writes to {out}")
    # Count *.parquet, matching verify_pc.py:204's convention. The first version
    # counted os.listdir and failed at 411 -- which turned out to be a real find
    # (a nested OneDrive folder, see C4b) but not a reason to block a launch.
    import glob as _g
    nreal = len(_g.glob(os.path.join(REAL_FLOORS, "*.parquet"))) \
        if os.path.isdir(REAL_FLOORS) else -1
    chk("C4 the real floors_v2\\ is intact and will not be opened",
        nreal == 409, f"{nreal} .parquet (expected 409)")
    # verify_pc.py's folder counts do NOT recurse, so a nested copy is invisible to
    # it. One was found on 2026-10-01 holding 112 byte-identical duplicates.
    nested = [d for d in os.listdir(REAL_FLOORS)
              if os.path.isdir(os.path.join(REAL_FLOORS, d))] \
        if os.path.isdir(REAL_FLOORS) else []
    chk("C4b no nested directory inside floors_v2\\ (invisible to verify_pc)",
        not nested, ", ".join(nested) if nested else "none")
    chk("C5 nperm is the registered Stage 2 value (amendment 5(5))",
        str(nperm) == "500", f"nperm={nperm}")
    if make:
        chk("C6 the scratch root resolves the dump through its junction",
            os.path.exists(os.path.join(SCRATCH, "data", "dump",
                                        (cells[0] if cells else "") + ".parquet")),
            f"{SCRATCH}\\data\\dump")
    return ok, out, nreal


def main(argv):
    opt = {}
    for i, a in enumerate(argv):
        if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            opt[a] = argv[i + 1]
    dry = "--dry-run" in argv
    nperm = opt.get("--nperm", "500")
    cf = opt.get("--cells")
    if not cf or not os.path.exists(cf):
        raise SystemExit("  --cells FILE is required (one cell name per line)")
    cells = [l.strip() for l in open(cf, encoding="utf-8") if l.strip()
             and not l.startswith("#")]

    ok, out, nreal = preflight(cells, nperm, make=not dry)
    if not ok:
        raise SystemExit("\n  PRE-FLIGHT FAILED -- nothing was launched.")

    nw = int(opt.get("--workers", 0)) or max(1, (os.cpu_count() or 4) - 3)
    env = dict(os.environ, VWAP_ROOT=SCRATCH, OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=CODE)
    print(f"\n  {len(cells)} cells, nperm {nperm}, {nw} workers, "
          f"VWAP_ROOT={SCRATCH}")
    if dry:
        print("\n  DRY RUN. Would run, per cell:")
        print(f"    set VWAP_ROOT={SCRATCH}")
        print(f"    python {os.path.join(CODE, 'floor_v2.py')} <CELL> {nperm}")
        print(f"  first three: {', '.join(cells[:3])}")
        print(f"\n  Nothing was written. The real floors_v2\\ still holds {nreal} "
              f"files.")
        return 0

    os.makedirs(LOGS, exist_ok=True)
    running, queue, done, t0 = {}, list(cells), 0, time.time()
    try:
        while queue or running:
            while queue and len(running) < nw:
                c = queue.pop(0)
                fh = open(os.path.join(LOGS, f"floors_{c}.log"), "w")
                p = subprocess.Popen(
                    [sys.executable, os.path.join(CODE, "floor_v2.py"), c, str(nperm)],
                    cwd=CODE, env=env, stdout=fh, stderr=subprocess.STDOUT)
                running[p] = (c, fh, time.time())
                print(f"  [start] {c:<26} ({len(running)} running, {len(queue)} queued)",
                      flush=True)
            time.sleep(2)
            for p in [p for p in running if p.poll() is not None]:
                c, fh, st = running.pop(p)
                fh.close()
                done += 1
                made = len([f for f in os.listdir(out) if f.startswith(c + "_side")])
                tag = (f"FAILED rc={p.returncode}" if p.returncode else
                       ("NO OUTPUT" if made == 0 else f"done, {made} sides"))
                print(f"  [{tag}] {c:<26} {time.time()-st:6.0f}s   {done}/{len(cells)}",
                      flush=True)
    except KeyboardInterrupt:
        print("\n  stopping -- finished cells stay finished")
        for p in running:
            p.terminate()
    print(f"\n  {done} cells in {(time.time()-t0)/60:.1f} min. Output in {out}")
    print(f"  the real floors_v2\\ still holds "
          f"{len(os.listdir(REAL_FLOORS))} files (was {nreal})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
