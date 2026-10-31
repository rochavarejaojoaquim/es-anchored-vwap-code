r"""
stage2_sides.py -- Stage 2 floors (nperm 500) for the 31 Romano-Wolf survivors, ONE PROCESS PER CELL+SIDE,
into a SCRATCH ROOT. Project root; rotates nothing; nothing under code\ is edited.

Rules it enforces (STAGE2_PROPOSAL_v2_2026-10-05.md, reviewer items 1-12):
  * the tag list comes from rw_check_independent_out.csv (SHA-256 checked), the pooled side read as the STRING "None"
    (pandas 3 turns an empty side into NaN and a group-by then drops those rows); count asserted = 31;
  * NO-LEAK: child output goes to logs_stage2_sealed\ (never opened); this file prints only counts and PASS/FAIL;
    finished outputs are hashed and set READ-ONLY before --check runs; --check prints no p, floor95 or edge;
  * the children get VWAP_ROOT = the scratch root (junctions to data\ and cores_v2\, an empty floors_v2\); a launcher
    asked to run without it REFUSES; the real floors_v2\ is hashed before and after in --preflight;
  * a MANIFEST is written before the first worker (code-set, producer hashes, tag-list hash, machine, versions).

    python stage2_sides.py --selftest                       toy fixtures + the real tag list (stage-1 numbers only)
    python stage2_sides.py --preflight                      separate scratch root, nperm 3 on the smallest cell+side
    python stage2_sides.py --plan [--part 1,2,3/5]          the tag list and its share, nothing else
    python stage2_sides.py --run --workers 8 [--part 1,2,3/5]
    python stage2_sides.py --check [--part ...]             K1-K3, counts only

    --set survivors (default; the 31, run 2026-10-05) | --set rest (the other 286 effective cell+sides, family B of
    STAGE2_AMENDMENT_DRAFT_v2). Every mode takes --set. 2026-10-06: edited after the 31-run; the pre-edit file is
    stale\stage2_sides_before_set_rest_2026-10-06.py (SHA-256 4292D167...). Run a --part with --set rest only after
    the same --selftest and --preflight --set rest pass on THAT machine.
"""
import glob
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, HERE)
RW_CSV = os.path.join(HERE, "rw_check_independent_out.csv")
RW_SHA = "44E9D780FE0C8C6C1C2812340887CFDBF41DDB618A3BE4F99C0D4BC47DF4B0B0"
SCRATCH = os.path.join(HERE, "stage2_root")
PF_ROOT = os.path.join(HERE, "stage2_pf_root")
SEALED_LOGS = os.path.join(HERE, "logs_stage2_sealed")
REAL_FLOORS = os.path.join(HERE, "floors_v2")
REAL_CORES = os.path.join(HERE, "cores_v2")
NPERM = 500
JOINT_SEED = 20260927
ATOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


# ------------------------------------------------------------------ the list
SET_EXPECT = {"survivors": 31, "rest": 286}


def tag_list(csv_path=RW_CSV, expect=None, check_sha=True, which="survivors"):
    """[(tag, weight)] heaviest first. Pooled side is the string 'None'.
    which = 'survivors' (the 31 Romano-Wolf survivors, Stage 2 as run on 2026-10-05) or 'rest' (the other 286 effective
    cell+sides, added 2026-10-06 for family B of STAGE2_AMENDMENT_DRAFT_v2). The CSV holds the 317 effective rows only."""
    if which not in SET_EXPECT:
        raise SystemExit(f"--set must be one of {sorted(SET_EXPECT)}")
    expect = SET_EXPECT[which] if expect is None else expect
    if check_sha and sha(csv_path) != RW_SHA:
        raise SystemExit(f"rw_check_independent_out.csv SHA-256 differs from the frozen {RW_SHA[:8]}...")
    t = pd.read_csv(csv_path)
    if len(t) != 317:
        raise SystemExit(f"rw_check_independent_out.csv has {len(t)} rows, expected the 317 effective ones")
    side = t["side"].astype(object).where(t["side"].notna(), "None").astype(str)
    t["tag"] = t["cell"].astype(str) + "_side" + side
    s = (t[t["rw_adj_p"] <= 0.05] if which == "survivors" else t[t["rw_adj_p"] > 0.05]).copy()
    if len(s) != expect:
        raise SystemExit(f"tag list ({which}) has {len(s)} rows, expected {expect}")
    s["w"] = s["nA"] + s["nB"]
    s = s.sort_values(["w", "tag"], ascending=[False, True])
    return list(zip(s["tag"], s["w"]))


def part_of(tags, part):
    """'1,2,3/5' = positions whose (i % 5) + 1 is in {1,2,3}; interleaved so each share is a fair mix."""
    if not part:
        return list(tags)
    try:
        which, k = part.split("/")
        want, k = {int(x) for x in which.split(",")}, int(k)
    except Exception:
        raise SystemExit(f"bad --part {part}")
    if not want <= set(range(1, k + 1)):
        raise SystemExit(f"bad --part {part}")
    return [t for i, t in enumerate(tags) if (i % k) + 1 in want]


# ---------------------------------------------------------------- safeguards
def child_env(root):
    """The environment every child gets. A launcher without a scratch root REFUSES (reviewer item 8)."""
    if not root or not os.path.isdir(root):
        raise SystemExit("REFUSED: no scratch root -- a child without VWAP_ROOT would resolve the REAL floors_v2.")
    for k, ok in (("VWAP_NULL_UNIT", (None, "day_strata")), ("VWAP_TAU", (None,))):
        if os.environ.get(k) not in ok:
            raise SystemExit(f"REFUSED: {k}={os.environ.get(k)!r} would change the null or tau.")
    return dict(os.environ, VWAP_ROOT=root, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", PYTHONPATH=CODE)


def shadowed(root_dir=HERE, code_dir=CODE):
    """Root-level .py files that share a name with a module in code\\ (sides.py puts only code\\ first)."""
    names = {os.path.basename(f) for f in glob.glob(os.path.join(code_dir, "*.py"))}
    return sorted(n for n in (os.path.basename(f) for f in glob.glob(os.path.join(root_dir, "*.py"))) if n in names)


def scratch_dirty(floors_dir, nperm=NPERM):
    """Resume trap (item 9): any file here whose nperm differs from the run's would be silently SKIPPED by sides.py."""
    bad = []
    for f in glob.glob(os.path.join(floors_dir, "*.parquet")):
        try:
            n = int(pd.read_parquet(f, columns=["nperm"])["nperm"].iloc[0])
        except Exception:
            n = -1
        if n != nperm:
            bad.append(os.path.basename(f))
    return bad


def real_floors_hash():
    h = hashlib.sha256()
    fs = sorted(glob.glob(os.path.join(REAL_FLOORS, "*.parquet")))
    for f in fs:
        h.update(os.path.basename(f).encode())
        h.update(sha(f).encode())
    return len(fs), h.hexdigest().upper()


def make_root(root):
    os.makedirs(root, exist_ok=True)
    from stage2_launch import junction
    r = [junction(os.path.join(root, "data"), os.path.join(HERE, "data")),
         junction(os.path.join(root, "cores_v2"), REAL_CORES)]
    os.makedirs(os.path.join(root, "floors_v2"), exist_ok=True)
    if any(x.startswith("FAILED") for x in r):
        raise SystemExit(f"junction failed: {r}")
    return r


def static_checks(tags, root, expect=31):
    out = []

    def add(name, ok, det=""):
        out.append((name, bool(ok), det))
    add(f"C1 tag list: {expect} cell+sides, no duplicates", len(tags) == expect and len(set(tags)) == expect, f"{len(tags)}")
    add("C2 every tag has a stored search and a stored floor",
        all(os.path.exists(f"{REAL_CORES}/{t}.parquet") and os.path.exists(f"{REAL_FLOORS}/{t}.parquet") for t in tags))
    add("C3 the write target is NOT the real floors_v2",
        os.path.normcase(os.path.abspath(os.path.join(root, "floors_v2"))) != os.path.normcase(os.path.abspath(REAL_FLOORS)))
    add("C4 no root-level .py shadows a module in code\\", not shadowed(), ", ".join(shadowed()) or "none")
    add("C5 VWAP_NULL_UNIT is unset or day_strata; VWAP_TAU is unset",
        os.environ.get("VWAP_NULL_UNIT") in (None, "day_strata") and os.environ.get("VWAP_TAU") is None)
    js = re.search(r"^JOINT_SEED\s*=\s*(\d+)", open(os.path.join(CODE, "floor_v2.py"), encoding="utf-8").read(), re.M)
    add("C6 floor_v2.JOINT_SEED is the registered 20260927", js and int(js.group(1)) == JOINT_SEED)
    return out


# ----------------------------------------------------------------- the pool
def run_pool(tags, workers, make_cmd, env, logdir, outdir, cwd=HERE):
    """Silent about results: prints only start/finish/rc. Returns the failed tags."""
    os.makedirs(logdir, exist_ok=True)
    queue, live, done, failed, t0, last = list(tags), {}, 0, [], time.time(), 0.0
    while queue or live:
        while queue and len(live) < workers:
            tag = queue.pop(0)
            if os.path.exists(os.path.join(outdir, tag + ".parquet")):
                done += 1
                continue
            fh = open(os.path.join(logdir, f"{tag}.log"), "w")
            live[subprocess.Popen(make_cmd(tag), cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)] = (tag, fh)
            print(f"  [start] {tag:<34} ({len(live)} running, {len(queue)} queued)", flush=True)
        for p in list(live):
            if p.poll() is None:
                continue
            tag, fh = live.pop(p)
            fh.close()
            if p.returncode == 0 and os.path.exists(os.path.join(outdir, tag + ".parquet")):
                done += 1
            else:
                failed.append(tag)
                print(f"  [FAIL ] {tag} rc={p.returncode} (log in {os.path.basename(logdir)}, not opened)", flush=True)
        if time.time() - last > 60:
            last = time.time()
            print(f"  ... {done}/{len(tags)} done, {(time.time()-t0)/60:.0f} min, {len(live)} running, {len(queue)} queued", flush=True)
        time.sleep(0.5)
    print(f"\n  {done} done, {len(failed)} failed, {(time.time()-t0)/60:.1f} min wall", flush=True)
    return failed


def seal_outputs(tags, floors_dir, manifest_path):
    """Hash every output, set it read-only, write the completion manifest. Prints counts only."""
    rec = {}
    for t in tags:
        p = os.path.join(floors_dir, t + ".parquet")
        rec[t] = sha(p)
        os.chmod(p, stat.S_IREAD)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"), outputs=rec), fh, indent=1)
    return len(rec)


def write_manifest(path, tags, part, root, which="survivors"):
    import gate_receipts as G
    m = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), nperm=NPERM, joint_seed=JOINT_SEED, part=part, set=which,
             n_tags=len(tags), tag_list_sha256=hashlib.sha256("\n".join(sorted(tags)).encode()).hexdigest().upper(),
             rw_csv_sha256=RW_SHA, code_set=G.code_set(), machine=G.machine_stamp(),
             producers={f: sha(os.path.join(HERE, f)) for f in ("sides.py", "stage2_sides.py", "stage2_launch.py", "gate_receipts.py")},
             stored_cores=hashlib.sha256("".join(sha(f"{REAL_CORES}/{t}.parquet") for t in sorted(tags)).encode()).hexdigest().upper(),
             stored_floors=hashlib.sha256("".join(sha(f"{REAL_FLOORS}/{t}.parquet") for t in sorted(tags)).encode()).hexdigest().upper(),
             dump_fingerprint=G.dump_fingerprint(os.path.join(HERE, "data", "dump")), scratch_root=root)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, default=str)
    return m


# ------------------------------------------------------------------ the check
def check_outputs(tags, new_dir, stored_dir=REAL_FLOORS, nperm=NPERM, say=print):
    """K1-K3 of the proposal. Prints PASS/FAIL and integer counts only. Returns (k1, k2, k3, nfound)."""
    k1 = k2 = k3 = found = 0
    for t in tags:
        pn = os.path.join(new_dir, t + ".parquet")
        if not os.path.exists(pn):
            continue
        found += 1
        a = pd.read_parquet(pn).iloc[0]
        b = pd.read_parquet(os.path.join(stored_dir, t + ".parquet")).iloc[0]
        tau = a["tau"]
        if (a["null_unit"] == "day_strata" and bool(a["joint"]) and int(a["joint_seed"]) == JOINT_SEED
                and float(tau) == 0.0 and int(a["nperm"]) == nperm):
            k1 += 1
        if abs(float(a["obs_score"]) - float(b["obs_score"])) <= ATOL:
            k2 += 1
        na, nb = np.array(json.loads(a["null_scores"]), float), np.array(json.loads(b["null_scores"]), float)
        if len(na) == nperm and len(nb) == 100:
            h = na[:100]
            fin = np.isfinite(h) & np.isfinite(nb)
            same = bool(np.allclose(h[fin], nb[fin], rtol=0, atol=ATOL) and np.array_equal(h[~fin], nb[~fin]))
            ob = float(b["obs_score"])
            if same and int((h >= ob).sum()) == int((nb >= ob).sum()):
                k3 += 1
    n = len(tags)
    say(f"  [{'PASS' if k1 == n else 'FAIL'}] K1 provenance (day_strata, joint, seed {JOINT_SEED}, tau 0, nperm {nperm})  {k1} of {n} (found {found})")
    say(f"  [{'PASS' if k2 == n else 'FAIL'}] K2 obs_score equals the stored value (atol {ATOL:g})  {k2} of {n}")
    say(f"  [{'PASS' if k3 == n else 'FAIL'}] K3 permutations 0-99 equal the stored nulls (atol {ATOL:g}, -inf by equality, k100 integer)  {k3} of {n}")
    return k1, k2, k3, found


# --------------------------------------------------------------- the preflight
def preflight(which="survivors"):
    tags = [t for t, _ in tag_list(which=which)]
    print(f"\nSTAGE 2 PRE-FLIGHT (set {which}) -- separate scratch root, nothing is written under the real floors_v2 / cores_v2\n")
    ok = True
    for name, good, det in static_checks(tags, PF_ROOT, expect=SET_EXPECT[which]):
        print(f"  [{'PASS' if good else 'FAIL'}] {name}   {det}")
        ok &= good
    try:
        child_env(None)
        refused = False
    except SystemExit:
        refused = True
    print(f"  [{'PASS' if refused else 'FAIL'}] M1 a launcher with no scratch root REFUSES")
    ok &= refused
    if not ok:
        raise SystemExit("\n  PRE-FLIGHT FAILED -- nothing was run.")
    n0, h0 = real_floors_hash()
    print(f"  real floors_v2: {n0} parquet, hash {h0[:16]}")
    make_root(PF_ROOT)
    pf_floors = os.path.join(PF_ROOT, "floors_v2")
    if glob.glob(os.path.join(pf_floors, "*.parquet")):
        raise SystemExit("  the preflight root already holds outputs; move it into stale\\ first.")
    small = tag_list(which=which)[-1][0]
    print(f"  smallest of the {SET_EXPECT[which]}: {small}  (nperm 3, about 4 searches)")
    cmd = [sys.executable, os.path.join(HERE, "sides.py"), "--one", small, "--nperm", "3"]
    os.makedirs(SEALED_LOGS, exist_ok=True)
    with open(os.path.join(SEALED_LOGS, "preflight.log"), "w") as fh:
        rc = subprocess.call(cmd, cwd=HERE, env=child_env(PF_ROOT), stdout=fh, stderr=subprocess.STDOUT)
    out = os.path.join(pf_floors, small + ".parquet")
    good = rc == 0 and os.path.exists(out)
    print(f"  [{'PASS' if good else 'FAIL'}] P1 the child ran with VWAP_ROOT and wrote under the scratch root (rc {rc})")
    ok &= good
    if good:
        a = pd.read_parquet(out).iloc[0]
        b = pd.read_parquet(f"{REAL_FLOORS}/{small}.parquet").iloc[0]
        na, nb = np.array(json.loads(a["null_scores"]), float), np.array(json.loads(b["null_scores"]), float)
        p2 = len(na) == 3 and bool(np.allclose(na, nb[:3], rtol=0, atol=ATOL))
        p3 = abs(float(a["obs_score"]) - float(b["obs_score"])) <= ATOL
        print(f"  [{'PASS' if p2 else 'FAIL'}] P2 permutations 0-2 equal the stored nulls (atol {ATOL:g})")
        print(f"  [{'PASS' if p3 else 'FAIL'}] P3 obs_score equals the stored value")
        ok &= p2 and p3
    n1, h1 = real_floors_hash()
    p4 = (n0, h0) == (n1, h1)
    print(f"  [{'PASS' if p4 else 'FAIL'}] P4 the real floors_v2 is unchanged ({n1} parquet, hash {h1[:16]})")
    ok &= p4
    print("\n  PRE-FLIGHT " + ("PASSED" if ok else "FAILED") + ". Move stage2_pf_root\\ into stale\\ before the real run.")
    return ok


# ------------------------------------------------------------------ self-test
def selftest():
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else ""), flush=True)
        ok[0] &= bool(cond)
    print("\nstage2_sides.py SELF-TEST -- toy fixtures, plus the real tag list (stage-1 numbers only)\n")
    # K0 real list: known numbers from rw_check_independent_out.csv
    tl = tag_list()
    tags = [t for t, _ in tl]
    chk("K0 the real list: 31 tags, 8 pooled ('None'), all distinct", len(tags) == 31 and sum(t.endswith("sideNone") for t in tags) == 8
        and len(set(tags)) == 31, f"{len(tags)} tags")
    chk("K0b heaviest first", [w for _, w in tl] == sorted([w for _, w in tl], reverse=True))
    a, b = part_of(tags, "1,2,3/5"), part_of(tags, "4,5/5")
    chk("K1 parts 1,2,3/5 and 4,5/5 are disjoint and exhaustive", set(a) | set(b) == set(tags) and not set(a) & set(b),
        f"{len(a)} + {len(b)}")
    # the pandas trap, demonstrated: a naive read loses the pooled rows
    naive = pd.read_csv(RW_CSV)
    naive = naive[naive.rw_adj_p <= 0.05]
    chk("K2 the trap is real: a naive group-by on side drops the pooled rows (31 -> 23)",
        int(naive.groupby("side").size().sum()) == 23)
    rest = [t for t, _ in tag_list(which="rest")]
    chk("K2b the 'rest' set: 286 distinct tags, disjoint from the 31, union = the 317 effective cell+sides",
        len(rest) == 286 and len(set(rest)) == 286 and not set(rest) & set(tags) and len(set(rest) | set(tags)) == 317,
        f"{len(rest)} + {len(tags)} = {len(set(rest) | set(tags))}")
    chk("K2c every 'rest' tag has a stored search and a stored floor (the Stage-1 nulls K3 will compare against)",
        all(os.path.exists(f"{REAL_CORES}/{t}.parquet") and os.path.exists(f"{REAL_FLOORS}/{t}.parquet") for t in rest))
    ra, rb = part_of(rest, "1,2,3/5"), part_of(rest, "4,5/5")
    chk("K2d the 'rest' shares 1,2,3/5 and 4,5/5 are disjoint and exhaustive", set(ra) | set(rb) == set(rest) and not set(ra) & set(rb),
        f"{len(ra)} + {len(rb)}")
    try:
        child_env(None)
        r = False
    except SystemExit:
        r = True
    chk("M1 a launcher with no scratch root REFUSES", r)
    d = tempfile.mkdtemp()
    open(os.path.join(d, "core_v2.py"), "w").close()
    os.makedirs(os.path.join(d, "code"))
    open(os.path.join(d, "code", "core_v2.py"), "w").close()
    chk("M2 a root-level module that shadows a code\\ module is caught", shadowed(d, os.path.join(d, "code")) == ["core_v2.py"])
    fl = os.path.join(d, "floors")
    os.makedirs(fl)
    pd.DataFrame([dict(nperm=3)]).to_parquet(os.path.join(fl, "X_side1.0.parquet"))
    chk("M3 a leftover nperm = 3 output in the scratch floors folder is caught (resume trap)", scratch_dirty(fl) == ["X_side1.0.parquet"])
    # pool + silence + seal
    out = os.path.join(d, "out")
    os.makedirs(out)
    toy = ["A_side1.0", "B_side-1.0", "C_sideNone", "D_side1.0", "E_side1.0"]
    stub = ("import sys,os;o=sys.argv[1];t=sys.argv[2];print('score 99.9 floor95 99.9 p 0.001');"
            "import pandas as pd;pd.DataFrame([dict(nperm=500)]).to_parquet(os.path.join(o,t+'.parquet'))")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        failed = run_pool(toy, 3, lambda t: [sys.executable, "-c", stub, out, t], dict(os.environ), os.path.join(d, "logs"), out, cwd=d)
    chk("K3 the pool runs all 5 toy jobs, 0 failed", not failed and len(glob.glob(os.path.join(out, "*.parquet"))) == 5)
    chk("K4 NO-LEAK: the parent's output carries no score, floor95 or p", "99.9" not in buf.getvalue() and "floor95" not in buf.getvalue())
    chk("K4b the child's output went to the log folder", any("99.9" in open(f).read() for f in glob.glob(os.path.join(d, "logs", "*.log"))))
    n = seal_outputs(toy, out, os.path.join(d, "done.json"))
    try:
        open(os.path.join(out, "A_side1.0.parquet"), "ab").close()
        ro = False
    except PermissionError:
        ro = True
    chk("K5 sealed outputs are hashed and READ-ONLY", n == 5 and ro)

    # the checker on toy rows
    def row(null, obs=0.04, **kw):
        r = dict(cell="C", side=1.0, tau=0.0, T=30, col="x", obs_score=obs, floor95=0.03, edge_over_floor=0.01, p=0.01, nperm=len(null),
                 null_unit="day_strata", joint=True, joint_seed=JOINT_SEED, null_scores=json.dumps(null))
        r.update(kw)
        return pd.DataFrame([r])
    st, nw = os.path.join(d, "stored"), os.path.join(d, "new")
    os.makedirs(st)
    os.makedirs(nw)
    rg = np.random.default_rng(5)
    base = list(rg.normal(0.02, 0.01, 100))
    base[7] = float("-inf")
    row(base).to_parquet(os.path.join(st, "T_side1.0.parquet"))

    def mk(null_new, **kw):
        row(null_new, **kw).to_parquet(os.path.join(nw, "T_side1.0.parquet"))
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            res = check_outputs(["T_side1.0"], nw, st)
        return res, buf2.getvalue()
    good_new = base + list(rg.normal(0.02, 0.01, 400))
    res, txt = mk(good_new)
    chk("K6 a faithful extension (first 100 equal, -inf kept) passes K1-K3", res[:3] == (1, 1, 1))
    chk("K6b the checker prints no p, floor95 or edge (the obs 0.04 is not printed)", "0.04" not in txt and "floor95" not in txt)
    perturbed = list(good_new)
    perturbed[3] += 1e-9
    chk("M4 a 1e-9 change in a first-100 null FAILS K3", mk(perturbed)[0][2] == 0)
    perturbed = list(good_new)
    perturbed[7] = 0.0
    chk("M5 a -inf replaced by a finite value FAILS K3", mk(perturbed)[0][2] == 0)
    chk("M6 a wrong joint seed FAILS K1", mk(good_new, joint_seed=1)[0][0] == 0)
    chk("M7 nperm 100 (not 500) FAILS K1", mk(base)[0][0] == 0)
    chk("M8 a changed obs_score FAILS K2", mk(good_new, obs=0.0400001)[0][1] == 0)
    print("\n" + ("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED"))
    return ok[0]


# ------------------------------------------------------------------ the run
def main(argv):
    opt = {a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--")}
    part = opt.get("--part")
    if "--selftest" in argv:
        sys.exit(0 if selftest() else 1)
    which = opt.get("--set", "survivors")
    if which not in SET_EXPECT:
        raise SystemExit(f"--set must be one of {sorted(SET_EXPECT)}")
    if "--preflight" in argv:
        sys.exit(0 if preflight(which) else 1)
    tags = part_of([t for t, _ in tag_list(which=which)], part)
    if "--plan" in argv:
        print(f"  {len(tags)} cell+sides in this share (part {part or 'ALL'}), heaviest first:")
        for t in tags:
            print("   ", t)
        return 0
    if "--check" in argv:
        k = check_outputs(tags, os.path.join(SCRATCH, "floors_v2"))
        return 0 if k[0] == k[1] == k[2] == len(tags) else 1
    if "--run" not in argv:
        print(__doc__)
        return 2
    if not selftest():
        raise SystemExit("self-test failed -- nothing launched")
    for name, good, det in static_checks(tags, SCRATCH, expect=len(tags)):
        print(f"  [{'PASS' if good else 'FAIL'}] {name}   {det}")
        if not good:
            raise SystemExit("PRE-FLIGHT FAILED")
    if os.path.isdir(PF_ROOT) and glob.glob(os.path.join(PF_ROOT, "floors_v2", "*.parquet")):
        raise SystemExit("stage2_pf_root\\ still holds preflight output -- move it into stale\\ first (it is not the run's root).")
    make_root(SCRATCH)
    fl = os.path.join(SCRATCH, "floors_v2")
    dirty = scratch_dirty(fl)
    if dirty:
        raise SystemExit(f"the scratch floors folder holds files with another nperm: {dirty[:3]} -- they would be silently skipped")
    n0, h0 = real_floors_hash()
    sfx = "" if which == "survivors" else f"_{which}"
    man = os.path.join(HERE, f"stage2{sfx}_manifest_{time.strftime('%Y-%m-%d_%H%M%S')}.json")
    m = write_manifest(man, tags, part, SCRATCH, which)
    print(f"  manifest {os.path.basename(man)}  code-set {m['code_set']}  real floors {n0} parquet {h0[:16]}")
    nw = int(opt.get("--workers", 0)) or max(1, (os.cpu_count() or 4) - 3)
    failed = run_pool(tags, nw, lambda t: [sys.executable, os.path.join(HERE, "sides.py"), "--one", t, "--nperm", str(NPERM)],
                      child_env(SCRATCH), SEALED_LOGS, fl)
    n1, h1 = real_floors_hash()
    print(f"  [{'PASS' if (n0, h0) == (n1, h1) else 'FAIL'}] the real floors_v2 is unchanged ({n1} parquet)")
    if failed:
        print(f"  {len(failed)} FAILED: {failed} -- outputs NOT sealed")
        return 1
    n = seal_outputs(tags, fl, os.path.join(HERE, f"stage2{sfx}_done_{time.strftime('%Y-%m-%d_%H%M%S')}.json"))
    print(f"  {n} outputs hashed and set read-only. NOTHING HAS BEEN READ. Next: python stage2_sides.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
