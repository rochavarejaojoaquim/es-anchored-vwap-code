r"""
calib_main.py -- the calibration of the Stage-1 null (HANDOFF 26.75-26.77).
Design: the reviewer's "arm 1'" (26.76), confirmed by the author 2026-10-03, with the main
session's corrections and the reviewer's round-4 changes (26.77). NOTHING STATISTICAL IS
REIMPLEMENTED: the search is core_v2.best_core, the null draw is
floor_v2.shuffle_within_block, the score is core_v2.score, and the Romano-Wolf stepdown is
taken (by ast) from rw_check_independent.py as installed.

WHAT THE PRODUCTION NULL MOVES (read from code, core_v2.py:146-147, :204; floor_v2.py:121-153):
only the setups' covariate rows. Setup hits, twin hits and the setup-twin pairing never move;
a twin inherits its source setup's cut. A dead frame here moves THE SAME THING and nothing
else: within each block, in bar order, the setups' covariate rows are CIRCULARLY SHIFTED.
It keeps every series' own persistence and day structure and breaks only their alignment.

REPLICATES. Replicate j uses ONE calendar anchor per block for ALL cells (cells that share
setups are shifted together): the covariates from the anchor date onward move to the start.
  j < 29   THE SEPARATED SET (SEP): anchors on a grid, at least 40 calendar days apart in BOTH
           blocks (block B, 1,676 days, has room for 29 such anchors and no more; the longest
           covariate persistence measured is 35.8 days). EVERY DECISION IS TAKEN ON SEP.
  j >= 29  extra anchors, iid uniform. They sharpen point estimates only.
ARM 1 (size).  For each of the 317 effective cell+sides, M dead frames. Each is searched ONCE
  and ranked against the 100 production null scores ALREADY STORED in floors_v2 for that
  cell+side: k = #{stored null >= dead score}, the production rule. Estimand: the size of the
  exact nulls that produced the 195 and the 31 (production JOINT_SEED).
  Interval: PIGEONHOLE bootstrap (Owen 2007) -- families and replicates resampled independently
  -- because a replicate is a shock common to every family and a family-only resample cannot
  see it (reviewer, round 4).
ARM F (family-wise). Replicate j = the j-th dead frame of every cell: 317 dead scores against
  the stored 317 x 100 null matrix through the installed Romano-Wolf stepdown (loo=True, the
  production setting). One Bernoulli per replicate: "any adjusted p <= 0.05". Decision interval:
  Clopper-Pearson on the 29 SEP replicates. No extra compute.
ARM 2 (strictness; synthetic positive controls, Schuemie et al. 2018 style). On replicate 0's
  dead frame of 60 size-stratified cell+sides, a known link is injected into the SETUP hits
  inside {covariate > its 85th percentile} (OFF the search's decile grid, core_v2.py:47), at
  delta = 0, 2, 4, 8 pp, in two scenarios:
     EASY  row-level flips, equal in both blocks
     HARD  flips concentrated in whole days, block B at delta/4
  Each test draws its own 100 production nulls (delta = 0 too, so the baseline is the SAME
  procedure). Also an ORACLE: the injected mask itself, no search, same permutations. Output:
  a detection CURVE (no pass threshold, E2), "right column" recorded beside "detected".

KNOWN-NUMBER CHECKS, per cell+side, before any dead frame is read (a FAIL writes no frames):
  K0a the unshifted frame reproduces the stored obs_score      K0b and the stored p
  K0c production permutations 0,1,2 recomputed now equal the stored null_scores[0:3]
In --merge, before any result: K1 317 of 317 complete, one provenance on every row; K2 the
installed stepdown on the REAL obs reproduces rw_check_independent_out.csv (31 of 317, every
adjusted p equal).

    python calib_main.py --selftest
    python calib_main.py --plan                 writes calib_main_plan.json (reads ONE cell)
    python calib_main.py --preflight            real-data path AND the merge path, temp dir
    python calib_main.py --arm 1 --workers 8 [--part 1,2,3/5]
    python calib_main.py --arm 2 --workers 8 [--part 1,2,3/5]
    python calib_main.py --merge

Project root; rotates nothing. Writes calib_main_parts\, calib_main_*_<date>.csv/json only.
"""
import ast
import contextlib
import datetime
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "calib_main_parts")
PLAN = os.path.join(HERE, "calib_main_plan.json")
SEED = 20261004
M_DEFAULT = 100
NPERM = 100
F_LO, F_HI = 0.15, 0.85
SEP, GRID_A = 29, 48                     # 0.7 x 1,676 d / 29 = 40.5 d; 0.7 x 2,762 d / 48 = 40.3 d
K0C_PERMS = 3
Q_INJ = 0.85
DELTAS = (2, 4, 8)                       # pp; a GRID (the range of F21's end-fifth edges), not "typical"
SCENARIOS = ("EASY", "HARD", "ZERO")     # ZERO = delta 0 through the same path
N_ARM2_CELLS = 60
PERSISTENT = ("speed", "W0", "value_area_atr", "rv_close")     # HANDOFF 26.75 M2
NOMINAL5, NOMINAL0, NOMINAL_P = 5 / 101, 1 / 101, 51 / 101
NBOOT_CI = 4000
PILOT_CELLS = [("N240_hv_ATR_8.0", "1.0"), ("N60_hh_SD_2.0", "-1.0"), ("N240_hv_ATR_4.0", "-1.0"),
               ("N160_hh_SD_1.5", "-1.0"), ("N200_hh_SD_1.0", "-1.0"), ("N60_session_ATR_6.0", "1.0"),
               ("N200_ll_ATR_4.0", "1.0"), ("N60_ll_SD_1.0", "1.0"), ("N60_hh_ATR_3.0", "None")]


def _imports():
    sys.path[:0] = [p for p in (HERE, os.path.join(HERE, "code")) if p not in sys.path]
    import core_v2
    import floor_v2
    return core_v2, floor_v2


def side_str(s):
    return "None" if s is None or pd.isna(s) else str(float(s))


def side_val(s):
    return None if s in ("None", "none") else float(s)


def family(cell):
    return cell.split("_", 1)[1]


def tag_of(cell, side_s):
    return f"{cell}_side{side_s}"


def read_parts(files):
    """'None' is a SIDE here, not a missing value: pandas reads the string None as NaN by
    default even with dtype=str (measured on pandas 3.0.5; reviewer round 4)."""
    return pd.concat([pd.read_csv(f, dtype={"side": str}, keep_default_na=False, na_values=[""])
                      for f in files], ignore_index=True)


# ------------------------------------------------------------------ dead frames
def anchor_fracs(j):
    j = int(j)
    if j < SEP:
        pa = np.random.default_rng([SEED, 12, 0]).permutation(GRID_A)[:SEP]
        pb = np.random.default_rng([SEED, 12, 1]).permutation(SEP)
        return {"A": float(F_LO + (pa[j] + 0.5) / GRID_A * (F_HI - F_LO)),
                "B": float(F_LO + (pb[j] + 0.5) / SEP * (F_HI - F_LO))}
    return {B: float(np.random.default_rng([SEED, 11, j, bi]).uniform(F_LO, F_HI))
            for bi, B in enumerate(("A", "B"))}


def shift_covariates(r, cols, fracs, bounds, dayint):
    """Circular shift of the covariate rows within each block, in bar order. Hits, days,
    blocks, bars untouched. Returns the dead frame and per-block (offset, n, 1st-percentile
    calendar distance of the shift in days)."""
    vals = r[cols].to_numpy()
    new = vals.copy()
    blocks = r.block.to_numpy()
    bars = r.bar.to_numpy()
    info = {}
    for B in np.unique(blocks):
        if B not in fracs:
            continue
        ix = np.where(blocks == B)[0]
        ix = ix[np.argsort(bars[ix], kind="stable")]
        n = len(ix)
        if n < 3:
            continue
        d = dayint[ix]
        lo, hi = bounds[B]
        anchor = lo + fracs[B] * (hi - lo)
        off = int(np.searchsorted(d, anchor, side="left"))
        off = min(max(off, 1), n - 1)
        src = np.roll(ix, -off)                       # destination i takes row i+off (mod n)
        new[ix] = vals[src]
        lag = np.abs(dayint[src] - d)
        info[B] = (off, n, float(np.percentile(lag, 1)))
    rd = r.copy()
    rd[cols] = new
    return rd, info


def k_count(obs, null):
    return int((np.asarray(null, float) >= obs).sum())


# ------------------------------------------------------------------ injection (arm 2)
def inject(hit, region, blocks, days, delta_by_block, form, rng):
    """Flip misses to hits inside `region` until each block's region hit rate is up by its
    delta. form 'row': flips at random rows. form 'day': whole days are filled, in random day
    order, until the count is reached. Returns new hits and the achieved delta per block."""
    out = hit.copy()
    got = {}
    for B, delta in delta_by_block.items():
        reg = np.where(region & (blocks == B))[0]
        if len(reg) == 0:
            got[B] = float("nan")
            continue
        miss = reg[out[reg] == 0]
        want = min(int(round(delta * len(reg))), len(miss))
        if form == "row":
            pick = rng.choice(miss, want, replace=False) if want else np.array([], int)
        else:
            dm = days[miss]
            order = rng.permutation(np.unique(dm))
            rank = {d: i for i, d in enumerate(order)}
            srt = miss[np.argsort(np.array([rank[d] for d in dm]), kind="stable")]
            pick = srt[:want]
        out[pick] = 1.0
        got[B] = float(len(pick) / len(reg))
    return out, got


def choose_inj_col(rd, cols, winner, idx, rng, core_v2):
    """Even arm-2 cells get a PERSISTENT column (order rotated by cell so it is not always
    `speed`), odd cells a seeded non-persistent one. Never the cell's stored winner."""
    if idx % 2 == 0:
        rot = (idx // 2) % len(PERSISTENT)
        cand = [c for c in PERSISTENT[rot:] + PERSISTENT[:rot] if c in cols]
    else:
        cand = rng.permutation([c for c in cols if c not in PERSISTENT]).tolist()
    blocks = rd.block.to_numpy()
    for col in cand:
        if col == winner or col.endswith("_xs"):
            continue
        v = rd[col].to_numpy(float)
        if not np.isfinite(v).any() or len(np.unique(v[np.isfinite(v)])) < 20:
            continue
        q = float(np.nanquantile(v, Q_INJ))
        m = v > q
        if m.sum() < core_v2.MIN_N or (~m).sum() < core_v2.MIN_N:
            continue
        if any((m & (blocks == B)).sum() < core_v2.MIN_N // 3 for B in ("A", "B")):
            continue
        return col, q
    return None, None


def is_right_col(found, injected):
    """The pooled search also holds `<col>_xs` (column x side): count it as the right column."""
    return bool(found in (injected, injected + "_xs"))


def oracle_score(frame, c, P, col, q, core_v2):
    v = frame[col].to_numpy(float)
    m = v > q
    cm = core_v2.twin_mask(m, P["src_idx"])
    eA = core_v2.edge_se(frame, c, P, m, cm, "A", nboot=500)
    eB = core_v2.edge_se(frame, c, P, m, cm, "B", nboot=500)
    if eA is None or eB is None or np.sign(eA[0]) != np.sign(eB[0]):
        return -np.inf
    return core_v2.score(eA, eB)


# ------------------------------------------------------------------ stamps, stored floors
def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def stamp():
    core_v2, floor_v2 = _imports()
    import gate_receipts
    m = gate_receipts.machine_stamp()
    return dict(code_set=gate_receipts.code_set(), producer_sha256=sha256_file(os.path.abspath(__file__)),
                plan_sha256=sha256_file(PLAN), joint_seed=int(floor_v2.JOINT_SEED),
                null_unit=str(floor_v2.NULL_UNIT), blocklen=5, seed=SEED, machine=m["node"],
                python=m["python"], numpy=m["numpy"], pandas=m["pandas"])


def stored_floor(cell, side_s):
    core_v2, floor_v2 = _imports()
    f = os.path.join(floor_v2.FLOORS, tag_of(cell, side_s) + ".parquet")
    t = pd.read_parquet(f).iloc[0]
    return dict(obs=float(t.obs_score), p=float(t.p), null=np.array(json.loads(t.null_scores), float),
                null_unit=str(t.null_unit), joint=bool(t.joint), joint_seed=int(t.joint_seed),
                nperm=int(t.nperm), col=str(t.col))


def load_plan():
    if not os.path.exists(PLAN):
        raise SystemExit("no calib_main_plan.json -- run: python calib_main.py --plan")
    with open(PLAN, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_plan():
    core_v2, floor_v2 = _imports()
    if os.path.exists(PLAN) and glob.glob(os.path.join(PARTS, "*", "*")):
        raise SystemExit("parts exist: the plan is frozen. Move calib_main_parts\\ aside first.")
    t = pd.read_csv(os.path.join(HERE, "rw_check_independent_out.csv"))
    if len(t) != 317:
        raise SystemExit(f"expected 317 effective rows, found {len(t)}")
    u = pd.read_csv(os.path.join(HERE, "rebuild_pseudo_2026-09-30", "unmatched_by_cell.csv"))
    size = dict(zip(u.cell, u.setups))
    rows = sorted(((int(size[c]), c, side_str(s)) for c, s in zip(t.cell, t.side)),
                  key=lambda x: (-x[0], x[1], x[2]))
    cells = [dict(cell=c, side=s, size=n) for n, c, s in rows]
    asc = cells[::-1]
    pos = sorted({int(round((i + 0.5) / N_ARM2_CELLS * (len(asc) - 1))) for i in range(N_ARM2_CELLS)})
    arm2 = [asc[i] for i in pos]
    big = cells[0]
    r, c, T = core_v2.load_cell(big["cell"], side_val(big["side"]))
    d = floor_v2._as_day_int(r.day)
    bounds = {B: [int(d[(r.block == B).to_numpy()].min()), int(d[(r.block == B).to_numpy()].max())]
              for B in ("A", "B")}
    plan = dict(seed=SEED, bounds=bounds, bounds_from=f"{big['cell']} side {big['side']}",
                cells=cells, arm2=arm2)
    with open(PLAN, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)
    print(f"  wrote calib_main_plan.json: {len(cells)} cell+sides, {len(arm2)} arm-2 cells, "
          f"block bounds (day ints) {bounds} from {plan['bounds_from']}")
    print(f"  arm 1: {len(cells)} jobs x (1 + {K0C_PERMS} + M) searches; "
          f"arm 2: {len(job_list('2'))} jobs x (1 + {NPERM}) searches")


# ------------------------------------------------------------------ jobs
def arm1_job(cell, side_s, M, parts=None, k0c=K0C_PERMS):
    core_v2, floor_v2 = _imports()
    parts = parts or os.path.join(PARTS, "arm1")
    os.makedirs(parts, exist_ok=True)
    tag = tag_of(cell, side_s)
    final = os.path.join(parts, tag + ".csv")
    partial = os.path.join(parts, tag + ".partial")
    if os.path.exists(final):
        return 0
    plan = load_plan()
    bounds = {B: tuple(v) for B, v in plan["bounds"].items()}
    st = stamp()
    S = stored_floor(cell, side_s)
    r, c, T = core_v2.load_cell(cell, side_val(side_s))
    cols = core_v2.search_cols(r, side_val(side_s))
    P = core_v2.prep(r, c)
    done = {}
    if os.path.exists(partial):
        done = {int(j): True for j in pd.read_csv(partial).j}
    else:
        # ---- known-number checks, BEFORE any dead frame
        cfg = (floor_v2.NULL_UNIT == S["null_unit"] == "day_strata" and floor_v2.JOINT_NULL
               and S["joint"] and S["joint_seed"] == floor_v2.JOINT_SEED and S["nperm"] == NPERM
               and len(S["null"]) == NPERM)
        obs = core_v2.best_core(r, c, P, cols)
        k0a = obs is not None and bool(np.isclose(obs["score"], S["obs"], rtol=0, atol=1e-12))
        k0b = bool(np.isclose((1 + k_count(S["obs"], S["null"])) / (1 + NPERM), S["p"], rtol=0, atol=1e-12))
        scratch = r.copy()
        rng = np.random.default_rng(0)                 # unused under JOINT_NULL + perm_index
        k0c_ok = True
        for k in range(k0c):
            b = core_v2.best_core(floor_v2.shuffle_within_block(
                r, cols, rng, into=scratch, unit=floor_v2.NULL_UNIT, perm_index=k), c, P, cols)
            v = b["score"] if b is not None else -np.inf
            same = (v == S["null"][k]) or bool(np.isclose(v, S["null"][k], rtol=0, atol=1e-12))
            k0c_ok &= bool(same)
        if not (cfg and k0a and k0b and k0c_ok):
            pd.DataFrame([dict(arm="1", cell=cell, side=side_s, status="K0 FAIL", cfg=cfg, k0a=k0a,
                               k0b=k0b, k0c=k0c_ok, **st)]).to_csv(final, index=False)
            print(f"  K0 FAIL {tag}: cfg {cfg} k0a {k0a} k0b {k0b} k0c {k0c_ok}", flush=True)
            return 3
    dayint = floor_v2._as_day_int(r.day)
    for j in range(M):
        if j in done:
            continue
        t0 = time.time()
        fr = anchor_fracs(j)
        rd, info = shift_covariates(r, cols, fr, bounds, dayint)
        b = core_v2.best_core(rd, c, P, cols)
        sc = b["score"] if b is not None else -np.inf
        nil = (0, 0, np.nan)
        row = dict(arm="1", cell=cell, side=side_s, status="ok", j=j, fA=fr["A"], fB=fr["B"],
                   offA=info.get("A", nil)[0], nA=info.get("A", nil)[1], lagA=info.get("A", nil)[2],
                   offB=info.get("B", nil)[0], nB=info.get("B", nil)[1], lagB=info.get("B", nil)[2],
                   score=sc, col=b["col"] if b else "", cut_pct=b["pct"] if b else np.nan,
                   sense=b["sense"] if b else "", k=k_count(sc, S["null"]), nperm=NPERM,
                   secs=time.time() - t0, **st)
        pd.DataFrame([row]).to_csv(partial, mode="a", index=False, header=not os.path.exists(partial))
    os.replace(partial, final)
    return 0


def arm2_job(cell, side_s, delta, scen, idx, parts=None, nperm=NPERM):
    core_v2, floor_v2 = _imports()
    parts = parts or os.path.join(PARTS, "arm2")
    os.makedirs(parts, exist_ok=True)
    tag = f"{tag_of(cell, side_s)}__d{delta}__{scen}"
    final = os.path.join(parts, tag + ".csv")
    if os.path.exists(final):
        return 0
    plan = load_plan()
    bounds = {B: tuple(v) for B, v in plan["bounds"].items()}
    st = stamp()
    S = stored_floor(cell, side_s)
    r, c, T = core_v2.load_cell(cell, side_val(side_s))
    cols = core_v2.search_cols(r, side_val(side_s))
    P = core_v2.prep(r, c)
    dayint = floor_v2._as_day_int(r.day)
    rd, info = shift_covariates(r, cols, anchor_fracs(0), bounds, dayint)
    key = zlib.crc32(f"{cell}|{side_s}".encode())
    col, q = choose_inj_col(rd, cols, S["col"], idx, np.random.default_rng([SEED, 21, key]), core_v2)
    if col is None:
        pd.DataFrame([dict(arm="2", cell=cell, side=side_s, delta=delta, scen=scen,
                           status="no injectable column", **st)]).to_csv(final, index=False)
        return 0
    region = rd[col].to_numpy(float) > q
    dB = delta / 400.0 if scen == "HARD" else delta / 100.0
    hit, got = inject(rd.hit.to_numpy().copy(), region, rd.block.to_numpy(), dayint,
                      {"A": delta / 100.0, "B": dB}, "day" if scen == "HARD" else "row",
                      np.random.default_rng([SEED, 22, key, delta, SCENARIOS.index(scen)]))
    ri = rd.copy()
    ri["hit"] = hit
    t0 = time.time()
    obs = core_v2.best_core(ri, c, P, cols)
    sc = obs["score"] if obs is not None else -np.inf
    orc = oracle_score(ri, c, P, col, q, core_v2)
    null, onull, scratch = np.empty(nperm), np.empty(nperm), ri.copy()
    rng = np.random.default_rng(0)
    for k in range(nperm):
        rp = floor_v2.shuffle_within_block(ri, cols, rng, into=scratch, unit=floor_v2.NULL_UNIT, perm_index=k)
        onull[k] = oracle_score(rp, c, P, col, q, core_v2)
        b = core_v2.best_core(rp, c, P, cols)
        null[k] = b["score"] if b is not None else -np.inf
    pd.DataFrame([dict(arm="2", cell=cell, side=side_s, status="ok", delta=delta, scen=scen,
                       inj_col=col, inj_persistent=idx % 2 == 0, inj_q=q, gotA=got.get("A"), gotB=got.get("B"),
                       score=sc, col=obs["col"] if obs else "", sense=obs["sense"] if obs else "",
                       cut_pct=obs["pct"] if obs else np.nan, k=k_count(sc, null),
                       right_col=bool(obs is not None and is_right_col(obs["col"], col)),
                       oracle=orc, k_oracle=k_count(orc, onull), nperm=nperm,
                       secs=time.time() - t0, **st)]).to_csv(final, index=False)
    return 0


def job_list(arm, part=None):
    plan = load_plan()
    if arm == "1":
        jobs = [("1", x["cell"], x["side"]) for x in plan["cells"]]
    else:
        jobs = [("2", x["cell"], x["side"], d, s, i) for i, x in enumerate(plan["arm2"])
                for d, s in [(0, "ZERO")] + [(d, s) for d in DELTAS for s in ("EASY", "HARD")]]
    if part:
        which, n = part.split("/")
        keep = {int(v) for v in which.split(",")}
        n = int(n)
        if not keep or min(keep) < 1 or max(keep) > n:
            raise SystemExit(f"bad --part {part}")
        jobs = [jb for i, jb in enumerate(jobs) if (i % n) + 1 in keep]
    return jobs


def run_pool(arm, workers, part, M):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    queue, live, done, failed = job_list(arm, part), {}, 0, 0
    print(f"  arm {arm}: {len(queue)} jobs" + (f" (part {part})" if part else "") + f", {workers} workers", flush=True)
    t0 = last = time.time()
    while queue or live:
        while queue and len(live) < workers:
            jb = queue.pop(0)
            spec = ":".join(str(v) for v in jb)
            cmd = [sys.executable, os.path.abspath(__file__), "--one", spec, "--M", str(M)]
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
            name = spec.replace(":", "__")
            log = open(os.path.join(HERE, "logs", f"calibmain_{name}.log"), "w")
            live[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)] = (name, log)
        for p in list(live):
            if p.poll() is None:
                continue
            name, log = live.pop(p)
            log.close()
            if p.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"  [FAIL] {name} rc={p.returncode} -- logs\\calibmain_{name}.log", flush=True)
        if time.time() - last > 300:
            last = time.time()
            print(f"  ... {done} done, {failed} failed, {len(live)} running, {len(queue)} queued, "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
        time.sleep(0.5)
    print(f"\n  arm {arm}: {done} done, {failed} failed, {(time.time() - t0) / 60:.1f} min wall")
    return failed


# ------------------------------------------------------------------ statistics
def pigeonhole_ci(v, fam, rep, nboot=NBOOT_CI, seed=1):
    """Mean of v with a percentile CI from resampling FAMILIES and REPLICATES independently
    (Owen 2007's pigeonhole bootstrap): a shock shared by a whole family, or by a whole
    replicate across all families, is carried into the interval."""
    v = np.asarray(v, float)
    fu, fi = np.unique(np.asarray(fam), return_inverse=True)
    ru, ri = np.unique(np.asarray(rep), return_inverse=True)
    A = np.zeros((len(fu), len(ru)))
    N = np.zeros((len(fu), len(ru)))
    np.add.at(A, (fi, ri), v)
    np.add.at(N, (fi, ri), 1.0)
    rng = np.random.default_rng(seed)
    est = np.empty(nboot)
    for b in range(nboot):
        F = rng.integers(0, len(fu), len(fu))
        R = rng.integers(0, len(ru), len(ru))
        est[b] = A[np.ix_(F, R)].sum() / N[np.ix_(F, R)].sum()
    return float(v.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


STATS = (("k<=4", lambda k: (k <= 4).astype(float), NOMINAL5, 100.0),
         ("k==0", lambda k: (k == 0).astype(float), NOMINAL0, 100.0),
         ("mean p", lambda k: (1 + k) / 101.0, NOMINAL_P, 1.0))


def size_stats(k, fam, rep, label, say=print, nboot=NBOOT_CI):
    k = np.asarray(k)
    out = {}
    for name, f, nom, sc in STATS:
        e, lo, hi = pigeonhole_ci(f(k), fam, rep, nboot=nboot)
        differs = bool(lo > nom or hi < nom)
        say(f"    {label:<24} {name:<7} {sc*e:8.4f}   95% CI [{sc*lo:.4f}, {sc*hi:.4f}]   nominal {sc*nom:.4f}"
            f"   -> {'DIFFERS FROM NOMINAL' if differs else 'nominal inside'}")
        out[name] = dict(est=e, lo=lo, hi=hi, nominal=nom, differs=differs)
    return out


def rw_funcs():
    """studentize() and romano_wolf() exactly as installed in rw_check_independent.py."""
    path = os.path.join(HERE, "rw_check_independent.py")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    ns = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("studentize", "romano_wolf"):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns["studentize"], ns["romano_wolf"]


def fwer_any(obs, NUL, studentize, romano_wolf):
    to, T = studentize(obs, NUL, loo=True)
    adj = romano_wolf(to, T)
    return int((adj <= 0.05).sum()), float(adj.min()), adj


def shrink_null(NUL, f=0.8):
    mu = NUL.mean(1, keepdims=True)
    return mu + f * (NUL - mu)


def cp(x, n):
    lo = _st.beta.ppf(0.025, x, n - x + 1) if x else 0.0
    hi = _st.beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return float(lo), float(hi)


def merge(parts=None, dry=False):
    core_v2, floor_v2 = _imports()
    parts = parts or PARTS
    plan = load_plan()
    st = stamp()
    today = datetime.date.today().isoformat()
    fs = sorted(glob.glob(os.path.join(parts, "arm1", "*.csv")))
    if not fs:
        raise SystemExit("no arm-1 parts yet")
    t = read_parts(fs)
    bad = t[t.status != "ok"]
    ok = t[t.status == "ok"].copy()
    ok["j"] = ok.j.astype(int)
    ok["k"] = ok.k.astype(int)
    print("\ncalib_main.py MERGE -- known-number checks first")
    have = {(c, s) for c, s in zip(ok.cell, ok.side)}
    want = have if dry else {(x["cell"], x["side"]) for x in plan["cells"]}
    M = int(ok.groupby(["cell", "side"]).j.nunique().min()) if len(ok) else 0
    full_j = bool((ok.groupby(["cell", "side"]).j.nunique() == M).all())
    prov = all(ok[c].nunique() == 1 and str(ok[c].iloc[0]) == str(st[c])
               for c in ("producer_sha256", "plan_sha256", "code_set", "seed", "joint_seed", "null_unit"))
    k1 = len(bad) == 0 and have == want and full_j and prov and (dry or M >= SEP)
    print(f"  [{'PASS' if k1 else 'FAIL'}] K1 every cell+side passed K0a-c, is complete, and every row carries "
          f"the live producer, plan, code-set and seeds   {len(have)} of {len(want)}, K0 failures {len(bad)}, "
          f"frames per cell {M}, one provenance {prov}")
    if len(bad):
        print(bad[["cell", "side", "status"]].to_string(index=False))
    studentize, romano_wolf = rw_funcs()
    ref = pd.read_csv(os.path.join(HERE, "rw_check_independent_out.csv"))
    keys = [(c, side_str(s)) for c, s in zip(ref.cell, ref.side)]
    SF = {kk: stored_floor(*kk) for kk in keys}
    NUL = np.vstack([SF[kk]["null"] for kk in keys])
    obs_real = np.array([SF[kk]["obs"] for kk in keys])
    n_real, min_real, adj_real = fwer_any(obs_real, NUL, studentize, romano_wolf)
    k2 = (n_real == 31 and bool(np.allclose(adj_real, ref.rw_adj_p.to_numpy(float), rtol=0, atol=1e-12))
          and int(np.isclose(adj_real, 1 / 101).sum()) == 17)
    print(f"  [{'PASS' if k2 else 'FAIL'}] K2 the installed stepdown on the REAL obs reproduces "
          f"rw_check_independent_out.csv   {n_real} of 317, min {min_real:.4f}, "
          f"{int(np.isclose(adj_real, 1/101).sum())} at the floor")
    if not (k1 and k2):
        raise SystemExit("  KNOWN-NUMBER CHECK FAILED -- no result is read.")
    ok["fam"] = ok.cell.map(family)
    if not dry:
        ok.to_csv(os.path.join(HERE, f"calib_main_arm1_{today}.csv"), index=False)
    sep_n = min(SEP, M)
    sep = ok[ok.j < sep_n]
    print(f"\nSTAMP: arm 1 | {len(have)} cell+sides x {M} covariate-shift dead frames = {len(ok):,} dead searches | "
          f"ranked against the STORED production nulls (floors_v2, day_strata, joint seed {st['joint_seed']}, "
          f"B = {NPERM}) | k = #{{null >= dead score}} | code-set {st['code_set']} | calib_main.py "
          f"{st['producer_sha256'][:16]} | plan {st['plan_sha256'][:16]} | machines {sorted(ok.machine.unique())} | "
          f"ES train A+B, sealed untouched")
    lag = np.minimum(ok.lagA, ok.lagB)
    print(f"  calendar distance of a frame's shift (1st percentile, days): min {lag.min():.0f}, median {np.median(lag):.0f}")
    print(f"\n  SIZE -- DECISION SET: the {sep_n} separated replicates ({len(sep):,} frames). Rule fixed before the run: "
          f"a rate DIFFERS when its pigeonhole 95% CI (families x replicates) excludes nominal.")
    res = dict(SEP=size_stats(sep.k, sep.fam, sep.j, "SEP, all cells"))
    surv = {kk for kk, a in zip(keys, adj_real) if a <= 0.05}
    is_s = np.array([(c, s) in surv for c, s in zip(sep.cell, sep.side)])
    print("  reported, never used to overturn:")
    if is_s.any() and (~is_s).any():
        res["the 31"] = size_stats(sep.k[is_s], sep.fam[is_s], sep.j[is_s], "SEP, the 31 survivors")
        res["the other 286"] = size_stats(sep.k[~is_s], sep.fam[~is_s], sep.j[~is_s], "SEP, the other 286")
    for a in ("hh", "ll", "hv", "session"):
        g = sep[sep.cell.str.split("_").str[1] == a]
        if g.fam.nunique() > 1:
            res[a] = size_stats(g.k, g.fam, g.j, f"SEP, anchor {a}")
    print(f"  ALL {M} replicates, POINT ESTIMATES ONLY (neighbouring anchors overlap; no interval): "
          f"k<=4 {100*(ok.k <= 4).mean():.4f}%   k==0 {100*(ok.k == 0).mean():.4f}%   "
          f"mean p {((1 + ok.k) / 101).mean():.4f}")
    pj = sep.groupby("j").k.apply(lambda s: (s <= 4).mean())
    ncell = len(have)
    print(f"  per-replicate k<=4 rate over SEP: sd {100*pj.std(ddof=1) if len(pj) > 1 else float('nan'):.2f} pp against "
          f"{100*np.sqrt(NOMINAL5*(1-NOMINAL5)/ncell):.2f} pp if cells were independent within a replicate (descriptive)")
    hist = np.bincount(np.minimum(sep.k.to_numpy() // 10, 9), minlength=10) / len(sep)
    print("  rank histogram over SEP, k in tenths (uniform ~0.099 each, last 0.109): " + " ".join(f"{h:.3f}" for h in hist))
    pil = ok[[(c, s) in set(PILOT_CELLS) for c, s in zip(ok.cell, ok.side)]]
    if len(pil):
        print(f"  the 9 pilot/archetype cell+sides here: k<=4 in {100*(pil.k <= 4).mean():.2f}% of {len(pil)} frames. "
              f"The pilot's hit-shift frames gave 1 of 22 -- A DIFFERENT DEAD-FRAME CONSTRUCTION, descriptive only.")
    kmut = np.array([k_count(s, shrink_null(SF[(c, sd)]["null"][None, :])[0]) for s, c, sd in zip(sep.score, sep.cell, sep.side)])
    m5 = float((kmut <= 4).mean())
    alive = m5 > res["SEP"]["k<=4"]["hi"]
    print(f"  [{'ALIVE' if alive else 'DEAD '}] mutation: stored nulls shrunk 20% toward their mean -> k<=4 "
          f"{100*m5:.2f}% (must exceed the CI's upper end {100*res['SEP']['k<=4']['hi']:.2f}%)")

    # ---- arm F
    D = ok.pivot_table(index=["cell", "side"], columns="j", values="score", aggfunc="first")
    kk_use = [kk for kk in keys if kk in have]
    D = D.reindex(pd.MultiIndex.from_tuples(kk_use, names=["cell", "side"]))
    rows = [i for i, kk in enumerate(keys) if kk in have]
    NULu, NULm = NUL[rows], shrink_null(NUL[rows])
    fj = [j for j in sorted(D.columns) if D[j].notna().all()]
    hit, nsurv, hit_m = {}, {}, {}
    for j in fj:
        n_, mn, _ = fwer_any(D[j].to_numpy(float), NULu, studentize, romano_wolf)
        hit[j], nsurv[j] = n_ > 0, n_
        hit_m[j] = fwer_any(D[j].to_numpy(float), NULm, studentize, romano_wolf)[0] > 0
    js = [j for j in fj if j < sep_n]
    x, n = int(sum(hit[j] for j in js)), len(js)
    lo, hi = cp(x, n) if n else (0.0, 1.0)
    xa, na = int(sum(hit.values())), len(hit)
    differs = bool(lo > 0.05 or hi < 0.05)
    print(f"\nARM F -- family-wise, the whole {len(kk_use)}-hypothesis family per replicate, installed Romano-Wolf "
          f"(loo=True), criterion any adjusted p <= 0.05, nominal 5%")
    print(f"    DECISION SET (SEP): at least one 'survivor' in {x} of {n} replicates = {100*x/max(n,1):.1f}%   "
          f"Clopper-Pearson 95% [{100*lo:.1f}%, {100*hi:.1f}%]   -> {'DIFFERS FROM NOMINAL' if differs else 'nominal inside'}")
    print(f"    all {na} replicates, point estimate only: {xa} of {na} = {100*xa/max(na,1):.1f}%;  'survivors' per "
          f"replicate: max {max(nsurv.values()) if nsurv else 0}, mean {np.mean(list(nsurv.values())) if nsurv else 0:.2f} (real data: 31)")
    xm = int(sum(hit_m.values()))
    print(f"  [{'ALIVE' if xm > xa else 'DEAD '}] mutation: nulls shrunk 20% -> {xm} of {na} replicates fire")
    out = dict(stamp=st, M=M, sep=sep_n, arm1=res, mutation_k5=m5,
               armF=dict(x=x, n=n, lo=lo, hi=hi, differs=differs, all_x=xa, all_n=na, mut=xm))

    # ---- arm 2
    f2 = sorted(glob.glob(os.path.join(parts, "arm2", "*.csv")))
    if f2:
        a = read_parts(f2)
        if not dry:
            a.to_csv(os.path.join(HERE, f"calib_main_arm2_{today}.csv"), index=False)
        g = a[a.status == "ok"].copy()
        prov2 = all(g[c].nunique() == 1 and str(g[c].iloc[0]) == str(st[c])
                    for c in ("producer_sha256", "plan_sha256", "code_set", "seed"))
        print(f"\nARM 2 -- synthetic positive controls on dead frames: {len(g)} tests ok of {len(a)} "
              f"({len(a) - len(g)} without an injectable column), {g.cell.nunique()} cells, one provenance {prov2}. "
              f"A detection CURVE; no pass threshold. delta 0 is the SAME procedure with nothing injected.")
        if not prov2 and not dry:
            raise SystemExit("  arm-2 rows carry mixed provenance -- not read.")
        print("    scen  delta   n   detected(k<=4)  right column  detected & right   ORACLE(no search) k<=4   achieved A / B pp")
        cur = []
        for (sc, d), q in g.groupby(["scen", "delta"]):
            det, rc = (q.k.astype(int) <= 4), q.right_col.astype(str).str.lower().eq("true")
            row = dict(scen=sc, delta=int(d), n=len(q), detected=float(det.mean()), right_col=float(rc.mean()),
                       both=float((det & rc).mean()), oracle=float((q.k_oracle.astype(int) <= 4).mean()),
                       gotA=float(100 * q.gotA.astype(float).mean()), gotB=float(100 * q.gotB.astype(float).mean()))
            cur.append(row)
            print(f"    {sc:<5} {int(d):>4} {len(q):>4}   {100*row['detected']:>10.1f}%   {100*row['right_col']:>9.1f}%"
                  f"   {100*row['both']:>12.1f}%   {100*row['oracle']:>16.1f}%        {row['gotA']:.2f} / {row['gotB']:.2f}")
        out["arm2"] = dict(curve=cur, injected_cols=g.inj_col.value_counts().to_dict())
        print(f"    injected columns: {out['arm2']['injected_cols']}")
    else:
        print("\nARM 2: no parts yet.")
    if not dry:
        with open(os.path.join(HERE, f"calib_main_{today}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"\n  wrote calib_main_arm1_{today}.csv" + (f", calib_main_arm2_{today}.csv" if f2 else "")
              + f" and calib_main_{today}.json")
    print("  NOT VALIDATED BY THIS RUN: the exact adjusted p of the 17 at the 1/101 floor; any other joint seed.")
    return out


# ------------------------------------------------------------------ self-test
def _sim(nsim, shrink, seed, ncell=315, fam_size=9, M=SEP, w_fam=0.25, w_rep=0.15, nboot=500, shift=0.0):
    """The validator on toy data with the DECISION SET's shape: cells in families, M separated
    replicates, one fixed set of 100 nulls per cell, a component shared by a family AND a
    component shared by a whole replicate across every family (the common-anchor shock).
    Returns, per statistic, the fraction of simulations whose pigeonhole CI excludes nominal."""
    rng = np.random.default_rng(seed)
    nf = ncell // fam_size
    fam = np.repeat(np.arange(nf), fam_size)
    F, R = np.repeat(fam, M), np.tile(np.arange(M), ncell)
    rej = {name: 0 for name, *_ in STATS}
    for _ in range(nsim):
        nul = np.sort(rng.normal(size=(ncell, 100)) * shrink, axis=1)
        d = (np.sqrt(w_rep) * rng.normal(size=(1, M)) + np.sqrt(w_fam) * rng.normal(size=(nf, M))[fam]
             + np.sqrt(1 - w_fam - w_rep) * rng.normal(size=(ncell, M))) + shift
        k = np.vstack([100 - np.searchsorted(nul[i], d[i], side="left") for i in range(ncell)]).ravel()
        for name, f, nom, sc in STATS:
            e, lo, hi = pigeonhole_ci(f(k), F, R, nboot=nboot, seed=int(rng.integers(1 << 30)))
            rej[name] += (lo > nom) or (hi < nom)
    return {name: v / nsim for name, v in rej.items()}


def selftest():
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else ""), flush=True)
        ok[0] &= bool(cond)
    print("\ncalib_main.py SELF-TEST -- toy data only, no dump.\n")
    rng = np.random.default_rng(1)

    def toy(n, seed, per_day):
        g = np.random.default_rng(seed)
        day = 15000 + np.sort(g.integers(0, 3000, n)) if per_day is None else 15000 + np.arange(n) // per_day
        blk = np.where(day < 15000 + (day.max() - 15000) // 2, "A", "B")
        regime = np.repeat(g.normal(0, 1, n // 50 + 1), 50)[:n]
        x1 = regime + g.normal(0, 0.3, n)
        hit = (g.random(n) < 0.5 + 0.15 * np.tanh(regime)).astype(float)
        return pd.DataFrame(dict(bar=np.arange(n), block=blk, day=day, hit=hit, x1=x1, x2=g.normal(size=n)))
    r = toy(6000, 2, 2)
    cols = ["x1", "x2"]
    bounds = {B: (int(r.day[r.block == B].min()), int(r.day[r.block == B].max())) for B in "AB"}
    di = r.day.to_numpy()
    rd, info = shift_covariates(r, cols, anchor_fracs(40), bounds, di)
    chk("K1 the shift moves ONLY covariates: hits, days, blocks, bars identical",
        all((rd[c] == r[c]).all() for c in ("hit", "day", "block", "bar")))
    chk("K2 each block keeps its exact multiset of covariate ROWS (pairs stay together)",
        all(sorted(map(tuple, rd.loc[rd.block == B, cols].to_numpy())) == sorted(map(tuple, r.loc[r.block == B, cols].to_numpy())) for B in "AB"))
    chk("K3 offsets are interior", all(0 < o < n for o, n, lg in info.values()), str(info))
    b0, b1 = np.corrcoef(r.x1, r.hit)[0, 1], np.corrcoef(rd.x1, rd.hit)[0, 1]
    chk("K4 the covariate-outcome link is broken", abs(b1) < abs(b0) / 3, f"{b0:+.3f} -> {b1:+.3f}")
    l1 = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
    xa, xb = r.x1[r.block == "A"].to_numpy(), rd.x1[rd.block == "A"].to_numpy()
    chk("K5 the covariate's own persistence is kept (lag-1 autocorrelation within 0.02)", abs(l1(xa) - l1(xb)) < 0.02, f"{l1(xa):.3f} -> {l1(xb):.3f}")
    same, _ = shift_covariates(r, cols, {}, bounds, di)
    chk("M1 no shift leaves the link intact (K4 is not vacuous)", abs(np.corrcoef(same.x1, same.hit)[0, 1] - b0) < 1e-12)
    r2 = toy(1500, 5, None)                                  # a sparse cell, other density
    _, inf2 = shift_covariates(r2, cols, anchor_fracs(40), bounds, r2.day.to_numpy())
    d1 = r.day[r.block == "A"].to_numpy()[info["A"][0]]
    d2 = r2.day[r2.block == "A"].to_numpy()[inf2["A"][0]]
    chk("K6 ONE calendar anchor for all cells: a dense and a sparse cell start from the same date (within 5 days)",
        abs(int(d1) - int(d2)) <= 5, f"day {d1} vs {d2}")
    chk("K7 replicates differ and are reproducible", anchor_fracs(3) == anchor_fracs(3) and anchor_fracs(3) != anchor_fracs(4)
        and all(F_LO <= v <= F_HI for j in (0, 28, 29, 99) for v in anchor_fracs(j).values()))
    spanA, spanB = 17529 - 14767, 20671 - 18995              # the plan's block spans, days
    fa = np.sort([anchor_fracs(j)["A"] for j in range(SEP)]) * spanA
    fb = np.sort([anchor_fracs(j)["B"] for j in range(SEP)]) * spanB
    chk("K7b THE SEPARATED SET: 29 anchors at least 40 calendar days apart in BOTH blocks",
        np.diff(fa).min() >= 40 and np.diff(fb).min() >= 40, f"min gap A {np.diff(fa).min():.1f} d, B {np.diff(fb).min():.1f} d")
    fx = np.sort([anchor_fracs(j)["B"] for j in range(SEP, 100)]) * spanB
    chk("M1b the extra replicates are NOT separated (so they must stay out of the decision set)", np.diff(fx).min() < 40,
        f"min gap {np.diff(fx).min():.1f} d")
    chk("K8 k counts null >= obs as an integer", k_count(0.5, [0.1, 0.5, 0.7, 0.2]) == 2)
    rt = read_parts([io.StringIO("cell,side,k\nX,None,3\nX,1.0,\n")])
    chk("K8b a part file's side 'None' is read as the string None, an empty number as NaN",
        rt.side.tolist() == ["None", "1.0"] and np.isnan(rt.k.iloc[1]))
    # injection (a 2-setups-per-day toy cannot show day concentration -- a day holds at most
    # one region miss; the first version failed K11 on that fixture, not on the code)
    r3 = toy(6000, 7, 40)
    di = r3.day.to_numpy()
    reg = r3.x2.to_numpy() > np.quantile(r3.x2, Q_INJ)
    blocks = r3.block.to_numpy()
    h0 = r3.hit.to_numpy()
    hr, gr = inject(h0.copy(), reg, blocks, di, {"A": 0.08, "B": 0.08}, "row", np.random.default_rng(3))
    hd, gd = inject(h0.copy(), reg, blocks, di, {"A": 0.08, "B": 0.02}, "day", np.random.default_rng(3))
    hz, gz = inject(h0.copy(), reg, blocks, di, {"A": 0.0, "B": 0.0}, "row", np.random.default_rng(3))
    up = lambda h, B: h[reg & (blocks == B)].mean() - h0[reg & (blocks == B)].mean()
    chk("K9 row injection raises the region's hit rate by delta in both blocks, nothing outside moves",
        abs(up(hr, "A") - 0.08) < 0.005 and abs(up(hr, "B") - 0.08) < 0.005 and (hr[~reg] == h0[~reg]).all(),
        f"A {100*up(hr,'A'):.2f} B {100*up(hr,'B'):.2f} pp")
    chk("K10 HARD: block B gets a quarter", abs(up(hd, "A") - 0.08) < 0.005 and abs(up(hd, "B") - 0.02) < 0.005,
        f"A {100*up(hd,'A'):.2f} B {100*up(hd,'B'):.2f} pp")
    tr = lambda h: len(np.unique(di[(h != h0) & (blocks == "A")]))
    chk("K11 day injection concentrates the same flips in FEWER days than row injection", tr(hd) < 0.7 * tr(hr), f"{tr(hd)} vs {tr(hr)} days")
    chk("K11b delta 0 changes nothing", (hz == h0).all())
    chk("K12 the injection boundary is off the search's decile grid", all(abs(Q_INJ - q) > 0.04 for q in np.linspace(0.10, 0.90, 9)))
    chk("K12b `speed_xs` counts as the right column for `speed`; another column does not",
        is_right_col("speed_xs", "speed") and is_right_col("speed", "speed") and not is_right_col("W0", "speed"))
    studentize, romano_wolf = rw_funcs()
    NUL = rng.normal(size=(40, 100))
    o = rng.normal(size=40)
    to, T = studentize(o[:1], NUL[:1], loo=False)
    chk("K13 the installed stepdown with one hypothesis is the plain permutation p",
        abs(romano_wolf(to, T)[0] - (1 + k_count(o[0], NUL[0])) / 101) < 1e-12)
    o2 = o.copy(); o2[5] = 12.0
    chk("M2 a planted +12 sd dead score makes the family-wise arm fire", fwer_any(o2, NUL, studentize, romano_wolf)[0] >= 1)
    fire = np.mean([fwer_any(rng.normal(size=40), NUL, studentize, romano_wolf)[0] > 0 for _ in range(400)])
    fire_m = np.mean([fwer_any(rng.normal(size=40), shrink_null(NUL, 0.8), studentize, romano_wolf)[0] > 0 for _ in range(400)])
    chk("M3 a null shrunk 20% raises the family-wise firing rate", fire_m > fire + 0.05, f"{100*fire:.1f}% -> {100*fire_m:.1f}%")
    lo, hi = cp(1, 29)
    chk("K14 Clopper-Pearson hand check: 1 of 29 -> [0.09%, 17.8%]", abs(lo - 0.00087) < 2e-4 and abs(hi - 0.1776) < 2e-3, f"[{100*lo:.2f}%, {100*hi:.2f}%]")
    # the validator itself, WITH a replicate-common shock, for all three statistics (C143)
    t0 = time.time()
    size = _sim(200, 1.0, 11)
    # POWER, stated as what it is. A null 20% too narrow has a true k<=4 rate of 9.4%; 10% too
    # narrow, 6.9%. With a STRONG replicate-common shock (15% of the variance) 29 replicates
    # cannot see the smaller one reliably; with none they can. A symmetric narrowing leaves
    # mean p unchanged, so mean p is tested against a SHIFT (the first M4 asked mean p to see a
    # narrowing -- a defect in the test, not in the code).
    p20 = _sim(60, 0.8, 12)
    p10 = _sim(60, 0.9, 14)
    p10_0 = _sim(60, 0.9, 15, w_rep=0.0)
    psh = _sim(60, 1.0, 16, shift=0.25)
    psh15 = _sim(60, 1.0, 17, shift=0.15)
    naive = _sim_family_only(200, 13)
    chk("K15 VALIDATOR SIZE with family AND replicate-common shocks: the pigeonhole CI excludes nominal in at most "
        "10% of calibrated runs, for k<=4, k==0 and mean p", all(v <= 0.10 for v in size.values()),
        "  ".join(f"{n} {100*v:.1f}%" for n, v in size.items()) + "  of 200")
    chk("M4 VALIDATOR POWER, strong replicate shock: a null 20% too narrow (true rate 9.4%) is flagged by k<=4 "
        "in at least 80% of runs", p20["k<=4"] >= 0.80, f"k<=4 {100*p20['k<=4']:.0f}%  k==0 {100*p20['k==0']:.0f}%  of 60")
    chk("M4b no replicate shock: a null 10% too narrow (true rate 6.9%) is flagged by k<=4 in at least 80% of runs",
        p10_0["k<=4"] >= 0.80, f"{100*p10_0['k<=4']:.0f}% of 60")
    print(f"  [INFO] strong replicate shock, null 10% too narrow: k<=4 flags {100*p10['k<=4']:.0f}% of 60 -- "
          f"the run CANNOT promise to see a 6.9% size if replicates share a strong shock")
    chk("M4c strong replicate shock: mean p sees a SHIFT of +0.25 sd in the dead scores in at least 80% of runs",
        psh["mean p"] >= 0.80, f"{100*psh['mean p']:.0f}% of 60   [{time.time()-t0:.0f}s]")
    print(f"  [INFO] the same with a shift of +0.15 sd: {100*psh15['mean p']:.0f}% of 60 (the first M4c demanded 80% "
          f"here; with a 15% shared shock 29 replicates give about 2 standard errors -- the bar was wrong, not the code)")
    chk("M5 a FAMILY-ONLY resample is fooled by the replicate-common shock (false alarms well above the pigeonhole's)",
        naive > size["k<=4"] + 0.05, f"family-only {100*naive:.1f}% vs pigeonhole {100*size['k<=4']:.1f}%")
    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


def _sim_family_only(nsim, seed, ncell=315, fam_size=9, M=SEP, w_fam=0.25, w_rep=0.15):
    rng = np.random.default_rng(seed)
    nf = ncell // fam_size
    fam = np.repeat(np.arange(nf), fam_size)
    rej = 0
    for _ in range(nsim):
        nul = np.sort(rng.normal(size=(ncell, 100)), axis=1)
        d = (np.sqrt(w_rep) * rng.normal(size=(1, M)) + np.sqrt(w_fam) * rng.normal(size=(nf, M))[fam]
             + np.sqrt(1 - w_fam - w_rep) * rng.normal(size=(ncell, M)))
        k = np.vstack([100 - np.searchsorted(nul[i], d[i], side="left") for i in range(ncell)])
        a = np.array([(k[fam == f] <= 4).sum() for f in range(nf)], float)
        S = rng.integers(0, nf, (500, nf))
        est = a[S].sum(1) / (fam_size * M * nf)
        rej += (np.percentile(est, 2.5) > NOMINAL5) or (np.percentile(est, 97.5) < NOMINAL5)
    return rej / nsim


def preflight():
    import tempfile
    tmp = tempfile.mkdtemp()
    plan = load_plan()
    pooled = [x for x in plan["cells"] if x["side"] == "None"][-1]
    sided = [x for x in plan["cells"] if x["side"] != "None"][-1]
    res = []
    for x in (pooled, sided):
        t0 = time.time()
        rc = arm1_job(x["cell"], x["side"], 3, parts=os.path.join(tmp, "arm1"), k0c=1)
        f = os.path.join(tmp, "arm1", tag_of(x["cell"], x["side"]) + ".csv")
        g = read_parts([f]) if os.path.exists(f) else pd.DataFrame()
        a = rc == 0 and len(g) == 3 and (g.status == "ok").all() and g.k.between(0, 100).all() and (g.side == x["side"]).all()
        res.append(a)
        print(f"  [{'PASS' if a else 'FAIL'}] arm 1 on {x['cell']} side {x['side']}: K0a-c and 3 dead frames, "
              f"{time.time()-t0:.0f}s (no calibration value shown)")
    y = plan["arm2"][0]
    for d, s, tol in ((8, "HARD", (0.08, 0.02)), (0, "ZERO", (0.0, 0.0))):
        t0 = time.time()
        arm2_job(y["cell"], y["side"], d, s, 0, parts=os.path.join(tmp, "arm2"), nperm=2)
        f = os.path.join(tmp, "arm2", f"{tag_of(y['cell'], y['side'])}__d{d}__{s}.csv")
        g = read_parts([f]) if os.path.exists(f) else pd.DataFrame()
        b = len(g) == 1 and g.status.iloc[0] == "ok" and abs(g.gotA.iloc[0] - tol[0]) < 0.01 and abs(g.gotB.iloc[0] - tol[1]) < 0.01
        res.append(b)
        print(f"  [{'PASS' if b else 'FAIL'}] arm 2 on {y['cell']} side {y['side']} delta {d} {s}: injection achieved its "
              f"deltas, search + oracle + 2 permutations ran, {time.time()-t0:.0f}s (no detection value shown)")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = merge(parts=tmp, dry=True)
        m = out["armF"]["all_n"] == 3 and out["M"] == 3 and "arm2" in out and len(out["arm2"]["curve"]) == 2
    except BaseException as e:                         # SystemExit included: the merge refused
        m = False
        print(f"      merge raised: {e}")
    res.append(m)
    print(f"  [{'PASS' if m else 'FAIL'}] the MERGE path runs end to end on these parts (pooled 'None' side kept, "
          f"3 family-wise replicates, arm-2 curve) -- output suppressed, no value shown")
    print(f"\n  PREFLIGHT {'PASS' if all(res) else 'FAIL'}  (temp dir {tmp}; nothing in the project)")
    return all(res)


if __name__ == "__main__":
    a = sys.argv[1:]
    M = int(a[a.index("--M") + 1]) if "--M" in a else M_DEFAULT
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    if "--one" in a:
        s = a[a.index("--one") + 1].split(":")
        if s[0] == "1":
            sys.exit(arm1_job(s[1], s[2], M))
        sys.exit(arm2_job(s[1], s[2], int(s[3]), s[4], int(s[5])))
    if "--plan" in a:
        make_plan(); sys.exit(0)
    if "--preflight" in a:
        sys.exit(0 if preflight() else 1)
    if "--merge" in a:
        merge(); sys.exit(0)
    if "--arm" in a:
        arm = a[a.index("--arm") + 1]
        if arm not in ("1", "2"):
            raise SystemExit("--arm 1 or --arm 2")
        if M < SEP:
            raise SystemExit(f"--M must be at least {SEP} (the decision set)")
        if not selftest():
            raise SystemExit("SELF-TEST FAILED")
        w = int(a[a.index("--workers") + 1]) if "--workers" in a else 8
        part = a[a.index("--part") + 1] if "--part" in a else None
        sys.exit(1 if run_pool(arm, w, part, M) else 0)
    print(__doc__)
