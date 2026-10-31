r"""
race_stage1.py -- Stage 1's filter search re-run with the MIRROR RACE as the outcome
(RACE_STAGE1_PROPOSAL v2, 2026-10-14). NOTHING STATISTICAL IS REIMPLEMENTED: the search is
core_v2.best_core on core_v2.search_cols, the null draw is floor_v2.shuffle_within_block
(day_strata, JOINT, seed 20260927), the Romano-Wolf stepdown is rw_check_independent's
studentize / romano_wolf taken by ast (calib_main.rw_funcs), the dead frame is
calib_main.shift_covariates and the planting is calib_main.inject.

THE ONE CHANGE: the outcome. Per cell+side, core_v2.load_cell's frame is restricted to
  setups whose race is resolved (race in {1, -1}); hit := (race == 1)  [VWAP before the mirror]
  twins whose OWN race is resolved AND whose source setup is resolved; hit := (race == 1)
Ties (race 0) and opens (race 9) are dropped, as in claim 2. The edge is P(VWAP first) on the
filtered setups minus the same on their twins: the mirror null, stated "among resolved setups".
No column is added; the searched column set equals Stage 1's (asserted).

FAMILY. The 317 effective cell+sides of rw_check_independent_out.csv (SHA-256 asserted). The
AUD-12 gate (core_v2.py:274-276) is applied to the RACE frame: each block >= 1000 setups and
>= 1000 twins. Fall-outs are LISTED, never substituted; the family is the cell+sides that pass
the gate and have a feasible search. Romano-Wolf runs over that family only.

NULLS. perm_index = OFFSET + k, OFFSET 2000, k = 0..99 (A14: indices at or above 2000; a later
confirmation of race cores must use >= 2100). B = 100, p = (1 + #{null >= obs}) / 101, the
adjusted-p floor is 1/101 = 0.0099 as in Stage 1.

READING, FIXED BEFORE THE RUN (also printed by --summarize):
  The expected yield is FEW OR NONE. On the record: the recent-block race score is non-negative
  only above about 3.4-4 pp at a 20% subset before the family bar (26.161 (1)); the full
  Romano-Wolf recall on planted fill effects was 48% at 8 pp equal in both periods, 1.7% at 4 pp,
  0% for decaying effects (26.105/26.107); the unconditional recent-block race gap is +0.78 pp.
  0 race cores    -> "no filter's race edge was detected at this power"; NEVER "the mirror does
                     not matter" or "no filter works".
  >0 race cores   -> a TRAIN-ONLY SCREENING list, in-sample, sizes carry the winner's curse, NOT
                     replicated; descriptive (28.1 decision 9: the mirror null is never a pass
                     condition); its list and SHA-256 join the USE-2 ingredient list.
  The raw p <= 0.05 count is printed beside the expected count under no effect (0.05 x family).
  --recall prints a planted-effect recall curve (1x, 2x, 4x NEED) with NO pass mark.

KNOWN NUMBERS (--check), before any race result:
  K0  pooled frames of the 144 registered cells: 3,260,501 resolved setups; setup race rate
      54.815% (A) / 50.737% (B); twin race rate over the twins of ALL setups 49.958% / 49.957%
      (claim 2, 26.65), to 0.0005 pp.
  K1  the job function with the outcome switched back to FILL and offset 0 reproduces, on 2
      cell+sides, the stored cores_v2 winner (col, cut, sense, score) and floors_v2 nulls 0..2.
  K1b offset liveness: the race null at offset 2000 equals an independent direct call
      (shuffle_within_block(perm_index=2000) -> best_core) and differs from offset 0.
  K2  search_cols(race frame) == search_cols(fill frame) as a list, on every checked cell.
  ISO the real cores_v2 and floors_v2 trees hash the same before and after (no write path of
      core_v2 / floor_v2 is called; this script writes only race_stage1\).

    python race_stage1.py --selftest
    python race_stage1.py --check
    python race_stage1.py --run --workers 8 [--part 1,2,3/5]
    python race_stage1.py --summarize
    python race_stage1.py --recall
Project root; rotates nothing (not in seal_certify's hashed set).
"""
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "race_stage1")
PARTS = os.path.join(OUTD, "parts")
RW_CSV = os.path.join(HERE, "rw_check_independent_out.csv")
RW_CSV_SHA = "44E9D780FE0C8C6C1C2812340887CFDBF41DDB618A3BE4F99C0D4BC47DF4B0B0"
OFFSET = 2000
NPERM = 100
GATE = 1000
SEED = 20261014
K0 = dict(n_res=3260501, rA=54.815, rB=50.737, tA=49.958, tB=49.957)
K1_TAGS = [("N200_hv_ATR_6.0", "1.0"), ("N100_hh_ATR_3.0", "-1.0")]
RECALL_N, RECALL_M = 20, (1, 2, 4)


def _imports():
    sys.path[:0] = [p for p in (HERE, os.path.join(HERE, "code")) if p not in sys.path]
    import core_v2
    import floor_v2
    return core_v2, floor_v2


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest().upper()


def side_val(s):
    return None if s in ("", "None", None) else float(s)


def side_s(s):
    return "None" if s in ("", "None", None) else str(float(s))


def tag_of(cell, s):
    return f"{cell}_side{side_s(s)}"


def family_list():
    if sha(RW_CSV) != RW_CSV_SHA:
        raise SystemExit("REFUSED: rw_check_independent_out.csv changed")
    t = pd.read_csv(RW_CSV, dtype={"side": str}, keep_default_na=False, usecols=["cell", "side"])
    out = [(c, side_s(s)) for c, s in zip(t.cell, t.side)]
    assert len(out) == 317 and len(set(out)) == 317
    return out


def stamp():
    import gate_receipts
    m = gate_receipts.machine_stamp()
    return dict(code_set=gate_receipts.code_set(), producer_sha256=sha(os.path.abspath(__file__)),
                offset=OFFSET, nperm=NPERM, machine=m["node"], python=m["python"],
                numpy=m["numpy"], pandas=m["pandas"])


def tree_fp(core_v2, floor_v2):
    h = hashlib.sha256()
    for d in (core_v2.OUT, floor_v2.FLOORS):
        for f in sorted(glob.glob(os.path.join(d, "*"))):
            h.update(f"{os.path.basename(f)}:{sha(f) if os.path.isfile(f) else 'DIR'}\n".encode())
    return h.hexdigest().upper()


# ------------------------------------------------------------------ the race frame
def race_from(r, c, x, side):
    """r, c = core_v2.load_cell's frames; x = the cell file's is_real/side/bar/src_bar/race."""
    xr = x[x.is_real == 1]
    if side is not None:
        xr = xr[np.sign(xr.side) == side]
    xc = x[x.is_real == 0]
    xc = xc[xc.src_bar.isin(set(xr.bar))]
    assert (xr.bar.to_numpy() == r.bar.to_numpy()).all(), "setup alignment"
    assert (xc.src_bar.to_numpy() == c.src_bar.to_numpy()).all(), "twin alignment"
    rr, cr = xr.race.to_numpy(), xc.race.to_numpy()
    kr = np.isin(rr, (1, -1))
    r2 = r[kr].reset_index(drop=True)
    r2["hit"] = (rr[kr] == 1).astype(float)
    kc = np.isin(cr, (1, -1)) & c.src_bar.isin(set(r2.bar)).to_numpy()
    c2 = c[kc].reset_index(drop=True)
    c2["hit"] = (cr[kc] == 1).astype(float)
    assert list(r2.columns) == list(r.columns) and list(c2.columns) == list(c.columns)
    return r2, c2


def race_frame(core_v2, cell, side):
    r, c, T = core_v2.load_cell(cell, side)
    x = pd.read_parquet(os.path.join(core_v2.DUMP, cell + ".parquet"),
                        columns=["is_real", "side", "bar", "src_bar", "race"])
    r2, c2 = race_from(r, c, x, side)
    return r, c, r2, c2, T


def gate_ok(r, c):
    return min(int((r.block == B).sum()) for B in "AB") >= GATE and \
        min(int((c.block == B).sum()) for B in "AB") >= GATE


def search_with_nulls(core_v2, floor_v2, r, c, side, nperm, offset, partial=None):
    """Stage 1 on one frame: obs = best_core, null k = best_core on the day_strata shuffle at
    perm_index offset + k. Returns obs, nulls, cols, P."""
    cols = core_v2.search_cols(r, side)
    P = core_v2.prep(r, c)
    obs = core_v2.best_core(r, c, P, cols)
    done = {}
    if partial and os.path.exists(partial):
        t = pd.read_csv(partial)
        done = dict(zip(t.k.astype(int), t.score.astype(float)))
    null = np.empty(nperm)
    scratch = r.copy()
    rng = np.random.default_rng(0)                     # unused under JOINT_NULL + perm_index
    for k in range(nperm):
        if k in done:
            null[k] = done[k]
            continue
        b = core_v2.best_core(floor_v2.shuffle_within_block(
            r, cols, rng, into=scratch, unit=floor_v2.NULL_UNIT, perm_index=offset + k), c, P, cols)
        null[k] = b["score"] if b is not None else -np.inf
        if partial:
            pd.DataFrame([dict(k=k, score=null[k])]).to_csv(partial, mode="a", index=False,
                                                            header=not os.path.exists(partial))
    return obs, null, cols, P


def dead_score(core_v2, floor_v2, r, c, cols, P, j=0):
    import calib_main as CM
    bounds = {B: tuple(v) for B, v in CM.load_plan()["bounds"].items()}
    rd, _ = CM.shift_covariates(r, cols, CM.anchor_fracs(j), bounds, floor_v2._as_day_int(r.day))
    b = core_v2.best_core(rd, c, P, cols)
    return (b["score"] if b is not None else -np.inf), rd


# ------------------------------------------------------------------ one job
def job(cell, s):
    core_v2, floor_v2 = _imports()
    os.makedirs(PARTS, exist_ok=True)
    tag = tag_of(cell, s)
    final = os.path.join(PARTS, tag + ".json")
    if os.path.exists(final):
        return 0
    side = side_val(s)
    st = stamp()
    r, c, r2, c2, T = race_frame(core_v2, cell, side)
    base = dict(cell=cell, side=s, T=T, n_setups=len(r), n_res_setups=len(r2), n_twins=len(c),
                n_res_twins=len(c2), resA=int((r2.block == "A").sum()), resB=int((r2.block == "B").sum()),
                twA=int((c2.block == "A").sum()), twB=int((c2.block == "B").sum()), **st)
    if not gate_ok(r2, c2):
        json.dump(dict(status="gated", **base), open(final, "w"), indent=1)
        return 0
    obs, null, cols, P = search_with_nulls(core_v2, floor_v2, r2, c2, side, NPERM, OFFSET,
                                           partial=os.path.join(PARTS, tag + ".partial"))
    if obs is None:
        json.dump(dict(status="no feasible cut", **base), open(final, "w"), indent=1)
        return 0
    d0, _ = dead_score(core_v2, floor_v2, r2, c2, cols, P, 0)
    p = (1 + int((null >= obs["score"]).sum())) / (1 + NPERM)
    out = dict(status="ok", **base, col=obs["col"], cut=obs["cut"], sense=obs["sense"], pct=obs["pct"],
               obs_score=obs["score"], eA=obs["eA"], eB=obs["eB"], seA=obs["seA"], seB=obs["seB"],
               nA=obs["nA"], nB=obs["nB"], p=p, null_scores=[float(v) for v in null],
               n_null_empty=int((~np.isfinite(null)).sum()), dead0_score=float(d0),
               null_unit=floor_v2.NULL_UNIT, joint=bool(floor_v2.JOINT_NULL), joint_seed=int(floor_v2.JOINT_SEED),
               n_cols=len(cols), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = final + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, final)
    pp = os.path.join(PARTS, tag + ".partial")
    if os.path.exists(pp):
        os.replace(pp, os.path.join(PARTS, tag + ".partial_done"))
    return 0


def job_list(part=None):
    jobs = family_list()
    if part:
        which, n = part.split("/")
        keep, n = {int(v) for v in which.split(",")}, int(n)
        if not keep or min(keep) < 1 or max(keep) > n:
            raise SystemExit(f"bad --part {part}")
        jobs = [jb for i, jb in enumerate(jobs) if (i % n) + 1 in keep]
    return jobs


def run_pool(workers, part):
    core_v2, floor_v2 = _imports()
    os.makedirs(os.path.join(OUTD, "logs"), exist_ok=True)
    fp0 = tree_fp(core_v2, floor_v2)
    man = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), part=part, workers=workers,
               jobs=len(job_list(part)), rw_csv_sha256=RW_CSV_SHA, trees_before=fp0, **stamp())
    with open(os.path.join(OUTD, f"manifest_{time.strftime('%Y-%m-%d_%H%M%S')}.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    queue = [jb for jb in job_list(part) if not os.path.exists(os.path.join(PARTS, tag_of(*jb) + ".json"))]
    live, done, failed = {}, 0, 0
    print(f"  race Stage 1: {len(queue)} jobs to do" + (f" (part {part})" if part else "") + f", {workers} workers", flush=True)
    t0 = last = time.time()
    while queue or live:
        while queue and len(live) < workers:
            cell, s = queue.pop(0)
            name = tag_of(cell, s)
            log = open(os.path.join(OUTD, "logs", name + ".log"), "w")
            env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--one", f"{cell}:{s}"],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            live[p] = (name, log)
        for p in list(live):
            if p.poll() is None:
                continue
            name, log = live.pop(p)
            log.close()
            if p.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"  [FAIL] {name} rc={p.returncode} -- race_stage1\\logs\\{name}.log", flush=True)
        if time.time() - last > 300:
            last = time.time()
            print(f"  ... {done} done, {failed} failed, {len(live)} running, {len(queue)} queued, "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
        time.sleep(0.5)
    fp1 = tree_fp(core_v2, floor_v2)
    print(f"\n  {done} done, {failed} failed, {(time.time() - t0) / 60:.1f} min wall")
    print(f"  [{'PASS' if fp0 == fp1 else 'FAIL'}] ISO real cores_v2 + floors_v2 unchanged by the run")
    return 1 if (failed or fp0 != fp1) else 0


# ------------------------------------------------------------------ checks
def check():
    core_v2, floor_v2 = _imports()
    import power_registered_set as PRS
    ok = [True]

    def chk(name, cond, extra=""):
        ok[0] &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}", flush=True)

    fp0 = tree_fp(core_v2, floor_v2)
    fam = family_list()
    chk("family list: 317 distinct cell+sides, rw csv hash", len(fam) == 317)
    # K0: claim 2 totals on the pooled frames of the 144 registered cells
    t0 = time.time()
    n_res = 0
    s = {B: [0, 0, 0, 0] for B in "AB"}            # setup vf, setup res, twin vf, twin res
    for cell in PRS.registered_cells():
        r, c, r2, c2, T = race_frame(core_v2, cell, None)
        n_res += len(r2)
        x = pd.read_parquet(os.path.join(core_v2.DUMP, cell + ".parquet"),
                            columns=["is_real", "block", "race"])
        tw = x[x.is_real == 0]
        for B in "AB":
            rb = r2[r2.block == B]
            s[B][0] += rb.hit.sum(); s[B][1] += len(rb)
            tb = tw[(tw.block == B) & tw.race.isin((1, -1))]
            s[B][2] += int((tb.race == 1).sum()); s[B][3] += len(tb)
    rate = {B: (100 * s[B][0] / s[B][1], 100 * s[B][2] / s[B][3]) for B in "AB"}
    chk("K0 resolved setups over the 144 pooled cells == 3,260,501", n_res == K0["n_res"], f"{n_res:,}")
    chk("K0 setup race rate A / B == 54.815 / 50.737 %",
        abs(rate["A"][0] - K0["rA"]) < 5e-4 and abs(rate["B"][0] - K0["rB"]) < 5e-4,
        f"{rate['A'][0]:.3f} / {rate['B'][0]:.3f}")
    chk("K0 twin race rate (twins of all setups) A / B == 49.958 / 49.957 %",
        abs(rate["A"][1] - K0["tA"]) < 5e-4 and abs(rate["B"][1] - K0["tB"]) < 5e-4,
        f"{rate['A'][1]:.3f} / {rate['B'][1]:.3f}  ({time.time() - t0:.0f}s)")
    # K1 / K1b / K2 on two cell+sides
    for cell, ss in K1_TAGS:
        side = side_val(ss)
        r, c, r2, c2, T = race_frame(core_v2, cell, side)
        st = pd.read_parquet(os.path.join(core_v2.OUT, tag_of(cell, ss) + ".parquet")).iloc[0]
        fl = pd.read_parquet(os.path.join(floor_v2.FLOORS, tag_of(cell, ss) + ".parquet")).iloc[0]
        obs, null, cols, P = search_with_nulls(core_v2, floor_v2, r, c, side, 3, 0)
        stored_null = np.array(json.loads(fl.null_scores), float)[:3]
        chk(f"K1 FILL outcome, offset 0, reproduces the stored winner  {tag_of(cell, ss)}",
            obs is not None and obs["col"] == st["col"] and obs["sense"] == st["sense"]
            and np.isclose(obs["cut"], st["cut"], rtol=0, atol=1e-12)
            and np.isclose(obs["score"], st["score"], rtol=0, atol=1e-12),
            f"{obs['col'] if obs else None} vs {st['col']}")
        chk(f"K1 FILL outcome nulls 0..2 == stored floors_v2 nulls  {tag_of(cell, ss)}",
            np.allclose(null, stored_null, rtol=0, atol=1e-12))
        cols_r = core_v2.search_cols(r2, side)
        chk(f"K2 search_cols(race) == search_cols(fill)  {tag_of(cell, ss)}", cols_r == cols,
            f"{len(cols_r)} columns")
        if gate_ok(r2, c2):
            _, rn, colsr, Pr = search_with_nulls(core_v2, floor_v2, r2, c2, side, 1, OFFSET)
            b = core_v2.best_core(floor_v2.shuffle_within_block(
                r2, colsr, np.random.default_rng(1), unit=floor_v2.NULL_UNIT, perm_index=OFFSET), c2, Pr, colsr)
            direct = b["score"] if b is not None else -np.inf
            _, rn0, _, _ = search_with_nulls(core_v2, floor_v2, r2, c2, side, 1, 0)
            chk(f"K1b offset live: race null at 2000 == direct call, != offset 0  {tag_of(cell, ss)}",
                np.isclose(rn[0], direct, rtol=0, atol=1e-12) and not np.isclose(rn[0], rn0[0], rtol=0, atol=1e-12),
                f"{rn[0]:.6f} vs direct {direct:.6f}, offset 0 {rn0[0]:.6f}")
        else:
            print(f"  [INFO] {tag_of(cell, ss)} race frame gated; K1b skipped on it")
    # gate fall-outs on the race frame (counts only, no outcome rate)
    gated = []
    for cell, ss in fam:
        r, c, r2, c2, T = race_frame(core_v2, cell, side_val(ss))
        if not gate_ok(r2, c2):
            gated.append(tag_of(cell, ss))
    print(f"  AUD-12 gate on the race frame: {317 - len(gated)} of 317 pass; {len(gated)} gated:")
    for g in gated:
        print(f"      {g}")
    fp1 = tree_fp(core_v2, floor_v2)
    chk("ISO real cores_v2 + floors_v2 unchanged by the check", fp0 == fp1)
    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, f"check_{datetime.date.today().isoformat()}.json")
    json.dump(dict(all_pass=ok[0], gated=gated, n_res=n_res, rates=rate, **stamp()), open(out, "w"), indent=1)
    print(f"\n  CHECK: {'ALL PASS' if ok[0] else 'FAILED'}  -> {out}  SHA-256 {sha(out)}")
    return 0 if ok[0] else 1


# ------------------------------------------------------------------ summary
def load_parts():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(PARTS, "*.json")))]
    return rows


def summarize():
    import calib_main as CM
    studentize, romano_wolf = CM.rw_funcs()
    rows = load_parts()
    fam = family_list()
    have = {tag_of(r["cell"], r["side"]) for r in rows}
    miss = [tag_of(*f) for f in fam if tag_of(*f) not in have]
    prov = {(r["producer_sha256"], r["code_set"], r["offset"], r["nperm"]) for r in rows}
    print(f"  [{'PASS' if not miss else 'FAIL'}] all 317 cell+sides present  missing {len(miss)}")
    print(f"  [{'PASS' if len(prov) == 1 else 'FAIL'}] one provenance on every row  {prov}")
    if miss or len(prov) != 1:
        return 1
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"  family: {len(ok)} ok; gated {sum(r['status'] == 'gated' for r in rows)}; "
          f"no feasible cut {sum(r['status'] == 'no feasible cut' for r in rows)}")
    obs = np.array([r["obs_score"] for r in ok])
    NUL = np.array([r["null_scores"] for r in ok], float)
    NULf = np.where(np.isfinite(NUL), NUL, np.nanmin(np.where(np.isfinite(NUL), NUL, np.nan), 1, keepdims=True))
    t_obs, T = studentize(obs, NULf, loo=True)
    adj = romano_wolf(t_obs, T)
    raw = np.array([r["p"] for r in ok])
    n = len(ok)
    print(f"\n  READING FIXED BEFORE THE RUN: see the docstring (few or none expected; 0 = 'not detected at this power').")
    print(f"  raw p <= 0.05: {int((raw <= 0.05).sum())} of {n}  (expected with no effect: {0.05 * n:.1f})")
    print(f"  Romano-Wolf adjusted p <= 0.05 over the family of {n}: {int((adj <= 0.05).sum())}  (floor 1/101)")
    df = pd.DataFrame([{k: r[k] for k in ("cell", "side", "col", "sense", "cut", "pct", "eA", "eB", "seA", "seB",
                                          "nA", "nB", "obs_score", "p")} for r in ok])
    df["t_obs"], df["rw_adj_p"] = t_obs, adj
    df["anchor"] = df.cell.str.split("_").str[1]
    df["trig_eps"] = df.cell.str.split("_", n=2).str[2]
    df["dedupe_group"] = df.anchor + "|" + df.trig_eps + "|" + df.side
    fill31 = pd.read_csv(RW_CSV, dtype={"side": str}, keep_default_na=False, usecols=["cell", "side", "rw_adj_p"])
    f31 = {tag_of(c, s) for c, s, p in zip(fill31.cell, fill31.side, fill31.rw_adj_p) if p <= 0.05}
    df["in_fill31"] = [tag_of(c, s) in f31 for c, s in zip(df.cell, df.side)]
    pas = df[df.rw_adj_p <= 0.05].sort_values("rw_adj_p")
    print(f"  race cores: {len(pas)} cell+sides in {pas.dedupe_group.nunique()} dedupe groups; "
          f"{int(pas.in_fill31.sum())} of them also among the fill-unit 31")
    pd.set_option("display.width", 250)
    if len(pas):
        show = pas[["cell", "side", "col", "sense", "cut", "pct", "eA", "eB", "seA", "seB", "rw_adj_p", "in_fill31"]].copy()
        for k in ("eA", "eB", "seA", "seB"):
            show[k] = (100 * show[k]).round(2)
        print(show.to_string(index=False))
    out = os.path.join(OUTD, f"race_stage1_{datetime.date.today().isoformat()}.csv")
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}  SHA-256 {sha(out)}")
    print("  STAMP: race unit (VWAP before mirror | resolved), twin-differenced, core_v2.best_core unchanged | "
          f"day_strata JOINT seed 20260927, perm_index {OFFSET}-{OFFSET + NPERM - 1} | {n} cell+sides | "
          f"code-set {ok[0]['code_set']} | producer {ok[0]['producer_sha256'][:16]} | ES train A+B, sealed untouched | "
          "in-sample screening, raw, no cost")
    return 0


def recall():
    """Planted race effects on dead frames, ranked (a) plain k <= 4 against the cell's own race
    nulls and (b) through Romano-Wolf over the family with every other member at its dead-frame
    score. NO pass mark (E2)."""
    core_v2, floor_v2 = _imports()
    import calib_main as CM
    studentize, romano_wolf = CM.rw_funcs()
    ok = [r for r in load_parts() if r["status"] == "ok"]
    ok.sort(key=lambda r: r["resB"])
    pick = [ok[int(i)] for i in np.linspace(0, len(ok) - 1, RECALL_N).round().astype(int)]
    obs0 = np.array([r["dead0_score"] for r in ok])
    NUL = np.array([r["null_scores"] for r in ok], float)
    NUL = np.where(np.isfinite(NUL), NUL, np.nanmin(np.where(np.isfinite(NUL), NUL, np.nan), 1, keepdims=True))
    idx = {tag_of(r["cell"], r["side"]): i for i, r in enumerate(ok)}
    res = []
    for ci, rr in enumerate(pick):
        side = side_val(rr["side"])
        r, c, r2, c2, T = race_frame(core_v2, rr["cell"], side)
        cols = core_v2.search_cols(r2, side)
        P = core_v2.prep(r2, c2)
        _, rd = dead_score(core_v2, floor_v2, r2, c2, cols, P, 0)
        key = zlib.crc32(tag_of(rr["cell"], rr["side"]).encode())
        col, q = CM.choose_inj_col(rd, cols, rr["col"], ci, np.random.default_rng([SEED, 1, key]), core_v2)
        if col is None:
            continue
        region = rd[col].to_numpy(float) > q
        cm = core_v2.twin_mask(region, P["src_idx"])
        need = {B: 2.6 * core_v2.edge_se(rd, c2, P, region, cm, B, nboot=500)[1] for B in "AB"}
        for m in RECALL_M:
            dl = {B: min(m * need[B], 0.95) for B in "AB"}
            hit, got = CM.inject(rd.hit.to_numpy().copy(), region, rd.block.to_numpy(),
                                 floor_v2._as_day_int(rd.day), dl, "row", np.random.default_rng([SEED, 2, key, m]))
            ri = rd.copy(); ri["hit"] = hit
            b = core_v2.best_core(ri, c2, P, cols)
            sc = b["score"] if b is not None else -np.inf
            i = idx[tag_of(rr["cell"], rr["side"])]
            k = int((NUL[i] >= sc).sum())
            v = obs0.copy(); v[i] = sc
            t_obs, TT = studentize(v, NUL, loo=True)
            adj = romano_wolf(t_obs, TT)[i]
            res.append(dict(tag=tag_of(rr["cell"], rr["side"]), m=m, col=col, needA=need["A"], needB=need["B"],
                            gotA=got["A"], gotB=got["B"], found_col=b["col"] if b else "",
                            right_col=bool(b is not None and CM.is_right_col(b["col"], col)),
                            plain=k <= 4, rw=adj <= 0.05))
    t = pd.DataFrame(res)
    print("  planted race effect (1x / 2x / 4x NEED in BOTH blocks), row flips; NO pass mark")
    for m in RECALL_M:
        s = t[t.m == m]
        print(f"   {m}x NEED: n {len(s)} | plain k<=4 {100 * s.plain.mean():.0f}% | full Romano-Wolf "
              f"{100 * s.rw.mean():.0f}% | right column {100 * s.right_col.mean():.0f}% | "
              f"median planted B {100 * s.gotB.median():.2f} pp")
    out = os.path.join(OUTD, f"recall_{datetime.date.today().isoformat()}.csv")
    t.to_csv(out, index=False)
    print(f"  wrote {out}  SHA-256 {sha(out)}")
    return 0


# ------------------------------------------------------------------ self-test (toy, no data)
def selftest():
    core_v2, floor_v2 = _imports()
    import calib_main as CM
    studentize, romano_wolf = CM.rw_funcs()
    ok = [True]

    def chk(name, cond, extra=""):
        ok[0] &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")

    rng = np.random.default_rng(7)
    n = 6000
    days = np.sort(rng.integers(15400, 16400, n))           # block A days (2012-2014)
    days[n // 2:] = np.sort(rng.integers(19100, 19900, n - n // 2))
    ts = pd.to_datetime(days, unit="D", utc=True) + pd.Timedelta(hours=15)
    blk = np.where(np.arange(n) < n // 2, "A", "B")
    cov = rng.normal(size=n)
    noise = rng.normal(size=n)
    race = np.where(rng.random(n) < 0.5 + 0.12 * (cov > 1.0), 1, -1)
    race[rng.random(n) < 0.08] = 9                           # opens
    race[rng.random(n) < 0.01] = 0                           # ties
    r = pd.DataFrame(dict(is_real=1, side=1.0, block=blk, ts=ts, bar=np.arange(n) * 10, src_bar=-1,
                          T=60, covx=cov, noise=noise))
    r["day"] = days
    r["hit"] = 0.0
    tw_src = np.repeat(r.bar.to_numpy(), 2)
    m2 = len(tw_src)
    twin_race = np.where(rng.random(m2) < 0.5, 1, -1)
    twin_race[rng.random(m2) < 0.15] = 9
    c = pd.DataFrame(dict(is_real=0, side=1.0, block=np.repeat(blk, 2), ts=np.repeat(ts, 2), bar=-1,
                          src_bar=tw_src, T=60, covx=np.nan, noise=np.nan))
    c["day"] = np.repeat(days, 2)
    c["hit"] = 0.0
    x = pd.concat([pd.DataFrame(dict(is_real=1, side=1.0, bar=r.bar, src_bar=-1, race=race)),
                   pd.DataFrame(dict(is_real=0, side=1.0, bar=-1, src_bar=tw_src, race=twin_race))], ignore_index=True)
    r2, c2 = race_from(r, c, x, 1.0)
    chk("K1 setups kept = race in {1,-1}", len(r2) == int(np.isin(race, (1, -1)).sum()), f"{len(r2)} of {n}")
    chk("K2 hit = (race == 1) on setups", np.array_equal(r2.hit.to_numpy(), (race[np.isin(race, (1, -1))] == 1).astype(float)))
    keep_src = set(r2.bar)
    want_tw = int((np.isin(twin_race, (1, -1)) & np.isin(tw_src, list(keep_src))).sum())
    chk("K3 twins kept = own race resolved AND source setup resolved", len(c2) == want_tw, f"{len(c2)}")
    chk("K4 no column added (setups, twins)", list(r2.columns) == list(r.columns) and list(c2.columns) == list(c.columns))
    xbad = x.copy(); xbad.loc[0, "bar"] = 123456789
    try:
        race_from(r, c, xbad, 1.0); al = False
    except AssertionError:
        al = True
    chk("M1 a misaligned cell file is refused", al)
    chk("K5 gate: 3,000-setup toy blocks fail the 1,000 / 1,000 rule only if a block is short",
        gate_ok(r2, c2) == (min((r2.block == B).sum() for B in "AB") >= GATE and min((c2.block == B).sum() for B in "AB") >= GATE))
    chk("M2 gate refuses a 900-setup block", not gate_ok(r2.iloc[:900].assign(block="A"), c2))
    # planted race effect on cov > 1 is found by the unchanged search
    P = core_v2.prep(r2, c2)
    b = core_v2.best_core(r2, c2, P, ["covx", "noise"])
    chk("K6 the unchanged search finds the planted race effect's column", b is not None and b["col"] == "covx"
        and b["sense"] == "hi", f"{b['col'] if b else None} {b['sense'] if b else ''} eA {100 * b['eA']:+.1f} eB {100 * b['eB']:+.1f}")
    # offset liveness on the toy
    sh0 = floor_v2.shuffle_within_block(r2, ["covx", "noise"], rng, unit="day_strata", perm_index=0)
    sh1 = floor_v2.shuffle_within_block(r2, ["covx", "noise"], rng, unit="day_strata", perm_index=OFFSET)
    chk("K7 perm_index 2000 gives a different day_strata draw from 0", not np.array_equal(sh0.covx.to_numpy(), sh1.covx.to_numpy()))
    sh1b = floor_v2.shuffle_within_block(r2, ["covx", "noise"], np.random.default_rng(99), unit="day_strata", perm_index=OFFSET)
    chk("K8 the draw at a given index does not depend on the rng passed (JOINT)", np.array_equal(sh1.covx.to_numpy(), sh1b.covx.to_numpy()))
    # Romano-Wolf: a single-hypothesis family reduces to the plain permutation p
    nul = rng.normal(size=(1, 100)); ob = np.array([nul[0].max() + 0.1])
    t_o, TT = studentize(ob, nul, loo=False)
    chk("K9 RW on one hypothesis == plain p = 1/101 when obs beats every null", np.isclose(romano_wolf(t_o, TT)[0], 1 / 101))
    print("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED")
    return ok[0]


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--one" in a:
        cell, s = a[a.index("--one") + 1].split(":")
        sys.exit(job(cell, s))
    if not selftest():
        sys.exit(1)
    if "--selftest" in a:
        sys.exit(0)
    if "--check" in a:
        sys.exit(check())
    if "--run" in a:
        w = int(a[a.index("--workers") + 1]) if "--workers" in a else max(1, (os.cpu_count() or 4) - 3)
        part = a[a.index("--part") + 1] if "--part" in a else None
        sys.exit(run_pool(w, part))
    if "--summarize" in a:
        sys.exit(summarize())
    if "--recall" in a:
        sys.exit(recall())
    print(__doc__); sys.exit(2)
