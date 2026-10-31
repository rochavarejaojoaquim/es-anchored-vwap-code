r"""mono_profile.py -- the SHAPE of each Stage 1 survivor's covariate effect. DESCRIPTIVE ONLY.

    python mono_profile.py --selftest                 toy fixtures only, reads no dump
    python mono_profile.py --k0-only=3                pre-flight: known-number check on 3 rows
    python mono_profile.py                            the 31 Romano-Wolf survivors (primary)
    python mono_profile.py --set=195                  the 195 at p <= 0.05 (screening)
    options: --rw=FILE (default rw_check_independent_out.csv)   --out=FILE

WHY THIS AND NOT mono_sweep.py (HANDOFF 26.59, brain/13 C138; the author chose option A,
2026-10-02). mono_sweep's "monotone" rule (rho >= 0.7, same sign in both blocks) has an
exact chance rate of 2 x (14/120)^2 = 2.72% per test and no null, so over ~12,000 tests
it would label ~330 noise covariates "monotone". The covariate VERDICTS already exist,
with a proper null: Stage 1's floors and Romano-Wolf. What is missing is the SHAPE --
is a winning cut a gradient or a step (HANDOFF.md:617-627)? HANDOFF.md:339-342 already
requires per-quintile edges wherever a monotone claim is made. This script reports them.
NO VERDICT IS DRAWN FROM ANY NUMBER IT PRINTS.

THE LIST IS FIXED BY STAGE 1, NOT CHOSEN HERE. --set=31: rw_adj_p <= 0.05 (F02). --set=195:
the registered p <= 0.05 (F01). Both are read from rw_check_independent.py's output, and the
counts must be EXACTLY 31 / 195 (integer compare) or the script stops.

NOTHING STATISTICAL IS REIMPLEMENTED (E24). Each bin's edge and SE come from the INSTALLED
core_v2.edge_se on the bin's mask -- the same estimator, the same 5-day moving-block
bootstrap (nboot=500, seed=1) and the same minimum (MIN_N//3 = 100 setups and 100 twins per
block) as every Stage 1 score. Twins follow their SOURCE setup's bin (core_v2.twin_mask),
exactly as the winner's mask does. Quintile cut points are POOLED over blocks A and B
within the cell+side, as core_v2.search pools its cut candidates (core_v2.py:198; AUD-34).

KNOWN-NUMBER CHECK, per row, BEFORE that row's profile is computed (D3): the stored winner's
mask, rebuilt from cores_v2's (col, cut, sense) exactly as core_v2.best_core does
(:242-246), must reproduce the stored eA, eB, seA, seB, nA, nB EXACTLY. One mismatch stops
the run. A cross-check also requires the rw file's row to agree with the cores_v2 file.

CAVEATS PRINTED WITH THE OUTPUT. (1) These cells were SELECTED as the best of a search, so
their profiles are biased away from zero (winner's curse; Faraggi & Simon 1996 put
maximally-selected effects ~10% high). (2) The per-block Spearman rho is DESCRIPTIVE; under
noise P(rho >= 0.7) for 5 bins is exactly 14/120 per block. (3) Edges are signed: negative
means the VWAP fills LESS often than the matched random level in that bin.

WRITES ONE CSV (default mono_profile_set<31|195>_<YYYY-MM-DD>.csv) carrying the stamp,
code-set and this script's SHA-256 on every row (brain/13 C137's rule).
"""
import datetime
import hashlib
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
import core_v2                                                        # noqa: E402

NQ = 5
NBOOT = 500                     # best_core's precise re-estimate (core_v2.py:245-246)
FIELDS = ["eA", "eB", "seA", "seB", "nA", "nB"]
RHO_CHANCE = "14/120"           # exact P(rho >= 0.7 | 5 bins, noise), one block


# ------------------------------------------------------------------ the core
def winner_mask(r, row):
    """The stored winner's mask, exactly as core_v2.best_core builds it (:242-243)."""
    v = r[str(row["col"])].to_numpy(float)
    return v <= float(row["cut"]) if str(row["sense"]) == "lo" else v > float(row["cut"])


def recompute_winner(r, c, P, row):
    m = winner_mask(r, row)
    cm = core_v2.twin_mask(m, P["src_idx"])
    eA = core_v2.edge_se(r, c, P, m, cm, "A", nboot=NBOOT)
    eB = core_v2.edge_se(r, c, P, m, cm, "B", nboot=NBOOT)
    if eA is None or eB is None:
        return None
    return dict(eA=eA[0], eB=eB[0], seA=eA[1], seB=eB[1], nA=eA[2], nB=eB[2])


def agree(stored, now, keys=FIELDS, rel=0.0):
    """rel=0: exact, as verify_speedup.py and blocklen_sensitivity.py compare."""
    for k in keys:
        a, b = float(stored[k]), float(now[k])
        if rel == 0.0:
            if a != b:
                return False
        elif abs(a - b) > rel * max(abs(a), abs(b), 1e-300):
            return False
    return True


def bins_of(v):
    """Pooled quintile bins over the FINITE values. NaN -> -1, never a real bin.

    np.digitize(nan, q) returns len(q), i.e. it would drop every missing value into the
    TOP bin silently; the guard below is what the self-test's M1 exists to prove.
    Ties can collapse cut points; the bins are then fewer than 5 and that is reported.
    Bins are RIGHT-CLOSED, (q[k-1], q[k]], which is core_v2's own "lo" convention
    (v <= cut). A cut equal to the maximum would leave an empty top bin, so it is
    dropped (self-test K7 caught np.digitize leaving an empty bin under ties).
    """
    fin = np.isfinite(v)
    q = np.unique(np.nanquantile(v[fin], np.linspace(0, 1, NQ + 1)[1:-1]))
    q = q[q < v[fin].max()]
    b = np.full(len(v), -1, dtype=np.int64)
    b[fin] = np.searchsorted(q, v[fin], side="left")
    return b, q


def profile(r, c, P, col):
    """Per (bin, block): signed edge, its block-bootstrap SE, setup count, value range."""
    v = r[col].to_numpy(float)
    b, q = bins_of(v)
    nb = len(q) + 1
    out = []
    for k in range(nb):
        m = b == k
        cm = core_v2.twin_mask(m, P["src_idx"])
        vk = v[m]
        for B in ("A", "B"):
            e = core_v2.edge_se(r, c, P, m, cm, B, nboot=NBOOT)
            out.append(dict(bin=k + 1, nbins=nb, block=B,
                            v_lo=float(vk.min()) if len(vk) else np.nan,
                            v_hi=float(vk.max()) if len(vk) else np.nan,
                            edge=np.nan if e is None else e[0],
                            se=np.nan if e is None else e[1],
                            n=0 if e is None else int(e[2]),
                            gated=e is None))
    return out, int((b == -1).sum())


def rho_by_block(rows):
    res = {}
    for B in ("A", "B"):
        e = [x["edge"] for x in rows if x["block"] == B]
        res[B] = (float(spearmanr(np.arange(len(e)), e).statistic)
                  if len(e) >= 3 and all(np.isfinite(e)) else np.nan)
    return res


# ------------------------------------------------------------------ self-test
def toy(plant, n=40000, D=400, seed=7, nan_hits=0):
    """Setups and 2 twins each. Real hit = 0.45 + plant(x); twin hit = 0.45 flat.
    Blocks A (days 0..D-1) and B (days D..2D-1). Returns r, c."""
    rng = np.random.default_rng(seed)
    blk = np.where(np.arange(n) < n // 2, "A", "B")
    day = np.where(blk == "A", rng.integers(0, D, n), rng.integers(D, 2 * D, n))
    x = rng.random(n)
    hit = (rng.random(n) < 0.45 + plant(x)).astype(float)
    if nan_hits:                      # NaN covariate on rows that HIT
        idx = np.flatnonzero(hit == 1)[:nan_hits]
        x[idx] = np.nan
    r = pd.DataFrame(dict(bar=np.arange(n), block=blk, day=day, hit=hit, x=x))
    src = np.repeat(np.arange(n), 2)
    cblk = blk[src]
    cday = np.where(cblk == "A", rng.integers(0, D, 2 * n), rng.integers(D, 2 * D, 2 * n))
    c = pd.DataFrame(dict(src_bar=src, block=cblk, day=cday,
                          hit=(rng.random(2 * n) < 0.45).astype(float)))
    return r, c


def selftest():
    print("mono_profile.py SELF-TEST -- detectors before any dump is read.\n")
    res = []

    def chk(name, ok, detail=""):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<70} {detail}")

    def edges(rows, B):
        return np.array([x["edge"] for x in rows if x["block"] == B])

    def ses(rows, B):
        return np.array([x["se"] for x in rows if x["block"] == B])

    # K1/K2 a planted GRADIENT 0,2,4,6,8 pp across quintiles is recovered, in order
    grad = lambda x: 0.02 * np.minimum((x * 5).astype(int), 4)
    r, c = toy(grad); P = core_v2.prep(r, c)
    rows, nmiss = profile(r, c, P, "x")
    want = np.array([0, .02, .04, .06, .08])
    zA = np.abs(edges(rows, "A") - want) / ses(rows, "A")
    zB = np.abs(edges(rows, "B") - want) / ses(rows, "B")
    chk("K1 planted gradient recovered in every bin, both blocks (|z| < 3.5)",
        (zA < 3.5).all() and (zB < 3.5).all(),
        "A " + " ".join(f"{e*100:+.2f}" for e in edges(rows, "A")) + " pp")
    rho = rho_by_block(rows)
    chk("K2 and its descriptive rho is high in both blocks (>= 0.9)",
        rho["A"] >= 0.9 and rho["B"] >= 0.9, f"rhoA {rho['A']:.2f} rhoB {rho['B']:.2f}")

    # K3 a planted STEP (+8 pp below the 40th percentile, 0 above) is a step, not a slope
    step = lambda x: np.where(x < 0.4, 0.08, 0.0)
    r, c = toy(step, seed=8); P = core_v2.prep(r, c)
    rows, _ = profile(r, c, P, "x")
    want = np.array([.08, .08, 0, 0, 0])
    ok = all((np.abs(edges(rows, B) - want) / ses(rows, B) < 3.5).all() for B in "AB")
    chk("K3 planted step recovered: bins 1-2 ~ +8 pp, bins 3-5 ~ 0",
        ok, "B " + " ".join(f"{e*100:+.2f}" for e in edges(rows, "B")) + " pp")

    # K4 no effect -> every bin consistent with 0
    r, c = toy(lambda x: 0.0 * x, seed=9); P = core_v2.prep(r, c)
    rows, _ = profile(r, c, P, "x")
    z = np.concatenate([np.abs(edges(rows, B)) / ses(rows, B) for B in "AB"])
    chk("K4 no planted effect -> every bin |z| < 3.5", (z < 3.5).all(), f"max |z| {z.max():.2f}")

    # K5 bins partition the finite rows exactly; counts sum to the block totals
    nsum = {B: sum(x["n"] for x in rows if x["block"] == B) for B in "AB"}
    chk("K5 bin counts sum to each block's setups (partition, nothing dropped)",
        nsum["A"] == int((r.block == "A").sum()) and nsum["B"] == int((r.block == "B").sum()),
        str(nsum))

    # K6 the per-bin SE is core_v2's block bootstrap: equal to a direct edge_se call
    m = (bins_of(r.x.to_numpy(float))[0] == 0)
    e = core_v2.edge_se(r, c, P, m, core_v2.twin_mask(m, P["src_idx"]), "A", nboot=NBOOT)
    b1 = [x for x in rows if x["bin"] == 1 and x["block"] == "A"][0]
    chk("K6 bin 1 edge and SE are EXACTLY core_v2.edge_se on that mask",
        e[0] == b1["edge"] and e[1] == b1["se"], f"{e[0]:.6f} / {e[1]:.6f}")

    # K7 ties: a covariate with 3 distinct values gives 3 bins, still a partition
    r3 = r.copy(); r3["x"] = np.floor(r3.x * 3)
    rows3, _ = profile(r3, c, P, "x")
    chk("K7 a 3-valued covariate yields 3 non-empty bins (ties reported, not hidden)",
        rows3[0]["nbins"] == 3 and sum(x["n"] for x in rows3) == len(r3)
        and not any(x["gated"] for x in rows3), f"nbins {rows3[0]['nbins']}")
    # K7b and the bins are right-closed like core_v2's "lo" mask: bin 1 == (v <= q[0])
    v = r.x.to_numpy(float); bb, qq = bins_of(v)
    chk("K7b bin 1 is exactly core_v2's 'lo' mask at the first cut (v <= q1)",
        np.array_equal(bb == 0, v <= qq[0]))

    # K8 the stored-winner known-number check: recompute_winner == direct edge_se
    row = dict(col="x", cut=float(np.quantile(r.x, 0.4)), sense="lo")
    got = recompute_winner(r, c, P, row)
    mm = winner_mask(r, row); cmm = core_v2.twin_mask(mm, P["src_idx"])
    dA = core_v2.edge_se(r, c, P, mm, cmm, "A", nboot=NBOOT)
    chk("K8 recompute_winner rebuilds the mask exactly as best_core (eA, seA, nA)",
        (got["eA"], got["seA"], got["nA"]) == dA, f"eA {got['eA']:+.6f}")

    # --- mutations: each must be CAUGHT ------------------------------------
    # M1 NaN covariates on rows that hit: the guard keeps them OUT of the top bin.
    r4, c4 = toy(lambda x: 0.0 * x, seed=10, nan_hits=3000); P4 = core_v2.prep(r4, c4)
    rows4, nmiss4 = profile(r4, c4, P4, "x")
    v = r4.x.to_numpy(float)
    naive = np.digitize(v, np.unique(np.nanquantile(v, [.2, .4, .6, .8])))
    top_naive = r4.hit.to_numpy()[naive == 4].mean()
    top_guard = r4.hit.to_numpy()[bins_of(v)[0] == 4].mean()
    chk("M1 naive np.digitize puts NaN rows in the TOP bin; the guard does not",
        nmiss4 == 3000 and top_naive - top_guard > 0.03,
        f"missing {nmiss4}; top-bin hit naive {top_naive:.3f} vs guarded {top_guard:.3f}")
    # M2 the comparator catches a 1-ulp change and passes an exact match
    s = dict(eA=0.1, eB=0.1, seA=0.01, seB=0.01, nA=100, nB=100)
    t = dict(s); t["seA"] = np.nextafter(0.01, 1)
    chk("M2 exact comparator flags a 1-ulp change in seA, passes an exact match",
        (not agree(s, t)) and agree(s, dict(s)))
    # M3 a wrong mask (sense flipped) must FAIL the known-number check
    bad = recompute_winner(r, c, P, dict(row, sense="hi"))
    chk("M3 flipping the stored sense breaks the known-number check", not agree(got, bad))

    # M4 K7 is not vacuous: the old np.digitize binning leaves an EMPTY bin under ties
    v3 = r3.x.to_numpy(float)
    old = np.digitize(v3, np.unique(np.nanquantile(v3, [.2, .4, .6, .8])))
    empties = sum(int((old == k).sum() == 0) for k in range(old.max() + 1))
    chk("M4 the superseded np.digitize binning leaves an empty bin on the same data",
        empties >= 1, f"{empties} empty bin(s)")

    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


# ------------------------------------------------------------------ the run
def sha256(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def code_set():
    import gate_receipts                     # the ONE code-set definition (pseudo_vwap.code_set)
    return gate_receipts.code_set()


def side_of(x):
    return None if pd.isna(x) else float(x)


def main():
    if "--selftest" in sys.argv:
        return 0 if selftest() else 1
    opt = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if "=" in a}
    which = opt.get("--set", "31")
    if which not in ("31", "195"):
        raise SystemExit("--set must be 31 or 195")
    rwf = opt.get("--rw", os.path.join(HERE, "rw_check_independent_out.csv"))
    k0_only = int(opt["--k0-only"]) if "--k0-only" in opt else None

    if not selftest():
        raise SystemExit("\n  SELF-TEST FAILED -- no data was read.")
    if not os.path.exists(rwf):
        raise SystemExit(f"\n  {rwf} not found. Regenerate it first:\n"
                         f"    python rw_check_independent.py results_table.csv")

    R = pd.read_csv(rwf)
    n31 = int((R.rw_adj_p <= 0.05).sum()); n195 = int((R.p <= 0.05).sum())
    print(f"\nLIST CHECK against brain/12 F01/F02 (integer counts), {os.path.basename(rwf)}:")
    print(f"  [{'PASS' if len(R) == 317 else 'FAIL'}] 317 effective rows          {len(R)}")
    print(f"  [{'PASS' if n31 == 31 else 'FAIL'}] rw_adj_p <= 0.05 -> 31       {n31}")
    print(f"  [{'PASS' if n195 == 195 else 'FAIL'}] p <= 0.05 -> 195             {n195}")
    if len(R) != 317 or n31 != 31 or n195 != 195:
        raise SystemExit("  LIST CHECK FAILED -- the input is not Stage 1's registered table.")
    L = R[R.rw_adj_p <= 0.05] if which == "31" else R[R.p <= 0.05]
    L = L.sort_values(["cell", "side"], na_position="first").reset_index(drop=True)
    if k0_only:
        L = L.head(k0_only)

    cs, me = code_set(), sha256(os.path.abspath(__file__))
    stamp = (f"mono_profile | set {which} ({len(L)} cell+sides, list fixed by Stage 1) | "
             f"quintiles pooled over blocks within cell+side | edge & SE = core_v2.edge_se, "
             f"5-day moving block, nboot {NBOOT}, seed 1, min 100 setups+twins per bin-block | "
             f"tau {core_v2.TAU} | code-set {cs} | mono_profile.py {me[:16]} | "
             f"{platform.node()} | ES, train blocks A+B, sealed 2018-2022 untouched | "
             f"raw, no null, DESCRIPTIVE ONLY")
    print("\nSTAMP: " + stamp)
    print("\nKNOWN-NUMBER CHECK FIRST on every row; one mismatch stops the run.\n")

    t0, out = time.time(), []
    for i, row in L.iterrows():
        cell, side = str(row.cell), side_of(row.side)
        f = os.path.join(core_v2.OUT, f"{cell}_side{side}.parquet")
        st = pd.read_parquet(f).iloc[0]
        if str(st["col"]) != str(row.col) or not agree(st, row, rel=1e-12):
            print(f"  [FAIL] {cell} side {side}: rw file and {os.path.basename(f)} disagree")
            return 1
        r, c, T = core_v2.load_cell(cell, side)
        P = core_v2.prep(r, c)
        got = recompute_winner(r, c, P, st)
        ok = got is not None and agree(st, got)
        tag = f"{cell} side {side} ({st['col']} {st['sense']} @{100*st['pct']:.0f}%)"
        print(f"  [{'PASS' if ok else 'FAIL'}] K0 stored winner reproduced exactly  {tag}"
              f"  {time.time() - t0:.0f}s", flush=True)
        if not ok:
            print("        " + ("edge_se returned None" if got is None else
                               "; ".join(f"{k}: stored {st[k]} vs now {got[k]}" for k in FIELDS
                                         if float(st[k]) != float(got[k]))))
            print("  A KNOWN-NUMBER MISMATCH IS A STOP CONDITION. Nothing was written.")
            return 1
        if k0_only:
            continue
        rows, nmiss = profile(r, c, P, str(st["col"]))
        rho = rho_by_block(rows)
        for x in rows:
            out.append(dict(set=which, cell=cell, side=side, col=str(st["col"]),
                            win_sense=str(st["sense"]), win_pct=float(st["pct"]),
                            win_cut=float(st["cut"]), win_eA=float(st["eA"]),
                            win_eB=float(st["eB"]), rw_adj_p=float(row.rw_adj_p),
                            p=float(row.p), n_missing=nmiss, rhoA=rho["A"], rhoB=rho["B"],
                            **x, code_set=cs, producer_sha256=me, stamp=stamp))
        for B in ("A", "B"):
            e = [x for x in rows if x["block"] == B]
            print(f"        {B}: " + "  ".join(
                ("   gated" if x["gated"] else f"{100*x['edge']:+6.2f}({100*x['se']:.2f})")
                for x in e) + f"   rho{B} {rho[B]:+.2f}")
    if k0_only:
        print(f"\n  --k0-only: {len(L)} of {len(L)} reproduce exactly, {time.time() - t0:.0f}s. "
              "Nothing written.")
        return 0

    df = pd.DataFrame(out)
    path = opt.get("--out", os.path.join(
        HERE, f"mono_profile_set{which}_{datetime.date.today().isoformat()}.csv"))
    df.to_csv(path, index=False)
    print(f"\n  {len(L)} of {len(L)} known-number checks PASS. Wrote {os.path.basename(path)} "
          f"({len(df)} rows). {time.time() - t0:.0f}s")
    print("  Edges in pp (SE); signed: negative = the VWAP fills LESS than the matched level.")
    print("  DESCRIPTIVE ONLY. Selected cells: magnitudes biased away from zero (winner's curse,")
    print(f"  Faraggi & Simon 1996, ~10%). Under noise P(rho >= 0.7) = {RHO_CHANCE} per block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
