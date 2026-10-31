r"""
baseline_check.py, VERSION 2 (2026-10-06) -- how big is the UNCONDITIONAL baseline inside the stored edges? Reviewer 26.109 objection 1; extended per the final A13 read (26.112, E-2, E-7).
Version 1 (M1b only, complement = ~mask) is superseded and was never run on real data.

THE POINT. core_v2's edge is rh.mean() - ch.mean() = the fill rate of the SUBSET's setups minus the fill rate of THEIR twins. It contains the unconditional claim-1 gap even for a filter that adds nothing.
B1 (A8) and the A13 list average sign x edge across groups; if the signs are not balanced the baseline does not cancel. This script measures, on TRAIN blocks A and B, for each of the 64 Stage-1 group
REPRESENTATIVES (a13_make_rlist.build: best Stage-1 t_obs, then smaller N; the stored winning filter), for BOTH twin schemes (M1b from the main file, M2 from the .m2.parquet, each twin carried by its
source setup, block of its source setup):
    unfiltered   all setups of the cell+side vs their twins
    subset S     the stored winner's subset, FINITE covariate only (equals the stored eA, eB: K0, for M1b)
    complement C the other setups of the cell+side with a FINITE covariate (NaN-covariate setups are in neither S nor C; their count is reported)
    conditional  gap(S) - gap(C)
and averages  sign x (each)  over (i) the 31's groups (the 18 groups with a member among the Romano-Wolf survivors), (ii) the A13 HEADLINE groups (the R-list, a13_make_rlist), per block and scheme.
THE REOPEN TRIGGER (A13.7, fixed BEFORE this is run): option (ii) (the conditional statistic as a co-criterion for B1) is REOPENED iff, on block B, in EITHER scheme,
    mean over the 31's groups of sign x (unfiltered edge)  >=  0.5 x  mean over the 31's groups of sign x (subset edge).   (a convention)
Results may change ONLY this trigger; they may not change any population, statistic or definition in A13.
READS: cell-level and stored-winner edges only; NO dial setting, bracket, tick, seal bar or bootstrap.

    python baseline_check.py --selftest
    python baseline_check.py        the real run (about 60 cells, single process, ESTIMATE 4-8 min with M2, NOT MEASURED)
"""
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
sys.path.insert(0, HERE)
SCHEMES = ("M1b", "M2")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest().upper()


def gap(r_block, r_hit, r_mask, tw_block, tw_hit, tw_src, mask, B):
    """Fill rate of the setups in `mask` minus the fill rate of their twins, block B. Twins are carried by their source setup (tw_src = index into the setups, -1 = not in this side)."""
    rs = (r_block == B) & r_mask & mask
    keep = tw_src >= 0
    ts = np.zeros(len(tw_src), bool)
    ts[keep] = (mask & r_mask)[tw_src[keep]]
    ts &= (tw_block == B)
    if not rs.any() or not ts.any():
        return np.nan, int(rs.sum()), int(ts.sum())
    return float(r_hit[rs].mean() - tw_hit[ts].mean()), int(rs.sum()), int(ts.sum())


def blocks(r_block, r_hit, tw, mask, fin):
    """{block: dict(unf, sub, comp, cond, n_S, n_C, t_S, t_C)} for one scheme. S = mask & finite, C = ~mask & finite, unfiltered = everything."""
    allm = np.ones(len(r_hit), bool)
    out = {}
    for B in ("A", "B"):
        unf, _, _ = gap(r_block, r_hit, allm, tw["block"], tw["hit"], tw["src"], allm, B)
        sub, nS, tS = gap(r_block, r_hit, fin, tw["block"], tw["hit"], tw["src"], mask & fin, B)
        comp, nC, tC = gap(r_block, r_hit, fin, tw["block"], tw["hit"], tw["src"], (~mask) & fin, B)
        out[B] = dict(unf=unf, sub=sub, comp=comp, cond=sub - comp, nS=nS, nC=nC, tS=tS, tC=tC)
    return out


def trigger(df, say=print):
    """The pre-stated reopen trigger on the 31's groups, block B, either scheme."""
    g = df[df.in31]
    res = {}
    for m in SCHEMES:
        unf = float(np.nanmean(g.sgn * g[f"{m}_unf_B"]))
        sub = float(np.nanmean(g.sgn * g[f"{m}_sub_B"]))
        res[m] = (unf, sub, unf >= 0.5 * sub)
        say(f"    {m}: mean sign x unfiltered {unf * 100:+.2f} pp | mean sign x subset (the stored edge) {sub * 100:+.2f} pp | unfiltered >= 0.5 x subset: {res[m][2]}")
    return any(v[2] for v in res.values()), res


def summarize(df, say=print):
    pops = (("the 31's groups (18)", df[df.in31]), ("the A13 HEADLINE groups (the R-list)", df[df.headline]), ("all 64 representatives", df))
    for name, g in pops:
        say(f"\n  {name}: n {len(g)} | positive {int((g.sgn > 0).sum())} / negative {int((g.sgn < 0).sum())} | mean(sign) {g.sgn.mean():+.3f}")
        for m in SCHEMES:
            for B in ("A", "B"):
                row = {k: float(np.nanmean(g.sgn * g[f"{m}_{k}_{B}"])) * 100 for k in ("unf", "sub", "comp", "cond")}
                say(f"    {m} block {B}: mean sign x [unfiltered {row['unf']:+.2f} | subset {row['sub']:+.2f} | complement {row['comp']:+.2f} | CONDITIONAL {row['cond']:+.2f}] pp")
    say("\n  THE PRE-STATED REOPEN TRIGGER (A13.7), the 31's groups, block B:")
    fire, _ = trigger(df, say)
    say(f"  >>> option (ii) {'IS REOPENED' if fire else 'is NOT reopened'} by the trigger as fixed before this run (a convention).")
    return fire


def _toy(n=300000, base=0.10, extra=0.0, nan_extra=0.0, seed=2):
    rg = np.random.default_rng(seed)
    blk = np.where(np.arange(n) < n // 2, "A", "B")
    x = rg.random(n)
    nanrow = rg.random(n) < 0.1
    hit = (rg.random(n) < 0.45 + base + extra * (x > 0.5) + nan_extra * nanrow).astype(float)
    x = np.where(nanrow, np.nan, x)          # a NaN covariate is False for BOTH senses (v <= cut, v > cut), so those setups fall into ~mask
    tws = {}
    for i, m in enumerate(SCHEMES):
        tw_hit = (rg.random(n) < 0.45).astype(float)
        tws[m] = dict(block=blk.copy(), hit=tw_hit, src=np.arange(n))
    return blk, x, nanrow, hit, tws


def selftest():
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else ""), flush=True)
        ok[0] &= bool(cond)
    print("\nbaseline_check.py v2 SELF-TEST -- toy fixtures only.\n")
    blk, x, nanrow, hit, tws = _toy(extra=0.06)
    fin = ~nanrow
    mask = x > 0.5
    res = blocks(blk, hit, tws["M1b"], mask, fin)
    chk("K1 conditional edge recovers the planted 6 pp filter and NOT the 10 pp baseline (tolerance 1.5 pp, n = 300k)", all(abs(res[B]["cond"] - 0.06) < 0.015 for B in "AB"), f"A {res['A']['cond']:.3f} B {res['B']['cond']:.3f}")
    chk("K2 the SUBSET edge contains the baseline (about 16 pp)", all(abs(res[B]["sub"] - 0.16) < 0.015 for B in "AB"), f"A {res['A']['sub']:.3f}")
    blk, x, nanrow, hit, tws = _toy(extra=0.0)
    fin = ~nanrow
    r0 = blocks(blk, hit, tws["M1b"], x > 0.5, fin)
    chk("M1 a filter that adds NOTHING shows the baseline in the subset edge (about 10 pp) and a conditional edge within 1.0 pp of 0 (about 3 SD)",
        all(abs(r0[B]["sub"] - 0.10) < 0.015 and abs(r0[B]["cond"]) < 0.01 for B in "AB"), f"sub {r0['A']['sub']:.3f} cond {r0['A']['cond']:+.4f}")
    # M2: the NaN rows have a higher base fill (e.g. warm-up setups): including them in C would bias the conditional edge; the finite-only C does not
    blk, x, nanrow, hit, tws = _toy(extra=0.0, nan_extra=0.15, seed=5)
    fin = ~nanrow
    rf = blocks(blk, hit, tws["M1b"], x > 0.5, fin)
    allc = blocks(blk, hit, tws["M1b"], x > 0.5, np.ones(len(x), bool))
    chk("M2 NaN-covariate setups in C bias the conditional edge (|bias| > 1.0 pp); the finite-only definition does not (|cond| < 1.0 pp)", abs(allc["A"]["cond"]) > 0.01 and abs(rf["A"]["cond"]) < 0.01,
        f"with NaN in C {allc['A']['cond']:+.4f} | finite only {rf['A']['cond']:+.4f}")
    # M3: both schemes are handled (M2 twins have their own hit column)
    r2 = blocks(blk, hit, tws["M2"], x > 0.5, fin)
    chk("M3 the second scheme runs on its own twin rows", np.isfinite(r2["A"]["sub"]) and r2["A"]["sub"] != rf["A"]["sub"], f"M1b {rf['A']['sub']:.4f} M2 {r2['A']['sub']:.4f}")
    # M4: the sign-mix leak and the trigger
    rows = []
    rg = np.random.default_rng(7)
    for i in range(18):
        sg = 1.0 if i < 12 else -1.0
        d = dict(in31=True, headline=False, sgn=sg)
        for m in SCHEMES:
            d[f"{m}_unf_B"] = 0.04 + rg.normal(0, 0.003)          # a 4 pp baseline in every group
            d[f"{m}_sub_B"] = 0.04 + sg * 0.0 + rg.normal(0, 0.003)  # a filter that adds nothing: the subset edge IS the baseline
        rows.append(d)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fire, _ = trigger(pd.DataFrame(rows), say=print)
    chk("M4 with a 12/6 sign mix, a 4 pp baseline and NO filter effect, mean(sign x subset) is about the baseline x 0.33 and the trigger FIRES (unfiltered >= 0.5 x subset)", fire,
        buf.getvalue().strip().splitlines()[0][:100])
    rows2 = []
    for i in range(18):
        d = dict(in31=True, headline=False, sgn=1.0)
        for m in SCHEMES:
            d[f"{m}_unf_B"] = 0.01
            d[f"{m}_sub_B"] = 0.09                                   # a big filter effect on a small baseline
        rows2.append(d)
    with contextlib.redirect_stdout(io.StringIO()):
        fire2, _ = trigger(pd.DataFrame(rows2), say=print)
    chk("M5 a small baseline under a large filter effect does NOT fire the trigger", not fire2)
    print("\n" + ("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED"))
    return ok[0]


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        raise SystemExit("self-test failed")
    import core_v2 as C
    import gate_receipts as G
    import a13_make_rlist as RL
    t = RL.load()
    s, reps, all_kept, rem, head = RL.build(t)
    hkeys = set(map(tuple, head[RL.KEY].to_numpy()))
    reps["headline"] = [tuple(x) in hkeys for x in reps[RL.KEY].to_numpy()]
    rows, t0 = [], time.time()
    for _, row in reps.iterrows():
        cell = row["cell"]
        side = None if row["side_s"] == "None" else float(row["side_s"])
        r, c, T = C.load_cell(cell, side)
        si = C.twin_index(r, c)
        st = pd.read_parquet(os.path.join(HERE, "cores_v2", f"{cell}_side{row['side_s']}.parquet")).iloc[0]
        v = r[st["col"]].to_numpy(float)
        fin = np.isfinite(v)
        m = (v <= float(st["cut"])) if st["sense"] == "lo" else (v > float(st["cut"]))
        rb, rh = r.block.to_numpy(), r.hit.to_numpy()
        tws = {"M1b": dict(block=c.block.to_numpy(), hit=c.hit.to_numpy(), src=si)}
        pos = {b: i for i, b in enumerate(r.bar.to_numpy())}
        m2 = pd.read_parquet(os.path.join(C.DUMP, cell + ".m2.parquet"), columns=["src_bar", "w_0.0"])
        s2 = np.array([pos.get(b, -1) for b in m2.src_bar.to_numpy()], dtype=np.int64)
        w = m2["w_0.0"].to_numpy(float)
        hit2 = ((w > 0) & (w <= T)).astype(float)
        blk2 = np.where(s2 >= 0, rb[np.maximum(s2, 0)], "")
        tws["M2"] = dict(block=blk2, hit=hit2, src=s2)
        rec = dict(cell=cell, side_s=row["side_s"], N=int(row["N"]), anchor=row["anchor"], trig=row["trig"], eps=row["eps"], in31=bool(row["in31"]), headline=bool(row["headline"]),
                   sgn=float(np.sign(st["eA"])), n_nan=int((~fin).sum()), n_setups=int(len(r)))
        for mm in SCHEMES:
            res = blocks(rb, rh, tws[mm], m, fin)
            for B in "AB":
                for k in ("unf", "sub", "comp", "cond", "nS", "nC", "tS", "tC"):
                    rec[f"{mm}_{k}_{B}"] = res[B][k]
        # K0: the M1b subset edge equals the stored eA, eB (S = mask & finite = mask for lo/hi senses)
        rec["k0"] = bool(abs(rec["M1b_sub_A"] - st["eA"]) <= 1e-12 and abs(rec["M1b_sub_B"] - st["eB"]) <= 1e-12)
        rows.append(rec)
        print(f"  {len(rows)}/{len(reps)} {cell} side {row['side_s']} K0 {'PASS' if rec['k0'] else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print(f"\n  [{'PASS' if df.k0.all() else 'FAIL'}] K0 the stored winner's M1b subset edge reproduces the stored eA and eB (1e-12) on {int(df.k0.sum())} of {len(df)} representatives")
    if not df.k0.all():
        raise SystemExit("K0 failed -- nothing summarised")
    print(f"  NaN-covariate setups per representative: median {df.n_nan.median():.0f}, max {int(df.n_nan.max())} of {int(df.n_setups.median())} (median group size); min rows in S/C/twins (train, any block, any scheme): "
          f"{int(min(df[[c for c in df.columns if c.split('_')[-2:-1] and c.split('_')[1] in ('nS', 'nC', 'tS', 'tC')]].min()))}")
    fire = summarize(df)
    df["code_set"] = G.code_set()
    df["script_sha256"] = sha(__file__)
    out = os.path.join(HERE, f"baseline_check_{time.strftime('%Y-%m-%d')}.csv")
    df.to_csv(out, index=False)
    print(f"\n  wrote {os.path.basename(out)} | stamp: 64 Stage-1 group representatives of the 195 (a13_make_rlist.build; pooled side = the string None) | stored winning filter | S = subset with FINITE covariate, C = the other finite-covariate setups |"
          f" gap(S), gap(C), unfiltered vs M1b and M2 twins, point estimates, train blocks A and B | code-set {df.code_set.iloc[0]} | ES train A+B, sealed untouched | raw, no cost | NO dial setting read | trigger fired: {fire}")


if __name__ == "__main__":
    main()
