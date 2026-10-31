r"""
a_only_shrinkage.py -- amendment A2's PRE-FREEZE MEASUREMENT (HANDOFF_section_28_AMENDMENTS_DRAFT.md A2; reviewer D4, 26.94).
STAGE-1 FILL-RATE NUMBERS ONLY: no dial setting, no bracket, no tick, no seal bar. Root file; rotates nothing.

QUESTION. Section 28 fixes the planning effect at 0.75 x the training edge (a 25% decay CONVENTION, McLean-Pontiff, search summary). How much does a
fill-rate edge actually shrink from block A to block B when the winner is chosen WITHOUT using block B? The 31 survivors cannot answer it: they were chosen on
min over BOTH blocks, which forces B/A toward 1. So: for each of the 317 effective cell+sides, run core_v2's own cutpoint search but score block A ALONE,
|eA| - 2.6 seA (the same pessimistic form), keep the best (col, cut, sense), then read its edge on block B. The TOP 31 by that A-only score (a CONVENTION: the
survivors' count, no threshold) are summarised; all 317 are reported beside them.

WHAT IT IS NOT. An era confound remains (B is later than A, the seal lies between them); cells overlap across N (24-74% shared setups), so rows are not
independent; the winner's curse in A is exactly what is being measured, and it is a measurement of THIS selection rule, not of a strategy.

    python a_only_shrinkage.py --selftest
    python a_only_shrinkage.py --k0 3                    the stored winners' eA/seA reproduce (3 smallest cells), nothing else
    python a_only_shrinkage.py --part 1/8 [--out DIR]    one share; run 8 terminals
    python a_only_shrinkage.py --summarize [--out DIR]   the aggregate over all shares
"""
import glob
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)
RW_CSV = os.path.join(HERE, "rw_check_independent_out.csv")
RW_SHA = "44E9D780FE0C8C6C1C2812340887CFDBF41DDB618A3BE4F99C0D4BC47DF4B0B0"
OUTDIR = os.path.join(HERE, "a_only_parts")
TOPK = 31


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def tag_frame(check_sha=True):
    """The 317 effective cell+sides, pooled side as the STRING 'None' (pandas 3 would drop it)."""
    if check_sha and sha(RW_CSV) != RW_SHA:
        raise SystemExit("rw_check_independent_out.csv differs from the frozen hash")
    t = pd.read_csv(RW_CSV)
    t["side_s"] = t["side"].astype(object).where(t["side"].notna(), "None").astype(str)
    eff = t[~t["dup_session"].astype(bool)].copy()
    if len(eff) != 317:
        raise SystemExit(f"{len(eff)} effective rows, expected 317")
    eff["tag"] = eff["cell"].astype(str) + "_side" + eff["side_s"]
    eff["w"] = eff["nA"] + eff["nB"]
    return eff.sort_values(["w", "tag"], ascending=[False, True]).reset_index(drop=True)


def part_of(df, part):
    if not part:
        return df
    try:
        which, k = part.split("/")
        want, k = {int(x) for x in which.split(",")}, int(k)
    except Exception:
        raise SystemExit(f"bad --part {part}")
    return df[[(i % k) + 1 in want for i in range(len(df))]]


# ---------------------------------------------------------------- the search
def edge_pack(C, r, c, P, m, B, nboot):
    cm = C.twin_mask(m, P["src_idx"])
    return C.edge_se(r, c, P, m, cm, B, nboot=nboot)


def mask_of(r, col, cut, sense):
    v = r[col].to_numpy(float)
    return v <= cut if sense == "lo" else v > cut


def search_A(C, r, c, P, cols, nboot=150):
    """core_v2.search with ONE change: the score is block A alone, |eA| - Z99*seA. Same candidates, same feasibility gates."""
    best = None
    for col in cols:
        v = r[col].to_numpy(float)
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            continue
        qs = np.nanquantile(v, np.linspace(C.QLO, C.QHI, C.NQ))
        for q in np.unique(qs):
            for sense in ("lo", "hi"):
                m = v <= q if sense == "lo" else v > q
                if m.sum() < C.MIN_N or (~m).sum() < C.MIN_N:
                    continue
                eA = edge_pack(C, r, c, P, m, "A", nboot)
                if eA is None:
                    continue
                sc = abs(eA[0]) - C.Z99 * eA[1]
                if best is None or sc > best["score"]:
                    best = dict(col=col, cut=float(q), sense=sense, score=sc)
    return best


def best_A(C, r, c, P, cols):
    """Rank with the cheap bootstrap, then re-estimate the winner with the precise one on A AND read it on B (as best_core does)."""
    b = search_A(C, r, c, P, cols, nboot=150)
    if b is None:
        return None
    m = mask_of(r, b["col"], b["cut"], b["sense"])
    eA = edge_pack(C, r, c, P, m, "A", 500)
    eB = edge_pack(C, r, c, P, m, "B", 500)
    if eA is None:
        return None
    b.update(eA=eA[0], seA=eA[1], nA=eA[2], scoreA=abs(eA[0]) - C.Z99 * eA[1],
             eB=(eB[0] if eB else np.nan), seB=(eB[1] if eB else np.nan), nB=(eB[2] if eB else 0))
    b["ratio"] = (np.sign(eA[0]) * b["eB"] / abs(eA[0])) if eB and eA[0] != 0 else np.nan
    return b


def one_tag(C, tag):
    cell, s = tag.rsplit("_side", 1)
    side = None if s == "None" else float(s)
    r, c, T = C.load_cell(cell, side)
    if min((r.block == B).sum() for B in "AB") < 1000 or min((c.block == B).sum() for B in "AB") < 1000:
        return None
    P = C.prep(r, c)
    return best_A(C, r, c, P, C.search_cols(r, side))


# ------------------------------------------------------------------- summary
def summarize(df, topk=TOPK, say=print):
    ok = df[df["status"] == "ok"].copy()
    top = ok.sort_values("scoreA", ascending=False).head(topk)

    def blk(x, name):
        r = x["ratio"].to_numpy(float)
        r = r[np.isfinite(r)]
        pooled = float((np.sign(x["eA"]) * x["eB"]).sum() / x["eA"].abs().sum())
        say(f"  {name:<34} n {len(x):>3} | signed B/A: median {np.median(r):+.3f}  mean {r.mean():+.3f}  p25 {np.quantile(r, .25):+.3f}  p75 {np.quantile(r, .75):+.3f}"
            f" | same sign in B {int((r > 0).sum())}/{len(r)} | pooled sum(sign*eB)/sum|eA| {pooled:+.3f}")
        return pooled
    say(f"\n  rows ok {len(ok)} of {len(df)}")
    p_top = blk(top, f"TOP {topk} by A-only score")
    p_all = blk(ok, "ALL effective cell+sides")
    rest = ok.drop(top.index)
    blk(rest, f"the other {len(rest)}")
    say(f"  A-only score: top-{topk} median {top['scoreA'].median()*100:.2f} pp, min {top['scoreA'].min()*100:.2f} pp | mean eA top {top['eA'].abs().mean()*100:.2f} pp, mean sign*eB {np.mean(np.sign(top['eA'])*top['eB'])*100:.2f} pp")
    return p_top, p_all


# ------------------------------------------------------------------ self-test
def _toy(effA, effB, seed=3, n=9000, D=300):
    """A core_v2-shaped frame: blocks A and B, a covariate x with a planted hit-rate effect inside {x > 0.5}, ONE twin per setup."""
    rg = np.random.default_rng(seed)
    blk = np.where(np.arange(n) < n // 2, "A", "B")
    day = np.where(blk == "A", rg.integers(0, D, n), 10_000 + rg.integers(0, D, n))
    x = rg.random(n)
    eff = np.where(blk == "A", effA, effB)
    hit = (rg.random(n) < 0.45 + eff * (x > 0.5)).astype(float)
    junk = rg.random(n)
    r = pd.DataFrame(dict(bar=np.arange(n), block=blk, day=day, hit=hit, x=x, junk=junk))
    c = pd.DataFrame(dict(src_bar=np.arange(n), block=blk, day=day, hit=(rg.random(n) < 0.45).astype(float)))
    return r, c


def selftest():
    import core_v2 as C
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else ""), flush=True)
        ok[0] &= bool(cond)
    print("\na_only_shrinkage.py SELF-TEST -- toy fixtures only, no dump.\n")
    res = {}
    for name, a, b in (("equal", 0.12, 0.12), ("decay", 0.12, 0.06), ("none_in_B", 0.12, 0.0)):
        r, c = _toy(a, b)
        P = C.prep(r, c)
        res[name] = best_A(C, r, c, P, ["x", "junk"])
    chk("K1 an effect planted in both blocks is found on column x", res["equal"]["col"] == "x", f"{res['equal']['col']} cut {res['equal']['cut']:.2f} {res['equal']['sense']}")
    chk("K2 equal effects -> signed B/A near 1", 0.6 < res["equal"]["ratio"] < 1.5, f"{res['equal']['ratio']:+.2f}")
    chk("K3 a planted decay to half -> B/A near 0.5", 0.2 < res["decay"]["ratio"] < 0.85, f"{res['decay']['ratio']:+.2f}")
    chk("K4 an effect ONLY in A (the A-only search finds it, B shows none) -> B/A near 0", abs(res["none_in_B"]["ratio"]) < 0.4 and res["none_in_B"]["col"] == "x", f"{res['none_in_B']['ratio']:+.2f}")
    # M1: the min-over-both-blocks search (core_v2.search) must NOT pick the A-only effect when B has none: the very selection bias this measurement removes
    r, c = _toy(0.12, 0.0)
    P = C.prep(r, c)
    b2 = C.search(r, c, P, ["x", "junk"], nboot=150)
    sa = res["none_in_B"]["scoreA"]
    chk("M1 core_v2.search (min over both blocks) scores the A-only effect far lower than the A-only search", b2 is None or b2["score"] < 0.5 * sa,
        f"min-over-blocks {None if b2 is None else round(b2['score'], 4)} vs A-only {sa:.4f}")
    # K5: the signed ratio flips sign correctly for a NEGATIVE edge
    r, c = _toy(-0.12, -0.12)
    P = C.prep(r, c)
    bn = best_A(C, r, c, P, ["x", "junk"])
    chk("K5 a NEGATIVE edge in both blocks gives a POSITIVE signed ratio", bn["eA"] < 0 and bn["ratio"] > 0.5, f"eA {bn['eA']:+.3f} ratio {bn['ratio']:+.2f}")
    # K6: summarize
    rows = []
    rg = np.random.default_rng(1)
    for i in range(60):
        eA = rg.normal(0.05, 0.02)
        rows.append(dict(tag=f"c{i}", status="ok", scoreA=abs(eA) - 0.01 * i / 60, eA=eA, eB=0.5 * eA, ratio=0.5))
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pt, pa = summarize(pd.DataFrame(rows), say=print)
    chk("K6 the summary reproduces a planted B/A of exactly 0.5 (pooled 0.500)", abs(pt - 0.5) < 1e-9 and abs(pa - 0.5) < 1e-9, f"{pt:.3f} {pa:.3f}")
    # K7: the real tag list
    tf = tag_frame()
    chk("K7 the real list: 317 effective cell+sides, 31 of them pooled-or-sided survivors known, pooled side kept", len(tf) == 317 and (tf["side_s"] == "None").sum() > 0,
        f"{len(tf)} rows, {(tf['side_s'] == 'None').sum()} pooled")
    a, b = part_of(tf, "1,2,3/5"), part_of(tf, "4,5/5")
    chk("K8 shares are disjoint and exhaustive", len(a) + len(b) == 317 and not set(a.tag) & set(b.tag))
    print("\n" + ("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED"))
    return ok[0]


def k0(n):
    """The stored winner's eA/seA must be reproduced by edge_pack on the n smallest cells (before any A-only number is read)."""
    import core_v2 as C
    tf = tag_frame().sort_values("w").head(n)
    good = True
    for _, row in tf.iterrows():
        cell, s = row["tag"].rsplit("_side", 1)
        side = None if s == "None" else float(s)
        r, c, T = C.load_cell(cell, side)
        P = C.prep(r, c)
        stored = pd.read_parquet(os.path.join(HERE, "cores_v2", row["tag"] + ".parquet")).iloc[0]
        m = mask_of(r, stored["col"], float(stored["cut"]), stored["sense"])
        eA = edge_pack(C, r, c, P, m, "A", 500)
        eB = edge_pack(C, r, c, P, m, "B", 500)
        same = (abs(eA[0] - stored["eA"]) <= 1e-12 and abs(eA[1] - stored["seA"]) <= 1e-12 and abs(eB[0] - stored["eB"]) <= 1e-12
                and abs(eB[1] - stored["seB"]) <= 1e-12)
        print(f"  [{'PASS' if same else 'FAIL'}] K0 {row['tag']}: stored winner's eA, seA, eB, seB reproduced (1e-12)")
        good &= bool(same)
    return good


def main(argv):
    opt = {a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--")}
    out = opt.get("--out", OUTDIR)
    if "--selftest" in argv:
        sys.exit(0 if selftest() else 1)
    if "--k0" in argv:
        sys.exit(0 if k0(int(opt.get("--k0", 3))) else 1)
    if "--summarize" in argv:
        fs = sorted(glob.glob(os.path.join(out, "a_only_*.csv")))
        if not fs:
            raise SystemExit("no parts")
        df = pd.concat([pd.read_csv(f, keep_default_na=False, dtype={"side": str}) for f in fs], ignore_index=True)
        for col in ("scoreA", "eA", "eB", "ratio"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        want = set(tag_frame()["tag"])
        have = list(df["tag"])
        c1 = len(have) == len(set(have)) and set(have) == want
        print(f"  [{'PASS' if c1 else 'FAIL'}] every one of the 317 appears exactly once   {len(set(have))} of {len(want)}, rows {len(have)}")
        if not c1:
            raise SystemExit("incomplete or duplicated shares -- nothing summarised")
        provs = {(x, df[x].nunique()) for x in ("code_set", "producer_sha256")}
        print("  one provenance:", all(n == 1 for _, n in provs), "| machines:", df["machine"].value_counts().to_dict())
        summarize(df)
        df.to_csv(os.path.join(HERE, f"a_only_shrinkage_{time.strftime('%Y-%m-%d')}.csv"), index=False)
        return 0
    if "--part" in argv or "--all" in argv:
        if not selftest():
            raise SystemExit("self-test failed")
        import core_v2 as C
        import gate_receipts as G
        tf = part_of(tag_frame(), opt.get("--part"))
        os.makedirs(out, exist_ok=True)
        stamp = dict(code_set=G.code_set(), producer_sha256=sha(__file__), machine=G.machine_stamp()["node"], python=sys.version.split()[0])
        rows = []
        part = (opt.get("--part") or "all").replace("/", "of").replace(",", "")
        for i, tag in enumerate(tf["tag"]):
            t0 = time.time()
            try:
                b = one_tag(C, tag)
                row = dict(tag=tag, status="ok" if b else "empty", **(b or {}), secs=round(time.time() - t0, 1), **stamp)
            except Exception as e:
                row = dict(tag=tag, status=f"ERR {type(e).__name__}: {e}", secs=round(time.time() - t0, 1), **stamp)
            rows.append(row)
            print(f"  {i + 1}/{len(tf)} {tag} {row['status']} {row['secs']}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, f"a_only_{part}.csv"), index=False)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
