"""
claim_checks_seal.py -- claims 1 and 2 (HANDOFF.md 21.17 / 21.19; section 28.4 Family A) on ONE period, for A15 (USE 1,
Tier 1; A15.2 and A15.6 'T1-claims').

claim_checks.py (SHA-256 004BF6CA..., UNCHANGED) cannot evaluate one period: its known numbers are the TRAIN rates and it
loops blocks A and B. This script IMPORTS its statistics -- load, arm_frame, claim1_block, claim2_block, boot_draws, ci --
and changes nothing in them; it only selects the rows of one period. Same day bootstrap (iid, B = 2000, seed 20261003) as
the registered verdict, the 5-day block bootstrap as robustness, as in claim_checks.py.

Modes:
  python claim_checks_seal.py --selftest
  python claim_checks_seal.py --known data\\dump
        KNOWN NUMBER: data\\dump's block B alone must reproduce 26.65's block-B results to 0.0006 pp (they are printed to
        3 decimals): claim 1 M1b +1.310 [+0.949, +1.678], M2 +1.368 [+1.010, +1.729]; claim 2 +0.780 [+0.387, +1.161].
        Then a MUTATION (the block-B setups' M1b twin outcomes replaced by block-A twins' outcomes) must FAIL it.
  python claim_checks_seal.py --descriptive DUMP --block B
        a fresh one-period train build (seal_build.py) against data\\dump on the SAME cells (only the cells present in
        DUMP): the same three results side by side, DESCRIPTIVE (a different twin draw of the same rule).
  python claim_checks_seal.py --seal DUMP
        the single pre-registered sealed test. REFUSED unless: A15 is registered (a 'REGISTERED A15 <sha>' line matching the
        registered copy); seal_certify.receipt_ok() (every certified file unchanged, this one included); the dump is the
        certified seal build; seal_use1_outputs\\freeze.json names the current receipt and this dump's fingerprint; and no
        claims_seal_*.json exists yet (one opening). Writes into seal_use1_outputs\\ only.
"""
import datetime, hashlib, json, os, platform, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "code"))
import claim_checks as CC                                   # noqa: E402
from power_registered_set import registered_cells          # noqa: E402

PUB_B = {"M1b": (1.310, 0.949, 1.678), "M2": (1.368, 1.010, 1.729), "race": (0.780, 0.387, 1.161)}   # 26.65, block B
TOL = 0.0006
OUTDIR = os.path.join(HERE, "seal_use1_outputs")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest().upper()


def one_period(df, period, kinds=("iid", "block")):
    """Claim 1 (M1b, M2) and claim 2 (M1b race) on the rows of ONE period, through claim_checks' own functions."""
    out = {}
    for kind in kinds:
        r = {}
        for tag in ("M1b", "M2"):
            s = CC.arm_frame(df, tag)
            r[tag] = CC.claim1_block(s[s.block == period], kind)
        s1 = CC.arm_frame(df, "M1b")
        r["race"] = CC.claim2_block(s1[s1.block == period], kind)
        out[kind] = r
    return out


def matches_pub(res):
    bad = []
    for k, (e, lo, hi) in PUB_B.items():
        x = res["iid"][k]
        if not (abs(x["est_pp"] - e) < TOL and abs(x["lo_pp"] - lo) < TOL and abs(x["hi_pp"] - hi) < TOL):
            bad.append(k)
    return bad


def show(res, label):
    print(f"  ----- {label} -----")
    for k in ("M1b", "M2", "race"):
        x = res["iid"][k]; y = res["block"][k]
        extra = f"   setup {100*x['rate_setup']:.3f}% vs twin {100*x['rate_twin']:.3f}%" if k == "race" else ""
        print(f"    {('claim 1 ' + k) if k != 'race' else 'claim 2 race':<14} {CC.fmt(x)}   5-day block CI [{y['lo_pp']:+.3f}, {y['hi_pp']:+.3f}]{extra}")
    v = all(res["iid"][k]["lo_pp"] > 0 for k in ("M1b", "M2")); w = res["iid"]["race"]["lo_pp"] > 0
    print(f"    by the 21.19 rule on this ONE period: claim 1 {'PROVEN' if v else 'NOT PROVEN'} (M1b AND M2), "
          f"claim 2 {'PROVEN' if w else 'NOT PROVEN'}")
    return v, w


def toy_frame(**kw):
    t = CC._toy(**kw)
    t["M1b_k"] = 2; t["M2_k"] = 2
    t["M1b_res"] = t.t_res; t["M1b_vf"] = t.t_vf; t["M2_res"] = t.t_res; t["M2_vf"] = t.t_vf
    return t


def selftest():
    ok = [True]
    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else "")); ok[0] &= bool(cond)
    print("claim_checks.py's own self-test (its statistics are what this script reuses):")
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        base = CC.selftest()
    chk("K0 claim_checks.selftest() passes", base, f"{buf.getvalue().count('[PASS]')} PASS lines")
    t = toy_frame(seed=21)
    r = one_period(t, "B", kinds=("iid",))
    s = CC.arm_frame(t, "M1b"); ref = CC.claim1_block(s[s.block == "B"], "iid")
    chk("K1 one_period(B) IS claim_checks.claim1_block on the block-B rows (no reimplementation)",
        r["iid"]["M1b"]["est_pp"] == ref["est_pp"] and r["iid"]["M1b"]["lo_pp"] == ref["lo_pp"])
    t2 = toy_frame(gap1=0.0, gap1_B=0.05, seed=22)
    rA = one_period(t2, "A", kinds=("iid",)); rB = one_period(t2, "B", kinds=("iid",))
    chk("K2 a gap planted in block B only is found on B and not on A", rB["iid"]["M1b"]["lo_pp"] > 0 and rA["iid"]["M1b"]["lo_pp"] <= 0,
        f"B {rB['iid']['M1b']['est_pp']:+.2f}, A {rA['iid']['M1b']['est_pp']:+.2f} pp")
    fake = {"iid": {k: dict(est_pp=e, lo_pp=lo, hi_pp=hi) for k, (e, lo, hi) in PUB_B.items()}}
    chk("K3 the comparator accepts the published numbers exactly", matches_pub(fake) == [])
    fake["iid"]["M2"]["lo_pp"] += 0.002
    chk("M1 the comparator rejects a 0.002 pp shift in one bound", matches_pub(fake) == ["M2"])
    t3 = t.copy(); t3["block"] = np.where(t3.block.to_numpy() == "B", "SEALED", t3.block.to_numpy())
    rS = one_period(t3, "SEALED", kinds=("iid",))
    chk("K4 the 'SEALED' label path gives exactly the block-B result when B is relabelled (no A / B keyed logic)",
        all(rS["iid"][k]["est_pp"] == r["iid"][k]["est_pp"] for k in ("M1b", "M2", "race")))
    print("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED")
    return ok[0]


def load_cells(dump, cells):
    df, bad_k, race_bad = CC.load(dump, cells)
    return df, bad_k, race_bad


def known(dump):
    if os.path.normcase(os.path.abspath(dump)) != os.path.normcase(os.path.join(HERE, "data", "dump")):
        raise SystemExit("--known is defined on data\\dump only")
    cells = registered_cells(); t0 = time.time()
    df, bad_k, race_bad = load_cells(dump, cells)
    print(f"{len(cells)} registered cells, {len(df):,} setups, {time.time()-t0:.0f}s; race violations {race_bad}; "
          f"setups without a twin: M1b {bad_k['M1b']}, M2 {bad_k['M2']} (printed counts)")
    res = one_period(df, "B")
    show(res, "data\\dump, block B ALONE")
    bad = matches_pub(res)
    print(f"  [{'PASS' if not bad else 'FAIL'}] T1-claims KNOWN NUMBER: block B alone reproduces 26.65's block-B claim 1 (M1b, M2) "
          f"and claim 2 to {TOL} pp" + (f"  -- differs: {bad}" if bad else ""))
    m = df.copy(); rng = np.random.default_rng(20261014)
    iB = np.where(m.block.to_numpy() == "B")[0]; iA = np.where(m.block.to_numpy() == "A")[0]
    take = rng.choice(iA, len(iB), replace=True)
    for c in ("M1b_f", "M1b_res", "M1b_vf", "M1b_k"):
        m.loc[m.index[iB], c] = m[c].to_numpy()[take]
    rm = one_period(m, "B", kinds=("iid",))
    mb = matches_pub({"iid": rm["iid"]})
    print(f"  [{'PASS' if mb else 'FAIL'}] MUTATION: block-B M1b twins replaced by block-A twins -> the known-number check "
          f"FAILS (it must): differs on {mb}; claim 1 M1b {rm['iid']['M1b']['est_pp']:+.3f} pp, claim 2 {rm['iid']['race']['est_pp']:+.3f} pp")
    ok = (not bad) and bool(mb)
    rec = dict(check="T1-claims", all_pass=ok, dump=dump, cells=len(cells), producer_sha=sha(__file__),
               claim_checks_sha=sha(os.path.join(HERE, "claim_checks.py")), machine=platform.node(),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"), results={k: res["iid"][k] for k in PUB_B})
    p = os.path.join(HERE, "seal_T1_claims_receipt.json"); json.dump(rec, open(p, "w"), indent=1, default=float)
    print(f"T1-CLAIMS: {'PASS' if ok else 'FAIL'}; receipt {os.path.basename(p)}")
    return 0 if ok else 1


def descriptive(dump, blk):
    have = set(f[:-8] for f in os.listdir(dump) if f.endswith(".parquet") and ".m2." not in f)
    cells = [c for c in registered_cells() if c in have]
    print(f"DESCRIPTIVE: {len(cells)} registered cells present in {dump}; block {blk}; the same cells from data\\dump beside it")
    a, _, _ = load_cells(os.path.join(HERE, "data", "dump"), cells); b, bk, rb = load_cells(dump, cells)
    show(one_period(a, blk), f"data\\dump, block {blk}, these {len(cells)} cells")
    show(one_period(b, blk), f"fresh build {dump}, block {blk} (a different twin draw)")
    print(f"  fresh build: race violations {rb}; setups without a twin M1b {bk['M1b']}, M2 {bk['M2']}")
    return 0


def seal(dump):
    import glob
    import seal_build as SB, seal_certify as SC, gate_receipts
    rp = os.path.join(HERE, "seal_T1_receipt.json")
    if not SB.a15_registered():
        raise SystemExit("REFUSED: needs 'REGISTERED A15 <sha>' in HANDOFF_section_28.md and the registered copy in stale\\")
    ok, why = SC.receipt_ok()                       # reviewer on v6, O2: every certified file unchanged NOW, this script included
    if not ok:
        raise SystemExit(f"REFUSED: {why}")
    done = glob.glob(os.path.join(OUTDIR, "claims_seal_*.json"))
    if done:                                        # ONE opening: a re-run needs a visible, recorded manual move
        raise SystemExit(f"REFUSED: the sealed claims have already been run ({[os.path.basename(x) for x in done]})")
    fz = os.path.join(OUTDIR, "freeze.json")
    if not os.path.exists(fz):
        raise SystemExit("REFUSED: no seal_use1_outputs\\freeze.json (seal_certify.py --freeze comes first)")
    f = json.load(open(fz))
    if f.get("t1_receipt_sha") != sha(rp) or f.get("seal_dump_fingerprint") != gate_receipts.dump_fingerprint(dump):
        raise SystemExit("REFUSED: the freeze does not name the current receipt or this dump")
    n_pq, content = SC.dump_content_hash(dump)          # reviewer on v7, N2(a): the rows read now are the rows frozen
    if content != f.get("seal_dump_content_sha") or n_pq != f.get("seal_dump_parquet_files"):
        raise SystemExit("REFUSED: the seal dump's parquet content differs from the freeze")
    root = os.path.dirname(os.path.dirname(os.path.abspath(dump)))          # <root>\data\dump -> <root>
    mk = os.path.join(root, "certified_build.json"); br = os.path.join(root, "seal_build_receipt.json")
    if os.path.normcase(root) != os.path.normcase(os.path.join(OUTDIR, "seal_root")) or not os.path.exists(mk) or not os.path.exists(br):
        raise SystemExit("REFUSED: the dump must be seal_use1_outputs\\seal_root\\data\\dump, built by seal_certify.py --build-seal")
    if json.load(open(mk)).get("t1_receipt_sha") != sha(rp) or json.load(open(br)).get("split") != "test":
        raise SystemExit("REFUSED: the seal build was not made under the current Tier-1 receipt, or is not a test-split build")
    return run_sealed(dump, OUTDIR, "claims_seal")


def run_sealed(dump, outdir, stem, relabel=None):
    """The accept branch of --seal: the one-period claims on the block labelled 'SEALED'. relabel=('B', 'SEALED') is used
    ONLY by --rehearse on a train build, so the accept path is run end to end before the single opening (reviewer on v7,
    N3); it prints train block-B results already known (26.65 / --descriptive), not sealed ones."""
    os.makedirs(outdir, exist_ok=True)
    have = set(f[:-8] for f in os.listdir(dump) if f.endswith(".parquet") and ".m2." not in f)
    cells = [c for c in registered_cells() if c in have]
    if relabel is None and len(cells) != len(registered_cells()):
        raise SystemExit(f"REFUSED: the seal dump holds {len(cells)} of the {len(registered_cells())} registered cells")
    df, bad_k, race_bad = load_cells(dump, cells)
    if relabel is not None:
        df["block"] = np.where(df.block.to_numpy() == relabel[0], relabel[1], "REHEARSAL-UNUSED")
    if not (df.block == "SEALED").any():
        raise SystemExit("REFUSED: no row is labelled 'SEALED'")
    print(f"{stem.upper()}: {len(cells)} cells, {int((df.block == 'SEALED').sum()):,} setups labelled SEALED; setups without a twin "
          f"M1b {bad_k['M1b']}, M2 {bad_k['M2']} (E8: dropped from that arm only); race violations {race_bad}")
    res = one_period(df, "SEALED")
    v, w = show(res, "SEALED 2018-2022" if relabel is None else f"REHEARSAL: train block {relabel[0]} relabelled SEALED")
    out = dict(claim1=v, claim2=w, results=res, dump=dump, rehearsal=relabel is not None, producer_sha=sha(__file__),
               machine=platform.node(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = os.path.join(outdir, f"{stem}_{datetime.date.today().isoformat()}.json"); json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"wrote {p}  SHA-256 {sha(p)}")
    return 0


def rehearse(dump):
    """--rehearse seal_root_B60\\data\\dump: the accept path on a TRAIN stand-in (block B relabelled SEALED), with the
    content hash and the fingerprint computed as the freeze would, written to the session-free folder rehearsal_outputs\\."""
    import seal_certify as SC, gate_receipts
    root = os.path.dirname(os.path.dirname(os.path.abspath(dump)))
    if os.path.normcase(os.path.abspath(dump)).startswith(os.path.normcase(OUTDIR)):
        raise SystemExit("REFUSED: the rehearsal never reads the seal folder")
    br = json.load(open(os.path.join(root, "seal_build_receipt.json")))
    if br.get("split") not in ("train", "trainA", "trainB"):
        raise SystemExit("REFUSED: the rehearsal runs on a train build only")
    n_pq, content = SC.dump_content_hash(dump)
    print(f"REHEARSAL on {dump} (split {br['split']}): fingerprint {gate_receipts.dump_fingerprint(dump)}; {n_pq} parquet files, "
          f"content hash {content[:16]}")
    return run_sealed(dump, os.path.join(HERE, "rehearsal_outputs"), "rehearsal_claims", relabel=("B", "SEALED"))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not selftest():
        sys.exit(1)
    if "--selftest" in a:
        sys.exit(0)
    if "--known" in a:
        sys.exit(known(a[a.index("--known") + 1]))
    if "--descriptive" in a:
        sys.exit(descriptive(a[a.index("--descriptive") + 1], a[a.index("--block") + 1] if "--block" in a else "B"))
    if "--seal" in a:
        sys.exit(seal(a[a.index("--seal") + 1]))
    if "--rehearse" in a:
        sys.exit(rehearse(a[a.index("--rehearse") + 1]))
    print(__doc__); sys.exit(2)
