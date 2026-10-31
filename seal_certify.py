"""
seal_certify.py -- the CERTIFYING SCRIPT of A15 (A15.6): it runs every train-side Tier-1 check of the seal path and writes
seal_T1_receipt.json, which --build-seal, seal_build.py --split test (through the marker) and claim_checks_seal.py --seal
refuse to run without. The receipt hashes every file in hashed_files() (all code\\*.py included) and the claims receipt.

  python seal_certify.py --selftest
  python seal_certify.py --train            (the author's terminal; loads train twice; ESTIMATE 10-15 min)
  python seal_certify.py --build-seal       (only after registration; ONE process, a fresh seal_use1_outputs\\seal_root)
  python seal_certify.py --seal-meta        (T1d, metadata only)
  python seal_certify.py --freeze --list USE2_LIST_FILE   (or --no-list "REASON"; the USE-1 freeze; claims refuse without it)

CHECKS (each PASS / FAIL, all must PASS for all_pass):
  T1-cut       on the real joint train load, seal_build.strata_from_edges(cutpoints) == control_matched.strata, M1b and M2
  T1c-strata   seal_build.strata_check (fresh A-alone / B-alone labels and pool flag; unpatched control > 1%)
  T1a, T1b     seal_build.compare_exact on seal_root_T1a (no patch) and seal_root_T1b (patch)
  T1e          seal_build.compare_exact on seal_root_T1e (patch; N 100, 160, 200 in one process)
  T1c-setups   seal_build.compare_setups on A60, B60, A240, B240 (the two-sided edge rule)
  receipts     every build receipt exists, names the expected split / N / cutpoints SHA, and a patched build says
               strata_live True (unpatched: null)
  T1-inv       on every twin of every build AND of data\\dump's N = 60 cells (the known number: 26.18 measured 0
               violations on data\\dump): another trade date than its setup; |bar - src_bar| > h_rec = 3T; the same block;
               the same M1b stratum label as its setup (Chicago hour // 4, ATR quintile under the train cutpoints, from the
               stored ct_hour and atr0); for M2 twins the same M1b part of the label (the momentum part is not stored);
               exactly 2 M1b twins per setup
  T1-claims    seal_T1_claims_receipt.json says all_pass and names the current claim_checks.py
  T1-plant     an ESTIMATOR-level positive control for claim 1 (M1b arm) on the rows of the fresh B-alone build: setup
               outcomes replaced by M2 twin outcomes (null CI must contain 0), then 3 x the claim-1 SE planted; it exercises
               claim1_block, not the build (the build's neutrality rests on T1a / T1b / T1c / T1e); no race or M2-arm plant
  dump         data\\dump untouched: every one of its files last written before the first build of 2026-10-14
               (the earliest receipt), and gate_receipts' dump fingerprint = 6025b02ef36c9b4d
"""
import hashlib, json, os, platform, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "code"))
import seal_build as SB                                     # noqa: E402

CUT = os.path.join(HERE, "seal_cutpoints_v2.json")
BUILDS = {"T1a": ("train", [60], False), "T1b": ("train", [60], True), "A60": ("trainA", [60], True),
          "B60": ("trainB", [60], True), "A240": ("trainA", [240], True), "B240": ("trainB", [240], True),
          # T1e (reviewer on v6, O7(c)): the lookbacks 100 / 160 / 200, never compared before, built in ONE process with several
          # N, as the seal build will be; compared exactly with data\\dump and checked for the twin rules
          "T1e": ("train", [100, 160, 200], True)}
DUMP_FP = "6025b02ef36c9b4d"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest().upper()


def hashed_files():
    """Every file whose change voids the receipt (reviewer on v6, O1): the root scripts of the seal path, the cutpoints, the
    cell list and the dump fingerprint code, and EVERY code\\*.py (dump_episodes imports pipeline, spec, walk_fast,
    controls_fast, control_matched, _walkextra, volume_features, presignal, linearity, rearm, safeio, pseudo_vwap, loader)."""
    root = ["seal_certify.py", "seal_build.py", "claim_checks_seal.py", "claim_checks.py", "seal_cutpoints_v2.json",
            "power_registered_set.py", "gate_receipts.py", "pseudo_null_pair.py",       # claim_checks.py imports it (reviewer, N5)
            os.path.join("data", "outright.parquet")]                                 # the raw bars themselves (N5)
    code = sorted(os.path.join("code", f) for f in os.listdir(os.path.join(HERE, "code")) if f.endswith(".py"))
    return root + code


def libs():
    import pyarrow
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__,
                node=platform.node())


def label_m1b(ct_hour, atr0, cp):
    e = np.array([float.fromhex(h) for h in cp["atr_edges_hex"]] if "atr_edges_hex" in cp else cp["atr_edges"], float)
    e[0] = -np.inf; e[-1] = np.inf
    a = np.asarray(atr0, float); q = np.full(len(a), -1, np.int64); ok = np.isfinite(a)
    q[ok] = pd.cut(a[ok], bins=e, labels=False, include_lowest=True).astype(np.int64)
    return (np.asarray(ct_hour, np.int64) // 4) * 100 + q


def invariants(cell, m2, cp, train=True):
    """Counts of violations, per rule, for one cell's setups and twins."""
    T = int(cell["T"].iloc[0]); h_rec = 3 * T
    s = cell[cell.is_real == 1].set_index("bar")
    slab = pd.Series(label_m1b(s.ct_hour, s.atr0, cp), index=s.index)
    out = {}
    for tag, tw in (("M1b", cell[cell.is_real == 0]), ("M2", m2)):
        src = tw.src_bar.to_numpy()
        miss = ~np.isin(src, s.index.to_numpy())
        tt = tw[~miss]; ss = s.loc[tt.src_bar.to_numpy()]
        tl = label_m1b(tt.ct_hour, tt.atr0, cp); sl = slab.loc[tt.src_bar.to_numpy()].to_numpy()
        out[f"{tag}_no_setup"] = int(miss.sum())
        out[f"{tag}_same_day"] = int((tt.day.to_numpy() == ss.day.to_numpy()).sum())
        out[f"{tag}_within_hrec"] = int((np.abs(tt.bar.to_numpy() - tt.src_bar.to_numpy()) <= h_rec).sum())
        out[f"{tag}_other_block"] = int((tt.block.to_numpy() != ss.block.to_numpy()).sum())
        out[f"{tag}_other_stratum"] = int((tl != sl).sum())
    k = cell[cell.is_real == 0].groupby("src_bar").size().reindex(s.index, fill_value=0)
    out["M1b_not_2"] = int((k != 2).sum()) if train else 0
    return out


INV_COLS = ["bar", "src_bar", "is_real", "day", "block", "ct_hour", "atr0", "T"]


def inv_folder(dump, cells, cp, train=True):
    tot = {}
    for c in cells:
        a = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=INV_COLS)
        b = pd.read_parquet(os.path.join(dump, c + ".m2.parquet"), columns=[x for x in INV_COLS if x not in ("is_real", "T")])
        for k, v in invariants(a, b, cp, train).items():
            tot[k] = tot.get(k, 0) + v
    return tot


def plant_check(dump):
    """T1-plant through claim_checks' own functions on a seal-path build (fresh B alone)."""
    import claim_checks as CC
    have = set(f[:-8] for f in os.listdir(dump) if f.endswith(".parquet") and ".m2." not in f)
    from power_registered_set import registered_cells
    cells = [c for c in registered_cells() if c in have]
    df, _, _ = CC.load(dump, cells)
    s = CC.arm_frame(df, "M1b"); s = s[(s.block == "B") & s["M2_f"].notna()].copy()
    # a NO-EFFECT frame: each setup's own outcome is replaced by an outcome of its M2 twins (a matched non-VWAP moment;
    # a twin mean of 0.5 becomes a seeded coin), so setup minus M1b twin is a twin-versus-twin contrast with no VWAP effect
    r0 = np.random.default_rng(7)
    m2 = s["M2_f"].to_numpy()
    s["rf"] = np.where(m2 == 1, 1.0, np.where(m2 == 0, 0.0, (r0.random(len(s)) < 0.5).astype(float)))
    base = CC.claim1_block(s, "iid")
    se = (base["hi_pp"] - base["lo_pp"]) / (2 * 1.96)
    target = 3 * se / 100.0                                    # as a fraction
    rng = np.random.default_rng(20261014)
    zero = np.where(s.rf.to_numpy() == 0)[0]
    k = int(round(target * len(s)))
    p = s.copy(); idx = rng.choice(zero, k, replace=False); col = p.columns.get_loc("rf"); p.iloc[idx, col] = 1.0
    planted = CC.claim1_block(p, "iid")
    back = CC.claim1_block(s, "iid")
    want = base["est_pp"] + 100 * k / len(s)
    ok0 = base["lo_pp"] <= 0 <= base["hi_pp"]
    ok1 = planted["lo_pp"] > 0 and abs(planted["est_pp"] - want) < se
    ok2 = back["est_pp"] == base["est_pp"]
    print(f"  T1-plant (no-effect frame: setup outcome replaced by its M2 twins'): null CI contains 0: {ok0}; "
          f"{len(cells)} cells, {len(s):,} block-B setups; unplanted {base['est_pp']:+.3f} [{base['lo_pp']:+.3f}, {base['hi_pp']:+.3f}] (SE {se:.3f}); planted "
          f"+{100*k/len(s):.3f} pp -> {planted['est_pp']:+.3f} [{planted['lo_pp']:+.3f}, {planted['hi_pp']:+.3f}] (want {want:+.3f}); "
          f"plant removed {back['est_pp']:+.3f}")
    return bool(ok0 and ok1 and ok2)


def run_train():
    if not selftest():
        return 1
    cp = json.load(open(CUT)); res = {}; t0 = time.time()
    def rec(name, ok, det=""):
        res[name] = bool(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name}  {det}", flush=True)
    # receipts
    first = None; okr = True; det = []
    for name, (split, Ns, patched) in BUILDS.items():
        p = os.path.join(HERE, f"seal_root_{name}", "seal_build_receipt.json")
        if not os.path.exists(p):
            okr = False; det.append(f"{name}: no receipt"); continue
        r = json.load(open(p)); first = min(first or r["finished"], r["finished"])
        good = r["split"] == split and r["N"] == Ns and (r.get("cutpoints_sha") == sha(CUT) if patched else r.get("cutpoints") is None)
        good &= (r.get("strata_live") is True) if patched else (r.get("strata_live") in (None, True))   # T1a ran before the null fix
        good &= r.get("dump_episodes_sha") == sha(os.path.join(HERE, "code", "dump_episodes.py"))
        good &= r.get("control_matched_sha") == sha(os.path.join(HERE, "code", "control_matched.py"))
        good &= r.get("loader_sha") == sha(os.path.join(HERE, "code", "loader.py"))
        okr &= bool(good); det.append(f"{name}:{'ok' if good else 'BAD'}(wrapper {str(r.get('wrapper_sha'))[:8]})")
    rec("receipts", okr, " ".join(det))
    # T1-cut
    import loader, control_matched
    J = loader.load("train")
    for m in (False, True):
        a = control_matched.strata(J, with_momentum=m); b = SB.strata_from_edges(J, m, cp)
        rec(f"T1-cut {'M2' if m else 'M1b'}", np.array_equal(a, b), f"{int((a != b).sum())} of {len(a):,} differ")
    del J
    rec("T1c-strata", SB.strata_check(CUT) == 0)
    rec("T1a", SB.compare_exact(os.path.join(HERE, "seal_root_T1a")) == 0)
    rec("T1b", SB.compare_exact(os.path.join(HERE, "seal_root_T1b")) == 0)
    rec("T1e (N 100/160/200, one process)", SB.compare_exact(os.path.join(HERE, "seal_root_T1e")) == 0)
    for name in ("A60", "B60", "A240", "B240"):
        rec(f"T1c-setups {name}", SB.compare_setups(os.path.join(HERE, f"seal_root_{name}"), name[0],
                                                    os.path.join(HERE, "seal_root_A60")) == 0)
    # T1-inv: the known number first
    dump = os.path.join(HERE, "data", "dump")
    c60 = sorted(f[:-8] for f in os.listdir(dump) if f.startswith("N60_") and f.endswith(".parquet") and ".m2." not in f)
    t = inv_folder(dump, c60, cp)
    rec("T1-inv KNOWN NUMBER on data\\dump N=60 (26.18: 0 violations)", sum(t.values()) == 0, str(t))
    for name in BUILDS:
        d = os.path.join(HERE, f"seal_root_{name}", "data", "dump")
        t = inv_folder(d, SB.cells_built(os.path.join(HERE, f"seal_root_{name}")), cp)
        rec(f"T1-inv {name}", sum(t.values()) == 0, str(t))
    cr = os.path.join(HERE, "seal_T1_claims_receipt.json")
    c = json.load(open(cr)) if os.path.exists(cr) else {}
    rec("T1-claims receipt", c.get("all_pass") is True and c.get("claim_checks_sha") == sha(os.path.join(HERE, "claim_checks.py"))
        and c.get("producer_sha") == sha(os.path.join(HERE, "claim_checks_seal.py")),     # reviewer on v6, O3
        f"claims receipt producer {str(c.get('producer_sha'))[:8]}, current claim_checks_seal.py {sha(os.path.join(HERE, 'claim_checks_seal.py'))[:8]}")
    rec("T1-plant", plant_check(os.path.join(HERE, "seal_root_B60", "data", "dump")))
    # data\dump untouched
    import gate_receipts
    try:
        fp = gate_receipts.dump_fingerprint(dump)
    except Exception as e:                                     # noqa: BLE001
        fp = f"error {e}"
    newest = max(os.path.getmtime(os.path.join(dump, f)) for f in os.listdir(dump))
    t_first = time.mktime(time.strptime(first, "%Y-%m-%d %H:%M:%S")) if first else 0
    rec("dump untouched", fp == DUMP_FP and newest < t_first - 60,
        f"fingerprint {fp}; newest file {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))}; first build finished {first}")
    # T1d's raw-bar check on a TRAIN year first (reviewer on v7, N1): its known number, measured 2026-10-14
    rb = raw_bar_meta(lo="2012-01-01", hi="2013-01-01")
    rec("T1d raw-bar check KNOWN NUMBER on train year 2012 (437,428 rows, 8 instruments, 0 duplicates, 0 non-monotone)",
        rb["n_rows"] == 437428 and rb["instruments"] == 8 and rb["dup_ts"] == 0 and rb["non_monotone"] == 0)
    allp = all(res.values())
    out = dict(all_pass=allp, checks=res, finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0),
               machine=dict(node=platform.node(), cpu=os.cpu_count(), python=platform.python_version()),
               hashes={p: sha(os.path.join(HERE, p)) for p in hashed_files()}, libs=libs(),
               claims_receipt_sha=sha(cr) if os.path.exists(cr) else None)
    json.dump(out, open(os.path.join(HERE, "seal_T1_receipt.json"), "w"), indent=1)
    print(f"\nSEAL TIER 1: {'ALL PASS' if allp else 'NOT ALL PASS'} ({sum(res.values())} of {len(res)}); wrote seal_T1_receipt.json")
    return 0 if allp else 1


def selftest():
    ok = [True]
    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}  {det}"); ok[0] &= bool(cond)
    cp = {"atr_edges": [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]}
    T = 10
    S = pd.DataFrame(dict(bar=[1000, 2000], src_bar=np.nan, is_real=1, day=[1, 2], block=["A", "A"], ct_hour=[1, 9],
                          atr0=np.float32([1.5, 2.5]), T=T))
    W = pd.DataFrame(dict(bar=[3000, 3100, 4000, 4100], src_bar=[1000, 1000, 2000, 2000], is_real=0, day=[3, 4, 5, 6],
                          block="A", ct_hour=[2, 3, 10, 11], atr0=np.float32([1.2, 1.9, 2.1, 2.9]), T=T))
    good = pd.concat([S, W], ignore_index=True); m2 = W.drop(columns=["is_real", "T"])
    chk("K1 a valid toy cell has 0 violations", sum(invariants(good, m2, cp).values()) == 0, str(invariants(good, m2, cp)))
    def mut(f, rule):
        g = good.copy(); f(g); r = invariants(g, m2, cp); return r[rule] > 0
    chk("M1 a twin on its setup's trade date is caught", mut(lambda g: g.__setitem__("day", np.where(g.bar == 3000, 1, g.day)), "M1b_same_day"))
    chk("M2 a twin within h_rec is caught", mut(lambda g: g.__setitem__("bar", np.where(g.bar == 3000, 1020, g.bar)), "M1b_within_hrec"))
    chk("M3 a twin from the other block is caught", mut(lambda g: g.__setitem__("block", np.where(g.bar == 4000, "B", g.block)), "M1b_other_block"))
    chk("M4 a twin in another ATR quintile is caught", mut(lambda g: g.__setitem__("atr0", np.where(g.bar == 4100, np.float32(3.5), g.atr0)), "M1b_other_stratum"))
    chk("M5 a twin in another hour bucket is caught", mut(lambda g: g.__setitem__("ct_hour", np.where(g.bar == 3100, 7, g.ct_hour)), "M1b_other_stratum"))
    g = good[good.bar != 4100]
    chk("M6 a setup with one M1b twin is caught", invariants(g, m2, cp)["M1b_not_2"] == 1)
    hf = hashed_files()
    need = [os.path.join("code", x) for x in ("dump_episodes.py", "pseudo_vwap.py", "pipeline.py", "walk_fast.py", "spec.py")]
    chk("K2 the receipt covers every code\\*.py and power_registered_set.py / gate_receipts.py",
        all(x in hf for x in need + ["power_registered_set.py", "gate_receipts.py"])
        and sum(x.startswith("code") for x in hf) == len([f for f in os.listdir(os.path.join(HERE, "code")) if f.endswith(".py")]),
        f"{len(hf)} files")
    import tempfile, contextlib, io
    td = tempfile.mkdtemp(); tsr = pd.Series(pd.date_range("2018-01-01 00:00", periods=6, freq="min", tz="UTC"))
    two = pd.concat([pd.DataFrame(dict(ts=tsr, instrument_id=1)), pd.DataFrame(dict(ts=tsr, instrument_id=2))], ignore_index=True)
    two.to_parquet(os.path.join(td, "ok.parquet"))                     # two contracts on the SAME minutes: correct data
    bad = two.copy(); bad.loc[3, "ts"] = bad.loc[2, "ts"]
    bad.to_parquet(os.path.join(td, "bad.parquet"))
    with contextlib.redirect_stdout(io.StringIO()):
        g1 = raw_bar_meta(os.path.join(td, "ok.parquet")); g2 = raw_bar_meta(os.path.join(td, "bad.parquet"))
    chk("K3 raw-bar check: two contracts sharing every minute (as the real file) give 0 duplicates, 0 non-monotone",
        g1["dup_ts"] == 0 and g1["non_monotone"] == 0 and g1["instruments"] == 2)
    chk("M7 raw-bar check: a duplicated minute WITHIN one contract is caught (duplicate AND non-monotone)",
        g2["dup_ts"] == 1 and g2["non_monotone"] == 1)
    old = (two.duplicated("ts").sum() > 0)
    chk("M8 the superseded rule (duplicates on ts alone) would have FAILED correct two-contract data (reviewer on v7, N1)", bool(old))
    print("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED")
    return ok[0]


SEAL_ROOT = os.path.join(HERE, "seal_use1_outputs", "seal_root")
RECEIPT = os.path.join(HERE, "seal_T1_receipt.json")


def receipt_ok():
    """The Tier-1 receipt says all_pass AND every file it hashed is unchanged now (so the certified code is the code that
    runs). Returns (ok, reason)."""
    if not os.path.exists(RECEIPT):
        return False, "no seal_T1_receipt.json"
    r = json.load(open(RECEIPT))
    if r.get("all_pass") is not True:
        return False, "the receipt does not say all_pass"
    if sorted(r["hashes"]) != sorted(hashed_files()):
        return False, "the receipt does not cover the current set of hashed files (a file was added, removed or the receipt is old)"
    changed = [p for p, h in r["hashes"].items() if not os.path.exists(os.path.join(HERE, p)) or sha(os.path.join(HERE, p)) != h]
    if changed:
        return False, f"changed since certification: {changed}"
    cr = os.path.join(HERE, "seal_T1_claims_receipt.json")
    if not os.path.exists(cr) or r.get("claims_receipt_sha") != sha(cr):
        return False, "seal_T1_claims_receipt.json is missing or differs from the one certified"
    if r.get("libs") != libs():                     # same machine and library versions as at certification (reviewer, N5)
        return False, f"machine or library versions differ from certification: {r.get('libs')} vs {libs()}"
    return True, "ok"


def build_seal():
    """THE SEAL BUILD (A15.6), only through this gate: 'REGISTERED A15' + a valid Tier-1 receipt; writes into
    seal_use1_outputs\\seal_root (the guarded folder of A15.5); all five lookbacks (dump_episodes builds every anchor at each
    N, which includes the 144 registered cells); ONE process (the multi-shard manifest race of 26.42 is not fixed in code\\)."""
    if not SB.a15_registered():
        raise SystemExit("REFUSED: A15 is not registered (needs 'REGISTERED A15 <sha>' in HANDOFF_section_28.md matching "
                         "stale\\AMENDMENT_A15_registered_copy_*.md)")
    ok, why = receipt_ok()
    if not ok:
        raise SystemExit(f"REFUSED: {why}")
    if os.path.exists(SEAL_ROOT):                    # reviewer on v6, O2: no resume, no mixed dump; a re-build needs a visible move
        raise SystemExit("REFUSED: seal_use1_outputs\\seal_root already exists. A seal build is made ONCE into a fresh folder; "
                         "a re-build after a failure (allowed only while no outcome column was opened) needs the old folder "
                         "moved into stale\\ by hand, and that is recorded.")
    os.makedirs(SEAL_ROOT)
    marker = dict(t1_receipt_sha=sha(RECEIPT), started=time.strftime("%Y-%m-%d %H:%M:%S"), machine=platform.node())
    json.dump(marker, open(os.path.join(SEAL_ROOT, "certified_build.json"), "w"), indent=1)
    rc = SB.build(SEAL_ROOT, "test", CUT, [60, 100, 160, 200, 240])
    br = json.load(open(os.path.join(SEAL_ROOT, "seal_build_receipt.json")))
    if rc != 0 or br.get("strata_live") is not True or br.get("split") != "test":
        print("SEAL BUILD NOT ACCEPTED: the strata patch was not live, or the build failed"); return 1
    n_pq, content = dump_content_hash(os.path.join(SEAL_ROOT, "data", "dump"))     # at the END of the build (reviewer on v8)
    marker.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), dump_parquet_files=n_pq, dump_content_sha=content)
    json.dump(marker, open(os.path.join(SEAL_ROOT, "certified_build.json"), "w"), indent=1)
    print(f"SEAL BUILD ACCEPTED: {n_pq} parquet files, content hash {content}")
    return 0


def seal_meta():
    """T1d: METADATA of the seal build only (28.5:170). Reads NO outcome column (w_*, mae_*, race, gap_at_T, mir_w): only
    bar / src_bar / is_real / day / block / ct_hour / atr0 / T of the cell files, the build receipt, and the ts and symbol
    columns of the raw bars of 2018-2022 (no price, no volume)."""
    from power_registered_set import registered_cells
    cp = json.load(open(CUT)); dd = os.path.join(SEAL_ROOT, "data", "dump")
    raw = raw_bar_meta()
    rc = json.load(open(os.path.join(SEAL_ROOT, "seal_build_receipt.json")))
    print(f"seal build receipt: split {rc['split']}, bars {rc.get('n_load'):,}, blocks {rc.get('blocks')}, strata live {rc.get('strata_live')}")
    cells = registered_cells(); tot = {}; rows = []
    for c in cells:
        a = pd.read_parquet(os.path.join(dd, c + ".parquet"), columns=INV_COLS)
        b = pd.read_parquet(os.path.join(dd, c + ".m2.parquet"), columns=[x for x in INV_COLS if x not in ("is_real", "T")])
        v = invariants(a, b, cp, train=False)
        for k, x in v.items():
            tot[k] = tot.get(k, 0) + x
        s = a[a.is_real == 1]; T = int(a["T"].iloc[0])
        k1 = a[a.is_real == 0].groupby("src_bar").size().reindex(s.bar, fill_value=0)
        k2 = b.groupby("src_bar").size().reindex(s.bar, fill_value=0)
        rows.append(dict(cell=c, setups=len(s), no_M1b=int((k1 == 0).sum()), one_M1b=int((k1 == 1).sum()),
                         no_M2=int((k2 == 0).sum()), end_tail_bars=3 * T + 2,
                         strata_used=int(len(np.unique(label_m1b(s.ct_hour, s.atr0, cp))))))
    m = pd.DataFrame(rows)
    print(m.to_string(index=False))
    print(f"TOTALS: setups {m.setups.sum():,}; without an M1b twin {m.no_M1b.sum():,}, with one {m.one_M1b.sum():,}; without an M2 twin "
          f"{m.no_M2.sum():,} (E8: dropped from that arm only)")
    print(f"T1-inv on all {len(cells)} seal cells (exactly-2 not required on the seal): {tot}")
    lab = []
    for c in cells:
        s = pd.read_parquet(os.path.join(dd, c + ".parquet"), columns=["is_real", "ct_hour", "atr0"]); s = s[s.is_real == 1]
        lab.append(label_m1b(s.ct_hour, s.atr0, cp))
    lab = np.concatenate(lab); q = lab % 100
    print("setup share by train ATR quintile (0 = calmest): " + " ".join(f"q{k} {100*np.mean(q == k):.1f}%" for k in range(5)))
    ok = all(v == 0 for k, v in tot.items() if k != "M1b_not_2") and raw["dup_ts"] == 0 and raw["non_monotone"] == 0
    out = dict(check="T1d", invariants_ok=ok, totals=tot, setups=int(m.setups.sum()), no_M1b=int(m.no_M1b.sum()),
               no_M2=int(m.no_M2.sum()), quintile_share=[float(np.mean(q == k)) for k in range(5)], raw=raw,
               build_receipt_sha=sha(os.path.join(SEAL_ROOT, "seal_build_receipt.json")), producer_sha=sha(__file__),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(out, open(os.path.join(HERE, "seal_use1_outputs", "seal_meta.json"), "w"), indent=1)
    print(f"T1d: invariants {'PASS' if ok else 'FAIL'}; wrote seal_use1_outputs\\seal_meta.json (metadata only)")
    return 0 if ok else 1


def raw_bar_meta(path=None, lo="2018-01-01", hi="2022-01-01"):
    """Outcome-blind raw-data checks of the bars in [lo, hi) (reviewer on v6 O10, on v7 N1): the raw file holds SEVERAL
    contracts per minute (the loader picks the front month), so duplicates are tested on (ts, instrument_id) and
    monotonicity WITHIN each instrument (0 required for both); rows and instruments per calendar year INSIDE the window
    only; first and last bar in UTC and Chicago. Reads only the ts and instrument_id columns (no price, no volume).
    KNOWN NUMBER: a train year, 2012, measured 2026-10-14: 437,428 rows, 8 instruments, 0 duplicates, 0 non-monotone."""
    path = path or os.path.join(HERE, "data", "outright.parquet")
    r = pd.read_parquet(path, columns=["ts", "instrument_id"])
    s = r[(r.ts >= pd.Timestamp(lo, tz="UTC")) & (r.ts < pd.Timestamp(hi, tz="UTC"))].reset_index(drop=True)
    del r
    dup = int(s.duplicated(["ts", "instrument_id"]).sum())
    nonmono = 0
    for _, g in s.groupby("instrument_id", sort=False):
        t = g.ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
        nonmono += int((np.diff(t) <= 0).sum())
    yr = s.groupby(s.ts.dt.year).agg(rows=("ts", "size"), instruments=("instrument_id", "nunique"))
    ch = "America/Chicago"
    first = s.ts.min() if len(s) else None; last = s.ts.max() if len(s) else None
    print("rows / instruments per calendar year in the window: " + " ".join(f"{int(y)} {int(v.rows):,}/{int(v.instruments)}" for y, v in yr.iterrows()))
    print(f"window {lo}..{hi}: {len(s):,} rows, {s.instrument_id.nunique()} instruments; duplicate (ts, instrument) {dup} "
          f"(0 required); non-monotone steps within an instrument {nonmono} (0 required)")
    if first is not None:
        print(f"first bar {first} UTC = {first.tz_convert(ch)}; last {last} UTC = {last.tz_convert(ch)}")
    return dict(n_rows=int(len(s)), instruments=int(s.instrument_id.nunique()), dup_ts=dup, non_monotone=nonmono,
                per_year={str(int(y)): dict(rows=int(v.rows), instruments=int(v.instruments)) for y, v in yr.iterrows()},
                first_utc=str(first), last_utc=str(last))


def dump_content_hash(dd):
    """SHA-256 of every parquet file of a dump, then one hash over the sorted 'name:hash' lines (reviewer on v7, N2(a):
    gate_receipts' fingerprint covers counts.json and the manifest, not the rows the claims read)."""
    names = sorted(f for f in os.listdir(dd) if f.endswith(".parquet"))
    lines = "".join(f"{f}:{sha(os.path.join(dd, f))}\n" for f in names)
    return len(names), hashlib.sha256(lines.encode("utf-8")).hexdigest().upper()


def freeze(list_file=None, no_list_reason=None):
    """The USE-1 FREEZE (A15.5/A15.6): written AFTER --seal-meta passed and BEFORE the claims run. It records the seal dump's
    fingerprint, the Tier-1 receipt, the claims receipt, the T1d output and (if given) the USE-2 ingredient list, and prints
    the SHA-256 of freeze.json for the dated freeze line. claim_checks_seal.py --seal refuses without it (reviewer on v6, O2)."""
    import gate_receipts
    ok, why = receipt_ok()
    if not ok:
        raise SystemExit(f"REFUSED: {why}")
    mp = os.path.join(HERE, "seal_use1_outputs", "seal_meta.json"); fz = os.path.join(HERE, "seal_use1_outputs", "freeze.json")
    if not os.path.exists(mp) or json.load(open(mp)).get("invariants_ok") is not True:
        raise SystemExit("REFUSED: seal_use1_outputs\\seal_meta.json is missing or did not pass")
    br = os.path.join(SEAL_ROOT, "seal_build_receipt.json")
    if json.load(open(mp)).get("build_receipt_sha") != sha(br):          # N2(b): T1d of THIS build
        raise SystemExit("REFUSED: seal_meta.json was produced for a different seal build")
    if not list_file and not no_list_reason:                             # N2(c): the author 26.174 item 3
        raise SystemExit("REFUSED: give the USE-2 ingredient list (--list FILE) or a recorded reason (--no-list \"REASON\")")
    if os.path.exists(fz):
        raise SystemExit("REFUSED: freeze.json already exists; the freeze is written once")
    dd = os.path.join(SEAL_ROOT, "data", "dump")
    n_pq, content = dump_content_hash(dd)
    mk = json.load(open(os.path.join(SEAL_ROOT, "certified_build.json")))
    if mk.get("dump_content_sha") != content or mk.get("dump_parquet_files") != n_pq:   # no edit between build and freeze
        raise SystemExit("REFUSED: the seal dump's parquet content differs from the hash taken at the end of the build")
    out = dict(seal_dump_fingerprint=gate_receipts.dump_fingerprint(dd), seal_dump_parquet_files=n_pq,
               seal_dump_content_sha=content, no_list_reason=no_list_reason, t1_receipt_sha=sha(RECEIPT),
               claims_receipt_sha=sha(os.path.join(HERE, "seal_T1_claims_receipt.json")), seal_meta_sha=sha(mp),
               build_receipt_sha=sha(os.path.join(SEAL_ROOT, "seal_build_receipt.json")),
               ingredient_list=list_file, ingredient_list_sha=sha(list_file) if list_file else None,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"), machine=platform.node())
    json.dump(out, open(fz, "w"), indent=1)
    print(json.dumps(out, indent=1)); print(f"FREEZE written: seal_use1_outputs\\freeze.json  SHA-256 {sha(fz)}  (put this in the dated freeze line)")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--freeze" in a:
        sys.exit(freeze(a[a.index("--list") + 1] if "--list" in a else None,
                        a[a.index("--no-list") + 1] if "--no-list" in a else None))
    if "--selftest" in a:
        sys.exit(0 if selftest() else 1)
    if "--train" in a:
        sys.exit(run_train())
    if "--build-seal" in a:
        sys.exit(build_seal())
    if "--seal-meta" in a:
        sys.exit(seal_meta())
    print(__doc__); sys.exit(2)
