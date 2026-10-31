"""
seal_build.py -- the SEAL PATH wrapper of A15 (draft v4, A15.6). Root-level: NOTHING under code\\ is edited.

It runs code\\dump_episodes.py UNCHANGED (runpy, as __main__) with three things replaced in memory:
  1. VWAP_ROOT = a scratch root (its own data\\ folder holding a HARD LINK to data\\outright.parquet), so every output
     lands in <root>\\data\\dump and the real data\\dump is never written (pipeline.py: OUT = VWAP_ROOT\\data\\dump).
  2. loader.load, so the build sees the period chosen here (dump_episodes.py:230 hard-codes load("train")):
        train   the joint train load (A + B), exactly as production
        trainA  a FRESH load of block A alone (loader.TRAIN_BLOCKS cut to A before the original load runs)
        trainB  a FRESH load of block B alone
        test    the sealed 2018-2022 period, load("test", unseal=True); REFUSED until A15 is registered
  3. optionally control_matched.strata, replaced by strata_train(): the same labels, but the ATR and 240-bar momentum
     quintile EDGES are the TRAIN ones read from a hashed cutpoints file (28.5:167), with open outer edges so a seal
     value outside the train range falls into the first / last quintile.

Modes (all for the author's terminal except --selftest):
  python seal_build.py --selftest
  python seal_build.py --make-cutpoints seal_cutpoints.json
        loads TRAIN, stores the qcut edges, and checks on the real train frame that strata_train() reproduces
        control_matched.strata() EXACTLY for both schemes (the core of T1b, without a build)
  python seal_build.py --root ROOT --split {train,trainA,trainB} [--cutpoints FILE] [--N 60 ...]
        a build into ROOT\\data\\dump; writes ROOT\\seal_build_receipt.json
  python seal_build.py --compare ROOT --exact          T1a / T1b: every file of the built cells equals data\\dump
  python seal_build.py --compare ROOT --setups A|B     T1c: setup rows of a one-block build equal data\\dump's rows

It reads no sealed bar unless --split test is given, and that is refused unless HANDOFF_section_28.md carries a line
"REGISTERED A15 <sha256>" matching a registered copy in stale\\, AND the root is seal_use1_outputs\\seal_root with the
marker seal_certify.py --build-seal writes (the only registered route). Comparisons read TRAIN outcome columns only.
"""
import hashlib, json, os, platform, runpy, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
REAL_DUMP = os.path.join(HERE, "data", "dump")
REAL_RAW = os.path.join(HERE, "data", "outright.parquet")
SPLITS = ("train", "trainA", "trainB", "test")
STATE = {}
FAMILY_A_COLS = ["ts", "side", "dist_atr", "atr0", "w_0.0", "mir_w", "race", "cens_T", "gap_at_T", "day", "block", "is_real"]
TIMING_KEYS = ("t_pseudo_match_s", "t_pseudo_walk_s")


def sha(p, n=None):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest().upper()
    return h if n is None else h[:n]


def a15_registered(p=None, stale_dir=None):
    """True only if HANDOFF_section_28.md has a line 'REGISTERED A15 <64-hex SHA-256>' AND a registered copy
    stale\\AMENDMENT_A15_registered_copy_*.md with exactly that SHA-256 exists (reviewer on v6, O7(b): a bare line, a
    draft or a stray handoff sentence does not unlock the seal)."""
    import re
    p = p or os.path.join(HERE, "HANDOFF_section_28.md"); sd = stale_dir or os.path.join(HERE, "stale")
    if not os.path.exists(p) or not os.path.isdir(sd):
        return False
    want = set()
    for l in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"REGISTERED A15\s+([0-9A-Fa-f]{64})\b", l)
        if m:
            want.add(m.group(1).upper())
    have = {sha(os.path.join(sd, f)) for f in os.listdir(sd) if f.startswith("AMENDMENT_A15_registered_copy_") and f.endswith(".md")}
    return bool(want & have)


SEAL_ROOT = os.path.join(HERE, "seal_use1_outputs", "seal_root")
T1_RECEIPT = os.path.join(HERE, "seal_T1_receipt.json")


def certified_root(root):
    """The only root a --split test build may write into: seal_use1_outputs\\seal_root, carrying the marker that
    seal_certify.py --build-seal writes, naming the CURRENT Tier-1 receipt (reviewer on v6, O7(a))."""
    mk = os.path.join(root, "certified_build.json")
    return (os.path.normcase(os.path.abspath(root)) == os.path.normcase(SEAL_ROOT) and os.path.exists(mk)
            and os.path.exists(T1_RECEIPT) and json.load(open(mk)).get("t1_receipt_sha") == sha(T1_RECEIPT))


def check_root(root):
    """The scratch root may not be the project, may not contain the real dump, and its dump may not resolve to it."""
    r = os.path.abspath(root)
    if os.path.normcase(r) == os.path.normcase(HERE):
        raise SystemExit("REFUSED: --root is the project root")
    tgt = os.path.join(r, "data", "dump")
    if os.path.exists(tgt) and os.path.exists(REAL_DUMP) and os.path.samefile(tgt, REAL_DUMP):
        raise SystemExit("REFUSED: <root>\\data\\dump resolves to the real data\\dump")
    if os.path.normcase(os.path.realpath(tgt)).startswith(os.path.normcase(os.path.realpath(REAL_DUMP))):
        raise SystemExit("REFUSED: <root>\\data\\dump lies inside the real data\\dump")
    return r


def prepare_root(root):
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    dst = os.path.join(root, "data", "outright.parquet")
    if not os.path.exists(dst):
        os.link(REAL_RAW, dst)            # same volume: a hard link, no copy
    if os.path.getsize(dst) != os.path.getsize(REAL_RAW):
        raise SystemExit("REFUSED: the scratch outright.parquet differs in size from the real one")


# ------------------------------------------------------------------ the strata patch
def strata_from_edges(d, with_momentum, cp):
    """control_matched.strata (control_matched.py:38-54) with the quintile edges taken from cp instead of pd.qcut on d.
    pd.qcut(x, 5, labels=False, duplicates='drop') == pd.cut(x, its own edges, labels=False, include_lowest=True); the
    outer edges are opened to -inf / +inf, which changes nothing for values inside the train range."""
    import numpy as np, pandas as pd
    def lab_of(x, edges):
        e = np.array(edges, dtype=float)
        if len(e) != 6:
            raise SystemExit(f"REFUSED: {len(e)} quintile edges, 6 expected (a duplicate edge was dropped)")
        e[0] = -np.inf; e[-1] = np.inf
        return pd.cut(x, bins=e, labels=False, include_lowest=True).astype(np.int64)
    atr = d.atr.to_numpy()
    hour = d.ts.dt.tz_convert("America/Chicago").dt.hour.to_numpy()
    hb = hour // 4
    ok = np.isfinite(atr)
    aq = np.full(len(d), -1)
    ea = [float.fromhex(h) for h in cp["atr_edges_hex"]] if "atr_edges_hex" in cp else cp["atr_edges"]
    er = [float.fromhex(h) for h in cp["ret_edges_hex"]] if "ret_edges_hex" in cp else cp["ret_edges"]
    aq[ok] = lab_of(atr[ok], ea)
    lab = hb.astype(np.int64) * 100 + aq
    if with_momentum:
        c = d.close.to_numpy()
        ret = np.full(len(d), np.nan)
        ret[240:] = (c[240:] - c[:-240]) / np.maximum(atr[240:], 1e-9)
        rok = np.isfinite(ret)
        rq = np.full(len(d), -1)
        rq[rok] = lab_of(ret[rok], er)
        lab = lab * 10 + rq
    return lab


def edges_of(d):
    """The qcut edges control_matched.strata uses on d, same arrays and same dtype."""
    import numpy as np, pandas as pd
    atr = d.atr.to_numpy(); ok = np.isfinite(atr)
    _, ea = pd.qcut(atr[ok], 5, labels=False, duplicates="drop", retbins=True)
    c = d.close.to_numpy(); ret = np.full(len(d), np.nan)
    ret[240:] = (c[240:] - c[:-240]) / np.maximum(atr[240:], 1e-9)
    rok = np.isfinite(ret)
    _, er = pd.qcut(ret[rok], 5, labels=False, duplicates="drop", retbins=True)
    return {"atr_edges": [float(x) for x in ea], "ret_edges": [float(x) for x in er],
            "atr_edges_hex": [float(x).hex() for x in ea], "ret_edges_hex": [float(x).hex() for x in er]}


# ------------------------------------------------------------------ the build
def install(root, split, cp_file):
    os.environ["VWAP_ROOT"] = root
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, CODE)
    import loader, control_matched
    orig0 = loader.load
    def orig(*a, **k):
        d = orig0(*a, **k)
        STATE["n_load"] = int(len(d)); STATE["blocks"] = {str(b): int(n) for b, n in d.block.value_counts().items()}
        return d
    if split == "test":
        if not a15_registered():
            raise SystemExit("REFUSED: --split test needs 'REGISTERED A15 <sha>' in HANDOFF_section_28.md and the registered copy")
        if not certified_root(root):
            raise SystemExit("REFUSED: --split test builds only through seal_certify.py --build-seal (seal_use1_outputs\\seal_root "
                             "with a marker naming the current seal_T1_receipt.json)")
        def load(split_="train", **kw):
            return orig("test", unseal=True)
    elif split in ("trainA", "trainB"):
        keep = [b for b in loader.TRAIN_BLOCKS if b[0] == split[-1]]
        def load(split_="train", **kw):
            saved = loader.TRAIN_BLOCKS
            loader.TRAIN_BLOCKS = keep
            try:
                return orig("train")
            finally:
                loader.TRAIN_BLOCKS = saved
    else:
        def load(split_="train", **kw):
            return orig("train")
    loader.load = load
    if cp_file:
        cp = json.load(open(cp_file))
        def patched(d, with_momentum):
            STATE["strata_calls"] = STATE.get("strata_calls", 0) + 1
            return strata_from_edges(d, with_momentum, cp)
        control_matched.strata = patched
        STATE["strata_patch"] = patched


def build(root, split, cp_file, Ns):
    root = check_root(root)
    prepare_root(root)
    if os.path.normcase(os.path.realpath(os.path.join(root, "data", "dump"))) == os.path.normcase(os.path.realpath(REAL_DUMP)):
        raise SystemExit("REFUSED")
    t0 = time.time(); wsha = sha(__file__)            # hashed at START, so an edit during the run cannot be misattributed
    install(root, split, cp_file)
    sys.argv = [os.path.join(CODE, "dump_episodes.py")] + [str(n) for n in Ns]
    g = runpy.run_path(os.path.join(CODE, "dump_episodes.py"), run_name="__main__")
    used = g.get("strata")
    if cp_file is None:
        live = None                                     # nothing patched: no liveness claim is made (T1a's evidence is the reproduction)
        print("STRATA: patch OFF; liveness n/a (the evidence for this build is its exact reproduction)")
    else:
        bound = used is STATE.get("strata_patch"); calls = STATE.get("strata_calls", 0)
        live = bool(bound and calls == 2)
        print(f"STRATA: patch ON; dump_episodes' strata IS the patched function: {bound}; patched calls {calls} "
              f"(2 expected, valid for dump_episodes.py {sha(os.path.join(CODE, 'dump_episodes.py'), 16)}) -> {'PASS' if live else 'FAIL'}")
    rec = dict(split=split, N=Ns, root=root, cutpoints=cp_file, cutpoints_sha=sha(cp_file) if cp_file else None,
               n_load=STATE.get("n_load"), blocks=STATE.get("blocks"), strata_calls=STATE.get("strata_calls", 0), strata_live=live,
               wrapper_sha=wsha, dump_episodes_sha=sha(os.path.join(CODE, "dump_episodes.py")),
               control_matched_sha=sha(os.path.join(CODE, "control_matched.py")), loader_sha=sha(os.path.join(CODE, "loader.py")),
               raw_size=os.path.getsize(REAL_RAW), wall_s=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
               machine=dict(node=platform.node(), cpu=os.cpu_count(), python=platform.python_version()))
    json.dump(rec, open(os.path.join(root, "seal_build_receipt.json"), "w"), indent=1)
    if cp_file is not None and not live:
        print(f"BUILD FAILED: the strata patch was not live (receipt written to {root} with strata_live False)")
        return 1
    print(f"BUILD DONE: split {split}, N {Ns}, {rec['wall_s']}s; receipt written to {root}")
    return 0


def make_cutpoints(out):
    sys.path.insert(0, CODE)
    import loader, control_matched
    import numpy as np
    d = loader.load("train")
    cp = edges_of(d)
    cp.update(source="loader.load('train')", rows=int(len(d)), producer_sha=sha(__file__),
              control_matched_sha=sha(os.path.join(CODE, "control_matched.py")))
    ok = True
    for mom in (False, True):
        a = control_matched.strata(d, with_momentum=mom); b = strata_from_edges(d, mom, cp)
        same = bool(np.array_equal(a, b)); ok &= same
        print(f"  [{'PASS' if same else 'FAIL'}] T1b core: strata_train == control_matched.strata on the real train frame, "
              f"{'M2' if mom else 'M1b'}: {int((a != b).sum())} of {len(a):,} labels differ; {len(np.unique(a))} distinct labels")
    if not ok:
        print("NOT WRITTEN: the patch is not neutral on train"); return 1
    json.dump(cp, open(out, "w"), indent=1)
    print(f"wrote {out}  SHA-256 {sha(out)}  (atr edges {len(cp['atr_edges'])}, ret edges {len(cp['ret_edges'])})")
    return 0


# ------------------------------------------------------------------ comparisons
def cells_built(root):
    dd = os.path.join(root, "data", "dump")
    return sorted(f[:-8] for f in os.listdir(dd) if f.endswith(".parquet") and ".m2." not in f)


# The 45 columns of data\dump's cell files that dump_episodes.py does NOT write: they are added by the later covariate
# stages (attach_context, enrich, cov_phase; Tier 2). MEASURED 2026-10-14 on N60_hh_ATR_3.0 (149 vs 104 columns), 26.183.
LATER_STAGE_COLS = ['L_all', 'L_settled', 'bars_since_touch', 'prior_touches', 'slope_15', 'slope_60', 'is_rth', 'dow',
    'rv_close', 'day_pos', 'days_to_exp', 'anchor_spike', 'xa_ll_dPV', 'xa_ll_dPC', 'xa_hv_dPV', 'xa_hv_dPC', 'xa_session_dPV',
    'xa_session_dPC', 'htf_5m_4h_dPV', 'htf_5m_4h_dPC', 'htf_5m_1d_dPV', 'htf_5m_1d_dPC', 'htf_15m_4h_dPV', 'htf_15m_4h_dPC',
    'htf_15m_1d_dPV', 'htf_15m_1d_dPC', 'slope_15_xs', 'slope_60_xs', 'd_PC_xs', 'trend_15m_xs', 'trend_1h_xs', 'xa_ll_dPV_xs',
    'xa_ll_dPC_xs', 'xa_hv_dPV_xs', 'xa_hv_dPC_xs', 'xa_session_dPV_xs', 'xa_session_dPC_xs', 'htf_5m_4h_dPV_xs',
    'htf_5m_4h_dPC_xs', 'htf_5m_1d_dPV_xs', 'htf_5m_1d_dPC_xs', 'htf_15m_4h_dPV_xs', 'htf_15m_4h_dPC_xs', 'htf_15m_1d_dPV_xs',
    'htf_15m_1d_dPC_xs',
    'xa_hh_dPV', 'xa_hh_dPC', 'xa_hh_dPV_xs', 'xa_hh_dPC_xs']   # the hh cross-anchor columns (absent from hh cells themselves)


def compare_exact(root):
    """T1a / T1b: every column the build WRITES equals data\\dump's same column exactly (values, dtype, row order, column
    order); the columns data\\dump has beyond them must be exactly LATER_STAGE_COLS (cell files; the cross-anchor xa_<own>
    columns are absent for a cell's own anchor, so the expected set is LATER_STAGE_COLS intersected with data\\dump's)."""
    import pandas as pd
    dd = os.path.join(root, "data", "dump"); cells = cells_built(root); bad = 0
    for c in cells:
        for suf in (".parquet", ".m2.parquet"):
            a = pd.read_parquet(os.path.join(REAL_DUMP, c + suf)); b = pd.read_parquet(os.path.join(dd, c + suf))
            extra = [x for x in a.columns if x not in b.columns]; alien = [x for x in b.columns if x not in a.columns]
            allowed = set(LATER_STAGE_COLS) if suf == ".parquet" else set()
            want = sorted(x for x in a.columns if x in allowed)
            if alien or sorted(extra) != want or len(extra) != (45 if suf == ".parquet" else 0):
                bad += 1; print(f"  [FAIL] {c}{suf}: columns not written by the build {extra[:6]}..., expected {len(want)}; columns only in the build {alien}")
                continue
            if [x for x in a.columns if x in b.columns] != list(b.columns):     # relative column order (reviewer on v6, O4(c))
                bad += 1; print(f"  [FAIL] {c}{suf}: the written columns are in a different relative order"); continue
            try:
                pd.testing.assert_frame_equal(a[list(b.columns)], b, check_exact=True)
            except AssertionError as e:
                bad += 1; print(f"  [FAIL] {c}{suf}: {str(e).splitlines()[0][:160]}")
        ja = json.load(open(os.path.join(REAL_DUMP, c + ".counts.json"))); jb = json.load(open(os.path.join(dd, c + ".counts.json")))
        for k in TIMING_KEYS:
            ja.pop(k, None); jb.pop(k, None)
        if ja != jb:
            bad += 1; print(f"  [FAIL] {c}.counts.json: keys differing {[k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k)]}")
    print(f"[{'PASS' if bad == 0 and cells else 'FAIL'}] EXACT: {len(cells)} cells x (cell file, .m2 file, counts.json; "
          f"timing keys excluded) compared with data\\dump, {bad} mismatches")
    return 0 if bad == 0 and cells else 1


def n_bars_A(roots):
    """Number of block-A bars of the train load (= the length of an A-alone load). Taken from an A-alone build's receipt
    if one is given and carries it, else measured by an A-alone load (about a minute)."""
    for r in roots:
        p = os.path.join(r, "seal_build_receipt.json")
        if os.path.exists(p):
            rec = json.load(open(p))
            if rec.get("split") == "trainA" and rec.get("n_load"):
                return int(rec["n_load"]), "receipt"
    sys.path.insert(0, CODE)
    import loader
    saved = loader.TRAIN_BLOCKS; loader.TRAIN_BLOCKS = [b for b in saved if b[0] == "A"]
    try:
        n = int(len(loader.load("train")))
    finally:
        loader.TRAIN_BLOCKS = saved
    return n, "measured by an A-alone load"


def _eq(x, y):
    import numpy as np
    if x.dtype.kind == "f":
        return np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=0, atol=1e-12)
    return np.array_equal(x, y)


def compare_setups(root, blk, nA_root=None):
    """T1c-setups, the TWO-SIDED EDGE RULE (reviewer, 26.179), fixed before any comparison was run:
    A-alone: the setups are EXACTLY the joint-A setups with bar < n_A - 3T - 2 (rearm.candidates drops the last
      h_rec + 2 = 3T + 2 bars of a load, rearm.py:60, spec.py:33); the joint-A setups at or beyond that bar are the
      'dropped' set and must be absent from the fresh build; A's bar indices are the same in both loads (A comes first).
    B-alone: the setups equal the joint-B setups EXCEPT before the first setup that both chains keep after the close of
      the joint chain's last A setup (rearm.greedy_keep has no segment bound, rearm.py:84-94); every difference is listed.
    Equality is by (ts, side); compared columns as COLS; any other difference is FAIL."""
    import numpy as np, pandas as pd
    COLS = FAMILY_A_COLS + ["T"]
    dd = os.path.join(root, "data", "dump"); cells = cells_built(root); bad = 0; n = 0; dropped = 0; listed = []
    nA, how = n_bars_A([nA_root or root])
    print(f"  n_A = {nA:,} block-A bars ({how})")
    for c in cells:
        a = pd.read_parquet(os.path.join(REAL_DUMP, c + ".parquet"), columns=COLS + ["bar", "t_0.0"])
        b = pd.read_parquet(os.path.join(dd, c + ".parquet"), columns=COLS + ["bar", "t_0.0"])
        b = b[b.is_real == 1].copy()
        T = int(a["T"].iloc[0])
        if blk == "A":
            a = a[(a.is_real == 1) & (a.block == "A")]
            cut = nA - 3 * T - 2
            exp, gone = a[a.bar < cut], a[a.bar >= cut]
            dropped += len(gone)
            ka = set(zip(exp.ts, exp.side)); kb = set(zip(b.ts, b.side)); kg = set(zip(gone.ts, gone.side))
            if ka != kb or (kg & kb):
                bad += 1; print(f"  [FAIL] {c}: expected {len(ka):,}, fresh {len(kb):,}, missing {len(ka - kb)}, extra {len(kb - ka)}, "
                                 f"dropped-but-present {len(kg & kb)}"); continue
            x = exp.sort_values(["ts", "side"]).reset_index(drop=True); y = b.sort_values(["ts", "side"]).reset_index(drop=True)
            if not np.array_equal(x.bar.to_numpy(), y.bar.to_numpy()):
                bad += 1; print(f"  [FAIL] {c}: bar indices differ")
        else:
            jA = a[(a.is_real == 1) & (a.block == "A")]
            a = a[(a.is_real == 1) & (a.block == "B")]
            s = int(jA.bar.max()); fv = jA.loc[jA.bar == s, "t_0.0"].iloc[0]
            f = int(fv) if np.isfinite(fv) else -1
            close_at = s + (f if 1 <= f <= T else T)
            ka = set(zip(a.ts, a.side)); kb = set(zip(b.ts, b.side))
            inj = np.array([k in ka for k in zip(b.ts, b.side)], dtype=bool)
            common_after = b[(b.bar.to_numpy() + nA > close_at) & inj].ts
            sync = common_after.min() if len(common_after) else None
            diff = (ka ^ kb)
            late = [k for k in diff if sync is None or k[0] >= sync]
            if late:
                bad += 1; print(f"  [FAIL] {c}: {len(late)} differences at or after the chains' re-synchronisation"); continue
            for k in sorted(diff):
                listed.append((c, str(k[0]), int(k[1]), "fresh only" if k in kb else "joint only"))
            keep = ka & kb
            x = a[[k in keep for k in zip(a.ts, a.side)]].sort_values(["ts", "side"]).reset_index(drop=True)
            y = b[[k in keep for k in zip(b.ts, b.side)]].sort_values(["ts", "side"]).reset_index(drop=True)
        n += len(x)
        for col in COLS:
            if not _eq(x[col].to_numpy(), y[col].to_numpy()):
                bad += 1; print(f"  [FAIL] {c}: column {col} differs on the common setups")
    for row in listed:
        print(f"  [LISTED] {row[0]}: setup at {row[1]} side {row[2]}: {row[3]} (inside the chain hand-over window)")
    print(f"[{'PASS' if bad == 0 and cells else 'FAIL'}] T1c-SETUPS block {blk}: {len(cells)} cells, {n:,} common setups, "
          f"columns {COLS} equal (floats to 1e-12); A-end setups dropped by the rule {dropped:,}; listed hand-over differences "
          f"{len(listed)}; {bad} failures. Twin rows are NOT compared here.")
    return 0 if bad == 0 and cells else 1


def strata_check(cp_file):
    """T1c-strata (reviewer, 26.179): with the train cutpoints, a FRESH A-alone and B-alone load give the SAME M1b and M2
    labels and the same twin-pool flag as the joint train load on every bar eligible in both (matched by ts; the pool
    flag outside the A-alone load's last 2,000 bars); without the patch the labels differ on more than 1% of eligible
    bars (positive control). Reads bars only, no outcome."""
    import numpy as np, pandas as pd
    sys.path.insert(0, CODE)
    import loader, control_matched
    cp = json.load(open(cp_file)); ok = True
    J = loader.load("train")
    Jlab = {m: control_matched.strata(J, with_momentum=m) for m in (False, True)}
    Jts = J.ts.to_numpy(); Jel = (J.eligible.to_numpy() & np.isfinite(J.atr.to_numpy()))
    for B in ("A", "B"):
        saved = loader.TRAIN_BLOCKS; loader.TRAIN_BLOCKS = [b for b in saved if b[0] == B]
        try:
            d = loader.load("train")
        finally:
            loader.TRAIN_BLOCKS = saved
        pos = np.searchsorted(Jts, d.ts.to_numpy())
        if not np.array_equal(Jts[pos], d.ts.to_numpy()):
            print(f"  [FAIL] block {B}: fresh bars not all found in the joint load"); ok = False; continue
        el = d.eligible.to_numpy() & np.isfinite(d.atr.to_numpy()) & Jel[pos]
        for m in (False, True):
            p = strata_from_edges(d, m, cp); u = control_matched.strata(d, with_momentum=m); j = Jlab[m][pos]
            nd = int((p[el] != j[el]).sum()); share_u = float((u[el] != j[el]).mean())
            okm = nd == 0 and share_u > 0.01; ok &= okm
            print(f"  [{'PASS' if okm else 'FAIL'}] block {B} {'M2' if m else 'M1b'}: patched labels differ from joint on {nd:,} of "
                  f"{int(el.sum()):,} eligible bars (0 required); UNPATCHED differ on {100*share_u:.1f}% (> 1% required: the control is live)")
        n = len(d); tail = np.arange(n) >= n - 2000
        pj = J.eligible.to_numpy()[pos] & np.isfinite(J.atr.to_numpy()[pos]); pf = d.eligible.to_numpy() & np.isfinite(d.atr.to_numpy())
        npool = int((pj[~tail] != pf[~tail]).sum()); ok &= npool == 0
        print(f"  [{'PASS' if npool == 0 else 'FAIL'}] block {B}: eligible-and-finite-ATR flag (the twin pool's bar-level part) equal "
              f"outside the last 2,000 bars: {npool} differ")
    print("T1c-STRATA: PASS" if ok else "T1c-STRATA: FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------ self-test
def selftest():
    import numpy as np, pandas as pd, tempfile
    sys.path.insert(0, CODE)
    import control_matched
    ok = True
    def chk(name, cond, info=""):
        nonlocal ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")
    rng = np.random.default_rng(20261014)
    n = 20000
    ts = pd.Series(pd.date_range("2015-01-01", periods=n, freq="min", tz="UTC"))
    atr = rng.gamma(2.0, 0.5, n).astype(np.float32); atr[rng.random(n) < 0.02] = np.nan
    atr[:3000] = np.round(atr[:3000] * 4) / 4                     # ties on a 0.25 grid
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    d = pd.DataFrame(dict(ts=ts, atr=atr, close=close))
    cp = edges_of(d)
    for mom in (False, True):
        a = control_matched.strata(d, with_momentum=mom); b = strata_from_edges(d, mom, cp)
        chk(f"K1 strata_from_edges == control_matched.strata on a float32 toy with ties and NaN ({'M2' if mom else 'M1b'})",
            np.array_equal(a, b), f"{int((a != b).sum())} differ of {n}")
    d2 = d.copy(); d2["atr"] = (d2.atr * 3).astype(np.float32)          # a 'seal' far above the train range
    b2 = strata_from_edges(d2, False, cp) % 100
    chk("K2 values beyond the train range fall into the last quintile, not NaN / -1", (b2[np.isfinite(d2.atr.to_numpy())] >= 0).all()
        and (b2[np.isfinite(d2.atr.to_numpy())] == 4).mean() > 0.5, f"share in top quintile {(b2[np.isfinite(d2.atr.to_numpy())] == 4).mean():.2f}")
    e2 = float(np.float32(cp["atr_edges"][2])); on_edge = int((d.atr.to_numpy() == e2).sum())
    bad = dict(cp); bad["atr_edges_hex"] = list(cp["atr_edges_hex"])
    bad["atr_edges_hex"][2] = float(np.nextafter(np.float32(e2), np.float32(-1e9))).hex()
    if on_edge:
        chk("M1 an edge moved one float32 step down is caught (rows sit exactly on that edge)",
            not np.array_equal(control_matched.strata(d, False), strata_from_edges(d, False, bad)), f"{on_edge} rows on the edge")
    else:
        print(f"  [INFO] M1 skipped: no toy row sits exactly on edge 2")
    bad2 = dict(cp); bad2["atr_edges_hex"] = [float(x).hex() for x in np.array(cp["atr_edges"]) * 1.01]
    chk("M2 edges moved by 1% are caught", not np.array_equal(control_matched.strata(d, False), strata_from_edges(d, False, bad2)))
    for r_, why in ((HERE, "the project root"),):
        try:
            check_root(r_); chk(f"K3 --root {why} is refused", False)
        except SystemExit:
            chk(f"K3 --root {why} is refused", True)
    t = tempfile.mkdtemp()
    try:
        check_root(t); chk("K4 a fresh scratch root is accepted", True)
    except SystemExit:
        chk("K4 a fresh scratch root is accepted", False)
    sd = os.path.join(t, "stale"); os.makedirs(sd)
    cpy = os.path.join(sd, "AMENDMENT_A15_registered_copy_toy.md"); open(cpy, "w").write("toy amendment text\n"); h = sha(cpy)
    p_no = os.path.join(t, "no.md"); open(p_no, "w").write("## A15 draft\nnot registered\n")
    p_bare = os.path.join(t, "bare.md"); open(p_bare, "w").write("## A15\nREGISTERED A15 2026-10-xx\n")
    p_wrong = os.path.join(t, "wrong.md"); open(p_wrong, "w").write("REGISTERED A15 " + "0" * 64 + "\n")
    p_yes = os.path.join(t, "yes.md"); open(p_yes, "w").write("## A15\nREGISTERED A15 " + h + " 2026-10-xx\n")
    chk("K5 registration needs 'REGISTERED A15 <sha>' AND a registered copy with that sha",
        a15_registered(p_yes, sd) and not a15_registered(p_no, sd), f"(the real file now: {a15_registered()})")
    chk("M5 a bare 'REGISTERED A15' line, or a sha with no matching copy, is refused",
        not a15_registered(p_bare, sd) and not a15_registered(p_wrong, sd))
    chk("K5b a --split test root other than the certified seal root is refused", not certified_root(t))
    # K6 / M3-M4: the two-sided edge rule of compare_setups on a toy dump (REAL_DUMP redirected to a temp folder)
    global REAL_DUMP
    saved_dump = REAL_DUMP
    try:
        jd = os.path.join(t, "joint"); os.makedirs(jd)
        def frame(rows):
            f = pd.DataFrame(rows, columns=["bar", "blk", "side", "fill"])
            return pd.DataFrame({"ts": pd.to_datetime(f.bar * 60, unit="s", utc=True), "side": f.side.astype(np.int8),
                                 "dist_atr": 3.0 + f.bar / 1e4, "atr0": 1.0, "w_0.0": (f.fill > 0).astype(float), "mir_w": 0.0,
                                 "race": 1.0, "cens_T": False, "gap_at_T": 0.5, "day": f.bar // 1440, "block": f.blk,
                                 "is_real": np.int64(1), "T": np.int64(10), "bar": f.bar.astype(np.int64), "t_0.0": f.fill.astype(np.int64)})
        nA = 1000                                            # A bars 0..999; cut = 1000 - 32 = 968
        joint = frame([(100, "A", 1, 3), (500, "A", -1, -1), (970, "A", 1, 5), (1005, "B", 1, 2), (1100, "B", -1, 4), (1300, "B", 1, 1)])
        joint.to_parquet(os.path.join(jd, "TOY.parquet")); REAL_DUMP = jd
        def fresh(name, rows, blk):
            r = os.path.join(t, name); os.makedirs(os.path.join(r, "data", "dump"))
            fr = frame(rows); fr["block"] = blk
            if blk == "B":                                    # the same real times as the joint rows (bar offset by n_A)
                fr["ts"] = pd.to_datetime((fr.bar + nA) * 60, unit="s", utc=True); fr["dist_atr"] = 3.0 + (fr.bar + nA) / 1e4
                fr["day"] = (fr.bar + nA) // 1440
            fr.to_parquet(os.path.join(r, "data", "dump", "TOY.parquet"))
            json.dump({"split": "train" + blk, "n_load": nA if blk == "A" else 400}, open(os.path.join(r, "seal_build_receipt.json"), "w"))
            return r
        import contextlib, io
        def run(r, blk):
            with contextlib.redirect_stdout(io.StringIO()):
                return compare_setups(r, blk, nA_root=rA)
        rA = fresh("fA", [(100, "A", 1, 3), (500, "A", -1, -1)], "A")
        chk("K6 A-alone: the joint-A setups before n_A - 3T - 2 match and the one at bar 970 is dropped by rule -> PASS", run(rA, "A") == 0)
        rA2 = fresh("fA2", [(100, "A", 1, 3)], "A")
        chk("M3 A-alone missing an expected setup -> FAIL", run(rA2, "A") == 1)
        rB = fresh("fB", [(1, "B", 1, 1), (5, "B", 1, 2), (100, "B", -1, 4), (300, "B", 1, 1)], "B")   # B bar = joint bar - 1000
        chk("K7 B-alone: an extra setup inside the hand-over window (before the chains re-synchronise) is listed -> PASS", run(rB, "B") == 0)
        rB2 = fresh("fB2", [(5, "B", 1, 2), (100, "B", -1, 4)], "B")
        chk("M4 B-alone missing a joint-B setup AFTER re-synchronisation -> FAIL", run(rB2, "B") == 1)
    finally:
        REAL_DUMP = saved_dump
    print("ALL DETECTORS ALIVE" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main(a):
    if "--selftest" in a:
        return selftest()
    if "--strata-check" in a:
        return strata_check(a[a.index("--strata-check") + 1])
    if "--make-cutpoints" in a:
        return make_cutpoints(a[a.index("--make-cutpoints") + 1])
    if "--compare" in a:
        root = check_root(a[a.index("--compare") + 1])
        if "--exact" in a:
            return compare_exact(root)
        if "--setups" in a:
            nr = a[a.index("--nA-from") + 1] if "--nA-from" in a else None
            return compare_setups(root, a[a.index("--setups") + 1], nr)
        print(__doc__); return 2
    if "--root" in a and "--split" in a:
        split = a[a.index("--split") + 1]
        if split not in SPLITS:
            raise SystemExit(f"--split must be one of {SPLITS}")
        cp = a[a.index("--cutpoints") + 1] if "--cutpoints" in a else None
        Ns = [int(x) for x in a[a.index("--N") + 1:] if x.isdigit()] if "--N" in a else [60]
        if selftest() != 0:
            return 1
        return build(a[a.index("--root") + 1], split, cp, Ns)
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
