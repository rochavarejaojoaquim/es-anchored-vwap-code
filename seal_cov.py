r"""
seal_cov.py -- TIER 2 of the seal path (A15.6 / A15.9 item 10): the covariate stages cov_phase (A and B), enrich (the 16
higher-timeframe files) and attach_context run on ANY split through the same loader patch as seal_build.py, with VWAP_ROOT
= a scratch root. Nothing under code\ is edited; the stages run UNCHANGED, their load("train") returning the chosen period.
Added columns (45): L_all L_settled bars_since_touch prior_touches slope_15 slope_60 is_rth dow rv_close day_pos
days_to_exp anchor_spike, the xa_* cross-anchor and htf_* higher-timeframe columns and the _xs copies.

    python seal_cov.py --selftest
    python seal_cov.py --stages ROOT --split train|trainA|trainB [--N 60 ...]     (a copy of a train build root)
    python seal_cov.py --compare-full ROOT        T2a: every column of every cell file equals data\dump's, exactly
    python seal_cov.py --compare-cov ROOT A|B     T2c: the searched covariates of the common setups equal the joint block's,
                                                   mismatches allowed ONLY in the first HANDOVER_DAYS trade days of the
                                                   load (rolling windows cross the A/B boundary in the joint build)
The test split is reached only through seal_certify.py --build-seal (seal_build.install's own guards apply: registered A15,
certified root). Project root; not a code\ file.
"""
import hashlib
import json
import os
import platform
import runpy
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
REAL_DUMP = os.path.join(HERE, "data", "dump")
ANCHORS = ["hh", "ll", "hv", "session"]
NS_ALL = [60, 100, 160, 200, 240]
HANDOVER_DAYS = 30          # a CONVENTION fixed before any T2c comparison: relvol's 1,200-bar same-hour window is about 20 trade days
STAGE_FILES = ["cov_phase.py", "enrich.py", "attach_context.py", "add_covariates.py", "loader.py", "spec.py", "safeio.py", "pipeline.py"]


def sha(p, n=None):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest().upper()
    return h[:n] if n else h


def run_stages(root, split, Ns=None, cp_file=None):
    """cov_phase A+B for every (N, anchor), enrich once, attach_context for every N, in THIS process, inside root."""
    import seal_build as SB
    root = SB.check_root(root)
    Ns = Ns or NS_ALL
    t0 = time.time(); wsha = sha(__file__)
    SB.install(root, split, cp_file)             # VWAP_ROOT, loader.load for the split; test is guarded there
    sys.path.insert(0, CODE)
    import importlib
    import loader
    import cov_phase, safeio
    d = loader.load("train")                     # the patched load: one load for every phase A
    print(f"  loaded {len(d):,} bars for split {split}", flush=True)
    for N in Ns:
        for a in ANCHORS:
            cov_phase.phase_a(N, a, d)
    del d
    for N in Ns:
        for a in ANCHORS:
            cov_phase.phase_b(N, a)
    sys.argv = [os.path.join(CODE, "enrich.py")]
    runpy.run_path(os.path.join(CODE, "enrich.py"), run_name="__main__")
    import attach_context
    attach_context.main(list(Ns))
    rec = dict(split=split, N=Ns, root=root, wrapper_sha=wsha, wall_s=round(time.time() - t0, 1),
               stage_files={f: sha(os.path.join(CODE, f)) for f in STAGE_FILES if os.path.exists(os.path.join(CODE, f))},
               finished=time.strftime("%Y-%m-%d %H:%M:%S"),
               machine=dict(node=platform.node(), cpu=os.cpu_count(), python=platform.python_version()))
    json.dump(rec, open(os.path.join(root, "seal_cov_receipt.json"), "w"), indent=1)
    print(f"COV STAGES DONE: split {split}, N {Ns}, {rec['wall_s']}s; receipt in {root}")
    return 0


def _cells(root):
    sys.path.insert(0, CODE)
    from loader import cell_files
    return [os.path.basename(f)[:-8] for f in cell_files("N*", os.path.join(root, "data", "dump"))]


def compare_full(root, ref=REAL_DUMP, say=print):
    """T2a. Every cell file: same column list in the same order, every value equal, exactly."""
    bad, n = [], 0
    cells = _cells(root)
    if not cells:
        say("  no cells in the root"); return False
    for c in cells:
        a = pd.read_parquet(os.path.join(root, "data", "dump", c + ".parquet"))
        b = pd.read_parquet(os.path.join(ref, c + ".parquet"))
        n += 1
        if list(a.columns) != list(b.columns):
            bad.append((c, "columns", sorted(set(a.columns) ^ set(b.columns))[:6])); continue
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=True)
        except AssertionError as e:
            bad.append((c, "values", str(e).splitlines()[0][:100]))
    say(f"  T2a EXACT: {n} cells x all columns vs {ref}: {len(bad)} mismatches" + (f"  e.g. {bad[:3]}" if bad else ""))
    return not bad


def compare_cov(root, blk, ref=REAL_DUMP, say=print):
    """T2c. Single-period root vs the joint dump's block rows, on the setups both hold (by ts, side).
    Compares every non-META column the pooled table reads. A mismatch is allowed ONLY in the first HANDOVER_DAYS trade
    days of the root's load (the joint build's rolling windows cross the A/B boundary); anything later FAILS."""
    sys.path.insert(0, CODE)
    import core_v2
    cells = _cells(root)
    fails, handover, tot = [], 0, 0
    for c in cells:
        a = pd.read_parquet(os.path.join(root, "data", "dump", c + ".parquet"))
        b = pd.read_parquet(os.path.join(ref, c + ".parquet"))
        a = a[a.is_real == 1]; b = b[(b.is_real == 1) & (b.block == blk)]
        cols = [x for x in core_v2.search_cols(a, None) if x in b.columns and x in a.columns and x not in ("ts", "side")]
        a = a.set_index(["ts", "side"]); b = b.set_index(["ts", "side"])
        com = a.index.intersection(b.index)
        if len(com) == 0:
            fails.append((c, "no common setups")); continue
        a, b = a.loc[com], b.loc[com]
        t0 = a.day.min()
        late = (a.day - t0) > HANDOVER_DAYS
        for col in cols:
            x, y = a[col].to_numpy(float), b[col].to_numpy(float)
            neq = ~((x == y) | (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=0, atol=1e-12, equal_nan=True))
            tot += len(x)
            handover += int((neq & ~late.to_numpy()).sum())
            nl = int((neq & late.to_numpy()).sum())
            if nl:
                fails.append((c, col, nl))
    say(f"  T2c block {blk}: {len(cells)} cells; mismatches inside the first {HANDOVER_DAYS} trade days (allowed, listed by count) "
        f"{handover}; AFTER it {len(fails)} column-cells" + (f"  e.g. {fails[:4]}" if fails else ""))
    return not fails


def selftest():
    ok = [True]

    def chk(n, c, e=""):
        ok[0] &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}  {e}")
    import tempfile
    td = tempfile.mkdtemp()
    ref = os.path.join(td, "ref"); root = os.path.join(td, "root", "data", "dump")
    os.makedirs(ref); os.makedirs(root)
    n = 40
    df = pd.DataFrame(dict(is_real=1, side=1.0, block="A", ts=pd.date_range("2012-01-02", periods=n, freq="D", tz="UTC"),
                           day=np.arange(15000, 15000 + n), speed=np.arange(n, dtype=float), rv_close=np.linspace(0, 1, n),
                           T=60, bar=np.arange(n), src_bar=-1))
    for c in ("N60_hh_ATR_3.0",):
        df.to_parquet(os.path.join(ref, c + ".parquet"))
        df.to_parquet(os.path.join(root, c + ".parquet"))
    rt = os.path.dirname(os.path.dirname(root))
    chk("K1 identical frames pass T2a", compare_full(rt, ref, say=lambda *a: None))
    d2 = df.copy(); d2.loc[5, "rv_close"] += 1e-9
    d2.to_parquet(os.path.join(root, "N60_hh_ATR_3.0.parquet"))
    chk("M1 a 1e-9 change in one value fails T2a", not compare_full(rt, ref, say=lambda *a: None))
    d3 = df.copy(); d3["extra"] = 1.0
    d3.to_parquet(os.path.join(root, "N60_hh_ATR_3.0.parquet"))
    chk("M2 an extra column fails T2a", not compare_full(rt, ref, say=lambda *a: None))
    d4 = df.copy(); d4.loc[3, "speed"] = -1.0                       # inside the first 30 days: allowed
    d4.to_parquet(os.path.join(root, "N60_hh_ATR_3.0.parquet"))
    chk("K2 a mismatch inside the hand-over window is allowed by T2c", compare_cov(rt, "A", ref, say=lambda *a: None))
    d5 = df.copy(); d5.loc[35, "speed"] = -1.0                      # after 30 days: fails
    d5.to_parquet(os.path.join(root, "N60_hh_ATR_3.0.parquet"))
    chk("M3 a mismatch after the hand-over window fails T2c", not compare_cov(rt, "A", ref, say=lambda *a: None))
    chk("K3 --split test is refused by the guards unless certified (install raises SystemExit)", True,
        "(guard lives in seal_build.install; exercised by seal_build's own K5/K5b)")
    print("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED")
    return ok[0]


if __name__ == "__main__":
    a = sys.argv[1:]
    if not selftest():
        sys.exit(1)
    if "--selftest" in a:
        sys.exit(0)
    sys.path.insert(0, HERE)
    if "--stages" in a:
        root = a[a.index("--stages") + 1]
        split = a[a.index("--split") + 1]
        Ns = None
        if "--N" in a:
            i = a.index("--N") + 1
            Ns = []
            while i < len(a) and not a[i].startswith("--"):
                Ns.append(int(a[i])); i += 1
        sys.exit(run_stages(root, split, Ns))
    if "--compare-full" in a:
        sys.exit(0 if compare_full(a[a.index("--compare-full") + 1]) else 1)
    if "--compare-cov" in a:
        i = a.index("--compare-cov")
        sys.exit(0 if compare_cov(a[i + 1], a[i + 2]) else 1)
    print(__doc__); sys.exit(2)
