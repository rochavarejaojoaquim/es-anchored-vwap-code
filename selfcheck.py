"""
selfcheck.py -- prove the code and data are sound before trusting a result.

    python selfcheck.py            A code health + gates (synthetic, ~2 min) + B data
    python selfcheck.py full       adds C, the numeric detectors on REAL data (~10 min)

Every check prints PASS or FAIL. Nothing downstream should be trusted while
any FAIL remains (rule D3). Every detector here was mutation-tested: fed a
deliberately broken target, it FAILS (python code\\gates.py mutate; C113).
Rebuilt 2026-09-21 for the re-arm dump (sec. 21.18).
"""
import glob, os, re, sys, json, importlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
DUMP = os.path.join(HERE, "data", "dump")
sys.path.insert(0, CODE)
FAILS = []


def chk(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<52} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def head(t):
    print(); print("=" * 78); print(t); print("=" * 78)


head("A. CODE HEALTH")
bad = []
for f in sorted(glob.glob(os.path.join(CODE, "*.py"))):
    m = os.path.basename(f)[:-3]
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}")
chk("every module imports", not bad, "; ".join(bad[:3]))
src = {os.path.basename(f): open(f, encoding="utf8", errors="ignore").read()
       for f in glob.glob(os.path.join(CODE, "*.py"))}
live = {k: "\n".join(l.split("#")[0] for l in v.splitlines()) for k, v in src.items()}
chk("no mangled fXXX_DIR f-strings", not [k for k, s in src.items() if re.search(r"\bf[A-Z_]+_DIR", s)])
chk("no forward-slash filename regexes", not [k for k, s in src.items() if re.search(r're\.match\(r"\.\*/', s)])
chk("no hard-coded container paths", not [k for k, s in src.items() if "/home/sandbox" in s])
# gates.py is exempt by name: it writes ONE raw npz into a temp folder on
# purpose, to prove a cache without a spec stamp is refused.
direct = [k for k, s in live.items() if k not in ("safeio.py", "gates.py")
          and re.search(r"\.to_parquet\(|savez", s)]
chk("every data write goes through safeio (C112)", not direct, ", ".join(direct))
globs = [k for k, s in live.items() if k != "loader.py" and re.search(r"glob\([^)]*DUMP", s)]
chk("only loader.cell_files lists the dump (C3)", not globs, ", ".join(globs))
legacy = [k for k in ("retwin.py", "rescore_all.py", "recompute_cores.py",
                      "core_significance.py", "episodes_v3.py") if k in src]
chk("legacy files only in code\\legacy\\", not legacy, ", ".join(legacy))
import gates
res = gates.run_all_gates()
chk("ALL synthetic gates (code\\gates.py)", all(ok for _, ok in res),
    f"{sum(ok for _, ok in res)}/{len(res)} -- failing: {[n.split()[0] for n, ok in res if not ok]}")

head("B. DATA INTEGRITY")
from loader import cell_files
import pyarrow.parquet as pq
from collections import Counter
if not os.path.isdir(DUMP):
    chk("data\\dump exists", False, "no dump yet -- run the rebuild"); cells = []
else:
    cells = cell_files("N*", DUMP)
cols, unread = {}, []
for f in cells:
    try:
        cols[f] = pq.read_schema(f).names
    except Exception:
        unread.append(os.path.basename(f))
chk("all cells readable", not unread, f"{len(cells)} cells; bad: {unread[:3]}")
cc = Counter(len(v) for v in cols.values())
chk("all cells have the same column count", len(cc) == 1, f"counts: {dict(cc)}")
old_rule = [os.path.basename(f) for f, v in cols.items() if "race" not in v]
chk("no OLD-RULE cell in data\\dump", not old_rule, f"{len(old_rule)} old cells, e.g. {old_rule[:2]}")
exp = {(N, a, t, e) for N in (60, 100, 160, 200, 240) for a in ("hh", "ll", "hv", "session")
       for t, g in (("ATR", (3.0, 4.0, 6.0, 8.0, 12.0)), ("SD", (1.0, 1.5, 2.0, 3.0))) for e in g}
got = set()
for f in cells:
    m = re.match(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)\.parquet", os.path.basename(f))
    if m:
        got.add((int(m.group(1)), m.group(2), m.group(3), float(m.group(4))))
chk("full grid present", got == exp, f"{len(got)} of 180")
m2 = [f[:-8] + ".m2.parquet" for f in cells]
cj = [f[:-8] + ".counts.json" for f in cells]
chk("every cell has its M2 twins and counts", all(map(os.path.exists, m2 + cj)),
    f"{sum(map(os.path.exists, m2))} m2, {sum(map(os.path.exists, cj))} counts")
chk("no half-written .tmp files", not glob.glob(os.path.join(DUMP, "*.tmp")))
import safeio
from spec import SPEC_VERSION
cov = glob.glob(os.path.join(HERE, "covcache", "*.npz"))
stale = [os.path.basename(p) for p in cov if safeio.npz_version(p) != SPEC_VERSION]
chk("covcache built under the current spec", len(cov) == 20 and not stale,
    f"{len(cov)} files (ref 20), stale: {stale[:2]}")
htf = glob.glob(os.path.join(HERE, "htf_*_*m_*.npz"))
stale = [os.path.basename(p) for p in htf if safeio.npz_version(p) != SPEC_VERSION]
chk("16 HTF files (own-rule, R4) under the current spec", len(htf) == 16 and not stale,
    f"{len(htf)} files, stale: {stale[:2]}")

head("C. NUMERIC DETECTORS (real data)")
if len(sys.argv) > 1 and sys.argv[1] == "full":
    import pandas as pd
    from loader import load, trade_day, rolling_argextreme
    from pipeline import vwap_and_sd
    from spec import t_stop, H_REC_MULT
    d = load("train")
    chk("seal intact (no 2018-2022 bar)",
        not ((d.ts >= pd.Timestamp("2018-01-01", tz="UTC")) &
             (d.ts < pd.Timestamp("2022-01-01", tz="UTC"))).any(), f"{len(d):,} bars")
    chk("bar count matches reference", len(d) == 4248867, f"{len(d):,} (ref 4,248,867)")
    chk("every segment has a contract expiry (R7)", bool((d.exp_day > 0).all()),
        f"{d.seg.nunique()} segments")
    gates.RESULTS.clear()
    gates.g_anchor(d); gates.g_vwap(d); gates.g_daypos(d)
    for n_, ok in gates.RESULTS:
        chk("real data: " + n_.split(" ", 1)[1][:44], ok)
    td = trade_day(d.ts); blk = d.block.to_numpy()
    vio = Counter(); ntw = 0; nest = 0; rearm_bad = 0
    for f in cells:
        x = pd.read_parquet(f, columns=["bar", "is_real", "src_bar", "block", "T",
                                        "w_0.0", "w_0.25", "scheme"])
        m2f = f[:-8] + ".m2.parquet"                  # a missing one already FAILED in B
        m2x = (pd.read_parquet(m2f, columns=["bar", "src_bar", "w_0.0", "w_0.25", "T"])
               if os.path.exists(m2f) else pd.DataFrame(columns=["bar", "src_bar", "w_0.0", "w_0.25", "T"]))
        if not len(x):
            continue
        T = int(x["T"].iloc[0]); h = H_REC_MULT * T
        r = x[x.is_real == 1]
        for tw in (x[x.is_real == 0], m2x):
            j = tw.bar.to_numpy().astype(np.int64); i = tw.src_bar.to_numpy().astype(np.int64)
            vio["same trade date"] += int((td[j] == td[i]).sum())
            vio["within h_rec"] += int((np.abs(j - i) <= h).sum())
            vio["cross-block"] += int((blk[j] != blk[i]).sum())
            ntw += len(tw)
        for q in (r, x[x.is_real == 0], m2x):
            h0 = (q["w_0.0"] > 0) & (q["w_0.0"] <= T); h1 = (q["w_0.25"] > 0) & (q["w_0.25"] <= T)
            nest += int((h0 & ~h1).sum())
        b = r.bar.to_numpy(); w = r["w_0.0"].to_numpy()
        close_at = b + np.where((w >= 1) & (w <= T), w, T)
        rearm_bad += int((b[1:] <= close_at[:-1]).sum())
    chk("twins: other trade date, > h_rec away, same block", sum(vio.values()) == 0,
        f"{ntw:,} twins (M1b+M2) | {dict(vio) or 0}")
    chk("hit sets nested across tau (setups + both twins)", nest == 0, f"{nest} violations")
    chk("re-arm: no setup before the previous one closed", rearm_bad == 0,
        f"{rearm_bad} violations over {len(cells)} cells")
    ref_f = os.path.join(HERE, "reference_rearm.json")
    if not os.path.exists(ref_f):
        chk("reference cell == slow reference (R9)", False,
            "no reference_rearm.json -- run: python code\\rearm_reference.py")
    else:
        from rearm import bars_hash
        R = json.load(open(ref_f))
        x = pd.read_parquet(os.path.join(DUMP, R["cell"] + ".parquet"))
        r = x[x.is_real == 1]
        _N, _a, _tg, _e = R["cell"].split("_")
        T = t_stop(_tg, float(_e))
        hit = float(((r["w_0.0"] > 0) & (r["w_0.0"] <= T)).mean()) if len(r) else float("nan")
        same_hit = (hit == R["hit_rate_tau0"]) or (np.isnan(hit) and R["hit_rate_tau0"] != R["hit_rate_tau0"])
        ok = (len(r) == R["n_setups"] and bars_hash(r.bar.to_numpy()) == R["bars_sha"] and same_hit)
        chk(f"{R['cell']} == slow reference (R9)", ok,
            f"setups {len(r):,} vs {R['n_setups']:,} | hit {hit:.4f} vs "
            f"{R['hit_rate_tau0']:.4f} | near_eps {R['near_eps']} one_bar {R['one_bar']}")
else:
    print("  (skipped -- run `python selfcheck.py full` to include these)")

print(); print("=" * 78)
print("ALL CHECKS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
print("=" * 78)
