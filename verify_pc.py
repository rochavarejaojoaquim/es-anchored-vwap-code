r"""
verify_pc.py -- ONE read-only check of the whole project folder.

    cd <project folder>
    python verify_pc.py

It never changes code\ or data\. The ONLY file it writes is verify_log.txt
(one line appended per run). Every line prints PASS, FAIL, WARN or INFO.
  PASS/FAIL  a check against a KNOWN number (rule D3)
  WARN       not wrong, but will change how a later step behaves
  INFO       harmless; reported so nothing is hidden
Version of 2026-09-21 (re-arm rebuild). It understands three states:
  INSTALLED  new code, no new dump yet          -> data section is INFO only
  PARTIAL    rebuild under way                  -> counts reported
  REBUILT    180 cells + 180 M2 + 180 counts    -> every data check applies
"""
import glob, hashlib, inspect, os, shutil, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")
DUMP = os.path.join(DATA, "dump")
FAILS, WARNS = [], []

KNOWN = {
    "code/_walkextra.py": "1262ff81d4dd294b61f11ef193663208ca738d80019e7ff0f2741aed623272a8",
    "code/add_covariates.py": "39deb415aba0df26fab0bdaedb77c9e8f6e8f09018bd585ce3fefc9dd35fd22a",
    "code/attach_context.py": "02b74ecee3a1cbdc1e98fafc976faaabbe331d9e691f3252944d217e3b43b7fb",
    "code/compare_old_new.py": "52f21ec23acb7d4297e2a080a32d16628174484c591b5259a151c41cd9ef0227",
    "code/control_matched.py": "16c308d039bb6d0b94c306cfd2819b6dfed055cc009013920a5a6660f172317c",
    "code/controls_fast.py": "c3da6c8923d8128be5a6adb2821d4c22ae321348986d676b0e3cf4a81fc73944",
    "code/core_v2.py": "52d08a94e7b3025fd76bb84f3127f1530054e60f0992c213baf3279e82924e6d",
    "code/cov_phase.py": "625e0e68fe1d3be0656c9df65e514c6bd40dbed852451ae2370fed772ae95355",
    "code/dump_episodes.py": "8499f5f16096c08b2036447f2aa2755eab98d4f33945231139871aa8dcd619a5",
    "code/enrich.py": "5bf14268df68495c1b66edc115a8ce256e1f09469817abd015a5c9fd78ba4824",
    "code/floor_v2.py": "ab3ab410fee18ddd055fe35152d5497924a8adc99687401f193ac09cc2d782f8",
    "code/gates.py": "90f46fce72aa294e1f19596725b5e5e0d63b621f30d41afc4b84fff465c02c52",
    "code/interactions.py": "f3d8e00655c98eebd78c44464d568b1d3b077e29322fc0f06290800819b33eb2",
    "code/linearity.py": "58f26f826bc248d5265d82b2408fd59c08b8f9022e8d96fe6d93eb915dec1a14",
    "code/loader.py": "e7baa6d3e4f91caffe675059adfd27e8d8f8d302f4c61f369965896749957f49",
    "code/mono_sweep.py": "f2d5783cb829a1735c388dd96888f217317ad9620f97c40fbb281e7fd69899a2",
    "code/peak_mem.py": "d07dade012a9495478eca04b15f9ee714adc1be6afec9ace0f9ceaf01ed7146b",
    "code/pipeline.py": "2612328c68cef65b6f47936bf8c5c497bb21dc58a85ad4f02c785e87c193e079",
    "code/presignal.py": "99899e7c393cdfdd7097da6f0a112bb0fc3f02cb443e6db202419238cbe761f5",
    # Re-stamped 2026-10-01 for the SYMMETRIC DRAW (section 25 amendment 7). The
    # pre-edit files are stale\pseudo_{vwap,gates}_before_symmetric_2026-10-01.py,
    # hash-verified against the PREVIOUS entries before anything was changed:
    #   pseudo_gates.py was 6e5540df38635de49e4cb463c35a9b88fa0ee76f91521528f6ed8681d346210c
    #   pseudo_vwap.py  was 748e2d8e6aec07661e2fa40b001dd297bcf546c008b603db3df3dd2ac25ce0ad
    # THE CODE-SET HASH ROTATES WITH THIS EDIT: 8bc9b3e3ff14 is superseded. Every
    # stamp from the next rebuild onward carries the new value, which this script
    # computes live (line ~222) and writes to verify_log.txt.
    # Re-stamped again 2026-10-02: the manifest-write race fix (see write_manifest).
    # a2a005fa... was the symmetric-draw-only version, never run to completion.
    # REVERTED 2026-10-02 (HANDOFF 26.57): the symmetric-draw rebuild was REJECTED
    # (26.44), so both files are back to their pre-symmetric bytes and the code-set
    # is 8bc9b3e3ff14 again -- the one the receipts and data\dump's results carry.
    # The rejected versions, which built rebuild_pseudo_2026-09-30\data\dump
    # (code-set 57154fc831ae, used by 26.46's out-of-sample validation), are kept:
    #   pseudo_gates.py was e0df1ebf2bb7f8baddfd099daf9fa08d01f67706a2c8d46bead5fd4e9a205ffd
    #                   = stale\pseudo_gates_WIP_symmetric_2026-10-01.py
    #   pseudo_vwap.py  was 246c971ccb682425d7c1f8291271795e7874d22d6d6ec30322085c3d92fe8e3b
    #                   = stale\pseudo_vwap_symmetric_racefix_REJECTED_2026-10-02.py
    "code/pseudo_gates.py": "6e5540df38635de49e4cb463c35a9b88fa0ee76f91521528f6ed8681d346210c",
    "code/pseudo_vwap.py": "748e2d8e6aec07661e2fa40b001dd297bcf546c008b603db3df3dd2ac25ce0ad",
    "code/rearm.py": "5f82e25c7c6e5a48f15432647cf3930e5a31a700bf1cf3f4a325062ae48ba218",
    "code/rearm_reference.py": "d03b98cd9f2aedc487b301e93407b06d8c1ea5279b7d4c0be3a4b502468a713b",
    "code/safeio.py": "84c2b49eee7f4dfa298644064153e9c599f82a7b017d5ad47d1aed2851057725",
    "code/sd_precision.py": "89f12f013b11ba724c363ba5031778377bc822bc7e112420fa08e66e3d646ac9",
    "code/spec.py": "d1a26c548ddf3f89ef0d319a344b75944a3c20df59ee70201e324ec434630b6b",
    "code/synthdata.py": "2d34f7e476287548e217f295d841b6f5f70348e4808e047a3f873eba4d17d8d9",
    "code/volume_features.py": "22bdd0f9947b4869a0a8610ef076c4b5bc6a61b463af63ae14d819f9d50a6a5f",
    "code/walk_fast.py": "3b1361e93f2ac71843b39ccd175ad00acd3dfd4f494e7abb57c53326a1ba452a",
    "code/within_core.py": "b869027caf3974ceb15bb9a8ef7625a59150d7ff8d4366e86d4aee475a08d4ab",
    "requirements.txt": "c1feb246a32327395562bd0ea71f2ac39b2e2d91f4553884eaa1a9f6cb6dbd6a",
    "run_all.py": "048d2690aff758385eb798a85cc98361f403b666b960e91a80a0592014de93b6",
    "selfcheck.py": "2df2e450400b363462631e1ad8d7e513a3b3441d48a140f127abe65c91876b36",
    # Root-level METHOD-BEARING scripts, added 2026-10-01 (concern C135, HANDOFF 26.28).
    # Only scripts that are FROZEN and produce a reported number belong here. Scripts
    # still under active development are deliberately left out: a tripwire that fires on
    # legitimate work gets silenced, and a reflex re-stamp is indistinguishable from an
    # unnoticed change. Their attribution comes from the producer_sha256 inside each
    # result's receipt instead (gate_receipts.py), which binds script to RESULT rather
    # than to "current". Adding keys here does NOT rotate the code-set hash: line 222
    # globs code\*.py only.
    "pseudo_null_pair.py": "19141a5237c0ccc0f8970db66da0b1f6a6b62f63c8656d3292e6c77d749c32cf",
    "pseudo_power.py": "6a804dc83fede4bcbf7fe130cebc6b10f65f2c2ab3c94d2d14d326410065f4d7"
}


def out(tag, name, detail=""):
    print(f"  [{tag}]  {name:<46} {detail}")
    if tag == "FAIL": FAILS.append(name)
    if tag == "WARN": WARNS.append(name)


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def head(t):
    print(); print("=" * 78); print(t); print("=" * 78)


def gb(path):
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try: tot += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return tot / 1e9


head("0. WHERE THIS IS RUNNING")
print(f"  folder : {ROOT}")
print(f"  python : {sys.version.split()[0]}  ({sys.executable})")
ok = os.path.isdir(CODE) and os.path.isdir(DATA)
out("PASS" if ok else "FAIL", "code\\ and data\\ sit beside this script",
    "" if ok else "run it from <project folder>")
if not ok:
    sys.exit(2)
try:
    import pandas, numpy, pyarrow
    out("INFO", "library versions", f"pandas {pandas.__version__} | numpy {numpy.__version__} | pyarrow {pyarrow.__version__}")
except Exception as e:
    out("FAIL", "pandas / numpy / pyarrow import", f"{type(e).__name__}")

head("1. CODE IS EXACTLY THE VERIFIED VERSION (SHA-256 fingerprints)")
bad = []
for rel, h in KNOWN.items():
    p = os.path.join(ROOT, *rel.split("/"))
    if not os.path.exists(p):
        bad.append(f"MISSING {rel}")
    elif sha(p) != h:
        bad.append(f"CHANGED {rel}")
out("PASS" if not bad and KNOWN else "FAIL", f"{len(KNOWN)} files match the known fingerprints",
    "; ".join(bad[:4]))
extra = sorted(os.path.basename(f) for f in glob.glob(os.path.join(CODE, "*.py"))
               if "code/" + os.path.basename(f) not in KNOWN)
out("PASS" if not extra else "FAIL", "no unexpected .py files in code\\", ", ".join(extra))
leg = [f for f in ("retwin.py", "rescore_all.py", "recompute_cores.py", "core_significance.py")
       if os.path.exists(os.path.join(CODE, "legacy", f))]
out("PASS" if len(leg) == 4 else "FAIL", "the 4 legacy files live in code\\legacy\\", f"{len(leg)} of 4")

head("2. PYTHON WILL RUN THIS COPY (not a stray duplicate)")
sys.path.insert(0, CODE)
try:
    import core_v2
    where = os.path.dirname(os.path.abspath(core_v2.__file__))
    out("PASS" if os.path.normcase(where) == os.path.normcase(CODE) else "FAIL",
        "core_v2 imports from code\\", where)
    out("PASS" if core_v2.TAU == "0.0" else "WARN", "core_v2 TAU (VWAP_TAU)", f"{core_v2.TAU}  (primary is 0.0)")
    # Generalises the core_v2 check above to EVERY module in code\ (2026-10-01).
    # A root-level copy of a code\ module shadows it for any script that does not
    # sys.path.insert(0, CODE) before importing. pseudo_gates.py was exactly that:
    # byte-identical today, unfingerprinted, and free to drift unnoticed. Checked by
    # file existence, never by import, so nothing is executed to run this test.
    dupes = sorted(os.path.basename(f) for f in glob.glob(os.path.join(CODE, "*.py"))
                   if os.path.exists(os.path.join(ROOT, os.path.basename(f))))
    out("PASS" if not dupes else "FAIL", "no code\\ module is shadowed by a root copy",
        ", ".join(dupes) + "  <- rename into stale\\" if dupes else "")
    import spec
    out("INFO", "spec version", spec.SPEC_VERSION)
except Exception as e:
    out("FAIL", "code imports", f"{type(e).__name__}: {e}")

head("3. DISK")
du = shutil.disk_usage(ROOT)
out("INFO", "free space on this drive", f"{du.free / 1e9:,.1f} GB of {du.total / 1e9:,.0f} GB")
for nm in ("dump", "dump_old_rule"):
    p = os.path.join(DATA, nm)
    if os.path.isdir(p):
        out("INFO", f"size of data\\{nm}\\", f"{gb(p):,.1f} GB")
for nm in ("covcache", "covcache_old_rule"):
    p = os.path.join(ROOT, nm)
    if os.path.isdir(p):
        out("INFO", f"size of {nm}\\", f"{gb(p):,.2f} GB")

head("4. DATA")
try:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(os.path.join(DATA, "outright.parquet"))
    n = pf.metadata.num_rows
    out("PASS" if n == 7749309 else "FAIL", "outright.parquet rows", f"{n:,} (ref 7,749,309)")
    out("INFO", "timestamp storage unit (AUD-20)", str(pf.schema_arrow.field("ts").type))
except Exception as e:
    out("FAIL", "outright.parquet readable", f"{type(e).__name__}"); pq = None
for f in ("GLBX-20260807-Q3DVY5LNUD.zip",):
    out("PASS" if os.path.exists(os.path.join(DATA, f)) else "FAIL", f"data\\{f} present")
out("PASS" if os.path.isdir(os.path.join(DATA, "extracted")) else "FAIL", "data\\extracted\\ present")
old_dump = os.path.join(DATA, "dump_old_rule")
out("INFO" if os.path.isdir(old_dump) else "WARN", "old-rule dump kept as data\\dump_old_rule\\",
    f"{len(glob.glob(os.path.join(old_dump, 'N*.parquet')))} files" if os.path.isdir(old_dump) else "not found")

cells = [f for f in glob.glob(os.path.join(DUMP, "N*.parquet")) if ".m2." not in f and ".tw2." not in f]
newc = [f for f in cells if os.path.exists(f[:-8] + ".counts.json")]
oldc = [f for f in cells if f not in newc]
state = "INSTALLED" if not cells else ("REBUILT" if len(newc) == 180 and not oldc else "PARTIAL")
out("INFO", "state", state)
out("PASS" if not oldc else "FAIL", "no OLD-RULE cell inside data\\dump\\",
    f"{len(oldc)} found -- rename data\\dump to data\\dump_old_rule" if oldc else "")
if state != "INSTALLED":
    m2 = glob.glob(os.path.join(DUMP, "*.m2.parquet")); cj = glob.glob(os.path.join(DUMP, "*.counts.json"))
    tag = "PASS" if state == "REBUILT" else "INFO"
    out(tag if len(newc) == 180 else "INFO", "cells (re-arm)", f"{len(newc)} / 180")
    out(tag if len(m2) == 180 else "INFO", "M2 twin files", f"{len(m2)} / 180")
    out(tag if len(cj) == 180 else "INFO", "counts files", f"{len(cj)} / 180")
    if pq:
        from collections import Counter
        cc = Counter()
        for f in newc:
            try: cc[len(pq.read_schema(f).names)] += 1
            except Exception: cc["unreadable"] += 1
        same = len(cc) == 1 and "unreadable" not in cc
        out(("PASS" if same else "FAIL") if newc else "INFO",      # 0 cells is not a failure
            "every cell has the same columns",
            (str(dict(cc)) if cc else "no re-arm cells yet") + "  (one distinct count = every cell identical)")
    done = glob.glob(os.path.join(DUMP, "*.done"))
    out(tag if len(done) == 20 else "INFO", "dump .done markers", f"{len(done)} / 20")
tmp = glob.glob(os.path.join(DUMP, "*.tmp")) + glob.glob(os.path.join(ROOT, "covcache", "*.tmp"))
out("PASS" if not tmp else "FAIL", "no half-written .tmp files", f"{len(tmp)} found" if tmp else "")
try:
    import safeio
    cov = glob.glob(os.path.join(ROOT, "covcache", "*.npz"))
    stale = [p for p in cov if safeio.npz_version(p) != spec.SPEC_VERSION]
    out("WARN" if stale else "INFO", "covcache\\", f"{len(cov) - len(stale)} current, {len(stale)} OLD"
        + ("  <- rename covcache to covcache_old_rule BEFORE the cov step" if stale else ""))
    htf = glob.glob(os.path.join(ROOT, "htf_*_*m_*.npz"))
    stale = [p for p in htf if safeio.npz_version(p) != spec.SPEC_VERSION]
    out("INFO", "HTF files, own-rule (R4)", f"{len(htf) - len(stale)} current / 16")
    oldh = [p for p in glob.glob(os.path.join(ROOT, "htf_*.npz")) if p not in htf]
    if oldh:
        out("INFO", "old HTF files (rolling-high, pre-R4)", f"{len(oldh)} -- unused, kept")
except Exception as e:
    out("FAIL", "cache stamps readable", f"{type(e).__name__}: {e}")
ref = os.path.join(ROOT, "reference_rearm.json")
out("INFO", "slow re-arm reference (R9)", "present" if os.path.exists(ref) else "not run yet")

head("5. OUTPUT FOLDERS (a file here makes a later run SKIP that cell)")
for d in sorted({"cores_v2", "floors_v2", "inter0", "mono"} | {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*_tau*")) + glob.glob(os.path.join(ROOT, "inter*"))}):
    k = glob.glob(os.path.join(ROOT, d, "*.parquet"))
    out("INFO" if not k else "WARN", f"{d}\\", f"{len(k)} result files" +
        ("  <- a new run would skip these" if k else "  (empty)"))

head("6. OTHER COPIES OF THE CODE (not used by Python; listed so nothing is hidden)")
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p) and d != "code":
        n = len(glob.glob(os.path.join(p, "**", "*.py"), recursive=True))
        if n:
            out("INFO", d + "\\", f"{n} .py files -- a kept copy, not used")

print(); print("=" * 78)
print("ALL CHECKS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
if WARNS: print(f"{len(WARNS)} WARNING(S): {WARNS}")
print("=" * 78)

import datetime
live = hashlib.sha256("".join(sha(f) for f in sorted(glob.glob(os.path.join(CODE, "*.py")))).encode()).hexdigest()[:12]
line = (f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"{'PASS' if not FAILS else 'FAIL ' + ';'.join(FAILS)} | state {state} | "
        f"code-set {live} | cells {len(newc)} new / {len(oldc)} old | python {sys.version.split()[0]}")
with open(os.path.join(ROOT, "verify_log.txt"), "a") as fh:
    fh.write(line + "\n")
print(f"logged -> verify_log.txt : {line}")
