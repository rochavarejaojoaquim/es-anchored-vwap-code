"""
run_all.py -- one entry point for the pipeline, to run on your machine.

It works out what has already been done and continues from there, so it is
safe to stop with Ctrl-C and start again. It STOPS at the first step that
fails (AUD-28) instead of running the next steps on missing inputs.

    python run_all.py status             what is done, what is left
    python run_all.py dump   [N ...]     episode cells (slowest; one N per window)
    python run_all.py cov    [N ...]     covariate cache A + join B
    python run_all.py htf                the 16 higher-timeframe files (R4)
    python run_all.py attach [N ...]     cross-anchor + HTF + _xs columns
    python run_all.py reference          the slow independent re-arm reference (R9)
    python run_all.py search             the core search (core_v2)
    python run_all.py floors             permutation floors (floor_v2)
    python run_all.py build              unzip -> outright.parquet (already done)

Rebuild ORDER (sec. 21.18): dump -> cov -> htf -> attach -> reference ->
selfcheck full -> compare_old_new -> verify_pc.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
DATA = os.path.join(HERE, "data")
DUMP = os.path.join(DATA, "dump")
ZIP = os.path.join(DATA, "GLBX-20260807-Q3DVY5LNUD.zip")
NS = [60, 100, 160, 200, 240]
ANCHORS = ["hh", "ll", "hv", "session"]


def sh(args):
    e = dict(os.environ); e["PYTHONPATH"] = CODE
    print("  >", " ".join(args), flush=True)
    rc = subprocess.call([sys.executable] + args, cwd=CODE, env=e)
    if rc != 0:                                          # AUD-28
        raise SystemExit(f"\nSTOPPED: `{' '.join(args)}` failed (exit code {rc}). "
                         f"Nothing after it was run. Fix it, then run the same command again.")


def ns_from(argv):
    return [int(x) for x in argv] or NS


def build():
    import zipfile
    os.makedirs(DATA, exist_ok=True)
    ex = os.path.join(DATA, "extracted")
    if not os.path.exists(ex):
        print("unzipping..."); zipfile.ZipFile(ZIP).extractall(ex)
    csvs = glob.glob(os.path.join(ex, "*.csv"))
    zsts = glob.glob(os.path.join(ex, "*.zst"))
    if not csvs and zsts:
        import zstandard as zstd
        out = zsts[0][:-4]
        with open(zsts[0], "rb") as f, open(out, "wb") as g:
            zstd.ZstdDecompressor().copy_stream(f, g)
        csvs = [out]
    out = os.path.join(DATA, "outright.parquet")
    if os.path.exists(out):
        print("outright.parquet already built"); return
    import pandas as pd
    sys.path.insert(0, CODE); import safeio
    df = pd.read_csv(csvs[0], usecols=["ts_event", "instrument_id", "symbol",
                                       "open", "high", "low", "close", "volume"])
    df = df[~df["symbol"].astype(str).str.contains("-")].copy()
    df["ts"] = pd.to_datetime(df["ts_event"]); df = df.drop(columns=["ts_event"])
    safeio.write_parquet(df, out)
    print("wrote", out, len(df), "rows")


def status():
    sys.path.insert(0, CODE)
    from loader import cell_files
    cells = cell_files("N*", DUMP) if os.path.isdir(DUMP) else []
    new = [f for f in cells if os.path.exists(f[:-8] + ".counts.json")]
    print(f"  outright.parquet : {'yes' if os.path.exists(os.path.join(DATA, 'outright.parquet')) else 'NO'}")
    print(f"  cells (re-arm)   : {len(new)} / 180" + (f"   ({len(cells) - len(new)} OLD-RULE cells present!)" if len(cells) > len(new) else ""))
    print(f"  M2 twin files    : {len(glob.glob(os.path.join(DUMP, '*.m2.parquet')))} / 180")
    print(f"  covcache         : {len(glob.glob(os.path.join(HERE, 'covcache', '*.npz')))} / 20")
    print(f"  HTF (own rule)   : {len(glob.glob(os.path.join(HERE, 'htf_*_*m_*.npz')))} / 16")
    print(f"  reference        : {'yes' if os.path.exists(os.path.join(HERE, 'reference_rearm.json')) else 'no'}")
    print(f"  core searches    : {len(glob.glob(os.path.join(HERE, 'cores_v2', '*.parquet')))} / 540")
    print(f"  floors           : {len(glob.glob(os.path.join(HERE, 'floors_v2', '*.parquet')))}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    rest = sys.argv[2:]
    if cmd == "dump":
        for N in ns_from(rest):
            sh(["dump_episodes.py", str(N)])
    elif cmd == "cov":
        for N in ns_from(rest):
            for a in ANCHORS:
                sh(["cov_phase.py", "A", str(N), a]); sh(["cov_phase.py", "B", str(N), a])
    elif cmd == "htf":
        sh(["enrich.py"])
    elif cmd == "attach":
        sh(["attach_context.py"] + [str(N) for N in ns_from(rest)])
    elif cmd == "reference":
        sh(["rearm_reference.py"])
    elif cmd == "search":
        sh(["core_v2.py"])
    elif cmd == "floors":
        for N in NS:
            sh(["floor_v2.py", f"N{N}"])
    elif cmd == "build":
        build()
    elif cmd == "status":
        status()
    else:
        raise SystemExit(__doc__)
