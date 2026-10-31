"""
mono_sweep.py -- the all-out search, with monotonicity as the test.

For every cell and every ENTRY-TIME covariate, split setups into quintiles and
compute the edge (setup hit - matched twin hit) in each, separately in block A
and block B.

WHY MONOTONICITY: a lucky bin is easy to find in a big search. A smooth
ordering across five bins is not.

    P(exact monotone order of 5 bins | noise)      = 2/120 = 1/60
    P(same monotone direction in BOTH blocks)      = 1/3600

Post-signal columns (within_core.POST) are EXCLUDED -- they cannot filter an
entry taken at the signal.
NOTE (AUD-34, open): quintile cut points are pooled over both blocks and the
size gates (1,500 setups, 100 per bin) differ from the other modules'.

Output per (cell, covariate): Spearman rho of edge vs quintile in each block,
the edge in the best and worst quintile, and sample sizes.
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import glob
import os
import re
import sys
import numpy as np
import safeio
import pandas as pd
from scipy.stats import spearmanr

DUMP = DUMP_DIR
NQ = 5
TAU = os.environ.get("VWAP_TAU", "0.0")   # PRIMARY 0.0 (A2: one stamp everywhere)

# Decision B (2026-09-21): tau lives in the output folder AND in every row, so a
# run at one tau can never be mistaken for, or skipped because of, another tau.
OUT = ROOT_DIR + "mono" + ("" if TAU == "0.0" else f"_tau{TAU}")
os.makedirs(OUT, exist_ok=True)

from within_core import POST   # AUD-9: ONE post-signal list (was a copy)
META = {"bar", "ts", "day", "block", "is_real", "src_bar", "N", "T", "anchor",
        "trigger", "eps", "underpowered", "hit", "side", "atr0", "scheme"}


def entry_covs(cols):
    out = []
    for c in cols:
        if c in META or c.startswith(("t_", "w_", "tlive_", "wlive_")):
            continue
        if any(c.startswith(p) or c == p for p in POST):
            continue
        out.append(c)
    return out


def run_cell(f):
    name = os.path.basename(f)[:-8]
    outf = f"{OUT}/{name}.parquet"
    if os.path.exists(outf):
        return None
    x = pd.read_parquet(f)
    T = int(x["T"].iloc[0])
    x["hit"] = ((x[f"w_{TAU}"] > 0) & (x[f"w_{TAU}"] <= T)).astype(float)
    r = x[x.is_real == 1]
    c = x[x.is_real == 0]
    if len(r) < 1500:
        safeio.write_parquet(pd.DataFrame(), outf); return 0
    m = re.match(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)", name)
    rows = []
    for cv in entry_covs(r.columns):
        v = r[cv].to_numpy(float)
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            continue
        try:
            q = np.nanquantile(v, np.linspace(0, 1, NQ + 1)[1:-1])
        except Exception:
            continue
        if len(np.unique(q)) < NQ - 1:
            continue
        tq = np.digitize(v, q)
        bmap = dict(zip(r.bar.to_numpy(), tq))
        ctq = c.src_bar.map(bmap).to_numpy()
        rec = dict(cell=name, tau=float(TAU), cov=cv, N=int(m.group(1)), anchor=m.group(2),
                   trig=m.group(3), eps=float(m.group(4)))
        good = True
        for B in ("A", "B"):
            mr = (r.block == B).to_numpy()
            mc = (c.block == B).to_numpy()
            eds, ns = [], []
            for k in range(NQ):
                rr = r.hit.to_numpy()[mr & (tq == k)]
                cc = c.hit.to_numpy()[mc & (ctq == k)]
                if len(rr) < 100 or len(cc) < 100:
                    good = False; break
                eds.append(rr.mean() - cc.mean()); ns.append(len(rr))
            if not good:
                break
            rho = spearmanr(np.arange(NQ), eds).statistic
            rec[f"rho{B}"] = float(rho)
            rec[f"lo{B}"] = float(eds[0]); rec[f"hi{B}"] = float(eds[-1])
            rec[f"best{B}"] = float(max(eds)); rec[f"worst{B}"] = float(min(eds))
            rec[f"n{B}"] = int(min(ns))
        if good:
            rows.append(rec)
    safeio.write_parquet(pd.DataFrame(rows), outf)
    return len(rows)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "N*"
    from loader import cell_files
    for f in cell_files(pat, DUMP):
        n = run_cell(f)
        if n is not None:
            print(f"  {os.path.basename(f):<34} {n}", flush=True)
