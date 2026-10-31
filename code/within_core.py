"""
within_core.py -- does a second filter ADD anything, tested properly?

WRONG test (what I did): compare CI of (core+X) against CI of (core).
Those overlap heavily because one sample is nested in the other.

RIGHT test: inside the core subset, split by X and compare the two halves.
    delta = edge(core & X=lo) - edge(core & X=hi)
with the SE of the DIFFERENCE. That is a comparison of disjoint groups drawn
from the same conditioned sample, and it answers "does X sort the core".

Run for EVERY variable, not a hand-picked few, in both blocks.
Forward selection is a search, so the number of variables tested is reported
and a candidate needs the same sign in both blocks.
"""
import os
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import sys
import numpy as np
import pandas as pd

DUMP = DUMP_DIR
TAU = os.environ.get("VWAP_TAU", "0.0")   # PRIMARY 0.0 (A2: one stamp everywhere)
# POST = measured AFTER the signal -> never an entry filter. Matched by exact
# name or prefix. 2026-09-21 rebuild adds the stored-only outcome columns:
# mae_fill_* (R11), mir_w / race (claim-2 race), cens_T (AUD-35).
# 2026-09-29 (§25): pseudo_* (pseudo-VWAP outcome), mae_all_wick, gap_at_T.
POST = ("er_p5", "er_p15", "er_p30", "er_ratio", "gap_p", "adv_", "anchor_expired_at", "bars_to_gap",
        "gap_at_break", "min_T", "spans_gap", "censored",
        "mae_fill", "mae_fillbar", "mir_w", "race", "cens_T",
        "pseudo_", "mae_all", "gap_at_T")
META = {"bar", "ts", "day", "block", "is_real", "src_bar", "N", "T", "anchor",
        "trigger", "eps", "underpowered", "hit", "atr0", "scheme"}


def load(cell, side=None):
    x = pd.read_parquet(f"{DUMP}/{cell}.parquet")
    T = int(x["T"].iloc[0])
    x["hit"] = ((x[f"w_{TAU}"] > 0) & (x[f"w_{TAU}"] <= T)).astype(float)
    r = x[x.is_real == 1]
    if side is not None:
        r = r[np.sign(r.side) == side]
    r = r.reset_index(drop=True)
    c = x[x.is_real == 0]
    c = c[c.src_bar.isin(set(r.bar))].reset_index(drop=True)
    return r, c, T


def terciles(r, col):
    v = r[col].to_numpy(float)
    if not np.isfinite(v).any() or np.nanstd(v) == 0:
        return None
    u = np.unique(v[np.isfinite(v)])
    if len(u) <= 2:
        m = v > np.nanmedian(v)
        return {"hi": m, "lo": ~m}
    q = np.nanquantile(v, [1 / 3, 2 / 3])
    if len(np.unique(q)) < 2:
        return None
    return {"lo": v <= q[0], "hi": v > q[1]}


def edge(r, c, mask, B):
    bm = dict(zip(r.bar.to_numpy(), mask))
    cm = c.src_bar.map(bm).fillna(False).to_numpy(bool)
    rr = r.hit.to_numpy()[(r.block == B).to_numpy() & mask]
    cc = c.hit.to_numpy()[(c.block == B).to_numpy() & cm]
    if len(rr) < 120 or len(cc) < 120:
        return None
    e = rr.mean() - cc.mean()
    var = rr.var() / len(rr) + cc.var() / len(cc)
    return e, len(rr), var


if __name__ == "__main__":
    cell = sys.argv[1]
    side = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    core_spec = sys.argv[3] if len(sys.argv) > 3 else None

    r, c, T = load(cell, side)
    core = np.ones(len(r), bool)
    if core_spec:
        for f in core_spec.split("+"):
            col, sd = f.rsplit("=", 1)
            t = terciles(r, col)
            core &= t[sd]
    print(f"CELL {cell} side={side} core={core_spec or '(none)'} "
          f"T={T} core n={core.sum():,}")
    for B in ("A", "B"):
        e = edge(r, c, core, B)
        if e:
            print(f"  core edge block {B}: {100*e[0]:+.2f}pp  "
                  f"n={e[1]}  +/-{200*np.sqrt(e[2]):.2f}pp")
    print()

    cols = [x for x in r.columns
            if x not in META and not x.startswith(("t_", "w_", "tlive_", "wlive_"))
            and not any(x.startswith(p) or x == p for p in POST)]
    rows = []
    for col in cols:
        t = terciles(r, col)
        if t is None:
            continue
        out = {}
        ok = True
        for B in ("A", "B"):
            el = edge(r, c, core & t["lo"], B)
            eh = edge(r, c, core & t["hi"], B)
            if el is None or eh is None:
                ok = False; break
            out[B] = (el[0] - eh[0], np.sqrt(el[2] + eh[2]), el[1], eh[1],
                      el[0], eh[0])
        if not ok:
            continue
        rows.append(dict(col=col, dA=out["A"][0], dB=out["B"][0],
                         seB=out["B"][1], nlo=out["B"][2], nhi=out["B"][3],
                         eloB=out["B"][4], ehiB=out["B"][5]))
    d = pd.DataFrame(rows)
    d["same"] = np.sign(d.dA) == np.sign(d.dB)
    d["t"] = d.dB.abs() / d.seB
    d = d[d.same & (d.t > 2)].sort_values("t", ascending=False)
    print(f"variables tested: {len(rows)} | sorting the core at |t|>2 "
          f"with matching sign in both blocks: {len(d)}")
    print()
    print("  %-20s %9s %9s %6s %10s %10s" %
          ("variable", "delta A", "delta B", "t", "edge lo B", "edge hi B"))
    for _, q in d.head(16).iterrows():
        print("  %-20s %+8.2fpp %+8.2fpp %6.1f %+9.2fpp %+9.2fpp"
              % (q.col, 100 * q.dA, 100 * q.dB, q.t, 100 * q.eloB, 100 * q.ehiB))
