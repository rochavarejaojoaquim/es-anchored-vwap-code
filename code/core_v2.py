"""
core_v2.py -- the core search, rebuilt with every correction inside the
selection criterion rather than applied afterwards.

CHANGES FROM v1
  twins        block-matched (C110), read from INSIDE each cell (is_real==0)
  errors       5-day block bootstrap, INSIDE the score, not bolted on later
  cutpoint     maximally selected over the 10th-90th percentile range
               (Lausen & Schumacher 1992), replacing the arbitrary tercile
  score        SYMMETRIC pessimistic edge at 99% (Decision D, 2026-09-21):
               min(|eA| - 2.6*SE_A, |eB| - 2.6*SE_B)  -- each block must clear
               its OWN error bar (A and B are independent replication blocks,
               §2). Was min(|eA|,|eB|) - 2.6*SE_B, which ignored A's error.
  direction    both tails; a negative edge is a continuation signal and
               competes on |edge|

The cutpoint search is itself a search, so the permutation floor (core_sig_v2)
covers BOTH the choice of variable and the choice of cutpoint -- which is the
multi-variable case Lausen, Hothorn, Bretz & Schumacher (2004) describe.
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import glob
import os
import sys
import numpy as np
import safeio
import pandas as pd
import pyarrow.parquet as pq

from within_core import META, POST
from loader import trade_day

DUMP = DUMP_DIR
TAU = os.environ.get("VWAP_TAU", "0.0")   # PRIMARY: pure sign crossing
# NOTE: core_v2 does NOT call walk(); VWAP_CHUNK has no effect here.
# Its memory is one cell parquet held in RAM -- measure with peak_mem.py.
TAU_SENS = "0.25"  # sensitivity only

# Decision B (2026-09-21): tau lives in the output folder AND in every row, so a
# run at one tau can never be mistaken for, or skipped because of, another tau.
OUT = ROOT_DIR + "cores_v2" + ("" if TAU == "0.0" else f"_tau{TAU}")
os.makedirs(OUT, exist_ok=True)
QLO, QHI, NQ = 0.10, 0.90, 9          # candidate cutpoints, 10th..90th
MIN_N = 300          # per side of the cut (cells can be 1,000 total)
NBOOT = 300
BOOT_MAX = 500       # best_core re-estimates the winner with 500 draws; the
#                      precomputed table must cover the LARGEST nboot used
#                      anywhere, or a slice silently returns fewer rows
Z99 = 2.6


def score(eA, eB):
    """THE score. eA, eB = (edge, SE, n) per block. Any floor or rescoring
    script must IMPORT this function, never retype the formula (C107)."""
    return min(abs(eA[0]) - Z99 * eA[1], abs(eB[0]) - Z99 * eB[1])


# ---- reading a cell -------------------------------------------------------
# A cell file has ~144 columns; the search uses ~78 of them. Reading only what
# is needed, once per cell instead of once per side, cuts the file reading by
# roughly two thirds. The kept set is derived from the file's own column names
# with the SAME rule search_cols uses, so the searched columns cannot differ
# (2026-09-24, before the floors -- which repeat this read many times over).
NEEDED = {"is_real", "side", "block", "ts", "bar", "src_bar", "T"}
_RAW = {}


def cell_columns(names):
    """The columns to read, IN THE FILE'S OWN ORDER.

    Order matters: the search walks the columns in order and keeps the FIRST of
    an exact tie, so reading them sorted alphabetically silently changed the
    winning column on tied cells (caught by verify_speedup.py on 3 of 10 real
    cells, 2026-09-24 -- the scores were identical, only the column differed).
    """
    extra = set(NEEDED) | {f"w_{TAU}"}
    return [c for c in names
            if c in extra
            or (c not in META
                and not c.startswith(("t_", "w_", "tlive_", "wlive_"))
                and not any(c.startswith(p) or c == p for p in POST))]


def raw_cell(cell):
    """The cell frame, columns subset, cached for the three sides in a row."""
    if cell not in _RAW:
        p = f"{DUMP}/{cell}.parquet"
        _RAW.clear()                                  # one cell at a time
        _RAW[cell] = pd.read_parquet(p, columns=cell_columns(pq.read_schema(p).names))
    return _RAW[cell]


def load_cell(cell, side=None):
    x = raw_cell(cell)
    T = int(x["T"].iloc[0])
    r = x[x.is_real == 1]
    if side is not None:
        r = r[np.sign(r.side) == side]
    r = r.reset_index(drop=True)
    r["day"] = trade_day(r.ts)                  # AUD-13: CME trade date (ints)
    r["hit"] = ((r[f"w_{TAU}"] > 0) & (r[f"w_{TAU}"] <= T)).astype(float)
    c = x[x.is_real == 0]          # twins live INSIDE the cell locally, block-matched
    c = c[c.src_bar.isin(set(r.bar))].reset_index(drop=True)
    c["day"] = trade_day(c.ts)
    c["hit"] = ((c[f"w_{TAU}"] > 0) & (c[f"w_{TAU}"] <= T)).astype(float)
    return r, c, T


def boot_index(D, nboot, blocklen=5, seed=1):
    """The day-blocks each bootstrap draw selects -- ONE row per draw.

    edge_se reseeds with a FIXED seed and draws only from the number of days,
    so every candidate subset was resampling the IDENTICAL blocks and paying
    for it ~1,400 times per search. Built once here instead (2026-09-24).
    The draws are generated exactly as the loop generated them, in the same
    order, so a 150-draw run is the first 150 rows of a 300-draw one.
    """
    rng = np.random.default_rng(seed)
    nblk = max(D // blocklen, 1)
    rows = []
    for _ in range(nboot):
        st = rng.integers(0, max(D - blocklen + 1, 1), nblk)   # AUD-6
        rows.append(np.concatenate([np.arange(s, min(s + blocklen, D)) for s in st]))
    w = max(len(x) for x in rows)
    if any(len(x) != w for x in rows):        # only when D < blocklen
        return rows
    return np.asarray(rows, dtype=np.int64)


def twin_index(r, c):
    """Where each twin's setup sits in r -- computed ONCE per cell.

    The search used to rebuild a python dict {bar: in_subset} and map it over
    every twin for EVERY candidate (~1,400 times per search); that was 56% of
    the run time. The mapping never changes, only the mask does, so:
        cmask = mask[src_idx]      (-1 = twin whose setup is not in this side)
    """
    pos = {b: i for i, b in enumerate(r.bar.to_numpy())}
    return np.array([pos.get(b, -1) for b in c.src_bar.to_numpy()], dtype=np.int64)


def twin_mask(mask, src_idx):
    return np.where(src_idx >= 0, mask[np.maximum(src_idx, 0)], False)


def prep(r, c):
    """Per-day sufficient statistics so the block bootstrap is O(days)."""
    out = {"src_idx": twin_index(r, c)}
    for B in ("A", "B"):
        rs = (r.block == B).to_numpy()
        cs = (c.block == B).to_numpy()
        days = np.unique(np.concatenate([r.day.to_numpy()[rs], c.day.to_numpy()[cs]]))
        ix = {d: i for i, d in enumerate(days)}
        out[B] = dict(days=days,
                      ri=np.array([ix[d] for d in r.day.to_numpy()[rs]]),
                      ci=np.array([ix[d] for d in c.day.to_numpy()[cs]]),
                      rs=rs, cs=cs, D=len(days),
                      SEL=boot_index(len(days), BOOT_MAX))
    return out


def edge_se(r, c, P, mask, cmask, B, blocklen=5, seed=1, nboot=NBOOT):
    p = P[B]
    rm = mask[p["rs"]]; cm = cmask[p["cs"]]
    rh = r.hit.to_numpy()[p["rs"]][rm]; ch = c.hit.to_numpy()[p["cs"]][cm]
    if len(rh) < MIN_N // 3 or len(ch) < MIN_N // 3:
        return None
    ri = p["ri"][rm]; ci = p["ci"][cm]; D = p["D"]
    rsum = np.bincount(ri, rh, D); rn = np.bincount(ri, None, D)
    csum = np.bincount(ci, ch, D); cn = np.bincount(ci, None, D)
    SEL = p.get("SEL")
    if SEL is None or seed != 1 or blocklen != 5 or len(SEL) < nboot:
        SEL = boot_index(D, nboot, blocklen, seed)     # any other setting: rebuild
    if isinstance(SEL, np.ndarray):
        S = SEL[:nboot]
        a1 = rn[S].sum(1); a2 = cn[S].sum(1)
        ok = (a1 > 0) & (a2 > 0)
        out = np.where(ok, rsum[S].sum(1) / np.where(ok, a1, 1)
                       - csum[S].sum(1) / np.where(ok, a2, 1), np.nan)
    else:                                              # ragged (D < blocklen)
        out = np.array([(rsum[s].sum() / rn[s].sum() - csum[s].sum() / cn[s].sum())
                        if rn[s].sum() and cn[s].sum() else np.nan for s in SEL[:nboot]])
    out = out[np.isfinite(out)]
    return float(rh.mean() - ch.mean()), float(np.std(out)), len(rh)


def search(r, c, P, cols, seed=1, nboot=150):
    """Best (variable, cutpoint, direction) by the 99% pessimistic edge."""
    best = None
    for col in cols:
        v = r[col].to_numpy(float)
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            continue
        qs = np.nanquantile(v, np.linspace(QLO, QHI, NQ))
        for q in np.unique(qs):
            for sense in ("lo", "hi"):
                m = v <= q if sense == "lo" else v > q
                if m.sum() < MIN_N or (~m).sum() < MIN_N:
                    continue
                cm = twin_mask(m, P["src_idx"])
                eA = edge_se(r, c, P, m, cm, "A", seed=seed, nboot=nboot)
                eB = edge_se(r, c, P, m, cm, "B", seed=seed, nboot=nboot)
                if eA is None or eB is None:
                    continue
                if np.sign(eA[0]) != np.sign(eB[0]):
                    continue
                sc = score(eA, eB)
                if best is None or sc > best["score"]:
                    best = dict(col=col, cut=float(q), sense=sense, score=sc,
                                eA=eA[0], eB=eB[0], seA=eA[1], seB=eB[1], nA=eA[2], nB=eB[2],
                                pct=float((v <= q).mean()))
    return best


def search_cols(r, side=None):
    """The columns the core search may use. ONE definition, shared by the core
    search and the permutation floor (C3, C107).

    AUD-39 (2026-09-21): the "_xs" columns (column x side) are dropped from a
    ONE-SIDE search. Within one side they are that column's sign-flip, so they
    carry no new information -- but they are not perfect duplicates, because a
    cut keeps ties on the other side (`v <= q` vs `v >= q`), and in 1 of 4
    measured cases that tie noise changed which column won. They stay in the
    POOLED search (side=None), where toward/away is real information."""
    cols = [x for x in r.columns
            if x not in META
            and not x.startswith(("t_", "w_", "tlive_", "wlive_"))
            and not any(x.startswith(p) or x == p for p in POST)]
    return cols if side is None else [x for x in cols if not x.endswith("_xs")]


def best_core(r, c, P, cols):
    """The WHOLE selection procedure: rank with a cheap bootstrap, then
    re-estimate the winner with a precise one. The core search and the floor
    both call THIS, so the null runs the identical procedure by construction."""
    b = search(r, c, P, cols, nboot=150)   # ranking noise ~6%, was ~11%
    if b is not None:
        v = r[b["col"]].to_numpy(float)
        m = v <= b["cut"] if b["sense"] == "lo" else v > b["cut"]
        cm = twin_mask(m, P["src_idx"])
        eA = edge_se(r, c, P, m, cm, "A", nboot=500)
        eB = edge_se(r, c, P, m, cm, "B", nboot=500)
        if eA and eB:
            b.update(eA=eA[0], eB=eB[0], seA=eA[1], seB=eB[1], nA=eA[2],
                     nB=eB[2], score=score(eA, eB))
    return b


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "N*"
    from loader import cell_files
    cells = cell_files(pat, DUMP)            # never .m2 / .tw2 (C3)
    if not cells:                            # AUD-44: never fail in silence
        raise SystemExit(f"no cells in data\\dump match '{pat}'. Give a cell name "
                         f"(N60_hh_ATR_3.0), a prefix (N60_hh) or N*.")
    print(f"  {len(cells)} cell(s) match '{pat}'", flush=True)
    for f in cells:
        cell = os.path.basename(f)[:-8]
        for side in (None, 1.0, -1.0):
            tag = f"{cell}_side{side}"
            outf = f"{OUT}/{tag}.parquet"
            if os.path.exists(outf):
                continue
            try:
                r, c, T = load_cell(cell, side)
            except Exception as e:                  # AUD-4: say WHY, loudly
                print(f"  !! {cell} side {side}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            if min((r.block == B).sum() for B in "AB") < 1000 or \
               min((c.block == B).sum() for B in "AB") < 1000:      # AUD-12
                safeio.write_parquet(pd.DataFrame(), outf); continue
            P = prep(r, c)
            # TWO-STAGE: cheap bootstrap to rank, precise one to report.
            b = best_core(r, c, P, search_cols(r, side))
            if b is None:
                safeio.write_parquet(pd.DataFrame(), outf); continue
            safeio.write_parquet(pd.DataFrame([dict(cell=cell, side=side, T=T,
                                                    tau=float(TAU), **b)]), outf)
            print("  %-24s %-5s %-18s cut@%.0f%% %s  eA %+6.2f eB %+6.2f "
                  "+/-%.2f/%.2f n=%5d SCORE %+6.2f"
                  % (cell[:24], str(side), b["col"][:18], 100 * b["pct"],
                     b["sense"], 100 * b["eA"], 100 * b["eB"], 260 * b["seA"], 260 * b["seB"],
                     b["nB"], 100 * b["score"]), flush=True)
