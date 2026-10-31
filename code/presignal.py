"""
presignal.py -- R6: the PRE-signal efficiency ratio (the author's hypothesis:
price TRENDING into the signal respects the line less; CONSOLIDATING, more).

Built in the 2026-09-20 chat as presignal_er.py, run on one cell, never
shipped; recovered verbatim from that transcript and ported here so it runs
INSIDE the dump (no separate in-place pass).

  er_pre{k}   Kaufman efficiency ratio over the k bars ENDING at the signal
              |close[t] - close[t-k]| / sum |close[i] - close[i-1]|
              0 = pure chop, 1 = a straight line. NaN if the k bars cross a
              roll. ENTRY covariate.
  er_anchor   the same, from the ANCHOR bar to the signal bar. ENTRY.
  er_ratio{k} er_p15 / er_pre{k}: did the market change character at the
              signal? Uses bars AFTER the signal -> POST (never an entry filter).

Example: closes 100, 101, 102, 103 -> net 3, path 3 -> ER 1.0.
         closes 100, 102, 100, 102 -> net 2, path 6 -> ER 0.33.
"""
import numpy as np

from spec import PRE_KS


def build_pre(close, seg):
    c = close.astype(float)
    step = np.abs(np.r_[0.0, np.diff(c)])
    cstep = np.concatenate([[0.0], np.cumsum(step)])
    idx = np.arange(len(c))
    out = {}
    for k in PRE_KS:
        lo = np.maximum(idx - k, 0)
        ok = (seg[lo] == seg[idx]) & (idx - k >= 0)
        net = np.abs(c - c[lo])
        path = cstep[idx + 1] - cstep[lo + 1]
        out[f"er_pre{k}"] = np.where(ok & (path > 0), net / np.maximum(path, 1e-9),
                                     np.nan).astype(np.float32)
    return out, cstep


def at_signals(PRE, cstep, close, sig, anc, er_p15):
    c = close.astype(float)
    out = {f"er_pre{k}": PRE[f"er_pre{k}"][sig] for k in PRE_KS}
    net = np.abs(c[sig] - c[anc])
    path = cstep[sig + 1] - cstep[anc + 1]
    out["er_anchor"] = np.where(path > 0, net / np.maximum(path, 1e-9), np.nan)
    for k in PRE_KS:
        out[f"er_ratio{k}"] = er_p15 / np.maximum(out[f"er_pre{k}"], 1e-6)
    return out
