"""
walk_fast.py -- vectorised replacement for the per-episode Python loop.

Each episode is a fixed-length forward scan over contiguous memory, so the
walk is a matrix operation:   t[k, j] = sig[k] + 1 + j
Chunked by VWAP_CHUNK (default 20,000,000 elements per matrix).

Correctness gate: selfcheck's synthetic loop reference (gates.py), which runs
anywhere with no data. The old verify() compared against episodes_v3.parquet,
a file not on the PC; it moved to code\\legacy\\ (AUD-32).

WICK fill = the bar's low (price above the level) or high (price below) TOUCHES
the level: `<=` / `>=`, not strictly through it (AUD-33 corrected wording).
"""
import os as _os_env
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import numpy as np
import pandas as pd

from spec import TAUS, XS
from _walkextra import extra, EXTRA_KEYS

CHUNK = 1500


def _first_true(mask, valid):
    """First column index where mask is True, per row; -1 if none. 1-based."""
    m = mask & valid
    any_ = m.any(axis=1)
    j = np.argmax(m, axis=1) + 1
    return np.where(any_, j, -1)


def walk(close, cum, ref, seg, sig, anc, atr0, g0, W0, h_rec, T=None,
         hi=None, lo=None):
    n = len(close)
    out = {f"tl_{t}": [] for t in TAUS}
    out.update({f"tf_{t}": [] for t in TAUS})
    out.update({f"tc_{t}": [] for t in TAUS})
    out.update({f"al_{x}": [] for x in XS})
    out.update({f"af_{x}": [] for x in XS})
    for k in ("min_live", "min_froz", "max_adv_live", "n_bars", "censored",
              "adv_live_T", "adv_froz_T", "min_froz_T"):
        out[k] = []
    # WICK touch: the bar's range reaches the level from the correct side
    # (touch, <=). Directional, not a range intersection.
    for t in TAUS:
        out[f"wf_{t}"] = []
        out[f"wl_{t}"] = []
    for k in EXTRA_KEYS:
        out[k] = []
    if T is None:
        T = h_rec

    # adaptive chunk: chunk*h_rec ~= VWAP_CHUNK elements per matrix (C87 OOM)
    _CH = int(_os_env.environ.get("VWAP_CHUNK", "20000000"))
    chunk = max(400, int(_CH // max(h_rec, 1)))
    off = np.arange(1, h_rec + 1)
    for s in range(0, len(sig), chunk):
        e = min(s + chunk, len(sig))
        S = sig[s:e]; A = anc[s:e]
        a0 = atr0[s:e][:, None]; G0 = g0[s:e][:, None]; W = W0[s:e][:, None]

        t = S[:, None] + off[None, :]
        valid = (t < n) & (seg[np.minimum(t, n - 1)] == seg[S][:, None])
        valid = np.logical_and.accumulate(valid, axis=1)   # stop at first break
        tc = np.minimum(t, n - 1)

        Wt = cum.vol(A[:, None], tc)
        Vt = cum.line(A[:, None], tc, Wt) + ref[S][:, None]
        V0 = (close[S][:, None] - G0)

        g_live = close[tc] - Vt
        g_froz = close[tc] - V0
        g_cf = G0 * (W / np.maximum(Wt, 1e-9))

        sg = np.sign(G0)
        if hi is not None:
            H = hi[tc]; L = lo[tc]
            for name, lvl in (("f", V0), ("l", Vt)):
                for tau in TAUS:
                    band = tau * a0
                    fill = np.where(sg > 0, L <= lvl + band, H >= lvl - band)
                    out[f"w{name}_{tau}"].append(_first_true(fill, valid))
        for name, g in (("l", g_live), ("f", g_froz), ("c", g_cf)):
            flip = np.sign(g) != sg
            for tau in TAUS:
                out[f"t{name}_{tau}"].append(
                    _first_true(flip | (np.abs(g) <= tau * a0), valid))
        for name, g in (("l", g_live), ("f", g_froz)):
            same = np.sign(g) == sg
            widen = np.where(same & valid, (np.abs(g) - np.abs(G0)) / a0, -np.inf)
            for X in XS:
                out[f"a{name}_{X}"].append(_first_true(widen >= X, valid))

        big = np.where(valid, np.abs(g_live) / a0, np.inf)
        out["min_live"].append(big.min(axis=1))
        out["min_froz"].append(np.where(valid, np.abs(g_froz) / a0, np.inf).min(axis=1))
        out["max_adv_live"].append(
            np.where(valid, (np.abs(g_live) - np.abs(G0)) / a0, -np.inf).max(axis=1))
        vT = valid[:, :T]
        out["adv_live_T"].append(
            np.where(vT, (np.abs(g_live[:, :T]) - np.abs(G0)) / a0, -np.inf).max(axis=1))
        out["adv_froz_T"].append(
            np.where(vT, (np.abs(g_froz[:, :T]) - np.abs(G0)) / a0, -np.inf).max(axis=1))
        out["min_froz_T"].append(
            np.where(vT, np.abs(g_froz[:, :T]) / a0, np.inf).min(axis=1))
        if hi is not None:
            ex = extra(out["wf_0.0"][-1], H, L, close[tc], close[S][:, None], G0,
                       sg, a0, valid, T)
            for k2 in EXTRA_KEYS:
                out[k2].append(ex[k2])
        nb = valid.sum(axis=1)
        out["n_bars"].append(nb)
        out["censored"].append(nb < h_rec)

    # zero-setup cells return every key with zero rows (AUD-25: always written)
    return {k: (np.concatenate(v) if len(v) else np.empty(0)) for k, v in out.items()}
