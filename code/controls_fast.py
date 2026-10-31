"""
controls_fast.py -- vectorised matched random-level control.

The control level is STATIC, so the walk is simpler than the episode walk:
    g_t = close[t] - level,   level = close[j] - sign(g0)*dist*atr[j]

Correctness gate: selfcheck's synthetic loop reference (gates.py). The old
gate re-walked control_M1.parquet (not on the PC) -> code\\legacy\\ (AUD-32).
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
from walk_fast import _first_true
from _walkextra import extra, EXTRA_KEYS


def walk_static(close, seg, ctl, atr0, g0, h_rec, T=None, hi=None, lo=None):
    """ctl: control bar indices. g0: signed starting gap (in price)."""
    n = len(close)
    out = {f"tl_{t}": [] for t in TAUS}
    out.update({f"al_{x}": [] for x in XS})
    out["max_adv"] = []
    out["adv_T"] = []
    for t in TAUS:
        out[f"w_{t}"] = []
    out["n_bars"] = []
    for k in EXTRA_KEYS:
        out[k] = []
    if T is None:
        T = h_rec
    # adaptive chunk: chunk*h_rec ~= VWAP_CHUNK elements per matrix (C87 OOM)
    _CH = int(_os_env.environ.get("VWAP_CHUNK", "20000000"))
    chunk = max(400, int(_CH // max(h_rec, 1)))
    off = np.arange(1, h_rec + 1)

    for s in range(0, len(ctl), chunk):
        e = min(s + chunk, len(ctl))
        J = ctl[s:e]
        a0 = atr0[s:e][:, None]
        G0 = g0[s:e][:, None]

        t = J[:, None] + off[None, :]
        valid = (t < n) & (seg[np.minimum(t, n - 1)] == seg[J][:, None])
        valid = np.logical_and.accumulate(valid, axis=1)
        tc = np.minimum(t, n - 1)

        level = close[J][:, None] - G0
        g = close[tc] - level
        sg = np.sign(G0)

        if hi is not None:
            H = hi[tc]; L = lo[tc]
            for tau in TAUS:
                band = tau * a0
                fill = np.where(sg > 0, L <= level + band, H >= level - band)
                out[f"w_{tau}"].append(_first_true(fill, valid))
        flip = np.sign(g) != sg
        for tau in TAUS:
            out[f"tl_{tau}"].append(
                _first_true(flip | (np.abs(g) <= tau * a0), valid))
        same = np.sign(g) == sg
        widen = np.where(same & valid, (np.abs(g) - np.abs(G0)) / a0, -np.inf)
        for X in XS:
            out[f"al_{X}"].append(_first_true(widen >= X, valid))
        out["max_adv"].append(
            np.where(valid, (np.abs(g) - np.abs(G0)) / a0, -np.inf).max(axis=1))
        vT = valid[:, :T]
        out["adv_T"].append(
            np.where(vT, (np.abs(g[:, :T]) - np.abs(G0)) / a0, -np.inf).max(axis=1))
        out["n_bars"].append(valid.sum(axis=1))
        if hi is not None:
            ex = extra(out["w_0.0"][-1], H, L, close[tc], close[J][:, None], G0,
                       sg, a0, valid, T)
            for k2 in EXTRA_KEYS:
                out[k2].append(ex[k2])

    # zero-setup cells return every key with zero rows (AUD-25: always written)
    return {k: (np.concatenate(v) if len(v) else np.empty(0)) for k, v in out.items()}


def draw_controls(d, sig_bars, sides, dists, strata_lab, K=2, seed=17,
                  match_block=True, min_gap=0):
    """Draw K matched control bars per episode. Same scheme as the loop version."""
    rng = np.random.default_rng(seed)
    n = len(d)
    atr = d.atr.to_numpy()
    from loader import trade_day
    day = trade_day(d.ts)                      # AUD-13: CME trade date (int days)
    pool_ok = (d.eligible.to_numpy() & np.isfinite(atr) & (strata_lab >= 0)
               & (np.arange(n) < n - 2000))
    # C110: a twin MUST come from the same training block as its setup.
    # Without this, 22.9% of twins crossed the A/B boundary and inherited the
    # wrong regime's conditions.
    blk = d.block.to_numpy() if match_block else np.zeros(n, dtype="U1")
    lab2 = np.array([f"{a}|{b}" for a, b in zip(strata_lab, blk)], dtype=object)
    pool = np.where(pool_ok)[0]
    order = np.argsort(lab2[pool].astype(str), kind="stable")
    pool = pool[order]
    plab = lab2[pool]
    ps = plab.astype(str)
    bounds = {L: (np.searchsorted(ps, L, "left"),
                  np.searchsorted(ps, L, "right")) for L in np.unique(ps)}

    src, ctl, sd_, dd_ = [], [], [], []
    for k, i in enumerate(sig_bars):
        b = bounds.get(f"{strata_lab[i]}|{blk[i]}")
        if b is None or b[1] - b[0] < 20:
            continue
        got = 0
        for _ in range(K * 6):
            if got >= K:
                break
            j = int(pool[rng.integers(b[0], b[1])])
            if day[j] == day[i] or abs(j - i) <= min_gap:   # AUD-7: no shared path
                continue
            src.append(i); ctl.append(j); sd_.append(sides[k]); dd_.append(dists[k])
            got += 1
    return (np.array(src), np.array(ctl, dtype=np.int64),
            np.array(sd_), np.array(dd_))
