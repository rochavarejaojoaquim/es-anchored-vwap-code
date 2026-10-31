"""
pseudo_vwap.py -- the anchor-identity placebo (§25, design from §23.11).

QUESTION  Does price return to the REAL anchored VWAP more than to a line
          anchored at an ARBITRARY bar in the same window, matched on
          distance, side, direction of travel, and volatility?

DESIGN, LOCKED before any pseudo-VWAP number exists (§25 pre-registration):
  FROZEN.  Computed ONCE at the signal bar, from the candidate anchor,
           exactly as the real primary result freezes its own line (§3:
           "all subsequent work uses the frozen line"). Neither line
           updates as the episode walks forward.
  WINDOW.  hh/ll/hv cells: same N-bar rolling lookback as the real anchor,
           excluding the real anchor bar itself (gate 2).
           session cells: any bar from session open to the signal bar.
  TOLERANCE.  |close - V'| / ATR0 within +-10% of eps. Redraw on miss.
  REDRAW CAP = 50, fixed here in writing before any real run (E1). A
           setup that exhausts 50 draws without a match is marked
           unmatched, excluded from the paired test, and COUNTED. If more
           than 10% of setups in the run are unmatched, the run is VOID
           (§25.5 gate 3) -- report the fraction, do not quietly proceed.
  WALK.    The SAME walk_fast function that fills the real line, called
           with the pseudo level in place of the real one (gate 4). No
           second implementation of "does price wick-touch this level."
  SIDE.    A candidate is accepted only if sign(close - V') == side, in
           addition to the distance check (added 2026-09-29, the author).
  NOT MATCHED, DEFERRED 2026-09-29: direction of travel. NOT MEASURED.
           Registered, not adopted as equivalent to the pre-registered match.
           The pseudo anchor is drawn from the SAME window of the SAME
           episode as the real one, so both see the identical price path up
           to the signal bar by construction (unlike M1b twins, which come
           from a different day).

INTEGRATION -- WIRED to pipeline.py's actual interface (2026-09-29):
  pipeline.vwap_and_sd() returns `cum` (pipeline.Cum: exact int64, RESET per
  segment, TWO arrays per term -- end Ve/Pe and start Vs/Ps) and `ref` (the
  segment's reference price). The live line at bar t anchored at `anc` is
      ref[t] + cum.line(anc, t)
  vwap_from_anchor() below is exactly that, and nothing else: the arithmetic
  lives in pipeline.py only. A window never crosses a segment, so candidate
  anchors are clipped to the real anchor's own rolling window, taken from
  loader.rolling_argextreme itself (window_lo) -- one segment rule, one place.
"""
import zlib

import numpy as np

from loader import rolling_argextreme

REDRAW_CAP = 50          # E1: fixed before any real run, see docstring
TOL_FRAC = 0.10          # +-10% of eps, per §25.5 gate 3
SEED = 20260929          # first use in this project; see SEED_RULE below
SEED_RULE = "default_rng([SEED, crc32(f'{cell}_{setup_id}')]) per setup"


def window_lo(n, N, seg):
    """First bar of the real anchor's N-bar rolling window, at every bar, clipped
    at the segment start. Taken FROM loader.rolling_argextreme: on a strictly
    increasing series the minimum of a window is its first bar, so the function
    returns exactly the lower bound it uses internally. -1 where the window is
    undefined (warm-up), which is also where the real anchor is undefined.
    Session cells do not use this: their window starts at the session open,
    which IS the real anchor."""
    return rolling_argextreme(np.arange(n, dtype=float), N, seg, maximum=False)


def same_side(close, level, side):
    """True if the candidate line sits on the same side of price as the real
    one. `side` is stored exactly as dump_episodes does: sign(close - real
    VWAP), so +1 = price ABOVE the line, -1 = price BELOW it."""
    return np.sign(close - level) == side


def vwap_from_anchor(cum, ref, anchor_idx, at_idx):
    """Anchored VWAP (in price points) at at_idx, anchored at anchor_idx, from
    pipeline's exact cumulative sums. anchor_idx <= at_idx, same segment."""
    return float(ref[at_idx] + cum.line(anchor_idx, at_idx))


def draw_candidate_anchor(window_lo, window_hi, real_anchor_idx, rng):
    """One candidate bar from [window_lo, window_hi], excluding the real
    anchor (gate 2). window_hi is the signal bar; a candidate at window_hi
    itself is excluded too (a zero-length anchor is not a line)."""
    choices = [i for i in range(window_lo, window_hi) if i != real_anchor_idx]
    if not choices:
        return None
    return choices[rng.integers(len(choices))]


def match_pseudo_anchor(cum, ref, window_lo, window_hi, real_anchor_idx,
                         close, atr0, eps, side, rng, cap=REDRAW_CAP):
    """Try up to `cap` candidate anchors; return the first whose distance
    is within TOL_FRAC of eps, or None with the draw count if none matched.

    Returns: (anchor_idx_or_None, level_or_None, n_draws)
    """
    lo, hi = (eps * (1 - TOL_FRAC), eps * (1 + TOL_FRAC))
    for k in range(cap):
        a = draw_candidate_anchor(window_lo, window_hi, real_anchor_idx, rng)
        if a is None:
            return None, None, k + 1
        V = vwap_from_anchor(cum, ref, a, window_hi)
        dist = abs(close - V) / atr0
        if same_side(close, V, side) and lo <= dist <= hi:
            return a, V, k + 1
    return None, None, cap


def pseudo_for_cell(cum, ref, sig, real_anc, wlo, session, close, atr0, target,
                    side, cell):
    """One pseudo anchor per setup of a cell. NO walking happens here: a redraw
    only evaluates the cumulative sums, so the walk is called once for the
    whole cell afterwards (by the caller), never once per redraw.

    sig, real_anc, atr0, target, side: per-setup arrays. wlo: window_lo() over
    all bars (unused for session cells, whose window starts at the real anchor,
    the session open). close: all bars. `target` is the distance to match, in
    ATR0 units, and is passed in by the caller.

    Each setup gets its own generator, default_rng([SEED, crc32(cell_signalbar)])
    (SEED_RULE), so a setup's draw does not depend on which other setups are in
    the run, the order, the resume point or the process it ran in.

    Returns (anchor_bar int64, -1 = unmatched; level float, nan = unmatched;
    n_draws int32)."""
    n = len(sig)
    pa = np.full(n, -1, dtype=np.int64)
    pv = np.full(n, np.nan)
    nd = np.zeros(n, dtype=np.int32)
    for k in range(n):
        s = int(sig[k])
        ra = int(real_anc[k])
        lo_k = ra if session else int(wlo[s])
        rng = np.random.default_rng([SEED, zlib.crc32(f"{cell}_{s}".encode())])
        a, V, d = match_pseudo_anchor(cum, ref, lo_k, s, ra, float(close[s]),
                                      float(atr0[k]), float(target[k]),
                                      float(side[k]), rng)
        nd[k] = d
        if a is not None:
            pa[k], pv[k] = a, V
    return pa, pv, nd


def code_set(code_dir):
    """The 12-hex code-set hash verify_pc.py logs: sha256 of the per-file
    sha256s of code\\*.py, in sorted order."""
    import glob
    import hashlib
    import os

    def sha(p):
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    files = sorted(glob.glob(os.path.join(code_dir, "*.py")))
    return hashlib.sha256("".join(sha(f) for f in files).encode()).hexdigest()[:12]


def write_manifest(out_dir, code_dir):
    """pseudo_manifest.json beside the rebuild output: the provenance pattern
    floor_v2.py uses (parameters + the code-set that produced them)."""
    import datetime
    import os
    import safeio
    m = {"date": datetime.date.today().isoformat(), "redraw_cap": REDRAW_CAP,
         "tol_frac": TOL_FRAC, "seed": SEED, "seed_rule": SEED_RULE,
         "code_set": code_set(code_dir),
         "distance_target": "the setup's own realized |close - V0| / ATR0 (dist_atr)",
         "side_matching": True,
         "direction_of_travel": "NOT IMPLEMENTED (amendment 2026-09-29 to section 25)"}
    safeio.write_json(m, os.path.join(out_dir, "pseudo_manifest.json"))
