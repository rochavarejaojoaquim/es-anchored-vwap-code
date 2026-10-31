"""
linearity.py -- R12, the author's definition of how LINEAR the anchored line was
between the anchor bar and the signal bar (2026-09-21).

  chord_t     straight line from V_anchor to V_signal
  dev_t       V_t - chord_t
  lin_maxdev  max |dev_t| / ATR0        the biggest gap from the chord
  lin_meandev mean |dev_t| / ATR0       how bowed the line is overall
              (mean over every bar anchor..signal, endpoints included)
  chord_slope (V_sig - V_anchor) / (sig - anchor) / ATR0   ATR per bar
  *_s         SETTLED versions (decision 9 = option a): the chord is drawn
              from bar anchor+SETTLE to the signal, so the line's violent
              birth neither enters the deviations nor tilts the chord.
              NaN when fewer than 2 points remain.

Continuous; no threshold is chosen here (E1). Causal: anchor -> signal only.
L_all is KEPT beside these -- it measures the single most violent jump, a
different thing; the data decides between them.

Example: V goes 100, 101, 102, 103, 104 (a straight line) -> every dev is 0,
lin_maxdev = 0. V goes 100, 101, 105, 103, 104 -> the chord at the middle bar
is 102, the line is at 105: lin_maxdev = 3 / ATR0.

D3 known numbers (gates.py): straight line -> 0.000; a planted hump of height
h on a straight line -> lin_maxdev = h exactly.
"""
import numpy as np



def lin_of(V, atr0):
    """(maxdev, meandev, chord_slope) for one line segment V[0..L]."""
    L = len(V) - 1
    if L < 1 or not np.all(np.isfinite(V)):
        return np.nan, np.nan, np.nan
    chord = V[0] + (V[-1] - V[0]) * np.arange(L + 1) / L
    dev = np.abs(V - chord)
    return dev.max() / atr0, dev.mean() / atr0, (V[-1] - V[0]) / L / atr0


def line_path(cum, ref, anc, sig):
    """The anchored VWAP from the anchor bar to the signal bar, rebuilt from the
    SAME exact integer cumulative sums as pipeline.vwap_and_sd."""
    t = np.arange(anc, sig + 1)
    W = cum.vol(anc, t)
    V = cum.line(anc, t, W) + ref[sig]
    V[W <= 0] = np.nan
    return V


def linearity_at(cum, ref, sig, anc, atr0, side, settle):
    n = len(sig)
    out = {k: np.full(n, np.nan) for k in
           ("lin_maxdev", "lin_meandev", "chord_slope",
            "lin_maxdev_s", "lin_meandev_s", "chord_slope_s")}
    for k in range(n):
        V = line_path(cum, ref, int(anc[k]), int(sig[k]))
        a, b, c = lin_of(V, atr0[k])
        out["lin_maxdev"][k], out["lin_meandev"][k], out["chord_slope"][k] = a, b, c
        if len(V) - 1 - settle >= 1:
            a, b, c = lin_of(V[settle:], atr0[k])
            out["lin_maxdev_s"][k], out["lin_meandev_s"][k], out["chord_slope_s"][k] = a, b, c
    out["chord_slope_xs"] = out["chord_slope"] * side      # + = line moving TOWARD price
    out["chord_slope_s_xs"] = out["chord_slope_s"] * side
    return out
