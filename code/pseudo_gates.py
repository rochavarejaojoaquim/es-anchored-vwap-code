"""
pseudo_gates.py -- §25.5 acceptance gates for pseudo_vwap.py, on synthetic
data with known answers. Run BEFORE touching real data (D1).

    python pseudo_gates.py            7 gates
    python pseudo_gates.py mutate     each target must FAIL when broken

Fixtures are built through pipeline.vwap_and_sd -- the REAL production path --
so the gates exercise pipeline.Cum, not a private copy of the arithmetic. The
known answers are hand-computed rational numbers (Python's unlimited ints /
Fraction, the §22.3 method), never read back from cum.line().
"""
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from loader import rolling_argextreme                              # noqa: E402
from pipeline import vwap_and_sd                                    # noqa: E402
from spec import PRICE_SCALE                                        # noqa: E402
from pseudo_vwap import (REDRAW_CAP, draw_candidate_anchor,        # noqa: E402
                          match_pseudo_anchor, vwap_from_anchor, window_lo)

# (high, low, close, volume), all on the 0.25 grid. Typical price = (H+L+C)/3.
BARS_1SEG = [(10.00, 10.00, 10.00, 1),     # typical 10
             (10.50, 9.50, 10.00, 2),      # typical 10
             (12.00, 10.00, 11.00, 3),     # typical 11
             (21.00, 19.00, 20.00, 1),     # typical 20
             (20.75, 19.25, 20.00, 4)]     # typical 20
# two segments (a contract roll after bar 3); segment 1 sits at ~50 so any
# leakage of segment 0's sums into segment 1 is glaring
BARS_2SEG = BARS_1SEG[:4] + [(50.00, 50.00, 50.00, 2),    # typical 50
                             (51.00, 49.00, 50.00, 1),    # typical 50
                             (54.00, 50.00, 52.00, 3),    # typical 52
                             (60.00, 58.00, 59.00, 2)]    # typical 59
SEG_2SEG = [0, 0, 0, 0, 1, 1, 1, 1]


def make_cum(bars, seg):
    """(cum, ref) from the REAL pipeline.vwap_and_sd. The anchor array passed
    in is irrelevant here: only cum and ref are used."""
    d = pd.DataFrame(dict(high=[b[0] for b in bars], low=[b[1] for b in bars],
                          close=[b[2] for b in bars], volume=[b[3] for b in bars],
                          seg=seg))
    _, _, _, cum, ref = vwap_and_sd(d, np.arange(len(d)))
    return cum, ref


def exact_vwap(bars, anchor, at):
    """Independent reference: unlimited-precision integers, no Cum involved.
    On the 0.25 grid 4*(H+L+C) is a whole number and typical = that / 12."""
    num = den = 0
    for h, l, c, v in bars[anchor:at + 1]:
        num += int(round((h + l + c) * 4)) * v
        den += v
    return Fraction(num, 12 * den)


def gate1_no_self_selection(quiet=False):
    """G1: the real anchor is NEVER drawn as a candidate. 10,000 draws."""
    rng = np.random.default_rng(1)
    real_anchor = 40
    hits = sum(draw_candidate_anchor(0, 60, real_anchor, rng) == real_anchor
               for _ in range(10_000))
    ok = hits == 0
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G1  real anchor drawn as "
              f"candidate: {hits} of 10,000 (must be 0)")
    return ok


def gate2_known_vwap(quiet=False):
    """G2: vwap_from_anchor (via the real pipeline.Cum) reproduces HAND-COMPUTED
    rationals. Single segment, BARS_1SEG. The hand values, worked on paper:
      anchor 1, t=4:  (10*2 + 11*3 + 20*1 + 20*4) / (2+3+1+4)   = 153/10
      anchor 0, t=4:  (10*1 + 10*2 + 11*3 + 20*1 + 20*4) / 11   = 163/11
      anchor 3, t=4:  (20*1 + 20*4) / 5                          = 20
      anchor 1, t=2:  (10*2 + 11*3) / 5                          = 53/5
      anchor 4, t=4:  a one-bar line is that bar's typical price = 20
    Each hand value is also re-derived independently with Fractions, so a typo
    in the literal cannot pass silently."""
    cum, ref = make_cum(BARS_1SEG, [0] * 5)
    hand = {(1, 4): Fraction(153, 10), (0, 4): Fraction(163, 11),
            (3, 4): Fraction(20), (1, 2): Fraction(53, 5), (4, 4): Fraction(20)}
    worst, ok = 0.0, True
    for (a, t), want in hand.items():
        ok &= exact_vwap(BARS_1SEG, a, t) == want           # the literal is right
        got = vwap_from_anchor(cum, ref, a, t)
        worst = max(worst, abs(got - float(want)))
    ok = bool(ok and worst < 1e-9)
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G2  5 hand-computed VWAPs via "
              f"pipeline.Cum: worst |error| {worst:.2e} points (need < 1e-9)")
    return ok


def gate3_redraw_cap_and_reporting(quiet=False):
    """G3: an IMPOSSIBLE match (eps far outside anything the window can
    produce) exhausts exactly REDRAW_CAP draws and reports unmatched --
    it does not loop forever, and it does not silently accept a bad match."""
    rng = np.random.default_rng(2)
    cum, ref = make_cum([(100.0, 100.0, 100.0, 1)] * 60, [0] * 60)  # flat: dist 0
    a, V, n = match_pseudo_anchor(cum, ref, 0, 59, 40,
                                   close=100.0, atr0=1.0, eps=999.0,
                                   side=1, rng=rng)   # impossible: eps=999
    ok = (a is None) and (n == REDRAW_CAP)
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G3  impossible match -> "
              f"anchor {a}, draws {n} (want None, {REDRAW_CAP})")
    return ok


def gate4_same_walk_function(quiet=False):
    """G4: pseudo_vwap.py contains NO bar-by-bar touch-detection loop of
    its own. Grep-level check: the module must not define a function whose
    body scans for a wick crossing -- that logic must be imported."""
    import inspect
    import pseudo_vwap
    src = inspect.getsource(pseudo_vwap)
    forbidden = ["low <=", "high >=", "for bar in", "while bar"]
    hit = [f for f in forbidden if f in src]
    ok = len(hit) == 0
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G4  no reimplemented walk "
              f"logic in pseudo_vwap.py (found: {hit or 'none'})")
    return ok


def gate5_cross_segment(quiet=False):
    """G5: correct ACROSS A SEGMENT BOUNDARY (the G7 bug of 2026-09-22: in the
    single-array form the slot at a segment's first bar held the previous
    segment's total, and window volumes came out negative). BARS_2SEG has a roll
    after bar 3. A window itself never spans a roll (Cum's own contract; G7
    below shows the draw window cannot), so the cases are the ones that broke:
      anchor 4 (segment 1's FIRST bar), t=7: (50*2+50*1+52*3+59*2)/8 = 53
      anchor 4, t=6:                         (50*2+50*1+52*3)/6      = 51
      anchor 5, t=7:                         (50*1+52*3+59*2)/6      = 54
      anchor 6, t=7:                         (52*3+59*2)/5           = 274/5
      anchor 2, t=3 (segment 0, unharmed):   (11*3+20*1)/4           = 53/4"""
    cum, ref = make_cum(BARS_2SEG, SEG_2SEG)
    hand = {(4, 7): Fraction(53), (4, 6): Fraction(51), (5, 7): Fraction(54),
            (6, 7): Fraction(274, 5), (2, 3): Fraction(53, 4)}
    worst, ok = 0.0, True
    for (a, t), want in hand.items():
        ok &= exact_vwap(BARS_2SEG, a, t) == want
        worst = max(worst, abs(vwap_from_anchor(cum, ref, a, t) - float(want)))
    ok = bool(ok and worst < 1e-9)
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G5  5 hand-computed VWAPs across a "
              f"segment boundary (incl. anchor ON the first bar): worst |error| "
              f"{worst:.2e}")
    return ok


def gate6_side_matching(quiet=False):
    """G6: every ACCEPTED match sits on the same side of price as the real
    setup (sign(close - V') == side). Every candidate is made exactly eps
    away but on a random side, so half are wrong-sided and must be redrawn.
    Also asserts rejections really happened (some match needed >1 draw)."""
    import pseudo_vwap as pv
    close, atr0, eps = 100.0, 1.0, 2.0
    rng, flip = np.random.default_rng(3), np.random.default_rng(4)
    orig = pv.vwap_from_anchor
    pv.vwap_from_anchor = lambda *a, **k: close + eps * atr0 * (1 if flip.random() < 0.5 else -1)
    bad = matched = multi = 0
    try:
        for side in (1, -1):
            for _ in range(500):
                a, V, n = match_pseudo_anchor(None, None, 0, 59, 40, close, atr0,
                                              eps, side, rng)
                if a is None:
                    continue
                matched += 1
                multi += int(n > 1)
                bad += int(np.sign(close - V) != side)
    finally:
        pv.vwap_from_anchor = orig
    ok = bad == 0 and matched > 900 and multi > 0
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G6  side matching: {bad} wrong-sided "
              f"of {matched} accepted (must be 0); {multi} needed a redraw (must be >0)")
    return ok


def gate7_window_clipping(quiet=False):
    """G7: the draw window is the real anchor's OWN window, clipped at the
    segment start, and comes from loader.rolling_argextreme (one segment rule).
    (a) hand-written expectation, n=12, two segments of 6, N=4: window_lo is
        [-1,-1,-1,0,1,2, -1,-1,-1,6,7,8]  (warm-up = -1; segment 1 never reaches
        back into segment 0).
    (b) on 3,000 random bars the REAL anchor rolling_argextreme returns always
        lies inside [window_lo, i], so a candidate drawn there is never outside
        the real anchor's window."""
    seg = np.array([0] * 6 + [1] * 6)
    want = [-1, -1, -1, 0, 1, 2, -1, -1, -1, 6, 7, 8]
    ok_a = list(window_lo(12, 4, seg)) == want
    r = np.random.default_rng(5)
    n, N = 3000, 60
    sg = np.repeat(np.arange(n // 500), 500)
    px = r.standard_normal(n).cumsum()
    real = rolling_argextreme(px, N, sg, maximum=True)
    lo = window_lo(n, N, sg)
    i = np.arange(n)
    defined = real >= 0
    ok_b = bool(np.array_equal(defined, lo >= 0)
                and np.all((real[defined] >= lo[defined]) & (real[defined] <= i[defined])))
    ok = bool(ok_a and ok_b)
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G7  window clipping: hand case "
              f"{'ok' if ok_a else 'WRONG'}; real anchor inside window on "
              f"{int(defined.sum())} bars: {'ok' if ok_b else 'WRONG'}")
    return ok


def gate8_cell_level(quiet=False):
    """G8: pseudo_for_cell on a synthetic two-segment cell (hh-style anchors).
    (a) every matched line is re-derived INDEPENDENTLY (exact Fractions, no Cum)
        and must equal the stored level; the anchor is inside the real anchor's
        window, in the signal's segment, never the real anchor; distance is
        within +-10% of the target and the side is the real setup's side.
    (b) per-setup seeding: an identical rerun, and a run on only HALF the
        setups, give the same answer for every setup they share.
    (c) unmatched setups are marked (-1 / nan) and some matches exist."""
    import pseudo_vwap as pv
    from pseudo_vwap import pseudo_for_cell
    r = np.random.default_rng(8)
    n, N = 1200, 60
    seg = np.array([0] * 600 + [1] * 600)
    c = 4000 + 0.25 * r.integers(-3, 4, n).cumsum()
    h = c + 0.25 * r.integers(0, 4, n)
    l = c - 0.25 * r.integers(0, 4, n)
    v = r.integers(1, 20, n)
    bars = list(zip(h, l, c, v))
    cum, ref = make_cum(bars, seg)
    real = rolling_argextreme(h, N, seg, maximum=True)
    wlo = window_lo(n, N, seg)
    sig = np.arange(80, n, 7)
    sig = sig[real[sig] >= 0]
    ra = real[sig]
    V0 = np.array([vwap_from_anchor(cum, ref, int(a), int(s)) for a, s in zip(ra, sig)])
    atr0 = np.full(len(sig), 2.0)
    keep = np.abs(c[sig] - V0) / atr0 > 0.3
    sig, ra, V0, atr0 = sig[keep], ra[keep], V0[keep], atr0[keep]
    side = np.sign(c[sig] - V0)
    target = np.abs(c[sig] - V0) / atr0
    pa, pl, nd = pseudo_for_cell(cum, ref, sig, ra, wlo, False, c, atr0, target, side, "SYN")
    m = pa >= 0
    ok_a = bool(m.any())
    for k in np.where(m)[0]:
        s, a = int(sig[k]), int(pa[k])
        ex = float(exact_vwap(bars, a, s))
        dist = abs(c[s] - ex) / atr0[k]
        ok_a &= (abs(pl[k] - ex) < 1e-9 and wlo[s] <= a < s and a != ra[k]
                 and seg[a] == seg[s] and np.sign(c[s] - ex) == side[k]
                 and 0.9 * target[k] - 1e-12 <= dist <= 1.1 * target[k] + 1e-12)
    ok_a = bool(ok_a)
    pa2, pl2, _ = pseudo_for_cell(cum, ref, sig, ra, wlo, False, c, atr0, target, side, "SYN")
    ok_b = bool(np.array_equal(pa, pa2) and np.array_equal(pl, pl2, equal_nan=True))
    half = np.arange(0, len(sig), 2)
    pa3, pl3, _ = pseudo_for_cell(cum, ref, sig[half], ra[half], wlo, False, c,
                                  atr0[half], target[half], side[half], "SYN")
    ok_b &= bool(np.array_equal(pa[half], pa3))
    ok_c = bool(np.all(pl[~m] != pl[~m]) and np.all(pa[~m] == -1) and (~m).any())
    ok = bool(ok_a and ok_b and ok_c)
    if not quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] G8  cell level: {int(m.sum())} of {len(m)} "
              f"matched; independent recompute/window/side/distance "
              f"{'ok' if ok_a else 'WRONG'}; rerun+subset identical "
              f"{'ok' if ok_b else 'WRONG'}; unmatched marked {'ok' if ok_c else 'WRONG'}")
    return ok


if __name__ == "__main__":
    gates = {"G1": gate1_no_self_selection, "G2": gate2_known_vwap,
              "G3": gate3_redraw_cap_and_reporting,
              "G4": gate4_same_walk_function,
              "G5": gate5_cross_segment,
              "G6": gate6_side_matching,
              "G7": gate7_window_clipping,
              "G8": gate8_cell_level}
    if len(sys.argv) > 1 and sys.argv[1] == "mutate":
        print("MUTATION TEST -- each target must FAIL\n")
        allok = True

        orig_draw = draw_candidate_anchor
        draw_candidate_anchor = lambda lo, hi, ra, rng: ra   # forces self-selection
        ok = not gate1_no_self_selection(quiet=True)
        draw_candidate_anchor = orig_draw
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G1 mutated (forced self-selection)")

        orig_vfa = vwap_from_anchor
        vwap_from_anchor = lambda *a, **k: 999.0
        ok = not gate2_known_vwap(quiet=True)
        vwap_from_anchor = orig_vfa
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G2 mutated (wrong VWAP value)")

        orig_match = match_pseudo_anchor
        match_pseudo_anchor = lambda *a, **k: (0, 100.0, 1)   # fakes an instant match
        ok = not gate3_redraw_cap_and_reporting(quiet=True)
        match_pseudo_anchor = orig_match
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G3 mutated (fake instant match)")

        import inspect as _insp
        real_getsource = _insp.getsource
        _insp.getsource = lambda mod: real_getsource(mod) + "\n    for bar in range(1): pass\n"
        ok = not gate4_same_walk_function(quiet=True)
        _insp.getsource = real_getsource
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G4 mutated (injected fake walk loop)")

        def _single_array_bug(cum, ref, anchor_idx, at_idx):
            # the 2026-09-22 G7 bug: the END array used for the anchor's base too,
            # so a segment's first bar subtracts the PREVIOUS segment's total
            W = cum.Ve[at_idx + 1] - cum.Ve[anchor_idx]
            P = cum.Pe[at_idx + 1] - cum.Pe[anchor_idx]
            return float(ref[at_idx] + P / W / PRICE_SCALE)
        vwap_from_anchor = _single_array_bug
        ok = not gate5_cross_segment(quiet=True)
        vwap_from_anchor = orig_vfa
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G5 mutated (single-array segment bug)")

        import pseudo_vwap as _pv
        orig_ss = _pv.same_side
        _pv.same_side = lambda *a, **k: True                  # side check switched off
        ok = not gate6_side_matching(quiet=True)
        _pv.same_side = orig_ss
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G6 mutated (side check disabled)")

        orig_wlo = window_lo
        window_lo = lambda n, N, seg: rolling_argextreme(     # segment rule ignored
            np.arange(n, dtype=float), N, np.zeros(n, dtype=int), maximum=False)
        ok = not gate7_window_clipping(quiet=True)
        window_lo = orig_wlo
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G7 mutated (window ignores segments)")

        class _CallOrderSeed:                 # seed depends on call order, not the setup
            n = 0

            @staticmethod
            def crc32(_b):
                _CallOrderSeed.n += 1
                return _CallOrderSeed.n
        orig_zlib = _pv.zlib
        _pv.zlib = _CallOrderSeed
        ok = not gate8_cell_level(quiet=True)
        _pv.zlib = orig_zlib
        allok &= ok
        print(f"  [{'ALIVE' if ok else 'DEAD'}] G8 mutated (per-setup seed replaced by call order)")

        print(f"\n  {'ALL GATES ALIVE' if allok else 'A GATE IS DEAD'}")
        sys.exit(0 if allok else 1)

    print("PSEUDO-VWAP GATES -- synthetic data, known answers\n")
    results = {k: g() for k, g in gates.items()}
    print(f"\n  {sum(results.values())} of {len(results)} gates pass")
    sys.exit(0 if all(results.values()) else 1)
