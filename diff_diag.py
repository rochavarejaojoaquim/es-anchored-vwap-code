r"""
diff_diag.py -- why do 4,742 bars differ between the old dump's setups and the
new dump's candidates? (follow-up to compare_old_new.py, 2026-09-23)

compare_old_new reported, per differing bar, how far its distance sat from eps,
and found bars far beyond it. That was the WRONG measurement: a candidate is the
FIRST bar of an excursion, so if the arithmetic nudges one bar across eps, the
excursion's starting bar SHIFTS -- both the old start and the new start are then
"differing", and the inner one can be far beyond eps.

This measures the right thing instead:
  * for every differing bar, the distance IN BARS to the nearest bar of the
    other set (1 = the start of the same excursion moved by one bar);
  * for paired bars, the distance of the bar that actually sits at the
    threshold (the one that decided it);
  * unpaired bars -- excursions that exist in one build and not the other --
    counted separately, since those are the only ones that are not a shift.

    python diff_diag.py                      (a spread of 6 cells)
    python diff_diag.py N100_hv_SD_1.0       (any cells you name)

Reads data\dump_old_rule and recomputes the new candidates; writes nothing.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from loader import load, rolling_argextreme                      # noqa: E402
from pipeline import session_anchor, vwap_and_sd                 # noqa: E402
from rearm import eligible, candidates                           # noqa: E402
from spec import SD_MIN_BAND                                     # noqa: E402
from spec import t_stop, H_REC_MULT                              # noqa: E402

def shift_gaps(diff, sn, so):
    """For each differing bar, the distance IN BARS to the nearest bar of the
    OTHER set. 1 = the same excursion, started one bar later or earlier."""
    out = []
    srt = {True: np.asarray(sorted(so), dtype=np.int64),      # sorted ONCE, not
           False: np.asarray(sorted(sn), dtype=np.int64)}     # once per bar
    for b in diff:
        other = srt[b in sn]
        k = np.searchsorted(other, b)
        near = [int(other[i]) for i in (k - 1, k) if 0 <= i < len(other)]
        out.append((min(abs(x - int(b)) for x in near) if near else 10 ** 9, near))
    return out


def old_arithmetic(d, anchor):
    """The PRE-decision-B line and SD: float32 typical prices, floating cumulative
    sums centred on each segment's first bar. This is what wrote the old dump."""
    vp = d.vp.to_numpy().astype(np.float32).astype(float)
    v = d.volume.to_numpy().astype(float); seg = d.seg.to_numpy()
    u = np.unique(seg)
    ref = np.array([dict(zip(u, vp[np.searchsorted(seg, u)]))[s_] for s_ in seg])
    x = vp - ref
    LD = np.longdouble
    cV = np.r_[0.0, np.cumsum(v)]
    c1 = np.concatenate([[LD(0)], np.cumsum((x * v).astype(LD))])
    c2 = np.concatenate([[LD(0)], np.cumsum((x * x * v).astype(LD))])
    i_ = np.arange(len(d)); safe = np.maximum(anchor, 0)
    W = cV[i_ + 1] - cV[safe]
    have = (anchor >= 0) & (W > 0)
    m1 = np.where(have, (c1[i_ + 1] - c1[safe]) / np.maximum(W, 1e-9), np.nan)
    m2 = np.where(have, (c2[i_ + 1] - c2[safe]) / np.maximum(W, 1e-9), np.nan)
    return (m1 + ref).astype(float), np.sqrt(np.maximum((m2 - m1 ** 2).astype(float), 0.0))


def gate(b, dist_new, dist_old, sd_new, sd_old, atr, trig, eps):
    """Which gate changed AT THIS BAR, if any?"""
    if (sd_new[b] > 0) != (sd_old[b] > 0):
        return "sd>0 flipped (AUD-37)"
    if trig == "SD" and ((sd_new[b] >= SD_MIN_BAND * atr[b]) != (sd_old[b] >= SD_MIN_BAND * atr[b])):
        return "band >= 1 ATR flipped"
    dn, do = dist_new[b], dist_old[b]
    if (dn >= eps) != (do >= eps) and not (np.isnan(dn) or np.isnan(do)):
        return "distance crossed eps"
    return None


def why(b, dist_new, dist_old, sd_new, sd_old, atr, trig, eps, ab_new, ab_old):
    """A candidate fires only if the PREVIOUS bar was not already beyond eps,
    so a flip at bar b-1 moves the fire onto b (or removes it) while b's own
    gates look untouched. Look at this bar first, then one bar back."""
    g = gate(b, dist_new, dist_old, sd_new, sd_old, atr, trig, eps)
    if g:
        return "at this bar: " + g
    if b > 0 and ab_new[b - 1] != ab_old[b - 1]:
        g2 = gate(b - 1, dist_new, dist_old, sd_new, sd_old, atr, trig, eps)
        return "bar before was beyond-eps in one build only -> " + (g2 or "a gate further back")
    return "unexplained -- neither this bar nor the one before"


def selftest():
    """Known numbers: {100,500,900} vs {101,500,2000} -> 100/101 are one bar
    apart (a shift), 900 and 2000 have no partner (real differences)."""
    sn, so = {100, 500, 900}, {101, 500, 2000}
    diff = np.array(sorted(sn ^ so), dtype=np.int64)
    g = [x[0] for x in shift_gaps(diff, sn, so)]
    # 100 and 101 are one bar apart; 900's nearest is 500 (400 away) and
    # 2000's is 900 (1,100 away) -- far, i.e. NOT a shift. "Unpaired" means
    # "no bar of the other set nearby", not "no bar at all".
    ok = sorted(g) == [1, 1, 400, 1100]
    print(f"  [{'PASS' if ok else 'FAIL'}]  self-test of the pairing: expected "
          f"gaps [1, 1, 400, 1100], got {sorted(g)}")
    return ok


HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "data", "dump_old_rule")
# ONE cell by default -- the worst one in compare_old_new (179 differing bars).
# Each EXTRA (lookback, anchor) group costs another ~2 minutes, because the
# anchor and the line have to be rebuilt over all 4.25M bars. Name more cells
# only if you want them, and group them by lookback+anchor to share that work.
CELLS = ["N100_hv_SD_1.0"]

if __name__ == "__main__":
    cells = [c for c in sys.argv[1:] if not c.startswith("--")] or CELLS
    if not selftest():
        raise SystemExit("pairing self-test failed -- the numbers below would be meaningless")
    if "--selftest" in sys.argv:
        raise SystemExit(0)
    import time
    t0 = time.time()
    print("  loading data ...", flush=True)
    d = load("train")
    seg = d.seg.to_numpy(); atr = d.atr.to_numpy(); close = d.close.to_numpy()
    cache = {}
    tot_pair = tot_unpair = 0
    print(f"  {'cell':<22} {'differ':>7} {'shift<=1':>9} {'shift<=3':>9} "
          f"{'unpaired':>9}  worst threshold gap")
    for cell in cells:
        N, akey, trig, eps = (int(cell.split("_")[0][1:]), cell.split("_")[1],
                              cell.split("_")[2], float(cell.split("_")[3]))
        if (N, akey) not in cache:
            cache.clear()
            print(f"  building the {akey} line at N={N} over {len(d):,} bars "
                  f"(~2 min) ... [{time.time() - t0:.0f}s]", flush=True)
            a = session_anchor(d) if akey == "session" else rolling_argextreme(
                d["high" if akey == "hh" else ("low" if akey == "ll" else "volume")]
                .to_numpy().astype(float), N, seg, akey != "ll")
            vw, sd, *_ = vwap_and_sd(d, a)
            vw_o, sd_o = old_arithmetic(d, a)
            cache[(N, akey)] = (a, vw, sd, vw_o, sd_o)
        a, vw, sd, vw_o, sd_o = cache[(N, akey)]
        el = eligible(d, a, vw, sd, atr, trig)
        scale = sd if trig == "SD" else atr
        dist = np.where(el, np.abs(close - vw) / np.maximum(scale, 1e-9), np.nan)
        new = candidates(dist, a, el, eps, len(d), H_REC_MULT * t_stop(trig, eps))
        old = pd.read_parquet(os.path.join(OLD, cell + ".parquet"), columns=["bar", "is_real"])
        old = old[old.is_real == 1].bar.to_numpy()
        sn, so = set(new.tolist()), set(old.tolist())
        diff = np.array(sorted(sn ^ so), dtype=np.int64)
        if not len(diff):
            print(f"  {cell:<22} {0:>7}"); continue
        el_o = eligible(d, a, vw_o, sd_o, atr, trig)
        scale_o = sd_o if trig == "SD" else atr
        dist_o = np.where(el_o, np.abs(close - vw_o) / np.maximum(scale_o, 1e-9), np.nan)
        from collections import Counter
        ab_n = np.where(np.isnan(dist), False, dist >= eps)
        ab_o = np.where(np.isnan(dist_o), False, dist_o >= eps)
        causes = Counter(why(int(b), dist, dist_o, sd, sd_o, atr, trig, eps, ab_n, ab_o)
                         for b in diff)
        gaps, thr = [], []
        for b, (g, near) in zip(diff, shift_gaps(diff, sn, so)):
            gaps.append(g)
            if g <= 3:                       # of the pair, one bar decided it
                cands = [abs(dist[b] - eps)] + [abs(dist[x] - eps) for x in near]
                thr.append(np.nanmin(cands) / eps)
        gaps = np.array(gaps)
        p1 = int((gaps <= 1).sum()); p3 = int((gaps <= 3).sum()); up = int((gaps > 3).sum())
        tot_pair += p3; tot_unpair += up
        w = np.nanmax(thr) if thr else float("nan")
        print(f"  {cell:<22} {len(diff):>7} {p1:>9} {p3:>9} {up:>9}  {w:.2e}")
        for k, c_ in causes.most_common():
            print(f"        {c_:>6}  {k}")
        far = np.sort(gaps[gaps > 3])
        if len(far):
            print(f"        unpaired bars, distance to the nearest bar of the other set: "
                  f"median {int(np.median(far))}, max {int(far.max())} bars "
                  f"(T = {t_stop(trig, eps)}); a gap under T is the same stretch of price")
    print(f"\n  paired (a shifted excursion start): {tot_pair} | unpaired: {tot_unpair}")
    print("  'worst threshold gap' = |dist - eps| / eps of the bar that DECIDED each pair;")
    print("  tiny = the arithmetic change alone moved it, which is the expected story.")
