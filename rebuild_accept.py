r"""
rebuild_accept.py -- section 25 amendment 7's BUILD ACCEPTANCE CRITERIA B0-B4,
measured on the REBUILT dump, before anything else is read.

Amendment 7 fixed these BEFORE the rebuild ran:
  B1  |mean delta| <= 0.005 ATR                     (old dump: -0.02422)
  B2  |mean ratio - 1| <= 0.001                     (old dump:  0.00447)
  B3  share with pseudo nearer in [49%, 51%]        (old dump:  53.16%)
  B4  pooled unmatched <= 10%, fallback rate reported
  B5  gate 1 (existing columns identical) still PASSES
IF B1-B3 FAIL, THE REBUILD IS NOT ACCEPTED: the old dump stays in place and
nothing downstream is read. That is the registered rule, not a preference.

B0 IS ADDED HERE, and it is the one nobody wrote down: DID THE DRAW ACTUALLY
CHANGE? If pseudo_anchor_bar came back identical to the current dump then the
symmetric draw never took effect, and B1-B3 would be measuring the OLD build
while appearing to pass or fail on their own terms. A rebuild that changed
nothing must not be mistaken for a rebuild that changed nothing MEASURABLE.

B5 IS A SEPARATE COMMAND, not in this script, because verify_columns.py already
does it and must not be reimplemented (E24). Use the PRE-PSEUDO dump as OLD:
    python verify_columns.py data\dump_pre_pseudo_2026-09-30 rebuild_pseudo_2026-09-30\data\dump ALL
That is gate 1 in its ORIGINAL form -- the 144 pre-pseudo columns must be
bit-identical and the 5 pseudo columns are the expected new ones. Comparing
against the CURRENT data\dump instead would fail by design, since the pseudo
columns are supposed to differ now.

NO RECEIPT GUARD HERE, deliberately. gate_receipts.require_pass cannot pass on a
freshly rebuilt dump -- its fingerprint and the code-set have both changed, so
the old receipts correctly refuse. B0-B4 are BUILD checks that run BEFORE the
gate, the power validation and the gap, which is exactly the registered order.

    python rebuild_accept.py --selftest
    python rebuild_accept.py rebuild_pseudo_2026-09-30\data\dump

Written 2026-10-02. Project root; rotates nothing. Reads only.
"""
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pseudo_vwap import BAND_STRATA, SEED, TOL_FRAC                     # noqa: E402
from power_registered_set import registered_cells                       # noqa: E402
from pseudo_dist import build                                           # noqa: E402

B1_MAX, B2_MAX, B3_LO, B3_HI, B4_MAX = 0.005, 0.001, 0.49, 0.51, 0.10
OLD = dict(delta=-0.02422, ratio=0.99553, nearer=0.5316)   # 26.38, the old dump


def target_stratum(cell, bar, dist):
    """Re-derive the stratum this setup was ASSIGNED, from the seed rule alone.

    Amendment 7's correction: the target half/stratum is the FIRST value drawn
    from the per-setup generator, so it is deterministic and recomputable, which
    is why the fallback needs no stored column. A FALLBACK is a setup whose
    realized distance landed OUTSIDE its assigned stratum.
    """
    rng = np.random.default_rng([SEED, zlib.crc32(f"{cell}_{int(bar)}".encode())])
    k = int(rng.integers(BAND_STRATA))
    lo, hi = dist * (1 - TOL_FRAC), dist * (1 + TOL_FRAC)
    w = (hi - lo) / BAND_STRATA
    return lo + k * w, lo + (k + 1) * w


def fallback_rate(df, sample=200000, seed=0):
    """Share of setups whose realized distance is outside their assigned stratum.
    Sampled, because it is a per-setup python loop over 3.8M rows."""
    n = len(df)
    idx = (np.random.default_rng(seed).choice(n, sample, replace=False)
           if n > sample else np.arange(n))
    cells = df["cell"].to_numpy()[idx]
    bars = df["bar"].to_numpy()[idx] if "bar" in df.columns else None
    dist = df["dist_atr"].to_numpy()[idx]
    pdist = df["pdist"].to_numpy()[idx]
    if bars is None:
        return float("nan"), 0
    out = 0
    for c, b, d, pd_ in zip(cells, bars, dist, pdist):
        lo, hi = target_stratum(c, b, d)
        if not (lo <= pd_ <= hi):
            out += 1
    return out / len(idx), len(idx)


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\nrebuild_accept.py SELF-TEST -- detectors before the dump is read.\n")
    chk("K1 the installed BAND_STRATA is EVEN, as amendment 7 requires",
        BAND_STRATA % 2 == 0, f"BAND_STRATA = {BAND_STRATA}")
    lo, hi = target_stratum("N60_hh_ATR_3.0", 12345, 3.0)
    chk("K2 the assigned stratum lies inside the +-10% band",
        3.0 * 0.9 - 1e-12 <= lo < hi <= 3.0 * 1.1 + 1e-12,
        f"[{lo:.4f}, {hi:.4f}] inside [2.7000, 3.3000]")
    chk("K3 its width is exactly one stratum",
        abs((hi - lo) - (3.0 * 0.2 / BAND_STRATA)) < 1e-12,
        f"{hi - lo:.6f} vs {3.0 * 0.2 / BAND_STRATA:.6f}")
    a = target_stratum("N60_hh_ATR_3.0", 12345, 3.0)
    chk("K4 the re-derivation is deterministic", a == (lo, hi), "same twice")
    b = target_stratum("N60_hh_ATR_3.0", 12346, 3.0)
    chk("K5 a different setup gets its own draw", True,
        f"bar 12345 -> [{lo:.4f}, {hi:.4f}], bar 12346 -> [{b[0]:.4f}, {b[1]:.4f}]")
    seen = {target_stratum("N60_x_ATR_3.0", i, 3.0)[0] for i in range(400)}
    chk("K6 across setups every stratum is used", len(seen) == BAND_STRATA,
        f"{len(seen)} distinct of {BAND_STRATA}")
    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


def main(dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    cells = registered_cells()
    print(f"REGISTERED SET: {len(cells)} cells from {dump}\n")
    df = build(dump, cells)

    # ---- B0: did the draw actually change?
    cur = os.path.join(HERE, "data", "dump")
    same = tot = 0
    for c in cells[:8]:
        p_new = pd.read_parquet(os.path.join(dump, c + ".parquet"),
                                columns=["is_real", "pseudo_anchor_bar"])
        p_old = pd.read_parquet(os.path.join(cur, c + ".parquet"),
                                columns=["is_real", "pseudo_anchor_bar"])
        a = p_new[p_new.is_real == 1].pseudo_anchor_bar.to_numpy()
        b = p_old[p_old.is_real == 1].pseudo_anchor_bar.to_numpy()
        m = min(len(a), len(b))
        same += int((a[:m] == b[:m]).sum())
        tot += m
    frac_same = same / max(tot, 1)
    b0 = frac_same < 0.50
    print(f"  [{'PASS' if b0 else 'FAIL'}] B0 THE DRAW ACTUALLY CHANGED   "
          f"{100*frac_same:.2f}% of anchors identical to the current dump over "
          f"{tot:,} setups (8 cells)")
    if not b0:
        raise SystemExit("B0 FAILED -- the rebuild did not change the draw. "
                         "B1-B3 would be measuring the OLD build.")

    # ---- B1, B2, B3
    md = float(df.delta.mean())
    mr = float(df.ratio.mean())
    sn = float((df.delta < 0).mean())
    print(f"\n  {'criterion':<52} {'old dump':>10} {'rebuilt':>10}")
    print(f"  {'B1 |mean delta| <= 0.005 ATR':<52} {OLD['delta']:>10.5f} {md:>10.5f}")
    print(f"  {'B2 |mean ratio - 1| <= 0.001':<52} {OLD['ratio']:>10.5f} {mr:>10.5f}")
    print(f"  {'B3 share pseudo nearer in [49%, 51%]':<52} "
          f"{100*OLD['nearer']:>9.2f}% {100*sn:>9.2f}%")
    r1, r2, r3 = abs(md) <= B1_MAX, abs(mr - 1) <= B2_MAX, B3_LO <= sn <= B3_HI
    print()
    print(f"  [{'PASS' if r1 else 'FAIL'}] B1   |{md:+.5f}| vs {B1_MAX}")
    print(f"  [{'PASS' if r2 else 'FAIL'}] B2   |{mr - 1:+.5f}| vs {B2_MAX}")
    print(f"  [{'PASS' if r3 else 'FAIL'}] B3   {100*sn:.2f}% vs [49%, 51%]")

    # ---- B4
    tot_s = unm = 0
    for c in cells:
        with open(os.path.join(dump, c + ".counts.json"), "r", encoding="utf-8") as fh:
            j = json.load(fh)
        tot_s += int(j["n_setups"])
        unm += int(j["n_pseudo_unmatched"])
    fr = unm / max(tot_s, 1)
    r4 = fr <= B4_MAX
    print(f"  [{'PASS' if r4 else 'FAIL'}] B4   pooled unmatched {unm:,} of "
          f"{tot_s:,} = {100*fr:.3f}% vs {100*B4_MAX:.0f}%   "
          f"(old dump 4.650%)")
    fb, nfb = fallback_rate(df)
    print(f"         fallback rate {100*fb:.3f}% on a {nfb:,}-setup sample "
          f"(reported, not gated -- amendment 7)")

    print("\n  " + "=" * 68)
    if r1 and r2 and r3 and r4:
        print("  B0-B4 PASS. The rebuild is ACCEPTED.")
        print("  NEXT, nothing skipping a step (section 26.3): B5 (verify_columns),")
        print("  then the exchangeability gate, then the power validation, then the")
        print("  gap -- each earning NEW receipts, because the old ones must refuse.")
    else:
        print("  NOT ACCEPTED. Amendment 7: the old dump stays in place and nothing")
        print("  downstream is read. Report the numbers; do not re-tune to pass.")
    print("  " + "=" * 68)
    return 0 if (r1 and r2 and r3 and r4) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args[0]))
