r"""
unmatched_adjust.py -- does the unmatched-vs-matched outcome difference SURVIVE
comparing like with like?  Follow-up to unmatched_outcomes.py (same estimator,
same trade-date clustering; real-line outcomes only, never a pseudo fill).

    python unmatched_adjust.py --selftest
    python unmatched_adjust.py DUMP_DIR

Strata, each a refinement of the last (an unmatched setup is compared only with
MATCHED setups in its own stratum; setups with no matched partner are dropped and
COUNTED):
  S0  cell x block                              (the original)
  S1  S0 x side                                 (+1 price above the line, -1 below)
  S2  S1 x distance quintile (dist_atr, within S1)
  S3  S2 x anchor-age quintile (age, within S1)
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from unmatched_outcomes import Acc, residuals

LEVELS = ["S0 cell x block", "S1 + side", "S2 + dist quintile", "S3 + age quintile"]


def strata_levels(blk, side, dist, age):
    """-> list of int-code arrays, one per LEVELS entry.  Quintiles are cut WITHIN
    (block, side), over matched AND unmatched setups together, by rank."""
    df = pd.DataFrame(dict(b=(blk == "B").astype(int), s=(side > 0).astype(int), d=dist, a=age))
    k1 = df.b * 2 + df.s
    q = lambda col: df.groupby(k1)[col].transform(lambda v: np.minimum((v.rank(method="first") - 1) * 5 // max(len(v), 1), 4)).astype(int)
    qd, qa = q("d"), q("a")
    s0 = df.b.to_numpy()
    s1 = k1.to_numpy()
    s2 = (k1 * 5 + qd).to_numpy()
    s3 = (s2 * 5 + qa).to_numpy()
    return [s0, s1, s2, s3]


def selftest():
    print("SELF-TEST (distance drives the outcome; the unmatched simply sit at larger distances)\n")
    rng = np.random.default_rng(5)
    res = []

    def check(name, ok):
        res.append(bool(ok))
        print(f"  [{'PASS'if ok else 'FAIL'}] {name}")
    n, D = 60000, 400
    dist = rng.gamma(4, 1, n)
    side = rng.choice([-1, 1], n)
    age = rng.integers(1, 300, n).astype(float)
    blk = np.where(rng.random(n) < 0.6, "A", "B")
    um = rng.random(n) < (0.03 + 0.25 * (dist > np.percentile(dist, 70)))       # unmatched sit at LARGE distance
    g = rng.integers(0, D, n)

    def eff(y, lev, levels):
        idx, r, refv = residuals(y, um, levels[lev])
        a = Acc(D)
        a.add(g[idx], r, refv)
        return a.result(), len(idx) / um.sum()
    levels = strata_levels(blk, side, dist, age)
    y = (rng.random(n) < np.clip(0.75 - 0.06 * dist, 0.02, 0.98)).astype(float)   # fill falls with distance; NO unmatched effect
    e0, _ = eff(y, 0, levels)
    e2, kept2 = eff(y, 2, levels)
    check(f"K1 outcome depends on distance only: S0 shows a spurious {100*e0['eff']:+.1f} pp, "
          f"S2 (distance quintiles) removes it: {100*e2['eff']:+.2f} pp [{100*e2['lo']:+.2f}, {100*e2['hi']:+.2f}] "
          f"(kept {100*kept2:.0f}% of unmatched)", e0["eff"] < -0.03 and e2["lo"] < 0.0 < e2["hi"] + 0.004)
    y2 = y + rng.normal(0, 0, n)
    y2 = (rng.random(n) < np.clip(0.75 - 0.06 * dist - 0.05 * um, 0.02, 0.98)).astype(float)   # a real -5 pp on top
    e2b, _ = eff(y2, 2, levels)
    check(f"K2 a REAL -5 pp unmatched effect on top of the distance effect survives S2: {100*e2b['eff']:+.2f} pp "
          f"[{100*e2b['lo']:+.2f}, {100*e2b['hi']:+.2f}]", e2b["lo"] < -0.05 < e2b["hi"] + 0.02 and e2b["hi"] < 0)
    noise = rng.random(n)
    lv = strata_levels(blk, side, noise, age)                                     # MUTATION: adjust on the WRONG variable
    e_bad, _ = eff(y, 2, lv)
    check(f"M1 stratifying on an unrelated variable instead of distance leaves the spurious {100*e_bad['eff']:+.1f} pp in place",
          e_bad["eff"] < -0.03)
    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


COLS = ["is_real", "pseudo_matched", "day", "block", "T", "w_0.0", "side", "dist_atr", "age", "gap_at_T", "cens_T"]


def main(dump):
    files = [f for f in sorted(glob.glob(os.path.join(dump, "N*.parquet"))) if ".m2." not in f]
    days = set()
    for f in files:
        days.update(pd.read_parquet(f, columns=["day"])["day"].unique().tolist())
    days = pd.Index(sorted(days))
    D = len(days)
    acc = {}          # (metric, level, group) -> Acc
    kept = {}         # (level, group) -> [kept unmatched rows, unmatched rows]
    sideshare = {}    # group -> [sum of side>0 among unmatched, n unmatched, sum among matched, n matched]

    def A(key):
        if key not in acc:
            acc[key] = Acc(D)
        return acc[key]
    for f in files:
        cell = os.path.basename(f)[:-8]
        mt = re.match(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)$", cell)
        N, akey = int(mt.group(1)), mt.group(2)
        if akey == "session" and N != 60:
            continue                                    # session family counted once (HANDOFF 22.4 #14)
        x = pd.read_parquet(f, columns=COLS)
        x = x[x.is_real == 1].reset_index(drop=True)
        um = (x.pseudo_matched != 1).to_numpy()
        T = x["T"].to_numpy()
        w = x["w_0.0"].to_numpy()
        hit = ((w >= 1) & (w <= T)).astype(float)
        gap = x["gap_at_T"].to_numpy(float)
        sd = x["side"].to_numpy(float)
        lev = strata_levels(x.block.to_numpy(), sd, x.dist_atr.to_numpy(float), x["age"].to_numpy(float))
        g = days.get_indexer(x["day"])
        for grp in ("ALL", f"anchor {akey}"):
            ss = sideshare.setdefault(grp, [0.0, 0, 0.0, 0])
            ss[0] += float((sd[um] > 0).sum()); ss[1] += int(um.sum())
            ss[2] += float((sd[~um] > 0).sum()); ss[3] += int((~um).sum())
            for li, st in enumerate(lev):
                for mname, y in (("fill", hit), ("gap_at_T", gap)):
                    idx, r, refv = residuals(y, um, st)
                    A((mname, li, grp)).add(g[idx], r, refv)
                    if mname == "fill":
                        k = kept.setdefault((li, grp), [0, 0])
                        k[0] += len(idx); k[1] += int(um.sum())
    print("STAMP: unmatched vs matched, REAL-line outcomes only | DEDUP set (session family counted once, N=60): "
          f"144 cells | independent units: {D:,} trade dates | clustered on trade date | raw (no null, no edge) | "
          "ES, train blocks A+B only, sealed 2018-2022 untouched")
    print("effect = unmatched MINUS matched in the same stratum; quintiles cut within (block, side) over matched+unmatched together\n")
    print("SIDE MIX: share of setups with side = +1 (price ABOVE the line)")
    for grp, s in sideshare.items():
        print(f"  {grp:<16} unmatched {100*s[0]/s[1]:5.1f}%   matched {100*s[2]/s[3]:5.1f}%   (unmatched rows {s[1]:,})")
    for mname, scale, unit in (("fill", 100, "pp"), ("gap_at_T", 1, "ATR0")):
        print(f"\n{mname}  ({unit})")
        print(f"  {'group':<16}{'stratification':<24}{'kept':>7}{'effect':>10}{'95% CI':>22}")
        for grp in ("ALL", "anchor hh", "anchor ll", "anchor hv", "anchor session"):
            for li, name in enumerate(LEVELS):
                r = acc[(mname, li, grp)].result()
                kp = kept[(li, grp)]
                print(f"  {grp:<16}{name:<24}{100*kp[0]/kp[1]:>6.1f}%{scale*r['eff']:>+10.3f}   [{scale*r['lo']:+.3f}, {scale*r['hi']:+.3f}]")
            print()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if not a:
        raise SystemExit(__doc__)
    if not selftest():
        raise SystemExit("\nSELF-TEST FAILED -- no data was read.")
    print()
    main(a[0])
