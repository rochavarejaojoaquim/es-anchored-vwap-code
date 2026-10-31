"""
paper_eps_tercile.py -- claim 1 by tercile of distance in TICKS, split by EPS GROUP (author note 8, 2026-10-15).

WHY. paper_robustness.py R4 splits claim 1 (M1b twins) by tercile of distance in ticks (dist_atr x atr0 / 0.25).
Distance in ticks is eps times ATR in ticks, so each tercile mixes eps cells and volatility states (reviewer,
round 4, C4). This script holds eps group fixed inside each tercile. TRAINING periods only (blocks A and B).
The held-out period is NOT read and must not be re-run without a dated amendment (A15).

IMPORTS UNCHANGED: paper_robustness.load / arm / gap / fmt (and through it claim_checks.py). The tercile edges are
the SAME as R4 (pooled over both periods on the M1b-arm setups), so the two tables are comparable cell for cell.

EPS GROUPS (the same definition as R5's small-eps corner, fixed before this script was written):
  small = ATR eps 3.0 or 4.0, or SD eps 1.0 or 1.5   (83.5% of setups, section 25 amendment 2)
  large = every other registered cell (ATR 6.0, 8.0, 12.0; SD 2.0, 3.0)
A finer table by (trigger, eps) is printed with the same omission rule and NO reading.

READING RULE, WRITTEN BEFORE ANY RUN (descriptive; no test, no cutoff other than the omission rule):
 (a) For each eps group and each tercile with at least 1,000 setups in BOTH periods, the claim-1 gap per period
     (day-bootstrap 95% interval), the share of the period, and the older-minus-recent difference (normal
     approximation on disjoint dates, as in R4). A tercile with fewer than 1,000 setups in either period is
     omitted from that group and listed.
 (b) For group g let C_g = (older-minus-recent in the smallest-tick tercile) minus (older-minus-recent in the
     largest-tick tercile), with an interval by normal approximation that treats the two terciles as independent
     (they share trade dates, so this is approximate). Four phrases, the only reading made:
       "persists"                       both terciles reported, C_g > 0 and the lower interval end of C_g > 0;
       "ordered as in R4, not distinguishable"   both reported, C_g > 0 but the lower end <= 0;
       "not ordered as in R4"           both reported and C_g <= 0;
       "cannot be read"                 either tercile omitted (it MUST then be reported as omitted).
     Also printed, no reading: C_g divided by the pooled R4 contrast (+4.40 - (-0.96) = +5.36), the magnitude of the
     tercile pattern that remains inside the group.
 (c) For a group with all three terciles reported, the two index-number decompositions of the older-to-recent
     decline (older gaps at the recent mix; recent gaps at the older mix) are printed. No reading is made of them;
     in the paper they stay labelled post hoc and descriptive, both orders printed.
 (d) FINER table (trigger x eps cells): printed with NO per-line reading. The only summary is the COUNT of
     (trigger, eps) cells in which both the smallest- and the largest-tick tercile are reported, and in how many of
     them the smallest-tick older-minus-recent difference exceeds the largest-tick one (point estimates). All lines
     are reported; none may be quoted alone.
 (e) Nothing here changes a registered statistic or verdict. Whatever it shows is reported in Section 3.2 beside R4.

DISCLOSURE (E1/E3). This is a follow-up designed AFTER R4's output (+4.40 / +0.65 / -0.96) had been seen: the
two-group split, the choice to read only the smallest against the largest tercile, and the 1,000-setup omission
floor (inherited from R4, not derived) were all fixed with R4 in view. Exploratory. "Written before any run of THIS
script" is true; it is not the relevant fact.
WHAT IT CANNOT SAY. Within one eps cell the variation in ticks is ATR0 in ticks, i.e. volatility state times price
level (median ATR0 1.55 ticks in 2010-2017 against 4.43 in 2022-2026, 26.87), so a within-group result is not a
tick or microstructure effect; anchor type is not controlled; the two eps groups share trade dates, so their
differences are not independent and must not be added or compared as independent. 'small' carries most of R4's
weight, so 'persists within small' is nearly guaranteed by composition. 'large' is likely to be omitted
(tercile 1 of large needs under 5.3 ticks, which ATR eps >= 6 rarely allows); if so the paper reports it omitted by
the pre-stated rule and does not write "holding eps fixed" as if both groups were tested.

KNOWN NUMBERS FIRST (any FAIL stops, nothing below is read): R4's edges 5.3 and 12.4 ticks; tercile 1 older +6.81 and
recent +2.42, tercile 3 older -0.10 and recent +0.86 (to 0.01 pp); the n-weighted mean of the two eps groups'
estimates equals the pooled tercile estimate (an identity, to 1e-9); the two groups partition each tercile.

Usage:  python paper_eps_tercile.py --selftest
        python paper_eps_tercile.py data\\dump
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_robustness as PR                                               # noqa: E402

MIN_N = 1000
KNOWN = {"edge1": 5.3, "edge2": 12.4, "T1_A": 6.81, "T1_B": 2.42, "T3_A": -0.10, "T3_B": 0.86}


def eps_group(trigger, eps):
    """'small' = ATR 3.0 / 4.0 or SD 1.0 / 1.5 (R5's small-eps corner); 'large' otherwise."""
    trigger = np.asarray(trigger)
    eps = np.asarray(eps, float)
    small = ((trigger == "ATR") & np.isin(eps, [3.0, 4.0])) | ((trigger == "SD") & np.isin(eps, [1.0, 1.5]))
    return np.where(small, "small", "large")


def diff(ga, gb):
    """older minus recent from two gap dicts (est, lo, hi) by normal approximation on disjoint dates."""
    sa, sb = (ga["hi"] - ga["lo"]) / 3.92, (gb["hi"] - gb["lo"]) / 3.92
    d = ga["est"] - gb["est"]
    se = (sa ** 2 + sb ** 2) ** 0.5
    return dict(est=d, lo=d - 1.96 * se, hi=d + 1.96 * se)


def decompose(ga, gb, ma, mb):
    """index-number decomposition of the older-to-recent decline in pp.
    ga, gb: per-tercile gap estimates (older, recent); ma, mb: tercile shares of the period (sum 1 each)."""
    ga, gb, ma, mb = (np.asarray(x, float) for x in (ga, gb, ma, mb))
    older, recent = float(ma @ ga), float(mb @ gb)
    older_at_recent_mix = float(mb @ ga)
    recent_at_older_mix = float(ma @ gb)
    return dict(older=older, recent=recent, decline=older - recent,
                A_mix=older - older_at_recent_mix, A_within=older_at_recent_mix - recent,
                B_mix=recent_at_older_mix - recent, B_within=older - recent_at_older_mix)


def contrast13(d1, d3):
    """C_g = (older-minus-recent, smallest-tick tercile) - (same, largest-tick tercile); normal approximation,
    the two terciles treated as independent (they share dates: approximate)."""
    se = (((d1["hi"] - d1["lo"]) / 3.92) ** 2 + ((d3["hi"] - d3["lo"]) / 3.92) ** 2) ** 0.5
    e = d1["est"] - d3["est"]
    return dict(est=e, lo=e - 1.96 * se, hi=e + 1.96 * se, se=se)


def reading(d1, d3, omitted):
    if omitted or d1 is None or d3 is None:
        return "cannot be read"
    c = contrast13(d1, d3)
    if c["est"] <= 0:
        return "not ordered as in R4"
    return "persists" if c["lo"] > 0 else "ordered as in R4, not distinguishable"


def selftest():
    ok = True
    rng = np.random.default_rng(5)
    # eps groups partition and match the stated definition
    tr = np.array(["ATR", "ATR", "ATR", "SD", "SD", "SD", "SD", "ATR"])
    ep = np.array([3.0, 4.0, 6.0, 1.0, 1.5, 2.0, 3.0, 12.0])
    g = eps_group(tr, ep)
    ok &= list(g) == ["small", "small", "large", "small", "small", "large", "large", "large"]
    # n-weighted identity: pooled mean difference equals the n-weighted mean of group estimates
    n = 5000
    df = pd.DataFrame(dict(day=rng.integers(0, 300, n), rf=(rng.random(n) < 0.55).astype(float),
                           tf=(rng.random(n) < 0.48).astype(float), w=1.0, grp=rng.choice(["small", "large"], n, p=[0.8, 0.2])))
    pooled = PR.gap(df, "rf", "w", "tf", "w")["est"]
    parts = [(len(s), PR.gap(s, "rf", "w", "tf", "w")["est"]) for _, s in df.groupby("grp")]
    ok &= abs(pooled - sum(a * b for a, b in parts) / sum(a for a, _ in parts)) < 1e-9
    # decomposition: hand case (older gaps 6,2,0 at mix .5,.3,.2; recent 2,1.5,1 at mix .1,.3,.6)
    d = decompose([6, 2, 0], [2, 1.5, 1], [.5, .3, .2], [.1, .3, .6])
    ok &= abs(d["older"] - 3.6) < 1e-9 and abs(d["recent"] - 1.25) < 1e-9
    ok &= abs(d["A_mix"] + d["A_within"] - d["decline"]) < 1e-9 and abs(d["B_mix"] + d["B_within"] - d["decline"]) < 1e-9
    ok &= abs(d["A_mix"] - (3.6 - (0.1 * 6 + 0.3 * 2 + 0.6 * 0))) < 1e-9
    # reading rule: four phrases only; a coin-flip ordering inside the noise must NOT read "persists"
    d1 = dict(est=4.4, lo=3.5, hi=5.3)
    d3 = dict(est=-1.0, lo=-1.8, hi=-0.2)
    ok &= reading(d1, d3, False) == "persists"
    ok &= reading(d3, d1, False) == "not ordered as in R4"
    n1 = dict(est=0.5, lo=-1.5, hi=2.5)
    n3 = dict(est=0.4, lo=-1.6, hi=2.4)
    ok &= reading(n1, n3, False) == "ordered as in R4, not distinguishable"
    ok &= reading(n1, n1, False) == "not ordered as in R4"          # equal estimates are NOT 'ordered'
    ok &= reading(d1, None, True) == "cannot be read"
    c = contrast13(d1, d3)
    ok &= abs(c["est"] - 5.4) < 1e-12 and abs(c["se"] - ((0.9 / 1.96) ** 2 + (0.8 / 1.96) ** 2) ** 0.5) < 1e-9
    # FINER count: 3 cells with both terciles reported, 2 ordered
    cells = {"a": (2.0, 0.0), "b": (1.0, -1.0), "c": (0.0, 1.0)}
    ok &= sum(1 for v in cells.values() if v[0] > v[1]) == 2
    # difference interval: symmetric and widens with the two SEs
    dd = diff(dict(est=5.0, lo=4.0, hi=6.0), dict(est=2.0, lo=1.0, hi=3.0))
    ok &= abs(dd["est"] - 3.0) < 1e-12 and abs((dd["hi"] - dd["lo"]) / 2 - 1.96 * ((1 / 1.96) ** 2 * 2) ** 0.5) < 1e-9
    print(f"self-test: {'ALL DETECTORS ALIVE' if ok else 'FAILED'}")
    return bool(ok)


def main(dump):
    if not (selftest() and PR.selftest()):
        raise SystemExit("self-test failed")
    t0 = time.time()
    cells = PR.CC.registered_cells()
    df = PR.load(dump, cells)
    m1 = PR.arm(df, "M1b")
    blocks = sorted(df.block.unique())
    A, B = blocks[0], blocks[-1]
    n_nan = int(m1.ticks.isna().sum())
    print(f"M1b-arm setups with NaN ticks (R4 would have put them in tercile 3): {n_nan:,}")
    m1 = m1[m1.ticks.notna()].copy()
    q = np.nanpercentile(m1.ticks, [100 / 3, 200 / 3])        # identical call to R4's, on the same frame
    m1["terc"] = np.digitize(m1.ticks, q)
    m1["grp"] = eps_group(m1.trigger.to_numpy(), m1.eps.to_numpy())
    # a REAL grouping check (the partition identity below holds for any partition)
    pairs = sorted({(str(a), float(b)) for a, b in zip(m1.trigger, m1.eps)})
    want = sorted([("ATR", e) for e in (3.0, 4.0, 6.0, 8.0, 12.0)] + [("SD", e) for e in (1.0, 1.5, 2.0, 3.0)])
    small_share = 100 * float((m1.grp == "small").mean())
    grp_ok = pairs == want and abs(small_share - 83.54) <= 0.5
    print(f"loaded {len(df):,} setups, {len(m1):,} in the M1b arm with ticks; tercile edges {q[0]:.2f}, {q[1]:.2f} ticks\n")
    out = dict(stamp=dict(dump=dump, n=len(m1), machine=platform.node(), date=datetime.date.today().isoformat(),
                          script=hashlib.sha256(open(__file__, "rb").read()).hexdigest().upper(),
                          paper_robustness=hashlib.sha256(open(PR.__file__, "rb").read()).hexdigest().upper(),
                          claim_checks=hashlib.sha256(open(PR.CC.__file__, "rb").read()).hexdigest().upper(),
                          note="TRAINING periods only; descriptive; held-out period not read"))
    # known numbers
    bad = False
    got_edges = (round(q[0], 1), round(q[1], 1))
    for lab, got, want in (("edge 1", got_edges[0], KNOWN["edge1"]), ("edge 2", got_edges[1], KNOWN["edge2"])):
        ok = abs(got - want) < 1e-9
        bad |= not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] known number {lab}: {got} vs {want}")
    pooled = {}
    for t in range(3):
        for b in (A, B):
            pooled[(t, b)] = PR.gap(m1[(m1.terc == t) & (m1.block == b)], "rf", "w", "tf", "w")
    for key, t, b in (("T1_A", 0, A), ("T1_B", 0, B), ("T3_A", 2, A), ("T3_B", 2, B)):
        ok = abs(pooled[(t, b)]["est"] - KNOWN[key]) < 0.005 + 1e-9
        bad |= not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] known number R4 {key}: {pooled[(t, b)]['est']:+.3f} vs {KNOWN[key]:+.2f}")
    # partition + identity per tercile and period
    for t in range(3):
        for b in (A, B):
            s = m1[(m1.terc == t) & (m1.block == b)]
            parts = [(len(x), PR.gap(x, "rf", "w", "tf", "w")["est"]) for _, x in s.groupby("grp")]
            wm = sum(a * e for a, e in parts) / sum(a for a, _ in parts)
            ok = sum(a for a, _ in parts) == len(s) and abs(wm - pooled[(t, b)]["est"]) < 1e-9
            bad |= not ok
            if not ok:
                print(f"  [FAIL] partition / identity, tercile {t + 1} block {b}")
    print(f"  [{'PASS' if grp_ok else 'FAIL'}] eps grouping: (trigger, eps) pairs = the registered nine {pairs == want}; "
          f"'small' share of setups {small_share:.2f}% vs 83.54% (26.56 counts, tolerance 0.5 pp)")
    bad |= not grp_ok
    print(f"  [{'FAIL' if bad else 'PASS'}] partition and n-weighted identity, all six tercile-periods")
    if bad:
        raise SystemExit("known-number check FAILED -- nothing below is to be read")
    print()

    pooled_c = contrast13(diff(pooled[(0, A)], pooled[(0, B)]), diff(pooled[(2, A)], pooled[(2, B)]))
    print(f"POOLED R4 CONTRAST C = d1 - d3 = {pooled_c['est']:+.2f} [{pooled_c['lo']:+.2f}, {pooled_c['hi']:+.2f}]\n")
    out["pooled_C"] = pooled_c
    print("PRIMARY: claim 1 (M1b) by tercile of distance in ticks WITHIN EPS GROUP, pp [95% CI] (share of that group-period)")
    res = {}
    for gname in ("small", "large"):
        gdf = m1[m1.grp == gname]
        print(f"\n  eps group '{gname}'  (setups: older {int((gdf.block == A).sum()):,}, recent {int((gdf.block == B).sum()):,})")
        r = {}
        omitted = []
        for t in range(3):
            row = {}
            for b in (A, B):
                d = gdf[(gdf.terc == t) & (gdf.block == b)]
                row[b] = None if len(d) < MIN_N else dict(PR.gap(d, "rf", "w", "tf", "w"),
                                                         share=100 * len(d) / max(1, int((gdf.block == b).sum())))
            if row[A] is None or row[B] is None:
                omitted.append(t + 1)
                print(f"    tercile {t + 1}: omitted (fewer than {MIN_N:,} setups in a period; "
                      f"older {int(((gdf.terc == t) & (gdf.block == A)).sum()):,}, recent {int(((gdf.terc == t) & (gdf.block == B)).sum()):,})")
                r[t + 1] = None
                continue
            dd = diff(row[A], row[B])
            r[t + 1] = dict(older=row[A], recent=row[B], diff=dd)
            print(f"    tercile {t + 1}: older {row[A]['est']:+.2f} [{row[A]['lo']:+.2f}, {row[A]['hi']:+.2f}] ({row[A]['share']:.0f}%)"
                  f"   recent {row[B]['est']:+.2f} [{row[B]['lo']:+.2f}, {row[B]['hi']:+.2f}] ({row[B]['share']:.0f}%)"
                  f"   older minus recent {dd['est']:+.2f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}]")
        d1 = r[1]["diff"] if r.get(1) else None
        d3 = r[3]["diff"] if r.get(3) else None
        verdict = reading(d1, d3, bool({1, 3} & set(omitted)))
        entry = dict(terciles=r, omitted=omitted, reading=verdict)
        if d1 is not None and d3 is not None:
            cg = contrast13(d1, d3)
            entry["C"] = cg
            entry["C_over_pooled"] = cg["est"] / pooled_c["est"]
            print(f"    C = d1 - d3 = {cg['est']:+.2f} [{cg['lo']:+.2f}, {cg['hi']:+.2f}] (terciles treated as independent); "
                  f"C / pooled C = {entry['C_over_pooled']:.2f}")
        else:
            print(f"    C: not computed (tercile(s) {sorted({1, 3} & set(omitted))} omitted; this MUST be reported)")
        print(f"    READING: the tick-scale pattern {verdict} within eps group '{gname}'")
        if not omitted:
            entry["decomposition"] = decompose([r[t]["older"]["est"] for t in (1, 2, 3)],
                                               [r[t]["recent"]["est"] for t in (1, 2, 3)],
                                               [r[t]["older"]["share"] / 100 for t in (1, 2, 3)],
                                               [r[t]["recent"]["share"] / 100 for t in (1, 2, 3)])
            e = entry["decomposition"]
            print(f"    decomposition (no reading): older {e['older']:+.2f}, recent {e['recent']:+.2f}, decline {e['decline']:+.2f}; "
                  f"order A mix {e['A_mix']:+.2f} within {e['A_within']:+.2f}; order B mix {e['B_mix']:+.2f} within {e['B_within']:+.2f}")
        res[gname] = entry
    out["primary"] = res

    print("\nFINER (descriptive, no reading): by (trigger, eps), terciles pooled as in R4; cells with >= 1,000 setups in both periods")
    fine = {}
    for (trg, eps), s in m1.groupby(["trigger", "eps"]):
        for t in range(3):
            row = {}
            for b in (A, B):
                d = s[(s.terc == t) & (s.block == b)]
                row[b] = None if len(d) < MIN_N else PR.gap(d, "rf", "w", "tf", "w")
            if row[A] is None or row[B] is None:
                continue
            dd = diff(row[A], row[B])
            fine[f"{trg}_{eps}_T{t + 1}"] = dict(older=row[A], recent=row[B], diff=dd)
            print(f"  {trg} eps {eps:>4} tercile {t + 1}: older {row[A]['est']:+.2f} (n {row[A]['n']:,})  recent {row[B]['est']:+.2f} "
                  f"(n {row[B]['n']:,})  older minus recent {dd['est']:+.2f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}]")
    out["fine"] = fine
    cells13 = {}
    for k, v in fine.items():
        base, t = k.rsplit("_T", 1)
        cells13.setdefault(base, {})[int(t)] = v["diff"]["est"]
    both = {k: v for k, v in cells13.items() if 1 in v and 3 in v}
    ordered = sum(1 for v in both.values() if v[1] > v[3])
    print(f"\nFINER COUNT (pre-stated): (trigger, eps) cells with tercile 1 and tercile 3 both reported: {len(both)} of 9; "
          f"smallest-tick older-minus-recent exceeds the largest-tick one in {ordered} of them. All lines above are reported.")
    out["fine_count"] = dict(both_reported=len(both), ordered=ordered, cells_total=9)

    path = os.path.join(HERE, f"paper_eps_tercile_{datetime.date.today().isoformat()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {os.path.basename(path)}  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
