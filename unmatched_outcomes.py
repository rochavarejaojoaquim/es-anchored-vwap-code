r"""
unmatched_outcomes.py -- are the setups the pseudo-VWAP could NOT match different
in outcome from the ones it matched?  (§25.5 gate 3 says an unmatched excess
"biases the matching"; this measures whether the EXCLUSION is a selection effect.)

    python unmatched_outcomes.py --selftest
    python unmatched_outcomes.py DUMP_DIR          all 180 cells, read-only

Uses REAL-LINE outcomes only (w_0.0, mae_*, gap_at_T, cens_T) -- never a pseudo
fill -- so it does not compute the real-minus-pseudo gap that §25.4 restricts.

ESTIMATOR (within cell AND block -- the same stratum, never across).
  For every unmatched setup i in stratum s:   r_i = y_i - mean(y over MATCHED setups in s)
  effect = mean of r_i over all unmatched setups  ("unmatched minus matched, same stratum").
  This is a stratified difference: a cell-mix difference (unmatched setups
  concentrating in cells with a different base rate) cannot fake an effect.
  INFERENCE clusters on TRADE DATE (F1): the effect is a ratio sum(S_d)/sum(N_d)
  over days, variance (D/(D-1)) * sum_d (S_d - effect*N_d)^2 / N^2.  The matched
  means' own sampling error is ignored (millions of matched rows; NOT propagated).
  Both counts are always reported: setups (rows) and trade dates (independent units).
Outcomes: fill within T (pp), censored (pp), mae_all_wick / gap_at_T /
  mae_fill_wick / mae_fill_close (ATR0 units, raw).  mae_fill_* exist only for
  FILLED setups, gap_at_T only for uncensored -- each is computed on the setups
  where it exists, so those rows are CONDITIONAL and say so.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


def residuals(y, um, strata):
    """y float (NaN = undefined), um bool (unmatched), strata int codes.
    -> (index of the unmatched rows used, r = y - matched mean in its stratum,
        reference (matched mean) for those rows)."""
    ok = np.isfinite(y)
    m, u = ok & ~um, ok & um
    ref = pd.Series(y[m]).groupby(strata[m]).mean()
    refv = ref.reindex(strata[u]).to_numpy()
    idx = np.flatnonzero(u)
    keep = np.isfinite(refv)
    return idx[keep], y[u][keep] - refv[keep], refv[keep]


class Acc:
    """Day-level sums for the ratio estimator of the stratified effect."""

    def __init__(self, D):
        self.S, self.N, self.D = np.zeros(D), np.zeros(D), D
        self.ref, self.rows_u, self.rows_m = 0.0, 0, 0

    def add(self, g_u, r, refv, n_matched=0):
        self.S += np.bincount(g_u, weights=r, minlength=self.D)
        self.N += np.bincount(g_u, minlength=self.D)
        self.ref += float(refv.sum())
        self.rows_u += len(r)
        self.rows_m += int(n_matched)

    def result(self):
        n = self.N.sum()
        if n == 0:
            return None
        eff = self.S.sum() / n
        days = self.N > 0
        D = int(days.sum())
        var = (D / max(D - 1, 1)) * np.sum((self.S[days] - eff * self.N[days]) ** 2) / n ** 2
        se = float(np.sqrt(var))
        return dict(eff=eff, se=se, lo=eff - 1.96 * se, hi=eff + 1.96 * se, n=int(n),
                    days=int((self.N > 0).sum()), ref=self.ref / n)


# ------------------------------------------------------------------ self-test
def shuffle_labels(um, s, rng):
    """Permute the unmatched labels WITHIN each stratum (keeps every stratum's share)."""
    um = um.copy()
    for k in np.unique(s):
        w = s == k
        um[w] = rng.permutation(um[w])
    return um


def synth(rng, delta, day_shock=0.0, S=40, D=500, n=300, day_share_sd=0.0):
    """S strata with very different base rates and unmatched shares (Simpson's
    confound: cells with a HIGH base rate also have MORE unmatched setups).
    Fill probability = base + delta*unmatched + a shared trade-date shock.
    day_share_sd > 0: unmatched status CLUSTERS on certain dates (lognormal
    per-date propensity, independent of the shock and of the outcome)."""
    base = rng.uniform(0.2, 0.8, S)
    share = 0.02 + 0.28 * (base - 0.2) / 0.6
    s = np.repeat(np.arange(S), n)
    g = rng.integers(0, D, S * n)
    dshare = np.exp(day_share_sd * rng.normal(0, 1, D) - day_share_sd ** 2 / 2)[g] if day_share_sd else 1.0
    um = rng.random(S * n) < np.minimum(share[s] * dshare, 0.95)
    shock = rng.normal(0, day_shock, D)[g] if day_shock else 0.0
    p = np.clip(base[s] + delta * um + shock, 0, 1)
    return (rng.random(S * n) < p).astype(float), um, s, g, D


def estimate(y, um, s, g, D):
    idx, r, refv = residuals(y, um, s)
    a = Acc(D)
    a.add(g[idx], r, refv)
    return a.result()


def naive(y, um):
    return y[um].mean() - y[~um].mean()


def selftest():
    print("SELF-TEST (planted effects with known answers; each mutation must FAIL)\n")
    rng = np.random.default_rng(11)
    res = []

    def check(name, ok):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    y, um, s, g, D = synth(rng, 0.05)
    e = estimate(y, um, s, g, D)
    check(f"K1 planted +5.0 pp recovered: {100*e['eff']:+.2f} pp [{100*e['lo']:+.2f}, {100*e['hi']:+.2f}]",
          e["lo"] < 0.05 < e["hi"])
    cover = null_rej = 0
    reps = 200
    for _ in range(reps):
        y, um, s, g, D = synth(rng, 0.0, day_shock=0.15)
        e = estimate(y, um, s, g, D)
        null_rej += (e["lo"] > 0) or (e["hi"] < 0)
        y, um, s, g, D = synth(rng, 0.05, day_shock=0.15)
        e = estimate(y, um, s, g, D)
        cover += e["lo"] < 0.05 < e["hi"]
    check(f"K2 no effect (with day-level shocks): CI excludes 0 in {100*null_rej/reps:.1f}% of {reps} draws (nominal 5%, need 1-10%)",
          0.01 <= null_rej / reps <= 0.10)
    check(f"K3 planted +5 pp: 95% CI covers the truth in {100*cover/reps:.1f}% of {reps} draws (need >= 90%)", cover / reps >= 0.90)
    y, um, s, g, D = synth(rng, 0.0)
    nv = naive(y, um)
    check(f"M1 WITHOUT stratifying (pooled difference) on a Simpson-confounded fixture with NO effect: "
          f"{100*nv:+.1f} pp -> is badly wrong; the stratified estimate is {100*estimate(y, um, s, g, D)['eff']:+.2f} pp",
          abs(nv) > 0.03 and abs(estimate(y, um, s, g, D)["eff"]) < 0.01)
    y, um, s, g, D = synth(rng, 0.05)                     # the effect is REAL in y ...
    e = estimate(y, shuffle_labels(um, s, rng), s, g, D)  # ... but the labels given to the estimator are shuffled
    check(f"M2 labels shuffled within stratum after a planted +5 pp: {100*e['eff']:+.2f} pp, CI [{100*e['lo']:+.2f}, {100*e['hi']:+.2f}] includes 0",
          e["lo"] < 0 < e["hi"])
    bad = good = 0
    reps = 200
    for _ in range(reps):                  # unmatched status clusters on dates; outcomes carry date shocks
        y, um, s, g, D = synth(rng, 0.0, day_shock=0.25, D=100, day_share_sd=1.0)
        idx, r, refv = residuals(y, um, s)
        se_iid = r.std(ddof=1) / np.sqrt(len(r))           # MUTATION: ignore trade-date clustering
        bad += abs(r.mean()) > 1.96 * se_iid
        e = estimate(y, um, s, g, D)
        good += (e["lo"] > 0) or (e["hi"] < 0)
    check(f"M3 IGNORING trade-date clustering: null rejected {100*bad/reps:.1f}% of {reps} draws (must be > 10%); "
          f"the clustered test on the same data: {100*good/reps:.1f}% (need 1-10%)",
          bad / reps > 0.10 and 0.01 <= good / reps <= 0.10)
    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


# ------------------------------------------------------------------------ run
METRICS = [("fill within T (pp)", 100, "hit"), ("censored by a roll (pp)", 100, "cens"),
           ("mae_all_wick (ATR0)", 1, "mae_all_wick"), ("gap_at_T (ATR0)  [uncensored only]", 1, "gap_at_T"),
           ("mae_fill_wick (ATR0)  [FILLED only]", 1, "mae_fill_wick"),
           ("mae_fill_close (ATR0)  [FILLED only]", 1, "mae_fill_close"),
           ("dist_atr at signal (ATR0)  [entry]", 1, "dist_atr"), ("age of the anchor, bars  [entry]", 1, "age")]
COLS = ["is_real", "pseudo_matched", "day", "block", "T", "w_0.0", "cens_T", "mae_all_wick", "gap_at_T",
        "mae_fill_wick", "mae_fill_close", "dist_atr", "age"]


def main(dump):
    files = [f for f in sorted(glob.glob(os.path.join(dump, "N*.parquet"))) if ".m2." not in f]
    days = set()
    for f in files:
        days.update(pd.read_parquet(f, columns=["day"])["day"].unique().tolist())
    days = pd.Index(sorted(days))
    D = len(days)
    groups = {}

    def acc(key):
        if key not in groups:
            groups[key] = {m[2]: Acc(D) for m in METRICS}
        return groups[key]
    cellsign = {}
    rows_tot = um_tot = 0
    for f in files:
        cell = os.path.basename(f)[:-8]
        mt = re.match(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)$", cell)
        N, akey, trig = int(mt.group(1)), mt.group(2), mt.group(3)
        x = pd.read_parquet(f, columns=COLS)
        x = x[x.is_real == 1].reset_index(drop=True)
        um = (x.pseudo_matched != 1).to_numpy()
        rows_tot += len(x)
        um_tot += int(um.sum())
        T = x["T"].to_numpy()
        w = x["w_0.0"].to_numpy()
        Y = dict(hit=((w >= 1) & (w <= T)).astype(float), cens=x.cens_T.astype(float).to_numpy())
        for c in ("mae_all_wick", "gap_at_T", "mae_fill_wick", "mae_fill_close", "dist_atr", "age"):
            Y[c] = x[c].to_numpy(float)
        g = days.get_indexer(x["day"])
        blk = x["block"].to_numpy()
        strata = (blk == "B").astype(int)                       # cell x block
        dedup = (akey != "session") or (N == 60)               # session family counted once (HANDOFF 22.4 #14)
        for name, y in Y.items():
            idx, r, refv = residuals(y, um, strata)
            if len(idx) == 0:
                continue
            nm = int(((~um) & np.isfinite(y)).sum())
            for setname in (("ALL180", "DEDUP") if dedup else ("ALL180",)):
                a = acc((setname, "ALL"))[name]
                a.add(g[idx], r, refv, nm)
                for b in ("A", "B"):
                    sel = blk[idx] == b
                    acc((setname, f"block {b}"))[name].add(g[idx][sel], r[sel], refv[sel])
                acc((setname, f"anchor {akey}"))[name].add(g[idx], r, refv)
                acc((setname, f"trigger {trig}"))[name].add(g[idx], r, refv)
        if dedup:
            yh = Y["hit"]
            if um.sum() >= 50 and (~um).sum() >= 50:
                cellsign[cell] = yh[um].mean() - yh[~um].mean()
    print("STAMP: pseudo-VWAP unmatched vs matched, REAL-line outcomes only | cells 180 | real setups "
          f"{rows_tot:,} (unmatched {um_tot:,} = {100*um_tot/rows_tot:.3f}%) | independent units: {D:,} trade dates | "
          "strata = cell x block | clustered on trade date | raw outcomes (no null, no edge) | "
          "ES, train blocks A+B only, sealed 2018-2022 untouched | DEDUP = session family counted once (N=60)")
    print("effect = unmatched MINUS matched, same cell and block.  + means the unmatched setups have the HIGHER value.\n")
    for label, scale, key in METRICS:
        print(f"{label}")
        print(f"  {'set / group':<22}{'unmatched rows':>15}{'days':>7}{'matched ref':>13}{'effect':>10}{'95% CI':>22}")
        for k in (("DEDUP", "ALL"), ("DEDUP", "block A"), ("DEDUP", "block B"), ("ALL180", "ALL"),
                  ("DEDUP", "anchor hh"), ("DEDUP", "anchor ll"), ("DEDUP", "anchor hv"), ("DEDUP", "anchor session"),
                  ("DEDUP", "trigger ATR"), ("DEDUP", "trigger SD")):
            if k in groups:
                r = groups[k][key].result()
                if r is None:
                    continue
                print(f"  {k[0] + ' ' + k[1]:<22}{r['n']:>15,}{r['days']:>7,}{scale*r['ref']:>13.3f}"
                      f"{scale*r['eff']:>+10.3f}   [{scale*r['lo']:+.3f}, {scale*r['hi']:+.3f}]")
        print()
    v = np.array(list(cellsign.values()))
    print(f"fill-rate difference per cell (DEDUP cells with >= 50 unmatched and >= 50 matched): {len(v)} cells; "
          f"unmatched higher in {int((v > 0).sum())}, lower in {int((v < 0).sum())}; "
          f"median {100*np.median(v):+.2f} pp, 10th/90th pct {100*np.percentile(v,10):+.2f}/{100*np.percentile(v,90):+.2f} pp")


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
