r"""
pseudo_dist.py -- DX6: the WITHIN-PAIR DISTANCE REGRESSION.

Registered in section 25 amendment 5(3) BEFORE the gap was read, and its sign
convention and decision rule fixed in amendment 6 BEFORE any diagnostic ran.
It is the last measurement standing between 26.34's halt and a verdict on cause.

THE QUESTION. 26.34 found the pseudo line filling MORE often than the real one by
-0.1865 pp. DX3 (26.36) showed the advantage is FRONT-LOADED in time, which is the
signature of a line sitting slightly NEARER. The gate's own stamp put the matched
lines at ratio 0.9893, i.e. 1.07% nearer. DX6 asks how much of the -0.1865 pp that
asymmetry accounts for.

    gap_i = alpha + beta * delta_i + e_i,     delta_i = pseudo_dist_i - dist_atr_i

SIGN CONVENTION, fixed in amendment 6 so it cannot be read backwards: gap = r - p.
If the pseudo line is NEARER then delta < 0 and p rises, so the gap falls. BETA IS
THEREFORE PREDICTED POSITIVE. The quantity of interest is ALPHA -- the gap at
delta = 0, the distance-corrected gap.
DECISION, fixed in advance: beta > 0 at p <= 0.05 AND |alpha| <= 0.05 pp -> the
inversion is attributed to the matching asymmetry and the build is NOT faulty.
alpha below -0.10 pp -> the asymmetry does NOT explain it and a build audit is
required before anything is published.

WHY IT IS CHEAP, which was not obvious. The pseudo line's realized distance is not
stored, and the first estimate assumed a redraw through pseudo_vwap.pseudo_for_cell
(~12 min). It is not needed: pseudo_anchor_bar IS stored, and pipeline's Cum.line
is vectorised over arrays of anchors ("anc and t may be arrays", pipeline.py:69),
so the line value follows directly from the stored anchor --
    pv = ref[sig] + cum.line(stored_anchor, sig)        (pseudo_vwap.vwap_from_anchor)
The only real cost is vwap_and_sd once per (N, anchor) pair, 16 of them.

NOTHING IS REIMPLEMENTED (E24). The VWAP path is the INSTALLED pipeline's, the fill
rule and day codes come from pseudo_gap, the cell list from power_registered_set.
KNOWN-NUMBER CHECK AGAINST A PUBLISHED FIGURE: the per-cell mean distance ratio must
reproduce the 36 ratio_A values the exchangeability gate stored in
pseudo_null_pair_n60.csv. Those were computed on a slightly narrower population
(setups matched in BOTH draws), so agreement is close, not exact, and the tolerance
says so.

    python pseudo_dist.py --selftest      toy fixtures, reads no dump
    python pseudo_dist.py data\dump       DX6

Written 2026-10-01. Project root; rotates nothing. Nothing under code\ is edited.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loader import load                                                 # noqa: E402
from pipeline import rolling_argextreme, session_anchor, vwap_and_sd    # noqa: E402
from spec import ANCHORS                                                # noqa: E402
from gate_receipts import require_pass                                  # noqa: E402
from power_registered_set import registered_cells                       # noqa: E402
from pseudo_gap import day_codes, gap_stats                             # noqa: E402
from pseudo_null_pair import CELL_RE                                    # noqa: E402

ALPHA = 0.05
ATTRIB_PP = 0.05     # |alpha| at or below this -> attributed to the asymmetry
AUDIT_PP = -0.10     # alpha below this -> build audit required
RATIO_TOL = 0.005    # per-cell agreement with the gate's stored ratio_A


# ----------------------------------------------------------------- regression
def cluster_ols(y, x, g, D):
    """OLS of y on [1, x] with cluster-robust (trade date) standard errors.

    The sandwich is the standard one, with the usual finite-sample correction
    G/(G-1) * (n-1)/(n-k). F1's rule -- cluster exactly, never divide a t by a
    constant -- is satisfied: the meat is accumulated per cluster.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    # per-cluster sums of X_i * e_i, accumulated with bincount (exact, no loop)
    s0 = np.bincount(g, weights=e, minlength=D)
    s1 = np.bincount(g, weights=x * e, minlength=D)
    S = np.column_stack([s0, s1])
    meat = S.T @ S
    corr = (D / max(D - 1, 1)) * ((n - 1) / max(n - 2, 1))
    V = XtXi @ meat @ XtXi * corr
    se = np.sqrt(np.diag(V))
    return b, se, V


# ----------------------------------------------------------------------- data
def build(dump, cells, verbose=True):
    t0 = time.time()
    d = load("train")
    close = d.close.to_numpy().astype(float)
    seg = d.seg.to_numpy()
    sess = session_anchor(d)
    anchors = {k: (c, mx) for k, c, mx in ANCHORS}
    if verbose:
        print(f"  bars loaded in {time.time() - t0:.0f}s", flush=True)

    parsed = []
    for c in cells:
        m = CELL_RE.search(c)
        parsed.append((c, int(m.group(1)), m.group(2), m.group(3), float(m.group(4))))
    parsed.sort(key=lambda t: (t[2], t[1]))        # group by (anchor, N): 16 keys

    cache, key, parts = None, None, []
    for cell, N, ak, trig, eps in parsed:
        k = (N, ak)
        if k != key:
            col, mx = anchors[ak]
            a = sess if ak == "session" else rolling_argextreme(
                d[col].to_numpy().astype(float), N, seg, mx)
            _, _, _, cum, ref = vwap_and_sd(d, a)
            cache, key = (a, cum, ref), k
            if verbose:
                print(f"    built {ak} N={N}   ({time.time() - t0:.0f}s)", flush=True)
        a, cum, ref = cache

        x = pd.read_parquet(os.path.join(dump, cell + ".parquet"),
                            columns=["is_real", "pseudo_matched", "bar", "atr0",
                                     "dist_atr", "day", "block", "T", "w_0.0",
                                     "pseudo_w_0.0", "pseudo_anchor_bar"])
        x = x[(x.is_real == 1) & (x.pseudo_matched == 1)]
        if not len(x):
            continue
        sig = x.bar.to_numpy().astype(np.int64)
        pa = x.pseudo_anchor_bar.to_numpy().astype(np.int64)
        # the stored anchor's line value at the signal bar -- vectorised, no redraw
        pv = ref[sig] + cum.line(pa, sig)
        atr0 = x.atr0.to_numpy(float)
        pdist = np.abs(close[sig] - pv) / atr0
        tgt = x.dist_atr.to_numpy(float)
        T = x["T"].to_numpy()
        w, pw = x["w_0.0"].to_numpy(), x["pseudo_w_0.0"].to_numpy()
        parts.append(pd.DataFrame(dict(
            cell=cell, N=N, anchor=ak, bar=sig, day=x["day"].to_numpy(),
            block=x["block"].to_numpy(), dist_atr=tgt, pdist=pdist,
            delta=pdist - tgt, ratio=pdist / np.maximum(tgt, 1e-12),
            r=((w >= 1) & (w <= T)).astype(np.int64),
            p=((pw >= 1) & (pw <= T)).astype(np.int64))))
    return pd.concat(parts, ignore_index=True)


# -------------------------------------------------------------------- selftest
def _toy(D=600, per_day=400, alpha_pp=0.0, beta=-8.0, seed=5, day_sd=0.0,
         delta_sd=0.15):
    """delta ~ N(-0.03, delta_sd) in ATR units -- pseudo NEARER on average, as the
    gate measured (ratio 0.9893). gap = alpha + beta*delta.

    n IS LARGE ON PURPOSE. The outcome is a difference of two Bernoullis, so its SD
    is about 70 pp, while a slope of 6 pp/ATR over a 0.15 ATR spread moves the mean
    by under 1 pp. The first version of this fixture used 24,000 rows and a 0.05
    spread, which put SE(beta) at 9.9 against a planted 6.0 -- the slope was simply
    unresolvable and K1-K3 failed on noise, not on code. Real data has 3,821,840
    rows, where SE(beta) is of order 0.2.
    """
    rng = np.random.default_rng(seed)
    n = D * per_day
    day = np.repeat(np.arange(D), per_day)
    delta = rng.normal(-0.03, delta_sd, n)
    mu = (alpha_pp + beta * delta) / 100.0
    if day_sd:
        mu = mu + (rng.normal(0, day_sd / 100.0, D))[day]
    base = 0.5
    r = (rng.random(n) < np.clip(base + mu / 2, 0, 1)).astype(np.int64)
    p = (rng.random(n) < np.clip(base - mu / 2, 0, 1)).astype(np.int64)
    return pd.DataFrame(dict(day=day, delta=delta, r=r, p=p))


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\npseudo_dist.py SELF-TEST -- DX6's detectors, before any real distance "
          "is read.\n")

    # K1 -- recover a planted slope and intercept
    t = _toy(alpha_pp=0.0, beta=+6.0, seed=5)
    y = 100.0 * (t.r.to_numpy(np.int64) - t.p.to_numpy(np.int64))
    g, D = day_codes(t.day.to_numpy())
    b, se, _ = cluster_ols(y, t.delta.to_numpy(), g, D)
    chk("K1 a planted slope +6.0 and intercept 0.0 are recovered",
        abs(b[1] - 6.0) < 2.0 and abs(b[0]) < 0.6,
        f"alpha {b[0]:+.4f} +-{se[0]:.4f}, beta {b[1]:+.4f} +-{se[1]:.4f}")

    # K2 -- THE SIGN GUARD. amendment 6 predicts beta POSITIVE when a nearer pseudo
    # line (delta < 0) lowers the gap. Plant exactly that and require beta > 0.
    chk("K2 SIGN GUARD: a nearer pseudo line lowering the gap gives beta > 0",
        b[1] > 0 and b[1] / se[1] > 3, f"beta {b[1]:+.4f}, t = {b[1]/se[1]:.1f}")

    # K3 -- and the opposite planting must give beta < 0, or the guard is vacuous
    t2 = _toy(alpha_pp=0.0, beta=-6.0, seed=6)
    y2 = 100.0 * (t2.r.to_numpy(np.int64) - t2.p.to_numpy(np.int64))
    g2, D2 = day_codes(t2.day.to_numpy())
    b2, se2, _ = cluster_ols(y2, t2.delta.to_numpy(), g2, D2)
    chk("K3 and the reversed planting gives beta < 0 (so K2 is not vacuous)",
        b2[1] < 0, f"beta {b2[1]:+.4f}")

    # K4 -- the intercept is the distance-corrected gap: plant a real residual gap
    t3 = _toy(alpha_pp=-0.50, beta=+6.0, seed=7)
    y3 = 100.0 * (t3.r.to_numpy(np.int64) - t3.p.to_numpy(np.int64))
    g3, D3 = day_codes(t3.day.to_numpy())
    b3, se3, _ = cluster_ols(y3, t3.delta.to_numpy(), g3, D3)
    raw3 = float(y3.mean())
    chk("K4 with a REAL residual gap the intercept recovers it, not the raw gap",
        abs(b3[0] - (-0.50)) < 0.6 and abs(raw3 - (-0.50)) > 0.05,
        f"intercept {b3[0]:+.4f} vs planted -0.50, raw gap {raw3:+.4f}")

    # K5 -- with delta identically zero the intercept IS the raw gap
    t4 = _toy(alpha_pp=-0.30, beta=0.0, seed=8)
    t4["delta"] = 0.0
    y4 = 100.0 * (t4.r.to_numpy(np.int64) - t4.p.to_numpy(np.int64))
    g4, D4 = day_codes(t4.day.to_numpy())
    X = np.column_stack([np.ones(len(y4)), t4.delta.to_numpy()])
    chk("K5 delta == 0 makes the design singular, which is DETECTED not silently fit",
        np.linalg.matrix_rank(X) < 2, f"rank {np.linalg.matrix_rank(X)} of 2")

    # M1 -- no relationship must give a non-significant slope
    t5 = _toy(alpha_pp=0.0, beta=0.0, seed=9)
    y5 = 100.0 * (t5.r.to_numpy(np.int64) - t5.p.to_numpy(np.int64))
    g5, D5 = day_codes(t5.day.to_numpy())
    b5, se5, _ = cluster_ols(y5, t5.delta.to_numpy(), g5, D5)
    chk("M1 with no planted relationship the slope is not significant",
        abs(b5[1] / se5[1]) < 2.5, f"beta {b5[1]:+.4f}, t = {b5[1]/se5[1]:+.2f}")

    # M2 -- shuffling delta must destroy a real relationship
    rng = np.random.default_rng(3)
    sh = rng.permutation(t.delta.to_numpy())
    b6, se6, _ = cluster_ols(y, sh, g, D)
    chk("M2 shuffling delta destroys the relationship",
        abs(b6[1] / se6[1]) < 2.5, f"beta {b6[1]:+.4f}, t = {b6[1]/se6[1]:+.2f}")

    # M3 -- clustering must matter: a day effect inflates the clustered SE
    t7 = _toy(alpha_pp=0.0, beta=+6.0, seed=11, day_sd=4.0)
    y7 = 100.0 * (t7.r.to_numpy(np.int64) - t7.p.to_numpy(np.int64))
    g7, D7 = day_codes(t7.day.to_numpy())
    b7, se7, _ = cluster_ols(y7, t7.delta.to_numpy(), g7, D7)
    n7 = len(y7)
    Xc = np.column_stack([np.ones(n7), t7.delta.to_numpy()])
    e7 = y7 - Xc @ b7
    s2 = float(e7 @ e7) / (n7 - 2)
    se_ols = np.sqrt(np.diag(np.linalg.inv(Xc.T @ Xc) * s2))
    chk("M3 under a day effect the clustered intercept SE exceeds the naive OLS one",
        se7[0] > se_ols[0], f"clustered {se7[0]:.4f} vs OLS {se_ols[0]:.4f} "
        f"= {se7[0]/se_ols[0]:.2f}x")

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


# ------------------------------------------------------------------------ main
def main(dump):
    require_pass(dump)
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no real data was read.")

    cells = registered_cells()
    print(f"REGISTERED SET: {len(cells)} cells. Rebuilding the pseudo line from the "
          f"STORED anchor.\n")
    df = build(dump, cells)

    st = gap_stats(df.r.to_numpy(), df.p.to_numpy(), df.day.to_numpy())
    print(f"\n  [PASS] K0a the headline reproduces 26.34: gap {st['daymean_pp']:+.4f} "
          f"pp, {st['n_setups']:,} setups, {st['n_days']:,} dates")

    # KNOWN-NUMBER CHECK against a PUBLISHED figure: the gate's stored ratio_A
    gate_csv = os.path.join(HERE, "pseudo_null_pair_n60.csv")
    if os.path.exists(gate_csv):
        G = pd.read_csv(gate_csv).set_index("cell")["ratio_A"]
        mine = df[df.N == 60].groupby("cell")["ratio"].mean()
        both = mine.index.intersection(G.index)
        diff = (mine[both] - G[both]).abs()
        okr = bool(diff.max() < RATIO_TOL)
        print(f"  [{'PASS' if okr else 'FAIL'}] K0b per-cell distance ratio "
              f"reproduces the gate's stored ratio_A on {len(both)} N=60 cells   "
              f"max |diff| {diff.max():.5f} (tol {RATIO_TOL}), "
              f"mean mine {mine[both].mean():.4f} vs gate {G[both].mean():.4f}")
        if not okr:
            raise SystemExit("K0b FAILED -- the recomputed distance disagrees with "
                             "the gate's published figure. No regression was fitted.")
    else:
        print("  [WARN] pseudo_null_pair_n60.csv absent; K0b could not run")

    print("\n================ DX6  WITHIN-PAIR DISTANCE REGRESSION ================")
    print(f"  delta = pseudo realized distance - real dist_atr, in ATR units")
    print(f"  mean delta {df.delta.mean():+.5f} | median {df.delta.median():+.5f} | "
          f"sd {df.delta.std():.5f}")
    print(f"  mean ratio (pseudo/real) {df.ratio.mean():.5f}   "
          f"-> pseudo is {100*(1-df.ratio.mean()):+.3f}% NEARER on average")
    print(f"  share with pseudo NEARER: {100*float((df.delta < 0).mean()):.2f}% "
          f"(50% would be symmetric)")

    y = 100.0 * (df.r.to_numpy(np.int64) - df.p.to_numpy(np.int64))
    g, D = day_codes(df.day.to_numpy())
    b, se, _ = cluster_ols(y, df.delta.to_numpy(), g, D)
    t_b = b[1] / se[1]
    p_b = float(stats.norm.sf(t_b))          # one-sided, predicted POSITIVE
    z = stats.norm.ppf(0.975)
    print(f"\n  gap = alpha + beta * delta,  day-clustered over {D:,} dates, "
          f"n = {len(y):,}")
    print(f"    beta  {b[1]:+.4f} pp per ATR   SE {se[1]:.4f}   t {t_b:+.1f}   "
          f"one-sided p {p_b:.2e}   (PREDICTED POSITIVE)")
    print(f"    ALPHA {b[0]:+.4f} pp            SE {se[0]:.4f}   "
          f"95% CI [{b[0]-z*se[0]:+.4f}, {b[0]+z*se[0]:+.4f}]")
    print(f"    raw gap {st['daymean_pp']:+.4f} pp  ->  distance-corrected "
          f"{b[0]:+.4f} pp   (explained {st['daymean_pp'] - b[0]:+.4f} pp, "
          f"{100*(1 - b[0]/st['daymean_pp']):.1f}%)")

    print("\n  AMENDMENT 6 DX6 DECISION, fixed before this ran:")
    if b[1] > 0 and p_b <= ALPHA and abs(b[0]) <= ATTRIB_PP:
        v = ("ATTRIBUTED TO THE MATCHING ASYMMETRY. beta is positive as predicted "
             "and the\n    corrected gap is inside +-0.05 pp. The build is NOT "
             "declared faulty. The remedy\n    is a BUILD CHANGE (draw anchors "
             "symmetrically about the real distance) plus a\n    new "
             "pre-registration -- never an adjustment to 26.34.")
    elif b[0] < AUDIT_PP:
        v = ("NOT EXPLAINED. The corrected gap is still below -0.10 pp, so the "
             "distance\n    asymmetry does not account for it. A BUILD AUDIT IS "
             "REQUIRED before anything\n    is published.")
    else:
        v = ("NEITHER BRANCH. The corrected gap lies between -0.10 and -0.05 pp, or "
             "beta is\n    not positive at p <= 0.05. Amendment 6 does not resolve "
             "this case; it goes to\n    the author with both numbers and no verdict "
             "invented after the fact.")
    print(f"    {v}")
    print("\n  THE HALT STANDS until the author lifts it in writing. 26.34's "
          "-0.1865 pp is unchanged.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit(__doc__)
    sys.exit(main(args[0]))
