r"""
dx6_joint.py -- section 25 AMENDMENT 9: amendment 8's V1/V2 recomputed with the CORRECT
variance of the difference, plus the registered LINEARITY SENSITIVITY of DX6's alpha.

WHAT WAS WRONG (HANDOFF 26.62 F1, brain/13 C139). Amendment 8 used
SE_diff = sqrt(SE_old^2 + SE_new^2) as if the two fits were independent. They share the
REAL arm entirely and about half the pseudo anchors, so Cov > 0 and the true variance of a
difference is V_old + V_new - 2 Cov. Fixing ONLY that term is the minimal correction; every
threshold and amendment 8's verdict ladder are kept VERBATIM (V1 FAIL -> the halt stays).

WHAT A RESULT MEANS, fixed before the run (review, 2026-10-03):
  * this is a DRAW-TO-DRAW REPRODUCIBILITY CHECK, not out-of-sample: same market data;
  * V1 needs the fits' correlation above ~0.998 to fail, so it is a FORMALITY and may never
    be cited as "slope validated";
  * V2 CAN fail (correlation above ~0.93). A V2 fail with |d_alpha| < 0.05 pp is a
    statistically detectable but practically negligible draw-to-draw difference; it goes
    to the author as PARTIAL, as amendment 8 says, with alpha_new and its CI;
  * the only truly independent test of the section 25 null is new market data (the seal).

THE JOINT VARIANCE. Both fits are pseudo_dist.cluster_ols (OLS of gap on [1, delta],
clustered on trade date, CR1). Scores are summed per date on a COMMON date universe
(union, zero-filled: an absent date adds nothing to that fit's sums). Cross term:
    C = A_old^-1 (S_old' S_new) A_new^-1 * sqrt(D1/(D1-1) * D2/(D2-1)) * sqrt(c1*c2),
    c_i = (n_i - 1)/(n_i - 2), D_i = each fit's OWN cluster count
so each diagonal block reproduces cluster_ols EXACTLY (K0b). Var(d) >= 0 is asserted.
Precedent: seemingly-unrelated estimation (Stata suest, Weesie 1999); comparing
coefficients across models on one sample (Clogg, Petkova & Haritou 1995). Search
summaries, PRIMARIES NOT READ.

LINEARITY SENSITIVITY (data\dump only), chosen AFTER results were seen, so reported as a
sensitivity and NEVER a criterion: alpha from a QUADRATIC fit (gap on [1, delta, delta^2],
clustered), and the gap by DELTA DECILE with day-clustered SEs. If the linear and quadratic
alphas disagree, the registered LINEAR fit remains the criterion; both are reported.

    python dx6_joint.py --selftest
    python dx6_joint.py data\dump rebuild_pseudo_2026-09-30\data\dump

Project root; rotates nothing. Reads only; writes dx6_joint_<date>.json (stamped).
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pseudo_dist import cluster_ols                                    # noqa: E402

PUB = dict(beta=7.8793, alpha=0.0066, se_beta=0.1048, se_alpha=0.0119,   # 26.38
           v_beta=7.8671, v_alpha=-0.0020, v_se_beta=0.1014, v_se_alpha=0.0115)  # 26.46
ATTRIB_PP = 0.05


# ------------------------------------------------------------------ joint sandwich
def scores(y, X, codes, U):
    """b, (X'X)^-1, per-date score sums on a universe of U dates, n, own D."""
    A = np.linalg.inv(X.T @ X)
    b = A @ (X.T @ y)
    e = y - X @ b
    S = np.column_stack([np.bincount(codes, weights=X[:, j] * e, minlength=U)
                         for j in range(X.shape[1])])
    return b, A, S, len(y), int(len(np.unique(codes)))


def vcov(A, S, n, D):
    k = A.shape[0]
    return A @ (S.T @ S) @ A * (D / (D - 1)) * ((n - 1) / (n - k))


def joint(fit1, fit2):
    b1, A1, S1, n1, D1 = fit1
    b2, A2, S2, n2, D2 = fit2
    k = A1.shape[0]
    V1, V2 = vcov(A1, S1, n1, D1), vcov(A2, S2, n2, D2)
    corr = np.sqrt(D1 / (D1 - 1) * D2 / (D2 - 1)) * np.sqrt((n1 - 1) / (n1 - k) * (n2 - 1) / (n2 - k))
    C = A1 @ (S1.T @ S2) @ A2 * corr
    Vd = V1 + V2 - C - C.T
    scale = np.diag(V1) + np.diag(V2)
    if (np.diag(Vd) < -1e-9 * scale).any():          # beyond rounding: a real negative
        raise ValueError(f"Var(d) negative: {np.diag(Vd)}")
    Vd[np.diag_indices_from(Vd)] = np.maximum(np.diag(Vd), 0.0)
    rho = np.diag(C) / np.sqrt(np.diag(V1) * np.diag(V2))
    return dict(b1=b1, b2=b2, V1=V1, V2=V2, C=C, Vd=Vd, rho=rho)


def common_codes(day1, day2):
    u = np.unique(np.concatenate([np.asarray(day1), np.asarray(day2)]))
    return np.searchsorted(u, day1), np.searchsorted(u, day2), len(u)


def design(delta, quad=False):
    cols = [np.ones(len(delta)), delta] + ([delta ** 2] if quad else [])
    return np.column_stack(cols)


def verdicts(J):
    db, da = abs(J["b2"][1] - J["b1"][1]), abs(J["b2"][0] - J["b1"][0])
    sb, sa = np.sqrt(J["Vd"][1, 1]), np.sqrt(J["Vd"][0, 0])
    v1 = db <= 1.96 * sb
    v2 = abs(J["b2"][0]) <= ATTRIB_PP and da <= 1.96 * sa
    return dict(d_beta=db, band_beta=1.96 * sb, d_alpha=da, band_alpha=1.96 * sa, V1=bool(v1),
                V2=bool(v2), naive_band_beta=1.96 * np.sqrt(J["V1"][1, 1] + J["V2"][1, 1]),
                naive_band_alpha=1.96 * np.sqrt(J["V1"][0, 0] + J["V2"][0, 0]))


def ladder(v):
    if v["V1"] and v["V2"]:
        return "V1 AND V2 PASS -- no evidence of draw-specificity (a REPRODUCIBILITY check, not out-of-sample)"
    if not v["V1"]:
        return "V1 FAILS -- amendment 8: NOT CONFIRMED; the halt STAYS and a build audit follows"
    tag = ("practically negligible (< 0.05 pp, pre-stated)" if v["d_alpha"] < ATTRIB_PP
           else "NOT negligible")
    return f"V2 FAILS with V1 passing -- PARTIAL, to the author with both fits; the difference is {tag}"


# ------------------------------------------------------------------ self-test
def _toy(n_days=500, per_day=200, beta=7.0, alpha=0.0, seed=1, day_sd=3.0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    delta = rng.normal(-0.024, 0.245, len(day))
    shock = rng.normal(0, day_sd, n_days)[day]
    y = alpha + beta * delta + shock + rng.normal(0, 70, len(day))
    return day, delta, y


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\ndx6_joint.py SELF-TEST -- detectors before any dump is read.\n")
    day, delta, y = _toy()
    c1, c2, U = common_codes(day, day)
    f1 = scores(y, design(delta), c1, U)
    # K1 the diagonal block IS cluster_ols, exactly
    b, se, _ = cluster_ols(y, delta, c1, int(len(np.unique(c1))))
    V = vcov(f1[1], f1[2], f1[3], f1[4])
    chk("K1 the joint diagonal reproduces pseudo_dist.cluster_ols exactly",
        np.allclose(b, f1[0], rtol=0, atol=1e-12) and np.allclose(se, np.sqrt(np.diag(V)), rtol=1e-12, atol=0),
        f"se {se[0]:.6f}/{se[1]:.6f}")
    # K2 identical data -> Var(d) = 0, rho = 1
    J = joint(f1, f1)
    chk("K2 identical data: Var(d) = 0 and rho = 1", np.allclose(J["Vd"], 0, atol=1e-10)
        and np.allclose(J["rho"], 1), f"rho {J['rho']}")
    # K3 independent data -> joint ~ naive
    d2, x2, y2 = _toy(seed=2)
    a1, a2, U2 = common_codes(day, d2 + 10_000)               # disjoint dates
    J = joint(scores(y, design(delta), a1, U2), scores(y2, design(x2), a2, U2))
    chk("K3 disjoint dates: the cross term is exactly 0, joint == naive",
        np.allclose(J["C"], 0), f"C max {np.abs(J['C']).max():.2e}")
    # K4 SHARED rows (real arm shared, half the second arm redrawn), planted slope shift:
    # the NAIVE band passes it, the JOINT band catches it -- the error amendment 9 fixes.
    rng = np.random.default_rng(3)
    yb = y + 0.9 * delta + np.where(rng.random(len(y)) < 0.5, rng.normal(0, 20, len(y)), 0)
    J = joint(scores(y, design(delta), c1, U), scores(yb, design(delta), c1, U))
    v = verdicts(J)
    chk("K4/M1 shared data + slope shift 0.9: naive band PASSES it, joint band CATCHES it",
        v["d_beta"] <= v["naive_band_beta"] and not v["V1"],
        f"|d| {v['d_beta']:.3f}, naive band {v['naive_band_beta']:.3f}, joint {v['band_beta']:.3f}, rho {J['rho'][1]:.3f}")
    # K5 unequal date sets: codes on the union; each diagonal still equals its own cluster_ols
    keep = day >= 3
    cA, cB, Uu = common_codes(day, day[keep])
    fA = scores(y, design(delta), cA, Uu)
    fB = scores(y[keep], design(delta[keep]), cB, Uu)
    _, seB, _ = cluster_ols(y[keep], delta[keep], np.unique(day[keep], return_inverse=True)[1],
                            int(len(np.unique(day[keep]))))
    VB = vcov(fB[1], fB[2], fB[3], fB[4])
    chk("K5 unequal date sets: union codes keep each fit's own SE exact",
        np.allclose(seB, np.sqrt(np.diag(VB)), rtol=1e-12, atol=0), f"{seB[1]:.6f}")
    # K6 the quadratic sensitivity recovers a planted curvature's intercept
    yq = 0.0 + 7.0 * delta + 30.0 * delta ** 2 + rng.normal(0, 70, len(delta))
    fq = scores(yq, design(delta, quad=True), c1, U)
    sq = np.sqrt(np.diag(vcov(fq[1], fq[2], fq[3], fq[4])))
    fl = scores(yq, design(delta), c1, U)
    chk("K6 planted curvature: quadratic alpha ~ 0 while the linear alpha is pulled away",
        abs(fq[0][0]) < 3 * sq[0] and fl[0][0] > fq[0][0],
        f"quad {fq[0][0]:+.3f}+-{sq[0]:.3f}, linear {fl[0][0]:+.3f}")
    # K7 ladder wording
    chk("K7 the ladder keeps amendment 8's consequence: V1 FAIL -> the halt STAYS",
        "halt STAYS" in ladder(dict(V1=False, V2=True, d_alpha=0.0)) and
        "negligible" in ladder(dict(V1=True, V2=False, d_alpha=0.0086)))
    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


# ------------------------------------------------------------------ main
def main(ref_dump, val_dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    from gate_receipts import dump_fingerprint, code_set
    from power_registered_set import registered_cells
    from pseudo_dist import build
    from pseudo_null_pair import day_clustered_mean
    with open(os.path.join(HERE, "receipt_gate.json"), encoding="utf-8") as fh:
        registered = json.load(fh)["dump_fingerprint"]
    f_ref, f_val = dump_fingerprint(ref_dump), dump_fingerprint(val_dump)
    if f_val == registered or f_val == f_ref:
        raise SystemExit("REFUSED: the validation dump must be neither the registered dump nor "
                         "the reference (the inverted guard of amendment 8).")
    if f_ref != registered:
        raise SystemExit("REFUSED: the reference dump is not the registered one.")
    t0 = time.time()
    cells = registered_cells()
    d1 = build(ref_dump, cells, verbose=True)
    d2 = build(val_dump, cells, verbose=True)
    y1 = 100.0 * (d1.r.to_numpy(np.int64) - d1.p.to_numpy(np.int64))
    y2 = 100.0 * (d2.r.to_numpy(np.int64) - d2.p.to_numpy(np.int64))
    x1, x2 = d1.delta.to_numpy(), d2.delta.to_numpy()
    c1, c2, U = common_codes(d1.day.to_numpy(), d2.day.to_numpy())
    F1, F2 = scores(y1, design(x1), c1, U), scores(y2, design(x2), c2, U)
    J = joint(F1, F2)
    se1, se2 = np.sqrt(np.diag(J["V1"])), np.sqrt(np.diag(J["V2"]))
    k0 = [("K0 reference refit reproduces 26.38 (beta, alpha)",
           abs(F1[0][1] - PUB["beta"]) <= 1e-3 and abs(F1[0][0] - PUB["alpha"]) <= 1e-3,
           f"{F1[0][1]:+.4f}, {F1[0][0]:+.4f}"),
          ("K0b diagonal SEs reproduce 26.38 and 26.46",
           all(abs(a - b) <= 1e-4 for a, b in ((se1[1], PUB["se_beta"]), (se1[0], PUB["se_alpha"]),
                                                (se2[1], PUB["v_se_beta"]), (se2[0], PUB["v_se_alpha"]))),
           f"ref {se1[1]:.4f}/{se1[0]:.4f}, val {se2[1]:.4f}/{se2[0]:.4f}"),
          ("K0c validation refit reproduces 26.46 (beta, alpha)",
           abs(F2[0][1] - PUB["v_beta"]) <= 1e-3 and abs(F2[0][0] - PUB["v_alpha"]) <= 1e-3,
           f"{F2[0][1]:+.4f}, {F2[0][0]:+.4f}")]
    for lab, okk, det in k0:
        print(f"  [{'PASS' if okk else 'FAIL'}] {lab}   {det}")
    if not all(k[1] for k in k0):
        raise SystemExit("A known-number check FAILED -- no verdict was computed.")
    v = verdicts(J)
    cs, me = code_set(), hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    stamp = (f"amendment 9 joint test | 144 registered cells | ref n {F1[3]:,} / {F1[4]:,} dates, "
             f"val n {F2[3]:,} / {F2[4]:,} dates, union {U:,} | cluster = trade date, CR1 | "
             f"ref {f_ref} val {f_val} | code-set {cs} | dx6_joint.py {me[:16]} | "
             f"{platform.node()} | ES train A+B, sealed untouched | OUT-OF-DRAW, not out-of-sample")
    print(f"\nSTAMP: {stamp}")
    print(f"  correlation of the two fits' estimates: alpha {J['rho'][0]:.4f}, beta {J['rho'][1]:.4f}")
    print(f"  V1 |d beta| {v['d_beta']:.4f} vs joint band {v['band_beta']:.4f} (naive was "
          f"{v['naive_band_beta']:.4f}) -> {'PASS' if v['V1'] else 'FAIL'}  [a FORMALITY: cannot "
          f"realistically fail]")
    print(f"  V2 alpha_new {F2[0][0]:+.4f} (|.| <= {ATTRIB_PP}) and |d alpha| {v['d_alpha']:.4f} vs "
          f"joint band {v['band_alpha']:.4f} (naive {v['naive_band_alpha']:.4f}) -> "
          f"{'PASS' if v['V2'] else 'FAIL'}")
    lad = ladder(v)
    print(f"  AMENDMENT 8 LADDER, verbatim consequences: {lad}")
    # V3, descriptive: the raw-gap shift; the real arm cancels exactly
    pred = F1[0][0] + F1[0][1] * x2.mean()
    print(f"  V3 DESCRIPTIVE (seen 2026-10-02, never a criterion): predicted raw gap {pred:+.4f}, "
          f"observed pooled {y2.mean():+.4f} pp")

    # linearity sensitivity on data\dump
    Fq = scores(y1, design(x1, quad=True), c1, U)
    seq = np.sqrt(np.diag(vcov(Fq[1], Fq[2], Fq[3], Fq[4])))
    print("\n  LINEARITY SENSITIVITY on data\\dump (chosen after results; NEVER a criterion):")
    print(f"    linear    alpha {F1[0][0]:+.4f} +- {se1[0]:.4f}")
    print(f"    quadratic alpha {Fq[0][0]:+.4f} +- {seq[0]:.4f}   beta {Fq[0][1]:+.4f}   "
          f"curvature {Fq[0][2]:+.4f} +- {seq[2]:.4f}")
    edges = np.quantile(x1, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, x1, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = dec == k
        r = day_clustered_mean(y1[m], c1[m], U)
        rows.append(dict(decile=k + 1, mean_delta=float(x1[m].mean()), gap_pp=float(r["mean"]),
                         se_pp=float(r["se"]), n=int(m.sum()),
                         linear_fit_pp=float(F1[0][0] + F1[0][1] * x1[m].mean())))
        print(f"    delta decile {k+1:>2}: mean delta {rows[-1]['mean_delta']:+.4f}  gap "
              f"{rows[-1]['gap_pp']:+.4f} +- {rows[-1]['se_pp']:.4f}  linear fit "
              f"{rows[-1]['linear_fit_pp']:+.4f}  n {rows[-1]['n']:,}")
    out = dict(stamp=stamp, code_set=cs, producer_sha256=me, rho=J["rho"].tolist(),
               ref=dict(b=F1[0].tolist(), se=se1.tolist()), val=dict(b=F2[0].tolist(), se=se2.tolist()),
               verdicts=v, ladder=lad, v3=dict(pred=pred, observed=float(y2.mean())),
               quadratic=dict(b=Fq[0].tolist(), se=seq.tolist()), deciles=rows)
    path = os.path.join(HERE, f"dx6_joint_{datetime.date.today().isoformat()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n  wrote {os.path.basename(path)}   {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(args[0], args[1]))
