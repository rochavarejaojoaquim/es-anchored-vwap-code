r"""
pseudo_null_pair.py -- HANDOFF section 25, amendment (a): the EXCHANGEABILITY GATE.

    python pseudo_null_pair.py --selftest         toy fixtures only, reads no real data
    python pseudo_null_pair.py DUMP_DIR           N=60 (36 cells), the registered sample
    python pseudo_null_pair.py DUMP_DIR --all-n   all five lookbacks

WHAT IT ASKS.  Two pseudo anchors drawn for the SAME setup by the SAME procedure,
differing only in their random stream, must fill at the same rate.  If they do not, the
pseudo-VWAP draw is not exchangeable and no real-minus-pseudo gap may be read.

    NULL_GAP      day-clustered mean of (fill_A - fill_B), in pp, with a 95% CI
    PASS          CI entirely inside [-0.25, +0.25]
    FAIL          CI entirely outside it
    INCONCLUSIVE  otherwise -- extend from N=60 to all N.  Still INCONCLUSIVE after all
                  N is NOT PASSED (inconclusive after all N): no real gap is read, the
                  test is reported UNRESOLVED (exchangeability not established), and the
                  tolerance is not loosened.  Exit code is non-zero for anything but PASS.
    DEGENERATE    A and B picked the same anchor on >= 99% of paired setups, so the
                  two draws are not independent and the gate is meaningless.  Checked
                  BEFORE the CI and overrides it.

The three CI cases in the amendment overlap (a CI can be partly inside the band, and a
wide CI can also be entirely outside), so they are applied in this fixed order: FAIL,
then PASS, then INCONCLUSIVE.  That ordering is this script's disambiguation, not the
amendment's words -- it is printed with the verdict.

A is the STORED draw (pseudo_anchor_bar / pseudo_w_0.0 in the dump).  B is drawn here
with an independent seed stream.  Before anything else the script re-draws A from the
stored seed rule and requires it to reproduce pseudo_anchor_bar on EVERY setup (D3); on
any mismatch it stops and reads nothing.

NO EDIT TO code\ IS NEEDED.  pseudo_vwap.pseudo_for_cell uses its `cell` argument in
exactly one place, the per-setup seed (code\pseudo_vwap.py:134):
    rng = np.random.default_rng([SEED, zlib.crc32(f"{cell}_{s}".encode())])
so passing a salted name ("<cell>#B") is a seed salt that leaves the matching, the
window, the redraw cap and the walk identical.  The salt is printed in the stamp.

It compares PSEUDO WITH PSEUDO only.  It never reads a real fill rate, so it can be run
and read before the section 25.4 power validation.  Read-only: writes one CSV.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, CODE)

from loader import load                                                  # noqa: E402
from pipeline import rolling_argextreme, session_anchor, vwap_and_sd     # noqa: E402
from spec import ANCHORS, H_REC_MULT                                     # noqa: E402
from walk_fast import walk                                               # noqa: E402
import pseudo_vwap as PV                                                 # noqa: E402

TOL_PP = 0.25          # amendment (a); a convention, a quarter of the 1.0pp floor
SALT = "#B"
DEGEN_FRAC = 0.99
CELL_RE = re.compile(r"N(\d+)_(\w+?)_(ATR|SD)_([\d.]+)$")
COLS = ["is_real", "bar", "day", "block", "T", "side", "dist_atr", "atr0",
        "pseudo_matched", "pseudo_anchor_bar", "pseudo_w_0.0"]


# ------------------------------------------------------------------ statistics
def day_clustered_mean(d, g, D):
    """Mean of d, clustered on trade date (F1: exact, never a divided t)."""
    n = len(d)
    if n == 0:
        return None
    S = np.bincount(g, weights=d, minlength=D)
    C = np.bincount(g, minlength=D)
    m = S.sum() / n
    used = C > 0
    Dd = int(used.sum())
    var = (Dd / max(Dd - 1, 1)) * np.sum((S[used] - m * C[used]) ** 2) / n ** 2
    se = float(np.sqrt(var))
    return dict(mean=float(m), se=se, lo=float(m - 1.96 * se), hi=float(m + 1.96 * se),
                n=int(n), days=Dd)


def gate_verdict(lo_pp, hi_pp, tol=TOL_PP):
    if lo_pp > tol or hi_pp < -tol:
        return "FAIL"
    if lo_pp >= -tol and hi_pp <= tol:
        return "PASS"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------- the cells
def cells_for(dump, all_n):
    out = []
    for f in sorted(glob.glob(os.path.join(dump, "N*.parquet"))):
        if ".m2." in f or ".tw2." in f:
            continue
        cell = os.path.basename(f)[:-8]
        m = CELL_RE.match(cell)
        N, ak, trig, eps = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        if not all_n and N != 60:
            continue
        out.append((f, cell, N, ak, trig, eps))
    return out


def main(dump, all_n, write_receipt=True):
    t0 = time.time()
    cells = cells_for(dump, all_n)
    if not cells:
        raise SystemExit(f"no cells found in {dump}")
    print(f"  {len(cells)} cells from {dump}   (loading the bars ...)", flush=True)
    d = load("train")
    close = d.close.to_numpy()
    hi = d.high.to_numpy().astype(float)
    lo_ = d.low.to_numpy().astype(float)
    seg = d.seg.to_numpy()
    nbars = len(d)
    sess = session_anchor(d)
    anchors = {k: (c, mx) for k, c, mx in ANCHORS}
    cache, wlo_cache = {}, {}
    print(f"  bars loaded in {time.time() - t0:.0f}s", flush=True)

    dA, dB, days_all, rows = [], [], [], []
    same_anchor, paired = 0, 0
    repro_bad = []
    for f, cell, N, ak, trig, eps in cells:
        if (N, ak) not in cache:
            cache.clear()
            col, mx = anchors[ak]
            a = sess if ak == "session" else rolling_argextreme(
                d[col].to_numpy().astype(float), N, seg, mx)
            _, _, _, cum, ref = vwap_and_sd(d, a)
            cache[(N, ak)] = (a, cum, ref)
        a, cum, ref = cache[(N, ak)]
        session = ak == "session"
        if not session and N not in wlo_cache:
            wlo_cache.clear()
            wlo_cache[N] = PV.window_lo(nbars, N, seg)
        wlo = None if session else wlo_cache[N]

        x = pd.read_parquet(f, columns=COLS)
        x = x[x.is_real == 1].reset_index(drop=True)
        sig = x.bar.to_numpy().astype(np.int64)
        atr0 = x.atr0.to_numpy(float)
        side = x.side.to_numpy(float)
        target = x.dist_atr.to_numpy(float)
        T = int(x["T"].iloc[0])
        h_rec = H_REC_MULT * T
        anc = a[sig]

        # ---- D3: re-draw A under the stored seed rule; it must reproduce the dump
        paA, plA, _ = PV.pseudo_for_cell(cum, ref, sig, anc, wlo, session, close,
                                         atr0, target, side, cell)
        stored = x.pseudo_anchor_bar.to_numpy().astype(np.int64)
        nbad = int((paA != stored).sum())
        if nbad:
            repro_bad.append((cell, nbad, len(stored)))

        # ---- B: the same procedure, an independent stream
        paB, plB, _ = PV.pseudo_for_cell(cum, ref, sig, anc, wlo, session, close,
                                         atr0, target, side, cell + SALT)
        both = (paA >= 0) & (paB >= 0)
        if not both.any():
            continue
        s_both = sig[both]
        wB = walk(close, cum, ref, seg, s_both, paB[both], atr0[both],
                  close[s_both] - plB[both], cum.vol(paB[both], s_both), h_rec,
                  T=T, hi=hi, lo=lo_)
        fB = ((wB["wf_0.0"] >= 1) & (wB["wf_0.0"] <= T)).astype(float)
        wA = x["pseudo_w_0.0"].to_numpy()[both]
        fA = ((wA >= 1) & (wA <= T)).astype(float)

        rA = np.abs(close[s_both] - plA[both]) / atr0[both] / target[both]
        rB = np.abs(close[s_both] - plB[both]) / atr0[both] / target[both]
        same = int((paA[both] == paB[both]).sum())
        same_anchor += same
        paired += int(both.sum())
        dA.append(fA)
        dB.append(fB)
        days_all.append(x["day"].to_numpy()[both])
        rows.append(dict(cell=cell, N=N, anchor=ak, trigger=trig, eps=eps,
                         setups=len(x), paired=int(both.sum()),
                         unmatched_A=int((paA < 0).sum()), unmatched_B=int((paB < 0).sum()),
                         fill_A=float(fA.mean()), fill_B=float(fB.mean()),
                         gap_pp=100 * float(fA.mean() - fB.mean()),
                         ratio_A=float(np.nanmean(rA)), ratio_B=float(np.nanmean(rB)),
                         same_anchor=same, repro_bad=nbad))
        print(f"  {cell:<22} paired {int(both.sum()):>7,}  fill A {100*fA.mean():5.2f}%  "
              f"B {100*fB.mean():5.2f}%  gap {100*(fA.mean()-fB.mean()):+6.3f}pp  "
              f"dist A {np.nanmean(rA):.3f} B {np.nanmean(rB):.3f}"
              f"{'  REPRO FAIL ' + str(nbad) if nbad else ''}", flush=True)

    if repro_bad:
        print("\n  [FAIL] D3: the re-drawn A does not reproduce the stored pseudo_anchor_bar")
        for c, b, t in repro_bad[:10]:
            print(f"         {c}: {b:,} of {t:,} setups differ")
        raise SystemExit("STOPPED: the stored draw cannot be reproduced from its own seed "
                         "rule, so the gate would test something other than the stored column.")
    print(f"\n  [PASS] D3: the re-drawn A reproduced the stored pseudo_anchor_bar on every "
          f"setup of all {len(rows)} cells")

    fA = np.concatenate(dA)
    fB = np.concatenate(dB)
    day = np.concatenate(days_all)
    uniq, g = np.unique(day, return_inverse=True)
    res = day_clustered_mean(100.0 * (fA - fB), g, len(uniq))
    frac_same = same_anchor / max(paired, 1)
    degenerate = frac_same >= DEGEN_FRAC
    verd = "DEGENERATE" if degenerate else gate_verdict(res["lo"], res["hi"])

    R = pd.DataFrame(rows)
    print("\nSTAMP: pseudo-vs-pseudo exchangeability gate (section 25 amendment (a)) | "
          f"{len(R)} cells ({'all N' if all_n else 'N=60'}) | paired setups {res['n']:,} "
          f"(matched in BOTH draws) | independent units: {res['days']:,} trade dates | "
          f"clustered on trade date | A = stored dump draw, B = seed salt '{SALT}' | "
          f"SEED {PV.SEED}, rule {PV.SEED_RULE} | REDRAW_CAP {PV.REDRAW_CAP}, TOL_FRAC "
          f"{PV.TOL_FRAC} | tau 0.0, wick fill within T | code-set {PV.code_set(CODE)} | "
          f"dump {dump} | ES, train blocks A+B, sealed 2018-2022 untouched | "
          "no real fill rate is read")
    print(f"\n  fill rate  A {100*fA.mean():6.3f}%   B {100*fB.mean():6.3f}%")
    print(f"  NULL_GAP (A - B) = {res['mean']:+.4f} pp   95% CI [{res['lo']:+.4f}, {res['hi']:+.4f}]"
          f"   SE {res['se']:.4f}")
    print(f"  distance ratio (realized / dist_atr):  A mean {R.ratio_A.mean():.4f}   "
          f"B mean {R.ratio_B.mean():.4f}   (matching band 0.900-1.100)")
    print(f"  unmatched: A {R.unmatched_A.sum():,}  B {R.unmatched_B.sum():,} of "
          f"{R.setups.sum():,} setups")
    print(f"  A and B picked the SAME anchor on {frac_same:.4%} of paired setups "
          f"(degenerate at >= {DEGEN_FRAC:.0%})")
    print(f"\n  tolerance +-{TOL_PP} pp | order: FAIL (CI entirely outside) -> PASS "
          f"(CI entirely inside) -> INCONCLUSIVE")
    print(f"  GATE: {verd}")
    if verd == "PASS":
        print("  -> the draw is exchangeable at this tolerance; the real gap may be read.")
    elif verd == "INCONCLUSIVE":
        print("  -> extend to all N (--all-n). More N adds setups, not trade dates, so the "
              "precision may improve little.\n     Still INCONCLUSIVE after all N: "
              "NOT PASSED (inconclusive after all N) -- no real gap is read, the test is "
              "reported\n     UNRESOLVED (exchangeability not established), and the tolerance "
              "is not loosened.")
    elif verd == "FAIL":
        print("  -> STOP. Do not read a real-minus-pseudo gap.")
    else:
        print("  -> the two draws are not independent; the gate is meaningless. Check the salt.")
    out = os.path.join(os.getcwd(), f"pseudo_null_pair_{'alln' if all_n else 'n60'}.csv")
    R.to_csv(out, index=False)
    print(f"\n  wrote {out}   ({time.time() - t0:.0f}s)")

    # ---- machine-readable receipt (26.28). The CSV above holds the per-cell frame
    # only; the POOLED statistic that decides this gate used to be printed and lost,
    # and it cannot be rebuilt from the CSV because the clustering is on trade date
    # pooled across cells. gate_receipts.require_pass() re-derives the verdict from
    # exactly these numbers rather than trusting the string.
    stats = dict(
        gap_pp=float(res["mean"]), se_pp=float(res["se"]),
        ci_lo_pp=float(res["lo"]), ci_hi_pp=float(res["hi"]),
        z=1.96, margin_pp=float(TOL_PP),
        n_setups=int(res["n"]), n_days=int(res["days"]),
        n_paired=int(paired), n_setups_total=int(R.setups.sum()),
        frac_same_anchor=float(frac_same), degen_frac=float(DEGEN_FRAC),
        n_cells=int(len(R)), repro_bad=int(len(repro_bad)),
    )
    if write_receipt:
        try:
            sys.path.insert(0, HERE)
            import gate_receipts
            path = gate_receipts.write(
                "gate", dump, verd, stats,
                extra=dict(all_n=bool(all_n), salt=SALT, seed=int(PV.SEED),
                           redraw_cap=int(PV.REDRAW_CAP), tol_frac=float(PV.TOL_FRAC),
                           fill_A=float(fA.mean()), fill_B=float(fB.mean()),
                           unmatched_A=int(R.unmatched_A.sum()),
                           unmatched_B=int(R.unmatched_B.sum()),
                           csv=os.path.basename(out), wall_s=round(time.time() - t0, 1)))
            print(f"  wrote {path}")
        except Exception as e:
            print(f"  [WARN] receipt NOT written: {type(e).__name__}: {e}\n"
                  f"         pseudo_gap.py will refuse to run until it exists.")
    return verd, stats


# ------------------------------------------------------------------ self-test
def _toy_bars(rng, n=1500):
    seg = np.repeat([0, 1], n // 2)
    c = 4000 + 0.25 * rng.integers(-3, 4, n).cumsum()
    h = c + 0.25 * rng.integers(0, 4, n)
    lo = c - 0.25 * rng.integers(0, 4, n)
    v = rng.integers(1, 20, n)
    d = pd.DataFrame(dict(high=h, low=lo, close=c, volume=v, seg=seg))
    _, _, _, cum, ref = vwap_and_sd(d, np.arange(n))
    return d, seg, c, h, lo, cum, ref


def selftest():
    print("SELF-TEST (toy fixtures; no dump is read)\n")
    res = []

    def check(name, ok):
        res.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    rng = np.random.default_rng(7)
    # ---- 1. the statistic, against an independent implementation
    import unmatched_outcomes as UO
    n, D = 40000, 600
    g = rng.integers(0, D, n)
    v = rng.normal(0, 1, n)
    mine = day_clustered_mean(v, g, D)
    acc = UO.Acc(D)
    acc.add(g, v, np.zeros(n))
    theirs = acc.result()
    check(f"K1 day-clustered mean matches unmatched_outcomes.Acc "
          f"(mean {mine['mean']:.6f} vs {theirs['eff']:.6f}, se {mine['se']:.6f} vs {theirs['se']:.6f})",
          abs(mine["mean"] - theirs["eff"]) < 1e-12 and abs(mine["se"] - theirs["se"]) < 1e-12)

    # ---- 2. the gate, at a realistic sample size
    def gate_on(delta, n=400_000, D=3000, seed=1):
        r = np.random.default_rng(seed)
        gg = r.integers(0, D, n)
        fa = (r.random(n) < 0.50).astype(float)
        fb = (r.random(n) < 0.50 + delta).astype(float)
        z = day_clustered_mean(100.0 * (fa - fb), gg, D)
        return z, gate_verdict(z["lo"], z["hi"])
    z0, v0 = gate_on(0.0)
    check(f"K2 no bias, {z0['n']:,} paired setups: NULL_GAP {z0['mean']:+.3f} pp "
          f"[{z0['lo']:+.3f}, {z0['hi']:+.3f}] -> {v0}", v0 == "PASS")
    z1, v1 = gate_on(0.01)
    check(f"M1 planted +1.0pp bias in B: NULL_GAP {z1['mean']:+.3f} pp "
          f"[{z1['lo']:+.3f}, {z1['hi']:+.3f}] -> {v1}", v1 == "FAIL")
    z2, v2 = gate_on(0.006)
    check(f"M2 planted +0.6pp bias in B: NULL_GAP {z2['mean']:+.3f} pp "
          f"[{z2['lo']:+.3f}, {z2['hi']:+.3f}] -> {v2}", v2 == "FAIL")
    z2b, v2b = gate_on(0.004)
    check(f"K2b BOUNDARY, planted +0.4pp with a +-{1.96*z2b['se']:.2f} pp half-width: "
          f"[{z2b['lo']:+.3f}, {z2b['hi']:+.3f}] overlaps the band -> {v2b} (NOT FAIL). The "
          f"gate cannot fail a bias of roughly 0.25-0.45pp at this precision; the "
          f"'still INCONCLUSIVE after all N = NOT PASSED' backstop is what covers it",
          v2b == "INCONCLUSIVE")
    z3, v3 = gate_on(0.0, n=4000, D=300, seed=3)
    check(f"M3 too small a sample ({z3['n']:,} setups, half-width {1.96*z3['se']:.2f} pp): -> {v3}",
          v3 == "INCONCLUSIVE")

    # ---- 3. the plumbing, on the REAL production functions
    d, seg, c, h, lo, cum, ref = _toy_bars(rng)
    N, nb = 60, len(c)
    real = rolling_argextreme(h, N, seg, maximum=True)
    wlo = PV.window_lo(nb, N, seg)
    sig = np.arange(120, nb, 9)
    sig = sig[real[sig] >= 0]
    V0 = np.array([PV.vwap_from_anchor(cum, ref, int(real[s]), int(s)) for s in sig])
    atr0 = np.full(len(sig), 2.0)
    keep = np.abs(c[sig] - V0) / atr0 > 0.4
    sig, V0, atr0 = sig[keep], V0[keep], atr0[keep]
    anc = real[sig]
    side = np.sign(c[sig] - V0)
    target = np.abs(c[sig] - V0) / atr0
    paA, plA, _ = PV.pseudo_for_cell(cum, ref, sig, anc, wlo, False, c, atr0, target, side, "TOY")
    paB, plB, _ = PV.pseudo_for_cell(cum, ref, sig, anc, wlo, False, c, atr0, target, side, "TOY" + SALT)
    both = (paA >= 0) & (paB >= 0)
    frac_same = float((paA[both] == paB[both]).mean())
    check(f"K3 the salt gives a DIFFERENT draw: same anchor on {frac_same:.1%} of "
          f"{int(both.sum())} paired setups (must be < {DEGEN_FRAC:.0%})", frac_same < DEGEN_FRAC)
    rA = np.abs(c[sig[both]] - plA[both]) / atr0[both] / target[both]
    rB = np.abs(c[sig[both]] - plB[both]) / atr0[both] / target[both]
    inband = float(((rA >= 0.9 - 1e-9) & (rA <= 1.1 + 1e-9)).mean())
    inbandB = float(((rB >= 0.9 - 1e-9) & (rB <= 1.1 + 1e-9)).mean())
    check(f"K4 both draws land inside the +-10% matching band (A {inband:.1%}, B {inbandB:.1%}); "
          f"mean ratio A {rA.mean():.3f} B {rB.mean():.3f}", inband == 1.0 and inbandB == 1.0)
    T = 30
    wk = walk(c, cum, ref, seg, sig[both], paB[both], atr0[both],
              c[sig[both]] - plB[both], cum.vol(paB[both], sig[both]), 3 * T, T=T,
              hi=h, lo=lo)
    f = ((wk["wf_0.0"] >= 1) & (wk["wf_0.0"] <= T)).astype(float)
    check(f"K5 the shared walk runs on the B draw and returns a fill rate "
          f"({100*f.mean():.1f}% of {len(f)} setups)", len(f) == int(both.sum()) and 0 < f.mean() < 1)
    paB2, _, _ = PV.pseudo_for_cell(cum, ref, sig, anc, wlo, False, c, atr0, target, side, "TOY")
    both2 = (paA >= 0) & (paB2 >= 0)
    frac2 = float((paA[both2] == paB2[both2]).mean())
    check(f"M4 MUTATION, B seed = A seed (no salt): same anchor on {frac2:.1%} -> DEGENERATE, "
          f"not PASS", frac2 >= DEGEN_FRAC)
    paC, _, _ = PV.pseudo_for_cell(cum, ref, sig, anc, wlo, False, c, atr0, target, side, "TOY")
    check("K6 the draw is reproducible: the same tag twice gives identical anchors (the D3 "
          "check the real run makes against the dump)", np.array_equal(paA, paC))
    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--all-n", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.dump:
        raise SystemExit(__doc__)
    if not selftest():
        raise SystemExit("\nSELF-TEST FAILED -- no real data was read.")
    print()
    v, _ = main(a.dump, a.all_n)
    sys.exit(0 if v == "PASS" else 1)
