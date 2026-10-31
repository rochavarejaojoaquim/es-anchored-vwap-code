r"""
dx6_validate.py -- section 25 amendment 8: the OUT-OF-SAMPLE validation of DX6.

DX6 (26.38) attributed the whole -0.1865 pp inversion to the distance asymmetry:
beta +7.8793 pp per ATR, corrected gap alpha +0.0066 pp, CI [-0.0167, +0.0300].
Its one real weakness is that it is A MODEL FITTED AFTER THE PROBLEM WAS SEEN.

The rebuild of 2026-10-02 was REJECTED on its own criteria (26.44) -- but it has a
DIFFERENT distance asymmetry, mean delta -0.02864 against -0.02422, with 54.74% of
pseudo anchors changed. It is therefore a dataset the model never saw, and refitting
there is an external validation. Mayo: post hoc carries no information about
SEVERITY; a model that predicts unseen data has been severely tested.

THE REJECTED DUMP STAYS REJECTED. Nothing this script prints can become the H1
answer or lift the halt. It validates a METHOD, not a hypothesis.

BOTH DUMPS ARE REFITTED IN ONE RUN, with the same code, rather than quoting 26.38's
numbers: that removes transcription risk (P2) and turns the old fit into a
known-number check against the published figures.

THE GUARD IS INVERTED, and that is the point. Reading a gap on an unregistered dump
is exactly what gate_receipts exists to prevent, so this script REFUSES unless the
validation dump's fingerprint DIFFERS from the one in receipt_gate.json. It cannot
be used to peek at a registered dump; it only works on one that is not.

    python dx6_validate.py --selftest
    python dx6_validate.py data\dump rebuild_pseudo_2026-09-30\data\dump

Written 2026-10-02. Project root; rotates nothing. Reads only.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gate_receipts import dump_fingerprint                             # noqa: E402
from power_registered_set import registered_cells                      # noqa: E402
from pseudo_dist import build, cluster_ols                             # noqa: E402
from pseudo_gap import day_codes, gap_stats                            # noqa: E402

PUB_BETA, PUB_ALPHA = 7.8793, 0.0066      # 26.38, for the K0 known-number check
K0_TOL_BETA, K0_TOL_ALPHA = 0.001, 0.001
ATTRIB_PP = 0.05                           # DX6's own attribution threshold (V2)


def fit(dump, cells, label):
    df = build(dump, cells, verbose=True)
    y = 100.0 * (df.r.to_numpy(np.int64) - df.p.to_numpy(np.int64))
    g, D = day_codes(df.day.to_numpy())
    b, se, _ = cluster_ols(y, df.delta.to_numpy(), g, D)
    st = gap_stats(df.r.to_numpy(), df.p.to_numpy(), df.day.to_numpy())
    out = dict(label=label, alpha=float(b[0]), se_alpha=float(se[0]),
               beta=float(b[1]), se_beta=float(se[1]),
               mean_delta=float(df.delta.mean()), n=len(df), D=D,
               raw_daymean=st["daymean_pp"], raw_pooled=st["pooled_pp"],
               anchors=df[["cell", "bar"]].assign(a=df.pdist).shape[0])
    print(f"\n  {label}: n {out['n']:,} over {D:,} dates | mean delta "
          f"{out['mean_delta']:+.5f} | raw gap {out['raw_daymean']:+.4f} pp")
    print(f"    alpha {out['alpha']:+.4f} +-{out['se_alpha']:.4f}   "
          f"beta {out['beta']:+.4f} +-{out['se_beta']:.4f}")
    return out


def selftest():
    ok = [True]

    def chk(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        ok[0] &= bool(cond)

    print("\ndx6_validate.py SELF-TEST -- detectors before any dump is read.\n")

    # the two-sample z on a difference of independent estimates
    def zdiff(a, sa, b, sb):
        s = np.sqrt(sa ** 2 + sb ** 2)
        return abs(a - b), 1.96 * s, abs(a - b) <= 1.96 * s

    d, lim, p = zdiff(7.8793, 0.1048, 7.95, 0.11)
    chk("K1 two estimates 0.07 apart with SE ~0.1 are NOT distinguishable", p,
        f"|diff| {d:.4f} vs limit {lim:.4f}")
    d, lim, p = zdiff(7.8793, 0.1048, 9.50, 0.11)
    chk("K2 two estimates 1.6 apart ARE distinguishable (so V1 can fail)", not p,
        f"|diff| {d:.4f} vs limit {lim:.4f}")
    d, lim, p = zdiff(0.0066, 0.0119, 0.0100, 0.0120)
    chk("K3 two near-zero intercepts agree", p, f"|diff| {d:.4f} vs limit {lim:.4f}")
    d, lim, p = zdiff(0.0066, 0.0119, 0.2000, 0.0120)
    chk("K4 an intercept that moved to +0.20 does NOT agree", not p,
        f"|diff| {d:.4f} vs limit {lim:.4f}")
    chk("K5 V2's absolute arm rejects a large |alpha| even if it 'agrees'",
        not (abs(0.30) <= ATTRIB_PP), "alpha 0.30 pp exceeds the 0.05 pp threshold")
    pred = PUB_ALPHA + PUB_BETA * (-0.02864)
    chk("K6 V3's predicted gap for the rejected dump re-derives to -0.2191 pp",
        abs(pred - (-0.2191)) < 0.0005, f"{pred:+.4f} pp")
    ctrl = PUB_ALPHA + PUB_BETA * (-0.02422)
    chk("K7 and the same formula reproduces the OLD dump's pooled -0.1842 pp "
        "(arithmetic control, NOT independent)",
        abs(ctrl - (-0.1842)) < 0.0005, f"{ctrl:+.4f} pp")
    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'A DETECTOR IS DEAD'}\n")
    return ok[0]


def main(ref_dump, val_dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")

    # ---- THE INVERTED GUARD
    rec = os.path.join(HERE, "receipt_gate.json")
    if not os.path.exists(rec):
        raise SystemExit("REFUSED: receipt_gate.json is absent, so the registered "
                         "dump cannot be identified and the guard cannot be applied.")
    with open(rec, "r", encoding="utf-8") as fh:
        registered = json.load(fh)["dump_fingerprint"]
    f_ref, f_val = dump_fingerprint(ref_dump), dump_fingerprint(val_dump)
    print(f"\n  registered dump fingerprint : {registered}")
    print(f"  reference dump              : {f_ref}  ({ref_dump})")
    print(f"  validation dump             : {f_val}  ({val_dump})")
    if f_val == registered:
        raise SystemExit(
            "REFUSED: the validation dump IS the registered one. This script exists "
            "to read an UNREGISTERED dump for method validation; pointing it at the "
            "registered dump would be a way around gate_receipts, which is exactly "
            "what it must not become.")
    if f_val == f_ref:
        raise SystemExit("REFUSED: the two dumps are identical, so there is nothing "
                         "out-of-sample about this.")
    if f_ref != registered:
        print("  [WARN] the reference dump is NOT the registered one; K0 will "
              "almost certainly fail.")
    print("  [OK] the validation dump is not the registered dump\n")

    cells = registered_cells()
    old = fit(ref_dump, cells, "REFERENCE (registered dump)")

    k0b = abs(old["beta"] - PUB_BETA) <= K0_TOL_BETA
    k0a = abs(old["alpha"] - PUB_ALPHA) <= K0_TOL_ALPHA
    print(f"\n  [{'PASS' if k0b and k0a else 'FAIL'}] K0 the reference refit "
          f"reproduces 26.38   beta {old['beta']:+.4f} vs {PUB_BETA}, "
          f"alpha {old['alpha']:+.4f} vs {PUB_ALPHA}")
    if not (k0b and k0a):
        raise SystemExit("K0 FAILED -- the reference fit does not reproduce the "
                         "published DX6 numbers. The validation dump was NOT read.")

    new = fit(val_dump, cells, "VALIDATION (rejected dump)")

    print("\n================ AMENDMENT 8  OUT-OF-SAMPLE VALIDATION ================")
    sd_b = float(np.sqrt(old["se_beta"] ** 2 + new["se_beta"] ** 2))
    sd_a = float(np.sqrt(old["se_alpha"] ** 2 + new["se_alpha"] ** 2))
    db, da = abs(new["beta"] - old["beta"]), abs(new["alpha"] - old["alpha"])
    v1 = db <= 1.96 * sd_b
    v2 = (abs(new["alpha"]) <= ATTRIB_PP) and (da <= 1.96 * sd_a)
    print(f"  V1 slope reproduces : |{new['beta']:+.4f} - {old['beta']:+.4f}| = "
          f"{db:.4f}  vs  1.96*SE_diff = {1.96*sd_b:.4f}   -> "
          f"{'PASS' if v1 else 'FAIL'}")
    print(f"  V2 corrected gap    : alpha_new {new['alpha']:+.4f} "
          f"(|.| <= {ATTRIB_PP}) and |diff| {da:.4f} vs {1.96*sd_a:.4f}   -> "
          f"{'PASS' if v2 else 'FAIL'}")
    pred = old["alpha"] + old["beta"] * new["mean_delta"]
    print(f"  V3 DESCRIPTIVE, not a criterion: predicted raw gap "
          f"{pred:+.4f} pp, observed {new['raw_daymean']:+.4f} pp "
          f"(diff {new['raw_daymean'] - pred:+.4f})")
    print(f"     the predicted SHIFT from the reference is only "
          f"{pred - old['raw_daymean']:+.4f} pp -- about 2 sigma, which is why this "
          f"scalar is reported and not used to decide.")

    print("\n  AMENDMENT 8 VERDICT, fixed before this ran:")
    if v1 and v2:
        print("    CONFIRMED OUT OF SAMPLE. The relationship holds on draws the model")
        print("    never saw, so the DX6 correction has survived a severe test. The")
        print("    paper's primary route becomes the corrected null, with the")
        print("    artefact, its cause and this validation all reported in full.")
    elif not v1:
        print("    NOT CONFIRMED. The slope is dump-specific, so DX6's attribution is")
        print("    incomplete. The halt STAYS and a build audit follows.")
    else:
        print("    PARTIAL. The slope holds but the corrected gap is not null on the")
        print("    new draws. Goes to the author with both fits; no verdict is invented")
        print("    after the fact.")
    print("\n  THE REJECTED DUMP STAYS REJECTED. Nothing above is the H1 answer and")
    print("  nothing above lifts the halt. 26.34 and 26.38 are unchanged.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(args[0], args[1]))
