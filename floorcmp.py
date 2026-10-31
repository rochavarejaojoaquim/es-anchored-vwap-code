"""
floorcmp.py -- compare two floor runs cell+side by cell+side.

    python floorcmp.py floors_cmp_day floors_cmp_strata

Built-in check (D3): obs_score does NOT depend on the null, so it MUST be
identical in both runs.  If it is not, something other than the shuffle
changed and the comparison is void.  Reported PASS/FAIL before any result.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "*.parquet")))
    if not fs:
        raise SystemExit(f"no .parquet files in {d}")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["tag"] = [os.path.basename(f)[:-8] for f in fs]
    return df.set_index("tag").sort_index()


def main(dA, dB, nameA="A", nameB="B"):
    A, B = load(dA), load(dB)
    common = A.index.intersection(B.index)
    print(f"  {nameA}: {len(A)} files   {nameB}: {len(B)} files   "
          f"in both: {len(common)}")
    if len(common) == 0:
        raise SystemExit("no overlapping cell+sides -- nothing to compare")
    only = set(A.index) ^ set(B.index)
    if only:
        print(f"  NOTE: {len(only)} cell+sides in one run only: {sorted(only)[:5]}")
    A, B = A.loc[common], B.loc[common]

    # ---- D3 known-number check -------------------------------------------
    d_obs = float(np.abs(A.obs_score - B.obs_score).max())
    ok = d_obs < 1e-12
    print(f"\n  [{'PASS' if ok else 'FAIL'}] obs_score identical in both runs "
          f"(max |diff| {d_obs:.3e}) -- the search must not depend on the null")
    if "null_unit" in A.columns and "null_unit" in B.columns:
        ua, ub = sorted(A.null_unit.unique()), sorted(B.null_unit.unique())
        print(f"  [{'PASS' if ua != ub and len(ua) == 1 == len(ub) else 'FAIL'}] "
              f"each run has ONE null_unit and they differ: {ua} vs {ub}")
    else:
        print("  [ -- ] no null_unit column: floors written before the stamp patch")
    if not ok:
        raise SystemExit("obs_score differs -- comparison void, stop here")

    # ---- the comparison ---------------------------------------------------
    out = pd.DataFrame({
        "obs": A.obs_score,
        f"floor_{nameA}": A.floor95,
        f"floor_{nameB}": B.floor95,
        "delta": B.floor95 - A.floor95,
        f"edge_{nameA}": A.edge_over_floor,
        f"edge_{nameB}": B.obs_score - B.floor95,
        f"p_{nameA}": A.p,
        f"p_{nameB}": B.p,
    })
    print("\n" + (100 * out).round(3).to_string())

    dl = out.delta
    print(f"\n  floor change, {nameB} minus {nameA}, in pp of the score:")
    print(f"    mean {100*dl.mean():+.3f}   median {100*dl.median():+.3f}   "
          f"min {100*dl.min():+.3f}   max {100*dl.max():+.3f}")
    print(f"    higher in {nameB}: {int((dl > 0).sum())} of {len(dl)}   "
          f"lower: {int((dl < 0).sum())}")

    sA = out[f"p_{nameA}"] <= 0.05
    sB = out[f"p_{nameB}"] <= 0.05
    print(f"\n  survival at the pre-registered p <= 0.05 (24.8):")
    print(f"    {nameA}: {int(sA.sum())} of {len(sA)}    "
          f"{nameB}: {int(sB.sum())} of {len(sB)}    "
          f"flipped: {int((sA != sB).sum())}")
    print(f"\n  READ IT AS: a POSITIVE mean delta means the {nameB} null is "
          f"HARDER to beat\n  and tonight's run on {nameA} would overstate the "
          f"survivor count.  A negative\n  mean means the opposite.  Either way "
          f"this is the number, not an argument.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python floorcmp.py <dirA> <dirB>")
    main(sys.argv[1], sys.argv[2],
         os.path.basename(sys.argv[1].rstrip("\\/")).replace("floors_cmp_", "") or "A",
         os.path.basename(sys.argv[2].rstrip("\\/")).replace("floors_cmp_", "") or "B")
