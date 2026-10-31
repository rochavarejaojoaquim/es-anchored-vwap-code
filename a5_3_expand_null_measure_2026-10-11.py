"""If the stored nulls were TOO NARROW by a factor 1/f (the 'liberal test' world the removal door exists for),
the repaired null is the stored null EXPANDED about its mean by g = 1/f. How many of the 31 survive then?
(shrink_null(NUL, g) with g > 1 expands.) Same instrument as shrink_only_measure.py; read-only."""
import os, sys
import numpy as np, pandas as pd
ROOT = r"."
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code"))
import calib_main as C
studentize, romano_wolf = C.rw_funcs()
ref = pd.read_csv(os.path.join(ROOT, "rw_check_independent_out.csv"))
keys = [(c, C.side_str(s)) for c, s in zip(ref.cell, ref.side)]
SF = {kk: C.stored_floor(*kk) for kk in keys}
NUL = np.vstack([SF[kk]["null"] for kk in keys]); obs = np.array([SF[kk]["obs"] for kk in keys])
n0, _, adj0 = C.fwer_any(obs, NUL, studentize, romano_wolf)
surv = adj0 <= 0.05
assert n0 == 31
print("g = expansion of the stored null's spread about its mean; stored nulls are 'too narrow by 1/g'")
print(f"{'g':>5} {'narrower by':>12} {'survivors of 317':>17} {'of the 31 still':>16} {'adj p max among 31':>19}")
for g in (1.0, 1.1, 1.25, 1.4, 1.5, 1.67, 2.0, 2.5):
    NG = C.shrink_null(NUL, g)
    n, mn, adj = C.fwer_any(obs, NG, studentize, romano_wolf)
    still = int(((adj <= 0.05) & surv).sum())
    print(f"{g:5.2f} {100*(1-1/g):10.0f}% {n:17d} {still:13d} of 31 {adj[surv].max():19.4f}")
