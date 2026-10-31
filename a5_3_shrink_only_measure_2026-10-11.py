"""Outcome-blind measurements for the A5(3) shrink-only rule (reviewer O2). Reads the frozen
calibration CSV, the stored floors (floors_v2), rw_check_independent.py. Writes nothing in the project.
M1  can a calibration of this design act on an INDIVIDUAL cell? per-cell k<=4 counts over the 29 SEP
    replicates against a binomial, and the dispersion of per-cell mean k.
M2  how much must the stored nulls be shrunk before this instrument says 'upward', and before any of
    the 31 survivors flips? (calib_main.shrink_null(NUL, f))
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy import stats

ROOT = r"."
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))
import calib_main as C

t0 = time.time()
t = C.read_parts([os.path.join(ROOT, "calib_main_arm1_2026-10-05.csv")])
t = t[t.status == "ok"].copy()
t["j"] = t.j.astype(int); t["k"] = t.k.astype(int)
t["fam"] = t.cell.map(C.family)
sep = t[t.j < C.SEP].copy()
cells = sorted(set(zip(sep.cell, sep.side)))
print(f"frames {len(t):,}  SEP frames {len(sep):,}  cell+sides {len(cells)}  (known: 31,700 / 9,193 / 317)")
assert len(t) == 31700 and len(sep) == 9193 and len(cells) == 317

# ---------------- M1
sep["lo"] = (sep.k <= 4).astype(int)
g = sep.groupby(["cell", "side"]).agg(x=("lo", "sum"), n=("lo", "size"), mk=("k", "mean"))
p = sep.lo.mean()
print(f"\nM1  pooled k<=4 rate over SEP {100*p:.4f}% ; per-cell counts out of 29:")
print("    observed distribution of X:", dict(sorted(g.x.value_counts().items())))
exp = {x: 317 * stats.binom.pmf(x, 29, p) for x in range(0, 9)}
print("    expected under one common rate (binomial):", {x: round(v, 1) for x, v in exp.items()})
var_obs, var_exp = g.x.var(ddof=1), 29 * p * (1 - p)
print(f"    var(X) observed {var_obs:.3f} vs binomial {var_exp:.3f} ; ratio {var_obs/var_exp:.2f} "
      f"(ratio > 1 = cell-to-cell heterogeneity beyond chance; < 1 = under-dispersed)")
chi = float(((g.x - 29 * p) ** 2 / (29 * p * (1 - p))).sum())
print(f"    dispersion chi2 {chi:.1f} on {len(g)-1} df, upper p {stats.chi2.sf(chi, len(g)-1):.4f}")
big = int((g.x >= 5).sum())
print(f"    cells with >= 5 of 29 frames at k<=4: {big}  (expected under common rate {317*stats.binom.sf(4, 29, p):.1f})")
fk = sep.groupby(["cell", "side"]).k.var(ddof=1).mean()
print(f"    per-cell mean k: observed var across cells {g.mk.var(ddof=1):.2f} vs {fk/29:.2f} expected from within-cell frame variance (ratio {g.mk.var(ddof=1)/(fk/29):.2f})")
pass

# ---------------- M2
studentize, romano_wolf = C.rw_funcs()
ref = pd.read_csv(os.path.join(ROOT, "rw_check_independent_out.csv"))
keys = [(c, C.side_str(s)) for c, s in zip(ref.cell, ref.side)]
SF = {kk: C.stored_floor(*kk) for kk in keys}
NUL = np.vstack([SF[kk]["null"] for kk in keys])
obs = np.array([SF[kk]["obs"] for kk in keys])
n0, mn0, adj0 = C.fwer_any(obs, NUL, studentize, romano_wolf)
assert n0 == 31
surv = np.array([a <= 0.05 for a in adj0])
print(f"\nM2  known number: real obs vs stored nulls -> {n0} survivors (PASS = 31); {time.time()-t0:.0f}s so far")
idx = {kk: i for i, kk in enumerate(keys)}
sep["ix"] = [idx[(c, s)] for c, s in zip(sep.cell, sep.side)]
sep["is31"] = surv[sep.ix.to_numpy()]
D = sep.pivot_table(index="ix", columns="j", values="score", aggfunc="first").reindex(range(len(keys)))
print(f"    {'f':>5} {'k<=4 %':>8} {'k==0 %':>8} {'CI(k<=4) %':>16} {'mean p':>7}  {'survivors of 31':>16} {'armF hits/29':>13}")
for f in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
    NF = C.shrink_null(NUL, f)
    k = np.array([C.k_count(s, NF[i]) for s, i in zip(sep.score, sep.ix)])
    e, lo, hi = C.pigeonhole_ci((k <= 4).astype(float), sep.fam, sep.j, nboot=1000)
    nF, _, adjF = C.fwer_any(obs, NF, studentize, romano_wolf)
    still = int(((adjF <= 0.05) & surv).sum())
    hitsF = 0
    for j in range(C.SEP):
        col = D[j].to_numpy(float)
        if np.isnan(col).any():
            continue
        if C.fwer_any(col, NF, studentize, romano_wolf)[0] > 0:
            hitsF += 1
    print(f"    {f:5.2f} {100*e:8.3f} {100*(k==0).mean():8.3f} [{100*lo:5.2f},{100*hi:5.2f}]    {((1+k)/101).mean():7.4f}  {still:>10d} of 31 {hitsF:>9d}/29")
print(f"\nnominal k<=4 {100*C.NOMINAL5:.3f}%  k==0 {100*C.NOMINAL0:.3f}%  mean p {C.NOMINAL_P:.4f}; total {time.time()-t0:.0f}s")
print("per-cell M1 for the 31 survivors only:")
g31 = sep[sep.is31].groupby(["cell", "side"]).lo.sum()
print("    X among the 31:", dict(sorted(g31.value_counts().items())), "n cells", len(g31))
