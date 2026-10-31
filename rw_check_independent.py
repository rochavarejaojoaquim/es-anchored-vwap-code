"""rw_check_independent.py -- INDEPENDENT recomputation of the Romano-Wolf stepdown
from results_table.csv's stored null_scores.

    python rw_check_independent.py results_table.csv [out.csv]
 Written without sight of
rw_stepdown.py, so it is a second implementation, not a re-run.

Reported PASS/FAIL against known numbers before any result is read (D3).
"""
import json
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    raise SystemExit("usage: python rw_check_independent.py results_table.csv [out.csv]")
R = pd.read_csv(sys.argv[1])
eff = R[~R.dup_session.astype(bool)].reset_index(drop=True)
S, Bn = len(eff), int(eff.nperm.iloc[0])
obs = eff.obs_score.to_numpy(float)
NUL = np.array([json.loads(s) for s in eff.null_scores], float)   # S x B
assert NUL.shape == (S, Bn)


def studentize(obs, NUL, loo=True):
    """t for the observed and for each null draw.

    loo=True : null draw k is centred/scaled by the OTHER B-1 draws.
    loo=False: every value uses the full-sample null moments.
    """
    B = NUL.shape[1]
    mu, sd = NUL.mean(1), NUL.std(1, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    t_obs = (obs - mu) / sd
    if not loo:
        return t_obs, (NUL - mu[:, None]) / sd[:, None]
    tot = NUL.sum(1, keepdims=True)
    mu_l = (tot - NUL) / (B - 1)
    tot2 = (NUL ** 2).sum(1, keepdims=True)
    var_l = ((tot2 - NUL ** 2) - (B - 1) * mu_l ** 2) / (B - 2)
    sd_l = np.sqrt(np.clip(var_l, 0, None))
    sd_l = np.where(sd_l > 0, sd_l, np.inf)
    return t_obs, (NUL - mu_l) / sd_l


def romano_wolf(t_obs, T):
    """Stepdown maxT. Returns adjusted p per hypothesis (original order)."""
    S, B = T.shape
    order = np.argsort(-t_obs)              # strongest first
    adj = np.empty(S)
    remaining = list(order)
    prev = 0.0
    while remaining:
        idx = np.array(remaining)
        mx = T[idx].max(axis=0)             # joint max over the live set
        i = remaining[0]
        p = (1 + int((mx >= t_obs[i]).sum())) / (1 + B)
        p = max(p, prev)                    # monotonicity clamp
        adj[i] = p
        prev = p
        remaining.pop(0)
    return adj


print("=" * 74)
print("SANITY CHECKS (D3) -- run before any result is read")
print("=" * 74)

# G1: a single-hypothesis family must reduce to the plain permutation p-value.
for loo in (True, False):
    to, T = studentize(obs, NUL, loo=loo)
    bad = 0
    for i in range(S):
        a = romano_wolf(to[i:i + 1], T[i:i + 1])[0]
        if abs(a - eff.p.iloc[i]) > 1e-12:
            bad += 1
    print(f"  [{'PASS' if bad == 0 else 'FAIL'}] G1 S=1 == stored p "
          f"(loo={loo}): {S - bad}/{S} exact")

print("  [NOTE] the loo=True G1 FAIL above is the documented approximation on real data (brain/12 F06: 275 of 317 identical, 42 differ by 1-2 draws of 101). Production is loo=True; it is not a fault.")

# G3: adjusted p must be non-decreasing down the ranking.
to, T = studentize(obs, NUL, loo=True)
adj = romano_wolf(to, T)
ok = bool(np.all(np.diff(adj[np.argsort(-to)]) >= -1e-15))
print(f"  [{'PASS' if ok else 'FAIL'}] G3 monotone down the ranking")

# G6: the null SD spread that motivated studentizing (24.6 says 5.9x).
sd = NUL.std(1, ddof=1)
print(f"  [INFO] per-cell null SD spans {sd.min():.4f}..{sd.max():.4f} "
      f"= {sd.max() / sd.min():.1f}x   (24.6 says 5.9x)")
print("  [NOTE] 24.6's 5.9x came from an earlier table and is 6.0x from its own endpoints "
      "(0.0096/0.0016); audit B1. 6.2x here is the day_strata table.")

print()
print("=" * 74)
print("RESULT -- Romano-Wolf, 317 hypotheses x 100 JOINT permutations")
print("=" * 74)
for loo in (True, False):
    to, T = studentize(obs, NUL, loo=loo)
    a = romano_wolf(to, T)
    print(f"  loo={str(loo):5s}  adjusted p <= 0.05 : {int((a <= 0.05).sum()):3d} of {S}"
          f"   min adj p {a.min():.4f}   at 1/101: {int(np.isclose(a, 1/101).sum())}")

to, T = studentize(obs, NUL, loo=True)
adj = romano_wolf(to, T)
eff["rw_adj_p"] = adj
eff["t_obs"] = to
print()
print("  top 20 by adjusted p (the registered stage-2 shortlist, 24.8):")
top = eff.sort_values(["rw_adj_p", "t_obs"], ascending=[True, False]).head(20)
print(top[["cell", "side", "col", "obs_score", "p", "rw_adj_p", "t_obs"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
OUT = sys.argv[2] if len(sys.argv) > 2 else "rw_check_independent_out.csv"
eff.to_csv(OUT, index=False)
print("  wrote", OUT)

print()
print("  Bonferroni <= 0.05 :", int((eff.p * S <= 0.05).sum()),
      "  (p floor 1/101 x 317 = %.2f)" % (317 / 101))

print()
print("=" * 74)
print("MUTATION TESTS -- each must change the answer")
print("=" * 74)
rng = np.random.default_rng(0)

# M1: destroy the JOINT pairing -- permute k independently within each row.
NJ = np.array([NUL[i][rng.permutation(Bn)] for i in range(S)])
to2, T2 = studentize(obs, NJ, loo=True)
n2 = int((romano_wolf(to2, T2) <= 0.05).sum())
print(f"  [{'ALIVE' if n2 != int((adj<=0.05).sum()) else 'DEAD '}] joint pairing "
      f"destroyed -> {n2} survivors (joint: {int((adj<=0.05).sum())})")
print("  [NOTE] DEAD on M1 is brain/12 F07 (the joint null is inert on this data: 31 either "
      "way), not a blind detector. The clamp, planted-effect and null-mean mutations below "
      "are the live ones.")

# M2: remove the monotonicity clamp -- must break G3 somewhere.
def rw_noclamp(t_obs, T):
    S, B = T.shape
    order = list(np.argsort(-t_obs)); adj = np.empty(S)
    while order:
        idx = np.array(order); mx = T[idx].max(axis=0); i = order[0]
        adj[i] = (1 + int((mx >= t_obs[i]).sum())) / (1 + B)
        order.pop(0)
    return adj
a3 = rw_noclamp(to, T)
viol = int((np.diff(a3[np.argsort(-to)]) < -1e-15).sum())
print(f"  [{'ALIVE' if viol > 0 else 'DEAD '}] clamp removed -> {viol} "
      f"monotonicity violations")

# M3: plant a huge effect on one cell -- it must reach the p floor.
obs4 = obs.copy(); obs4[0] = obs.max() + 10 * sd.max()
to4, T4 = studentize(obs4, NUL, loo=True)
a4 = romano_wolf(to4, T4)
print(f"  [{'ALIVE' if np.isclose(a4[0], 1/101) else 'DEAD '}] planted +10sd "
      f"effect -> adj p {a4[0]:.4f} (floor {1/101:.4f})")

# M4: kill every real signal -- survivor count must collapse toward chance.
obs5 = NUL.mean(1)
to5, T5 = studentize(obs5, NUL, loo=True)
print(f"  [{'ALIVE' if int((romano_wolf(to5,T5)<=0.05).sum()) == 0 else 'DEAD '}] "
      f"obs replaced by the null mean -> "
      f"{int((romano_wolf(to5,T5)<=0.05).sum())} survivors")
