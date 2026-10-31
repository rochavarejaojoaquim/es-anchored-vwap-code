"""
ROMANO-WOLF STEPDOWN for the anchored VWAP cell family.

WHAT IT IS.  An established multiple-testing procedure (Romano & Wolf 2005,
"Stepwise Multiple Testing as Formalized Data Snooping"), the stepwise
successor to White's Reality Check -- which the existing per-cell floor
already is, applied one cell at a time.  It controls the FAMILY-WISE error
rate: the probability of even one false rejection across the whole family.

WHY IT BEATS BONFERRONI HERE.  Bonferroni across 317 tests demands
p < 0.05/317 = 0.00016.  With 100 permutations the smallest attainable p is
1/101 = 0.0099, so Bonferroni can never reject anything at this permutation
count.  Romano-Wolf does not divide by 317; it uses the joint null of the
whole family, so its adjusted p can reach 1/101 and still control FWER.
That is the only route to a verdict at nperm 100.

DOWNSIDES, stated on first use:
  1. It needs the nulls drawn JOINTLY -- the same day relabelling in every
     cell for permutation k.  With per-cell seeds it still controls FWER but
     throws away the dependence and loses power.  (G5 measures this.)
  2. The max across hypotheses is only meaningful if the statistics are on a
     comparable scale.  MEASURED on the real file: the per-cell null SD spans
     0.0016 to 0.0096, a 5.9x range, so a RAW maxT would be dominated by the
     high-variance cells.  This module therefore studentizes each cell by its
     own null moments, leave-one-out.  Needs nothing beyond null_scores, so
     no extra columns and no re-run.
  3. It controls FWER, which is strict.  With 317 hypotheses and one weak
     real effect it may reject nothing.  That is the price of a verdict that
     survives review, not a fault.
  4. It says nothing about the winner's curse and does not replace the
     sealed 2018-2022 holdout.

Run:  python rw_stepdown.py          gates, then the (VOID) plumbing run
      python rw_stepdown.py mutate   break each target; the gate must FAIL
"""
import json
import sys

import numpy as np
import pandas as pd

ALPHA = 0.05


# --------------------------------------------------------------- the method
def studentize(T, N, loo=True):
    """Put every hypothesis on a common scale using its OWN null moments.

    loo=True leaves draw b out of the moments used to standardize draw b, so
    a draw does not help set its own scale.  With B=100 the difference is
    about 1%, but it is free.
    """
    B = N.shape[0]
    mu, sd = N.mean(0), N.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)          # a constant null cannot be beaten
    Tz = (T - mu) / sd
    if not loo:
        return Tz, (N - mu) / sd
    tot, tot2 = N.sum(0), (N ** 2).sum(0)
    mu_b = (tot - N) / (B - 1)
    var_b = (tot2 - N ** 2 - (B - 1) * mu_b ** 2) / (B - 2)
    sd_b = np.sqrt(np.maximum(var_b, 0))
    sd_b = np.where(sd_b > 0, sd_b, np.inf)
    return Tz, (N - mu_b) / sd_b


def rw_stepdown(T, N, stepdown=True, monotone=True):
    """Adjusted p-values controlling FWER.

    T : (S,)   observed statistic, larger = stronger
    N : (B,S)  null draws.  Row b must be the SAME relabelling in every column.
    """
    S, B = len(T), N.shape[0]
    order = np.argsort(-T, kind="stable")
    active = np.ones(S, bool)
    padj = np.empty(S)
    prev = 0.0
    for s in order:
        M = N[:, active].max(axis=1) if stepdown else N.max(axis=1)
        p = (1 + int((M >= T[s]).sum())) / (1 + B)
        prev = max(prev, p) if monotone else p
        padj[s] = prev
        active[s] = False
    return padj


def bonferroni(T, N):
    B, S = N.shape
    raw = np.array([(1 + int((N[:, s] >= T[s]).sum())) / (1 + B) for s in range(S)])
    return np.minimum(raw * S, 1.0)


# ----------------------------------------------------------------- fixtures
def null_world(reps, S, B, rho, rng, effect=0.0, n_effect=0):
    """reps families of S exchangeable hypotheses, correlation rho across
    hypotheses, drawn jointly.  One draw becomes the observation, B the null."""
    f = rng.standard_normal((reps, B + 1, 1))
    e = rng.standard_normal((reps, B + 1, S))
    X = np.sqrt(rho) * f + np.sqrt(1 - rho) * e
    if n_effect:
        X[:, 0, :n_effect] += effect          # planted only in the OBSERVED draw
    return X[:, 0, :], X[:, 1:, :]


# -------------------------------------------------------------------- gates
def gates(quiet=False, stepdown=True, monotone=True, break_joint=False, g5_rho=0.7):
    res = {}

    def say(tag, ok, msg):
        # bool(): a numpy bool_ is not the Python singleton, so the mutation
        # harness's `is True` test silently read every numpy-valued gate as
        # failed.  Coerce once, here, rather than at each comparison site.
        ok = bool(ok)
        res[tag] = ok
        if not quiet:
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag:<4} {msg}")

    rng = np.random.default_rng(11)

    # G1  KNOWN ANSWER: one hypothesis -> the ordinary permutation p-value
    T, N = null_world(300, 1, 100, 0.0, rng)
    pa = np.array([rw_stepdown(T[i], N[i], stepdown, monotone)[0] for i in range(300)])
    pp = np.array([(1 + int((N[i][:, 0] >= T[i][0]).sum())) / 101 for i in range(300)])
    say("G1", np.allclose(pa, pp),
        "S=1 reduces exactly to the permutation p-value (1+#null>=obs)/(1+B)")

    # G2  KNOWN ANSWER: FWER must sit at alpha when everything is null
    for rho, tgt in ((0.0, "independent"), (0.6, "correlated")):
        T, N = null_world(4000, 50, 100, rho, rng)
        fw, fb, fu = [], [], []
        for i in range(4000):
            Tz, Nz = studentize(T[i], N[i])
            if break_joint:                       # mutation: destroy the pairing
                Nz = np.take_along_axis(
                    Nz, rng.permuted(np.tile(np.arange(100)[:, None], (1, 50)), axis=0), 0)
            fw.append(rw_stepdown(Tz, Nz, stepdown, monotone).min() <= ALPHA)
            fb.append(bonferroni(Tz, Nz).min() <= ALPHA)
            fu.append(((1 + (Nz >= Tz).sum(0)) / 101).min() <= ALPHA)
        say(f"G2{'i' if rho == 0 else 'c'}",
            abs(np.mean(fw) - ALPHA) < 0.012,
            f"FWER, all-null {tgt} family (S=50, target 5.0%): "
            f"Romano-Wolf {np.mean(fw):5.1%} | Bonferroni {np.mean(fb):5.1%} "
            f"| uncorrected {np.mean(fu):5.1%}")

    # G3  adjusted p-values are non-decreasing down the ordering
    T, N = null_world(200, 40, 100, 0.4, rng)
    ok = True
    for i in range(200):
        Tz, Nz = studentize(T[i], N[i])
        pa = rw_stepdown(Tz, Nz, stepdown, monotone)
        ok &= bool(np.all(np.diff(pa[np.argsort(-Tz, kind="stable")]) >= -1e-12))
    say("G3", ok, "adjusted p is monotone non-decreasing down the ranking")

    # G4  POWER: plant one real effect; RW must find it, and beat Bonferroni
    T, N = null_world(2000, 50, 100, 0.5, rng, effect=4.0, n_effect=1)
    hw = hb = 0
    for i in range(2000):
        Tz, Nz = studentize(T[i], N[i])
        hw += int(rw_stepdown(Tz, Nz, stepdown, monotone)[0] <= ALPHA)
        hb += int(bonferroni(Tz, Nz)[0] <= ALPHA)
    say("G4", hw / 2000 > hb / 2000,
        f"power on a planted +4sd effect: Romano-Wolf {hw/2000:.1%} vs "
        f"Bonferroni {hb/2000:.1%} (Bonferroni cannot reach 0.05 at B=100)")

    # G5  the joint seed earns its keep: with the SAME data, nulls drawn
    #     independently across cells lose power under positive dependence
    T, N = null_world(2000, 50, 100, g5_rho, rng, effect=3.5, n_effect=1)
    hj = hi = 0
    for i in range(2000):
        Tz, Nz = studentize(T[i], N[i])
        hj += int(rw_stepdown(Tz, Nz, stepdown, monotone)[0] <= ALPHA)
        Ni = np.take_along_axis(Nz, rng.permuted(
            np.tile(np.arange(100)[:, None], (1, 50)), axis=0), 0)
        hi += int(rw_stepdown(Tz, Ni, stepdown, monotone)[0] <= ALPHA)
    # margin, not a bare '>': with rho=0 there is nothing for the joint draw
    # to exploit and the two must come out level, which is the mutation.
    say("G5", hj / 2000 > hi / 2000 + 0.03,
        f"joint nulls beat per-cell nulls under dependence (rho={g5_rho}): "
        f"{hj/2000:.1%} vs {hi/2000:.1%} power, margin {100*(hj-hi)/2000:+.1f}pp")

    return res


# ------------------------------------------------------------------ the run
def plumbing_run(path):
    R = pd.read_csv(path)
    eff = R[~R.dup_session.astype(bool)].reset_index(drop=True)
    T = eff.obs_score.to_numpy()
    N = np.array([json.loads(s) for s in eff.null_scores])  # (S,B)
    N = N.T                                                 # -> (B,S)
    print(f"  loaded {N.shape[1]} hypotheses x {N.shape[0]} permutations")
    Tz, Nz = studentize(T, N)
    pa = rw_stepdown(Tz, Nz)
    eff["rw_p"] = pa
    print(f"  adjusted p: min {pa.min():.4f}  |  <= 0.05: {int((pa <= ALPHA).sum())}"
          f"  |  at the 1/101 floor: {int((pa <= 1/101 + 1e-12).sum())}")
    print(f"\n  top 8 by studentized statistic:")
    e2 = eff.assign(z=Tz).sort_values("z", ascending=False).head(8)
    print("   " + e2[["cell", "side", "obs_score", "p", "z", "rw_p"]]
          .round(4).to_string(index=False).replace("\n", "\n   "))
    return eff


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mutate":
        print("MUTATION TEST -- break the target, the named gate must FAIL\n")
        clean = gates(quiet=True)
        muts = {
            "G3": ("drop the monotonicity clamp", dict(monotone=False)),
            "G5": ("remove the dependence the joint seed exploits", dict(g5_rho=0.0)),
            "G2c": ("destroy the joint pairing of the nulls", dict(break_joint=True)),
        }
        allok = True
        for tag, (desc, kw) in muts.items():
            now = gates(quiet=True, **kw).get(tag)
            good = clean.get(tag) is True and now is False
            allok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {tag:<4} {desc:<40} "
                  f"clean={clean.get(tag)}  mutated={now}")
        # single-step vs stepdown: stepdown must never be MORE conservative
        rng = np.random.default_rng(5)
        T, N = null_world(400, 40, 100, 0.5, rng, effect=3.0, n_effect=2)
        worse = 0
        for i in range(400):
            Tz, Nz = studentize(T[i], N[i])
            worse += int((rw_stepdown(Tz, Nz, True) >
                          rw_stepdown(Tz, Nz, False) + 1e-12).any())
        print(f"  [{'PASS' if worse == 0 else 'FAIL'}] SD   stepdown is never more "
              f"conservative than single-step ({worse} of 400 violations)")
        allok &= (worse == 0)
        print(f"\n  {'ALL GATES ALIVE' if allok else 'A GATE IS DEAD'}")
        sys.exit(0 if allok else 1)

    print("ROMANO-WOLF STEPDOWN -- gates on synthetic data with known answers\n")
    r = gates()
    print(f"\n  {sum(r.values())} of {len(r)} gates pass")
    path = sys.argv[1] if len(sys.argv) > 1 else "results_table.csv"
    S_ = pd.read_csv(path, usecols=["dup_session", "null_unit", "joint", "nperm"])
    nu = sorted(set(S_.null_unit.astype(str)))
    jt = sorted(set(S_.joint.astype(str)))
    nperms = sorted(set(S_.nperm.astype(int)))
    n_eff = int((~S_.dup_session.astype(bool)).sum())
    print("=" * 74)
    print(f"STEPDOWN on {path}")
    print(f"  stamp: {n_eff} effective / {len(S_)} rows | null_unit {nu} | "
          f"joint {jt} | nperm {nperms} | ES only")
    if len(nu) > 1 or len(jt) > 1 or len(nperms) > 1:
        raise SystemExit("  REFUSED: the file mixes conventions (A2).")
    if nu != ["day_strata"] or jt != ["True"]:
        print("  THE NUMBERS BELOW ARE VOID: not the day_strata joint null.")
        print("  This exercises the code path only.")
    else:
        print("  Validated null (day_strata, joint).  These numbers are a result.")
    print("=" * 74)
    plumbing_run(path)
