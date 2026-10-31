r"""a10_power_preview.py -- the BASE-bracket joint-power preview of section 28, amendment A10.  (version 2)

REGISTERED TEXT (HANDOFF_section_28.md, A10): "The BASE-bracket joint power of the 31 under (c) and (a) at
K = 1 and K = 5 is computed BEFORE S3-S5 are registered, on stage-1-defined BASE results, and reported to
The author." "The S2 go/no-go preview decides ONLY whether to open the seal for strategies; it may not change
any grid, dial or K." The algorithm is 28.6 "POWER AND NOT TESTED" with A3's SE rule. NO dial setting is read.

WHAT IT COMPUTES, per Stage-1 Romano-Wolf survivor (31), on its DISCOVERED subset (stored column, cut, sense):
  direction  FADE if the stored fill-rate edge is positive, CONTINUATION if negative (28.6, S0).
  BASE bracket (28.6 S1): FADE target = the frozen VWAP level, stop = its mirror; CONTINUATION the reverse;
             an unresolved trade is marked at the close of bar T. Outcome in ATR0 units (E13):
             FADE: race 1 -> +dist_atr ; race -1 -> -dist_atr ; race 9 -> dist_atr - side*gap_at_T ;
             CONTINUATION = the negative of those; a same-bar TIE (race 0) is STOP FIRST = -dist_atr in BOTH
             directions (E4). cens_T rows excluded (E8). E6 (flat 16:10), E7 (news) OFF; E11 (minimum bracket)
             OFF in the registered columns and ON in one labelled sensitivity column.
  three arms (28.4): gross; M1b-differenced; M2-differenced (setup outcome minus the mean of its usable twins
             of that scheme, traded in the same direction, E12).
  per train block b: arm statistics as ratios of day sums; SE_b,k = max(SE_iid, SE_block5) from 2,000 resamples
             of the block's CALENDAR trade dates (universe from N60_hh_SD_1.0: 1,941 in A, 1,188 in B; iid
             default_rng(20261005); block core_v2.boot_index(D, 2000, 5, 1)); R_b from the iid draws;
             SeSeal = SE x sqrt(D_b / 1030).
  power_b    P(three correlated normals with means m_k / SeSeal_b,k all exceed norm.ppf(1 - 0.05/K)),
             100,000 draws, default_rng(20261007). JOINT POWER = min(power_A, power_B).
  planning effects m_k: (c) 0.75 x edge [ADOPTED, a convention]; (a) edge - 2.6 SE [rejected form]; 0.5 x edge
             (A2); the point estimate. K = 1 and K = 5 as registered; K = 18 (the dedupe groups of the 31) as an
             extra labelled column.
READING, FIXED BEFORE THE RUN: DESCRIPTIVE. NO PRINTED NUMBER TRIGGERS ANYTHING; whether to go on to strategy
work is the author's decision; nothing here may change a grid, dial or K, or pick or drop a core. The edges are
the in-sample, selection-inflated ones; the 31 are not the gate passers and the removal loop is not run; power
in ATR units is not power on the seal's tick condition; E6/E7 would remove about 17% of trades (26.88).

KNOWN NUMBERS, all before any power is printed: the rw CSV hash; 31 survivors = 16 FADE + 15 CONTINUATION;
sign(eA) == sign(eB) on all 31; each rebuilt subset reproduces the stored nA, nB; every open trade (setups and
both twin schemes) lies between the two levels; the calendar universe is 1,941 / 1,188 dates; joint_power
reproduces 26.85's single-arm 81.6%. COMPARISONS printed (tolerances stated, not gates): the race edges of the
16 positive cores against 26.91 (median +8.30 pp A, +4.88 B) and of the 15 negative ones (-0.37, +0.12).

    python a10_power_preview.py --selftest
    python a10_power_preview.py data\dump        (about 1.5-2 min, single process, MEASURED 3 s per core on 2)
"""
import os, sys, time, hashlib, platform, warnings
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import core_v2                                              # noqa: E402
import mono_profile as MP                                   # noqa: E402

D_REG = {"A": 1941, "B": 1188}; D_SEAL = 1030; NRES = 2000; NMC = 100000
RW_SHA = "44E9D780FE0C8C6C1C2812340887CFDBF41DDB618A3BE4F99C0D4BC47DF4B0B0"
NEED = ["is_real", "side", "bar", "src_bar", "day", "block", "dist_atr", "race", "gap_at_T", "cens_T", "atr0"]
FORMS = (("c", lambda e, s: 0.75 * e), ("a", lambda e, s: e - 2.6 * s), ("half", lambda e, s: 0.5 * e), ("pt", lambda e, s: e))


def fade_outcome(race, dist, side, gap):
    race = np.asarray(race); dist = np.asarray(dist, float); sg = np.sign(np.asarray(side, float)); gap = np.asarray(gap, float)
    return np.where(race == 1, dist, np.where((race == -1) | (race == 0), -dist, dist - sg * gap))


def base_outcome(race, dist, side, gap, direction, tie_negated=False):
    """BASE outcome in the traded direction; a tie is a LOSS in both directions (E4). tie_negated=True is the
    superseded rule, kept only for the self-test mutation and the printed sensitivity."""
    fo = fade_outcome(race, dist, side, gap)
    if tie_negated:
        return direction * fo
    return np.where(np.asarray(race) == 0, -np.asarray(dist, float), direction * fo)


def arm_table(s, day, t1, t2, universe):
    """Per CALENDAR day sums [sum s, n, sum(s-t1), n1, sum(s-t2), n2] on the block's date universe."""
    codes = np.searchsorted(universe, day); D = len(universe)
    assert (universe[codes] == day).all(), "a trade date outside the calendar universe"
    ok = np.isfinite(s); h1 = ok & np.isfinite(t1); h2 = ok & np.isfinite(t2)
    tab = np.zeros((D, 6)); one = np.ones(len(s))
    for j, (val, m) in enumerate(((s, ok), (one, ok), (s - t1, h1), (one, h1), (s - t2, h2), (one, h2))):
        tab[:, j] = np.bincount(codes[m], np.asarray(val, float)[m], D)
    return tab


def arm_stats(T):
    D = len(T)
    f = lambda S: np.column_stack([S[:, 0] / S[:, 1], S[:, 2] / S[:, 3], S[:, 4] / S[:, 5]])
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = f(T.sum(0)[None, :])[0]
        idx = np.random.default_rng(20261005).integers(0, D, size=(NRES, D))
        bi = f(T[idx].sum(1)); bb = f(T[core_v2.boot_index(D, NRES, 5, 1)].sum(1))
        se_i, se_b = np.nanstd(bi, 0), np.nanstd(bb, 0)
        full = bi[np.isfinite(bi).all(1)]
        Rm = np.corrcoef(full.T) if len(full) > 10 else None
    return est, np.fmax(se_i, se_b), Rm, se_i, se_b, T.sum(0)[[1, 3, 5]].astype(int)


def joint_power(m, se_seal, R, K):
    z = norm.ppf(1 - 0.05 / K)
    L = np.linalg.cholesky(R + 1e-9 * np.eye(3))
    X = np.random.default_rng(20261007).standard_normal((NMC, 3)) @ L.T + (m / se_seal)[None, :]
    return float((X > z).all(1).mean())


def powers(est, se, R, block):
    """NaN (an undefined arm or correlation) propagates as NaN, never as power 0."""
    out = {}
    se_seal = se * np.sqrt(D_REG[block] / D_SEAL)
    for form, fn in FORMS:
        for K in (1, 5, 18):
            m = fn(est, se)
            out[f"{form}_K{K}"] = joint_power(m, se_seal, R, K) if (R is not None and np.isfinite(m / se_seal).all()) else np.nan
    return out


# ------------------------------------------------------------------ self-test
def selftest():
    ok = True
    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    print("SELF-TEST (toy fixtures, no data)")
    race, dist, side, gap = [1, -1, 0, 9, 9], [3, 3, 3, 3, 3], [1, 1, -1, 1, -1], [0, 0, 0, 1.0, -2.0]
    o = base_outcome(race, dist, side, gap, 1.0)
    chk("K1 FADE: win +dist, loss and tie -dist, open = dist - side*gap", np.allclose(o, [3, -3, -3, 2.0, 1.0]), str(o.tolist()))
    oc = base_outcome([1, -1, 0, 9], [3, 3, 3, 3], [1, 1, 1, 1], [0, 0, 0, 1.0], -1.0)
    chk("K1b CONTINUATION: the negative, EXCEPT a tie, which is a loss (stop first, E4)", np.allclose(oc, [-3, 3, -3, -2.0]), str(oc.tolist()))
    om = base_outcome([1, -1, 0, 9], [3, 3, 3, 3], [1, 1, 1, 1], [0, 0, 0, 1.0], -1.0, tie_negated=True)
    chk("M0 MUTATION the superseded rule (tie negated) fails K1b", not np.allclose(om, [-3, 3, -3, -2.0]), str(om.tolist()))
    chk("M0b MUTATION a flipped gap sign puts an open trade outside the levels", abs(fade_outcome([9], [3], [1], [-4.0])[0]) > 3)
    rng = np.random.default_rng(3); nd, per = 1200, 6; n = nd * per; day = np.repeat(np.arange(nd), per); uni = np.arange(nd)
    def world(mu_s, mu_t, sd=3.0, dayfx=0.0):
        e = np.repeat(rng.standard_normal(nd), per) * dayfx
        s = mu_s + sd * rng.standard_normal(n) + e; t1 = mu_t + sd * rng.standard_normal(n) / np.sqrt(2) + e; t2 = mu_t + sd * rng.standard_normal(n) / np.sqrt(2) + e
        return arm_stats(arm_table(s, day, t1, t2, uni))
    est, se, R, se_i, se_b, nn = world(0.30, 0.0)
    an = 3.0 / np.sqrt(n)
    chk("K2 gross edge and SE recover a planted +0.30 with SD 3", abs(est[0] - 0.30) < 3 * an and abs(se_i[0] / an - 1) < 0.1, f"edge {est[0]:+.3f}, SE_iid {se_i[0]:.4f} vs {an:.4f}")
    chk("K3 a differenced arm is noisier by about sqrt(1.5) with independent twins", abs(se_i[1] / se_i[0] - np.sqrt(1.5)) < 0.12, f"{se_i[1]/se_i[0]:.3f} vs {np.sqrt(1.5):.3f}")
    chk("K4 correlation gross vs differenced about 1/sqrt(1.5)", abs(R[0, 1] - 1 / np.sqrt(1.5)) < 0.06, f"{R[0,1]:.3f} vs {1/np.sqrt(1.5):.3f}")
    # zero days do not change a ratio estimator's SE: half the calendar empty
    s = 0.3 + 3.0 * rng.standard_normal(n); nanv = np.full(n, np.nan)
    e1 = arm_stats(arm_table(s, day, nanv, nanv, uni))[3][0]; e2 = arm_stats(arm_table(s, day * 2, nanv, nanv, np.arange(2 * nd)))[3][0]
    chk("K4b empty calendar days leave the iid SE unchanged (within 5%)", abs(e2 / e1 - 1) < 0.05, f"{e1:.4f} vs {e2:.4f}")
    p1 = joint_power(np.array([2.8, 2.8, 2.8]), np.ones(3), np.ones((3, 3)) - 1e-12 * (1 - np.eye(3)), 1)
    chk("K5 three identical arms: joint power == one-arm power", abs(p1 - (1 - norm.cdf(norm.ppf(0.95) - 2.8))) < 0.01, f"{p1:.3f}")
    p3 = joint_power(np.array([2.8, 2.8, 2.8]), np.ones(3), np.eye(3), 1)
    chk("K6 three independent arms: joint power == the product", abs(p3 - (1 - norm.cdf(norm.ppf(0.95) - 2.8)) ** 3) < 0.01, f"{p3:.3f}")
    pk = joint_power(np.array([1.0, 1.0, 1.0]), np.full(3, 0.31), np.ones((3, 3)) - 1e-12 * (1 - np.eye(3)), 5)
    chk("K6b KNOWN NUMBER 26.85: one arm, edge 1.0, SE 0.31, K = 5 -> 81.6%", abs(pk - 0.816) < 0.005, f"{pk:.3f}")
    pw = powers(np.array([0.30, 0.30, 0.30]), np.array([0.05, 0.06, 0.06]), np.full((3, 3), 0.7) + 0.3 * np.eye(3), "B")
    chk("K7 ordering: point >= (c) >= half and (c) >= (a) here; K=1 >= K=5 >= K=18", pw["pt_K1"] >= pw["c_K1"] >= pw["half_K1"] and pw["c_K1"] >= pw["c_K5"] >= pw["c_K18"], str({k: round(v, 3) for k, v in pw.items() if k.startswith("c_") or k == "pt_K1"}))
    est, se, R, se_i, se_b, nn = world(0.0, 0.0, dayfx=1.5)
    chk("K8 a shared day effect cancels in the differenced arms", se[1] < se[0], f"gross {se[0]:.4f}, diff {se[1]:.4f}")
    chk("M1 one arm negative -> joint power about 0", powers(np.array([-0.2, 0.3, 0.3]), np.full(3, 0.05), np.eye(3), "A")["pt_K1"] < 0.001)
    chk("M1b an undefined arm gives NaN, not 0", np.isnan(powers(np.array([0.3, np.nan, 0.3]), np.full(3, 0.05), np.eye(3), "A")["c_K1"]))
    e2 = np.repeat(np.repeat(rng.standard_normal(nd // 10), 10), per)[:n] * 2.0
    st = arm_stats(arm_table(3.0 * rng.standard_normal(n) + e2, day, nanv, nanv, uni))
    chk("M2 under a 10-day regime the 5-day block SE exceeds the iid SE (the max rule works)", st[4][0] > 1.3 * st[3][0], f"block {st[4][0]:.4f} vs iid {st[3][0]:.4f}")
    print("ALL DETECTORS ALIVE" if ok else "SELF-TEST FAILED"); return ok


# ------------------------------------------------------------------ main
def twins_mean(tw, bars, direction):
    o = base_outcome(tw.race, tw.dist_atr, tw.side, tw.gap_at_T, direction)
    o = np.where(np.asarray(tw.cens_T, bool), np.nan, o)
    return pd.Series(o).groupby(np.asarray(tw.src_bar)).mean().reindex(bars).to_numpy()   # mean skips NaN


def open_between(x):
    fo = fade_outcome(x.race, x.dist_atr, x.side, x.gap_at_T); m = (np.asarray(x.race) == 9) & np.isfinite(fo)
    return bool((np.abs(fo[m]) <= np.asarray(x.dist_atr, float)[m] + 1e-9).all())


def race_edge(r, m, tw1, B):
    rb = r[m & (r.block == B).to_numpy()]; t = tw1[tw1.src_bar.isin(set(rb.bar))]
    a = rb.race[rb.race.isin([1, -1])]; b = t.race[t.race.isin([1, -1])]
    return 100 * float((a == 1).mean() - (b == 1).mean())


def main(dump, limit=None):
    if not selftest():
        return 1
    t0 = time.time()
    rwp = os.path.join(HERE, "rw_check_independent_out.csv")
    if hashlib.sha256(open(rwp, "rb").read()).hexdigest().upper() != RW_SHA:
        print("[STOP] rw_check_independent_out.csv is not the frozen file"); return 1
    R = pd.read_csv(rwp); S = R[R.rw_adj_p <= 0.05].reset_index(drop=True)
    ref = pd.read_parquet(os.path.join(dump, "N60_hh_SD_1.0.parquet"), columns=["day", "block", "is_real"])
    n_set = {B: int(ref.day[(ref.block == B) & (ref.is_real == 1)].nunique()) for B in "AB"}     # the registered 1,941 / 1,188
    uni = {B: np.unique(ref.day[ref.block == B].to_numpy()) for B in "AB"}                        # setups and twins: the calendar
    if n_set != D_REG:
        print(f"[STOP] setup trade dates of the reference cell {n_set} != registered {D_REG}"); return 1
    print(f"calendar universe (reference cell, setup or twin dates): A {len(uni['A'])}, B {len(uni['B'])}; registered D for the scaling: {D_REG}")
    if len(S) != 31:
        print(f"[STOP] survivors {len(S)}"); return 1
    if limit:
        S = S.iloc[:limit]
    rows, bad = [], []
    for _, row in S.iterrows():
        cell, side = str(row.cell), MP.side_of(row.side)
        st = pd.read_parquet(os.path.join(core_v2.OUT, f"{cell}_side{side}.parquet")).iloc[0]
        col, cut, sense = str(st["col"]), float(st["cut"]), str(st["sense"])
        x = pd.read_parquet(os.path.join(dump, cell + ".parquet"), columns=sorted(set(NEED + [col])))
        r = x[x.is_real == 1]
        if side is not None:
            r = r[np.sign(r.side) == side]
        r = r.reset_index(drop=True)
        v = r[col].to_numpy(float); m = (v <= cut) if sense == "lo" else (v > cut)
        tw1 = x[(x.is_real == 0) & x.src_bar.isin(set(r.bar))]
        tw2 = pd.read_parquet(os.path.join(dump, cell + ".m2.parquet"), columns=["src_bar", "side", "dist_atr", "race", "gap_at_T", "cens_T"])
        tw2 = tw2[tw2.src_bar.isin(set(r.bar))]
        k = dict(counts=(int((m & (r.block == "A")).sum()) == int(st["nA"])) and (int((m & (r.block == "B")).sum()) == int(st["nB"])),
                 sign=np.sign(st["eA"]) == np.sign(st["eB"]), opens=open_between(r) and open_between(tw1) and open_between(tw2))
        if not all(k.values()):
            bad.append((cell, str(row.side), k))
        direction = 1.0 if float(st["eA"]) > 0 else -1.0
        cens = r.cens_T.to_numpy(bool)
        s = np.where(cens, np.nan, base_outcome(r.race, r.dist_atr, r.side, r.gap_at_T, direction))
        s_old = np.where(cens, np.nan, base_outcome(r.race, r.dist_atr, r.side, r.gap_at_T, direction, tie_negated=True))
        t1 = twins_mean(tw1, r.bar.to_numpy(), direction); t2 = twins_mean(tw2, r.bar.to_numpy(), direction)
        tk = r.atr0.to_numpy(float) / 0.25
        e11 = (r.dist_atr.to_numpy(float) >= 1.0) & (r.dist_atr.to_numpy(float) * tk >= 4.0)
        rec = dict(cell=cell, side=str(row.side), col=col, sense=sense, dir="FADE" if direction > 0 else "CONT",
                   grp=f"{cell.split('_')[1]}_{cell.split('_')[2]}_{cell.split('_')[3]}_{row.side}")
        pw, pw11 = {}, {}
        for B in ("A", "B"):
            b = (r.block == B).to_numpy(); sub = b & m
            est, se, Rm, se_i, se_b, nn = arm_stats(arm_table(s[sub], r.day.to_numpy()[sub], t1[sub], t2[sub], uni[B]))
            pw[B] = powers(est, se, Rm, B)
            s11 = sub & e11
            e2, se2, R2, _, _, n11 = arm_stats(arm_table(s[s11], r.day.to_numpy()[s11], t1[s11], t2[s11], uni[B]))
            pw11[B] = powers(e2, se2, R2, B)["c_K5"]
            ties = int((r.race.to_numpy()[sub] == 0).sum())
            rec.update({f"n{B}": int(nn[0]), f"n_d1{B}": int(nn[1]), f"n_d2{B}": int(nn[2]), f"ties{B}": ties, f"seal_n{B}": float(nn[0] * D_SEAL / D_REG[B]),
                        f"days_share{B}": float((arm_table(s[sub], r.day.to_numpy()[sub], t1[sub], t2[sub], uni[B])[:, 1] > 0).mean()),
                        f"g{B}": est[0], f"d1{B}": est[1], f"d2{B}": est[2], f"t_g{B}": est[0] / se[0], f"t_d1{B}": est[1] / se[1], f"t_d2{B}": est[2] / se[2],
                        f"blk_iid{B}": se_b[0] / se_i[0], f"ticks{B}": float(np.nanmean(s[sub] * tk[sub])), f"ticks_tieneg{B}": float(np.nanmean(s_old[sub] * tk[sub])),
                        f"seal_se_ticks{B}": float(se[0] * np.sqrt(D_REG[B] / D_SEAL) * np.nanmean(tk[sub])), f"e11_kept{B}": float(e11[sub].mean()),
                        f"race{B}": race_edge(r, m, tw1, B)})
        for kk in pw["A"]:
            rec[kk] = np.nan if np.isnan([pw["A"][kk], pw["B"][kk]]).any() else min(pw["A"][kk], pw["B"][kk])
        rec["c_K5_E11"] = np.nan if np.isnan([pw11["A"], pw11["B"]]).any() else min(pw11["A"], pw11["B"])
        rows.append(rec)
    D = pd.DataFrame(rows)
    nf, nc = int((D.dir == "FADE").sum()), int((D.dir == "CONT").sum())
    print(f"\n{len(D)} survivors ({nf} FADE, {nc} CONTINUATION), {D.grp.nunique()} dedupe groups, {time.time() - t0:.0f}s")
    if bad or (limit is None and (nf, nc) != (16, 15)):
        print("[STOP] a known number failed; no power is printed.", bad[:3], (nf, nc)); return 1
    print("  [PASS] subset counts == stored nA, nB; sign(eA) == sign(eB); every open trade (setups, M1b, M2) lies between the levels; calendar 1,941 / 1,188; 16 + 15")
    for d, refA, refB in (("FADE", 8.30, 4.88), ("CONT", -0.37, 0.12)):
        q = D[D.dir == d]
        if len(q):
            print(f"  COMPARISON (not a gate) race edge of the {d} cores, median pp: A {q.raceA.median():+.2f} (26.91: {refA:+.2f}), B {q.raceB.median():+.2f} (26.91: {refB:+.2f})")
    stamp = (f"A10 BASE-bracket joint-power preview v2 | the 31 Stage-1 Romano-Wolf survivors, discovered subsets | BASE symmetric bracket, tie = stop first, opens marked at T, "
             f"E6/E7 OFF, E11 OFF except the labelled column | arms gross / M1b-diff / M2-diff, ATR0 units | SE = max(iid, 5-day block) on the calendar dates, {NRES} resamples | "
             f"seal SE x sqrt(D_b/1030), D_A 1941, D_B 1188 | MC {NMC}, seed 20261007 | (c) = 0.75 x edge, A CONVENTION | {dump} | rw csv {RW_SHA[:16]} | "
             f"a10_power_preview.py {hashlib.sha256(open(__file__, 'rb').read()).hexdigest().upper()} | {platform.node()} | python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__} | "
             f"ES train A+B, sealed untouched | raw, no cost, IN-SAMPLE AND SELECTION-INFLATED, DESCRIPTIVE; no number here triggers anything")
    print("\n" + stamp + "\n")
    pd.set_option("display.width", 340); pd.set_option("display.max_rows", 100)
    print(D[["cell", "side", "col", "dir", "nA", "nB", "ticksA", "ticksB", "t_gA", "t_gB", "t_d1B", "t_d2B", "c_K1", "c_K5", "c_K18", "a_K5", "half_K5", "pt_K5", "c_K5_E11"]].round(2).to_string(index=False))
    for d in ("FADE", "CONT"):
        q = D[D.dir == d]
        if not len(q):
            continue
        cnt = lambda c: int((q[c] >= 0.8).sum())
        print(f"\n{d} ({len(q)}): joint power >= 80%  (c) K=1 {cnt('c_K1')}, K=5 {cnt('c_K5')}, K=18 {cnt('c_K18')} | (a) K=1 {cnt('a_K1')}, K=5 {cnt('a_K5')} | half K=5 {cnt('half_K5')} | point K=1 {cnt('pt_K1')}, K=5 {cnt('pt_K5')} | "
              f"(c) K=5 with E11 ON {cnt('c_K5_E11')} | undefined {int(q.c_K5.isna().sum())}")
        print(f"   gross edge / SE: block A median {q.t_gA.median():+.2f} (max {q.t_gA.max():+.2f}), block B median {q.t_gB.median():+.2f} (max {q.t_gB.max():+.2f}); all three arms > 0 in both blocks: "
              f"{int(((q[['gA', 'gB', 'd1A', 'd1B', 'd2A', 'd2B']] > 0).all(1)).sum())}")
        print(f"   gross ticks per trade: median A {q.ticksA.median():+.2f}, B {q.ticksB.median():+.2f}; max A {q.ticksA.max():+.2f}, B {q.ticksB.max():+.2f}; with ties negated (superseded rule) median A {q.ticks_tienegA.median():+.2f}, B {q.ticks_tienegB.median():+.2f}; ties per core median {int(q.tiesA.median())} / {int(q.tiesB.median())}")
        print(f"   seal SE of the gross mean in ticks: median A {q.seal_se_ticksA.median():.2f}, B {q.seal_se_ticksB.median():.2f} (26.84: 0.10 / 0.31); expected seal trades per core: min {q[['seal_nA','seal_nB']].min().min():.0f}; "
              f"E11 keeps a median {100*q.e11_keptA.median():.0f}% (A) / {100*q.e11_keptB.median():.0f}% (B) of trades")
    print(f"\nSE_block5 / SE_iid, gross arm (A3): block A median {D.blk_iidA.median():.2f} (max {D.blk_iidA.max():.2f}); block B median {D.blk_iidB.median():.2f} (max {D.blk_iidB.max():.2f})")
    out = os.path.join(HERE, f"a10_power_preview_{time.strftime('%Y-%m-%d')}.csv")
    D.assign(stamp=stamp).to_csv(out, index=False)
    print(f"wrote {os.path.basename(out)}   {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2].split("=")[1]) if len(sys.argv) > 2 and sys.argv[2].startswith("--limit=") else None))
