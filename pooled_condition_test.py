r"""pooled_condition_test.py -- which conditions are ASSOCIATED WITH THE RACE RATE, pooled over the 144 registered
cells (version 2, after the reviewer's read, HANDOFF 26.169). The concise replacement (the author, 2026-10-13:
"yes lets do this") for the Stage-1-style race-unit search (design v4, set aside). UNREGISTERED until the A5(2)
text is confirmed. EXPLORATORY and IN-SAMPLE: it is not needed for the paper's two claims.

OUTCOME   race: among setups whose race is RESOLVED, hit = 1 if the frozen VWAP level was touched before its
          mirror (claim 2's outcome). No twin enters. Not distorted by the twins' hour (26.155).
STATISTIC per condition c (every core_v2 search column, pooled sides) and train block b: an EXTREME-GROUPS
          contrast, theta_b(c) = mean centred hit in the TOP group - mean centred hit in the BOTTOM group.
          Inside each (cell, block) unit: a column with more than 10 distinct finite values: bottom = v <= its
          20th percentile, top = v > its 80th percentile (percentiles of the unit's own finite rows); a column
          with 2..10 distinct values: bottom = the lowest value, top = the highest value. A unit with fewer than
          100 rows in either group is DROPPED for that column (counted). Hits are centred on the unit's mean hit
          over its finite rows. Units are then pooled (rows as weights). ct_hour is ordered by session position
          (ct_hour - 17) mod 24, so its contrast is LATE session versus EARLY session.
RANGE     (added 2026-10-13 at the author's request, 26.175) a SECOND contrast per condition, named '<col>|mid':
          mean centred hit in the MIDDLE (every finite row that is in neither end group) minus the two END groups
          together. It sees "works inside a range" and "works at both extremes", which top-vs-bottom cannot. It is
          in the same family (the family doubles). A five-fifths profile is printed per condition, no verdict.
INFERENCE trade DATES are the units. Three day bootstraps per block, B draws each: iid dates, and moving blocks of
          28 and of 56 calendar days. Whole dates are resampled, so every cell's rows of a date move together.
          For each: t = theta / SE*, SE* = the bootstrap SD; Romano-Wolf step-down max-|t| over the family of
          ALL block-level contrasts (2 x the number of conditions kept), on the centred draws. adjusted p =
          the LARGEST of the three (no block length is picked from data). A CONDITION PASSES iff both its block
          contrasts have adjusted p <= 0.05 AND the same sign.
ORDER     --check (tables, known numbers, group sizes, SE ratios; NO contrast printed)  ->  --worlds (dead
          frames on the REAL outcomes: calib_main.shift_covariates, one anchor for all cells)  ->
          --summarize-worlds  ->  --run (refuses unless 29 world files from THIS script version exist).
READING OF THE WORLDS, FIXED BEFORE THEY RUN (the reviewer's rule, 26.170): per column and block, ratio = SD of
          theta over the 29 separated worlds / mean SE used (the largest of the three). In --run each contrast's
          OBSERVED t is divided by f = max(1, its ratio): no cutoff, no median; the bootstrap draws keep their
          own scale; it can only remove passes. The count of worlds with any rejection is printed with its
          Clopper-Pearson interval and triggers nothing (29 worlds cannot tell 5% from about 12%). Known limits:
          a dead frame gives one event different donor rows in different N-cells, so the ratio may be somewhat
          optimistic; an event is counted once per cell it falls in (no dedupe).
WHAT A PASS MEANS: in the training data, within cells, the race rate among resolved setups differs between the
          column's top and bottom groups, the same way in both periods, at a family-wise 5% bar. Setup-weighted
          (small-eps cells dominate). NOT VWAP-specific. Not independent of Stage 1 (same data).
WHAT IT CANNOT SEE: a middle band, both tails the same way, an hour window inside the session, effects that point
          different ways in different situations (a signed column pooled over both sides can cancel: its _xs
          twin is the test), effects in one period only. A non-pass is "not detected by this contrast", never
          "no effect". No sentence "the test sees effects above X" is written.

    python pooled_condition_test.py --selftest
    python pooled_condition_test.py data\dump --check
    python pooled_condition_test.py data\dump --worlds=29 --part=1/8      (... 8/8; about 2-3 min per world, ESTIMATE)
    python pooled_condition_test.py data\dump --summarize-worlds
    python pooled_condition_test.py data\dump --run
"""
import os, sys, time, glob, hashlib, platform, warnings
import numpy as np, pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import core_v2                                              # noqa: E402
from power_registered_set import registered_cells           # noqa: E402

B_BOOT, SEED, L_BLOCK, MIN_G = 5000, 20261013, 28, 100
KINDS = ("iid", 28, 56)          # the pass rule takes the LARGEST adjusted p of the three; no block length is picked from data
PUB = dict(n=3260501, A=54.815, B=50.737)
CACHE = os.path.join(HERE, "pooled_condition_cache.npz")
WDIR = os.path.join(HERE, "pooled_worlds_v4")      # version 4 (range contrast added); older versions wrote pooled_worlds\


def groups(v):
    """One (cell, block) unit. codes: 0 not used, 1 bottom, 2 top, 3 other finite. All 0 if the unit is dropped."""
    v = np.asarray(v, float); fin = np.isfinite(v); code = np.zeros(len(v), np.int8)
    if fin.sum() < 2 * MIN_G:
        return code
    vals = np.unique(v[fin])
    if len(vals) < 2:
        return code
    if len(vals) <= 10:
        bot, top = fin & (v == vals[0]), fin & (v == vals[-1])
    else:
        q20, q80 = np.quantile(v[fin], [0.2, 0.8])
        bot, top = fin & (v <= q20), fin & (v > q80)
    if bot.sum() < MIN_G or top.sum() < MIN_G:
        return code
    code[fin] = 3; code[bot] = 1; code[top] = 2
    return code


def unit_tables(v, hit, pos, D):
    """Rows 0-3: [sum centred hit, n] for bottom and top on the block's day axis (the TOP-vs-BOTTOM contrast).
    Rows 4-7: the same for ENDS (bottom + top together) and MIDDLE (every other finite row): the RANGE contrast,
    middle minus ends; left at zero when the middle group has fewer than MIN_G rows. None if the unit is dropped."""
    code = groups(v)
    if not code.any():
        return None, code
    h = hit[code > 0].mean(); t = np.zeros((8, D))
    for j, c in ((0, 1), (2, 2)):
        m = code == c
        t[j] = np.bincount(pos[m], hit[m] - h, D); t[j + 1] = np.bincount(pos[m], None, D)
    mid = code == 3
    if mid.sum() >= MIN_G:
        t[4] = t[0] + t[2]; t[5] = t[1] + t[3]
        t[6] = np.bincount(pos[mid], hit[mid] - h, D); t[7] = np.bincount(pos[mid], None, D)
    return t, code


def fifths_profile(v, hit):
    """DESCRIPTIVE: [sum centred hit, n] in each fifth of the unit's finite values (more than 10 distinct values only)."""
    v = np.asarray(v, float); fin = np.isfinite(v); out = np.zeros((5, 2))
    if fin.sum() < 5 * MIN_G or len(np.unique(v[fin])) <= 10:
        return out
    q = np.quantile(v[fin], [0.2, 0.4, 0.6, 0.8]); k = np.searchsorted(q, v[fin], side="left"); c = hit[fin] - hit[fin].mean()
    out[:, 0] = np.bincount(k, c, 5); out[:, 1] = np.bincount(k, None, 5)
    return out


def mid_class(TA, TB, C):
    """Per base column: how its RANGE contrast may be read. From the pooled group means in both blocks:
    middle-peak (middle above BOTH ends in both blocks), both-ends (middle below BOTH ends in both blocks), otherwise
    echo (a one-tail effect also moves middle-minus-ends, by arithmetic: reviewer 26.175). Only the first two count as a
    range finding."""
    out = []
    for j in range(C):
        up = dn = True
        for T in (TA, TB):
            with np.errstate(invalid="ignore", divide="ignore"):
                bot, top, mid = T[j, 0].sum() / T[j, 1].sum(), T[j, 2].sum() / T[j, 3].sum(), T[j + C, 2].sum() / T[j + C, 3].sum()
            up &= bool(mid > bot and mid > top); dn &= bool(mid < bot and mid < top)
        out.append("middle-peak" if up else "both-ends" if dn else "echo")
    return out


def stack_hyp(T, names):
    """(C, 8, D) -> (2C, 4, D): the top-vs-bottom hypotheses, then the range hypotheses named '<col>|mid'."""
    return np.concatenate([T[:, :4], T[:, 4:8]], 0), list(names) + [f"{n}|mid" for n in names]


def theta(T):
    with np.errstate(invalid="ignore", divide="ignore"):
        return T[..., 2] / T[..., 3] - T[..., 0] / T[..., 1]


def boot(Tb, kind, seed, B=B_BOOT):
    """Tb: (C, 4, D). Returns theta (C,), draws (B, C)."""
    C, _, D = Tb.shape
    if kind == "iid":
        W = np.random.default_rng(seed).multinomial(D, np.full(D, 1.0 / D), size=B).astype(float)
    else:
        S = core_v2.boot_index(D, B, int(kind), seed); W = np.zeros((B, D))
        for i in range(B):
            W[i] = np.bincount(S[i], minlength=D)
    X = W @ Tb.reshape(C * 4, D).T
    return theta(Tb.sum(2)), theta(X.reshape(B, C, 4))


def stepdown(t_obs, t_star):
    """Romano-Wolf step-down max-|t| adjusted p-values; t_star (B, H) centred and scaled draws."""
    H = len(t_obs); a = np.abs(t_obs); order = np.argsort(-a); adj = np.empty(H)
    Ts = np.abs(t_star)[:, order]
    cm = np.maximum.accumulate(Ts[:, ::-1], axis=1)[:, ::-1]
    p = (1 + (cm >= a[order][None, :]).sum(0)) / (1 + len(Ts))
    adj[order] = np.maximum.accumulate(p)
    return adj


def analyse(TA, TB, B=B_BOOT, fA=None, fB=None, kinds=KINDS):
    """Returns the per-column table, the column indices dropped for an undefined contrast or draw, and the family size.
    fA, fB: per-column dead-frame factors (>= 1). They divide the OBSERVED t only; the bootstrap draws keep their own
    scale (dividing both would change nothing: reviewer, 26.170 objection 1)."""
    C = TA.shape[0]; res = {}; good = np.ones(C, bool)
    fA = np.ones(C) if fA is None else np.asarray(fA, float); fB = np.ones(C) if fB is None else np.asarray(fB, float)
    for ki, kind in enumerate(kinds):
        for bi, (nm, T) in enumerate((("A", TA), ("B", TB))):
            th, dr = boot(T, kind, (SEED if kind == "iid" else 0) + 1 + bi + 10 * ki, B)
            good &= np.isfinite(th) & np.isfinite(dr).all(0)
            res[(kind, nm)] = (th, dr)
    out = pd.DataFrame(index=range(C)); adjs = []
    for kind in kinds:
        thA, drA = res[(kind, "A")]; thB, drB = res[(kind, "B")]
        seA, seB = drA[:, good].std(0), drB[:, good].std(0)
        t_obs = np.r_[thA[good] / (seA * fA[good]), thB[good] / (seB * fB[good])]
        t_star = np.c_[(drA[:, good] - thA[good][None, :]) / seA[None, :], (drB[:, good] - thB[good][None, :]) / seB[None, :]]
        assert np.isfinite(t_obs).all() and np.isfinite(t_star).all(), "an undefined contrast survived the screen"
        adjs.append(stepdown(t_obs, t_star))
        k = "iid" if kind == "iid" else f"b{kind}"
        out.loc[good, f"se_{k}A"] = 100 * seA; out.loc[good, f"se_{k}B"] = 100 * seB
    n = int(good.sum()); adj = np.maximum.reduce(adjs)
    out.loc[good, "thA"] = 100 * res[(kinds[0], "A")][0][good]; out.loc[good, "thB"] = 100 * res[(kinds[0], "B")][0][good]
    out["se0A"] = out[[c for c in out.columns if c.startswith("se_") and c.endswith("A")]].max(axis=1)
    out["se0B"] = out[[c for c in out.columns if c.startswith("se_") and c.endswith("B")]].max(axis=1)
    out["fA"] = fA; out["fB"] = fB; out["seA"] = out.se0A * fA; out["seB"] = out.se0B * fB
    out["tA"] = out.thA / out.seA; out["tB"] = out.thB / out.seB
    out.loc[good, "adjA"] = adj[:n]; out.loc[good, "adjB"] = adj[n:]
    out["passes"] = (out.adjA <= 0.05) & (out.adjB <= 0.05) & (np.sign(out.thA) == np.sign(out.thB))
    for nm, T in (("A", TA), ("B", TB)):
        out[f"nbot{nm}"] = T[:, 1].sum(1).astype(int); out[f"ntop{nm}"] = T[:, 3].sum(1).astype(int)
    return out, np.where(~good)[0], 2 * n


# ------------------------------------------------------------------ self-test
def _toy(seed, eff=0.0, eff_B=None, cols=10, cells=5, nd=400, per=8, dayfx=0.0, flip_B=False, regime=0, nan_col=False, binary=False, base_shift=0.0, center=True, shape="top", full=False):
    rng = np.random.default_rng(seed); T = {b: np.zeros((cols, 8, nd)) for b in "AB"}
    de = rng.standard_normal(2 * nd) * dayfx
    if regime:
        de = np.repeat(rng.standard_normal(2 * nd // regime + 1), regime)[:2 * nd] * dayfx
    for c in range(cells):
        n = 2 * nd * per; day = np.repeat(np.arange(2 * nd), per); isA = day < nd
        X = rng.standard_normal((n, cols))
        if regime:
            X[:, 1] = np.repeat(np.repeat(rng.standard_normal(2 * nd // regime + 1), regime)[:2 * nd], per) + 0.3 * rng.standard_normal(n)
        if binary:
            X[:, 2] = (rng.random(n) < 0.3).astype(float)
        if nan_col:
            X[rng.random(n) < 0.3, 3] = np.nan
        e = np.where(isA, eff, eff if eff_B is None else eff_B) * np.where(~isA & flip_B, -1, 1)
        topX0 = np.zeros(n, bool)
        for m in (isA, ~isA):
            topX0[m] = (X[m, 0] > np.quantile(X[m, 0], 0.8)) if shape == "top" else ((X[m, 0] > np.quantile(X[m, 0], 0.2)) & (X[m, 0] <= np.quantile(X[m, 0], 0.8)))
        p = 0.52 + c * base_shift + de[day] + e * topX0
        hit = (rng.random(n) < np.clip(p, 0.01, 0.99)).astype(float)
        for bname, m in (("A", isA), ("B", ~isA)):
            pos = day[m] - (0 if bname == "A" else nd)
            for j in range(cols):
                t, code = unit_tables(X[m, j], hit[m] if center else hit[m] - hit[m].mean() + 0.0, pos, nd)
                if t is not None:
                    T[bname][j] += t
    return (stack_hyp(T["A"], range(cols))[0], stack_hyp(T["B"], range(cols))[0]) if full else (T["A"][:, :4], T["B"][:, :4])


def selftest():
    ok = True
    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    print("SELF-TEST (toy fixtures, no data)")
    c = groups(np.r_[np.arange(1000.0), np.full(50, np.nan)])
    chk("K1 continuous: 200 bottom, 200 top, 600 other, NaN unused", (c == 1).sum() == 200 and (c == 2).sum() == 200 and (c == 3).sum() == 600 and (c[-50:] == 0).all())
    c2 = groups(np.r_[np.zeros(700), np.ones(300)])
    chk("K2 two-valued: bottom = the low value, top = the high value", (c2 == 1).sum() == 700 and (c2 == 2).sum() == 300)
    c3 = groups(np.r_[np.zeros(950), np.ones(50)]); c4 = groups(np.full(1000, 3.0)); c5 = groups(np.r_[np.arange(700.0), np.full(300, 999.0)])
    chk("K2b dropped units: a group under 100 rows; a constant column; a column capped at its maximum for 30% of rows", not c3.any() and not c4.any() and not c5.any())
    c6 = groups(np.repeat(np.arange(5.0), 200))
    chk("K2c a 5-valued column (weekday): bottom = the lowest value only, top = the highest only", (c6 == 1).sum() == 200 and (c6 == 2).sum() == 200)
    rng = np.random.default_rng(1); t_obs = np.array([5.0, 0.1, 3.0]); ts = rng.standard_normal((4000, 3)); adj = stepdown(t_obs, ts)
    chk("K3 step-down: monotone in |t|; first step = brute-force max-|t| p", adj[0] <= adj[2] <= adj[1] and abs(adj[0] - (1 + (np.abs(ts).max(1) >= 5.0).sum()) / 4001) < 1e-12, str(np.round(adj, 4).tolist()))
    W = 60; hits = 0; tt = []
    for s in range(W):
        r, _, _ = analyse(*_toy(100 + s, nan_col=True, binary=True), B=800)
        hits += int(((r.adjA <= 0.05) | (r.adjB <= 0.05)).any()); tt.append(np.r_[r.tA.dropna(), r.tB.dropna()])
    tt = np.concatenate(tt); lo, hi = beta.ppf(.025, hits, W - hits + 1) if hits else 0.0, beta.ppf(.975, hits + 1, W - hits)
    chk("K4 SIZE on a toy null with NaN and a binary column: share of |t| > 1.96 near 5% and the any-contrast family-wise rate's CI contains 5%",
        abs((np.abs(tt) > 1.96).mean() - 0.05) < 0.02 and lo <= 0.05 <= hi, f"|t|>1.96: {100*(np.abs(tt) > 1.96).mean():.1f}%; family-wise {hits} of {W} [{100*lo:.1f}, {100*hi:.1f}]%")
    tA = []; tB28 = []
    for s in range(40):
        TA, TB = _toy(300 + s, dayfx=0.06, regime=20)
        th, dr = boot(TA, "iid", 5, 600); tA.append(th[1] / dr[:, 1].std()); th2, dr2 = boot(TA, 56, 1, 600); tB28.append(th2[1] / dr2[:, 1].std())
    chk("M1 a column and the outcome both living in 20-day regimes: the iid-date t is too large, the 56-day block t much less so (why block variants are in the rule; they do not fully repair it)",
        np.std(tA) > 1.25 and np.std(tB28) < np.std(tA), f"SD of t over 40 worlds: iid {np.std(tA):.2f}, 56-day block {np.std(tB28):.2f} (1.0 is correct)")
    r, _, _ = analyse(*_toy(7, eff=0.06), B=1500)
    chk("K6 a planted +6 pp top-group effect in column 0 passes, and only column 0", bool(r.passes[0]) and int(r.passes.sum()) == 1, f"thA {r.thA[0]:+.2f}, thB {r.thB[0]:+.2f} pp")
    r, _, _ = analyse(*_toy(8, eff=0.06, eff_B=0.0), B=1500)
    chk("M2 an effect in block A only does NOT pass", not bool(r.passes[0]), f"adjA {r.adjA[0]:.3f}, adjB {r.adjB[0]:.3f}")
    r, _, _ = analyse(*_toy(9, eff=0.06, flip_B=True), B=1500)
    chk("M3 opposite signs in the two blocks do NOT pass", not bool(r.passes[0]), f"thA {r.thA[0]:+.2f}, thB {r.thB[0]:+.2f}")
    TA, TB = _toy(11, base_shift=0.05); th = theta(TA.sum(2))
    chk("K7 cells with different base rates leave no contrast when hits are centred per unit", np.abs(th).max() < 0.03, f"max |theta| {100*np.abs(th).max():.2f} pp")
    TA, TB = _toy(10, eff=0.06); r1, d1, n1 = analyse(TA, TB, B=1200); TA2 = TA.copy(); TA2[4] = 0.0; r2, d2, n2 = analyse(TA2, TB, B=1200)
    chk("K8 an undefined column is dropped from the family and listed, not scored as 0", list(d2) == [4] and n2 == n1 - 2 and np.isnan(r2.adjA[4]))
    f2 = np.full(TA.shape[0], 2.5); r3, _, _ = analyse(TA, TB, B=1200, fA=f2, fB=f2)
    chk("K9 a dead-frame factor of 2.5 scales the reported SE and STRICTLY raises the planted column's adjusted p (it acts on the observed t only)",
        np.allclose(r3.seA, 2.5 * r1.seA) and r3.adjA[0] > r1.adjA[0] and r3.adjB[0] > r1.adjB[0] and not bool(r3.passes[0]),
        f"adjA {r1.adjA[0]:.4f} -> {r3.adjA[0]:.4f}, adjB {r1.adjB[0]:.4f} -> {r3.adjB[0]:.4f}")
    RA, RB = _toy(21, eff=0.06, shape="mid", full=True); rr, _, _ = analyse(RA, RB, B=1500)
    chk("K10 RANGE: a +6 pp effect planted in the MIDDLE three fifths of column 0 passes on the range contrast (row 10) and is invisible to top-vs-bottom (row 0)",
        bool(rr.passes[10]) and not bool(rr.passes[0]) and rr.thA[10] > 0 and int(rr.passes.sum()) == 1, f"range thA {rr.thA[10]:+.2f}, thB {rr.thB[10]:+.2f}; top-vs-bottom thA {rr.thA[0]:+.2f} pp")
    RA, RB = _toy(22, eff=0.06, full=True); rr, _, _ = analyse(RA, RB, B=1500)
    chk("K11 a top-fifth effect passes on top-vs-bottom (row 0); its range contrast is negative (the ends beat the middle), as arithmetic requires", bool(rr.passes[0]) and rr.thA[10] < 0 and rr.thB[10] < 0,
        f"top-vs-bottom thA {rr.thA[0]:+.2f}; range thA {rr.thA[10]:+.2f} pp")
    pf = fifths_profile(np.arange(1000.0), np.r_[np.zeros(800), np.ones(200)])
    chk("K12 the five-fifths profile: 200 rows per fifth, centred sums add to 0, the planted top fifth is +0.8", (pf[:, 1] == 200).all() and abs(pf[:, 0].sum()) < 1e-9 and abs(pf[4, 0] / 200 - 0.8) < 1e-12)
    PA, PB = _toy(21, eff=0.06, shape="mid", full=True); EA, EB = _toy(22, eff=0.06, full=True)
    chk("K13 reading guard: the planted middle effect is classed middle-peak; the planted top-fifth effect is classed echo, so its range row can never be reported as a range finding",
        mid_class(PA, PB, 10)[0] == "middle-peak" and mid_class(EA, EB, 10)[0] == "echo")
    th, dr = boot(TA, "iid", 3, 400); se = dr.std(0); a1 = stepdown(th / se, (dr - th) / se); a2 = stepdown(th / (2.5 * se), (dr - th) / (2.5 * se))
    chk("M4 the superseded form (factor on both the observed and the bootstrap t) changes NOTHING, which is why it was a defect", np.array_equal(a1, a2))
    print("ALL DETECTORS ALIVE" if ok else "SELF-TEST FAILED"); return ok


# ------------------------------------------------------------------ data
def axis_of(dump):
    ref = pd.read_parquet(os.path.join(dump, "N60_hh_SD_1.0.parquet"), columns=["ts", "block"])
    out = {}
    for b in "AB":
        d = np.asarray(core_v2.trade_day(ref.ts[ref.block == b])); out[b] = (int(d.min()) - 10, int(d.max()) + 10)
    return out


def build(dump, world=None, plant=None, cells=None, hours=False):
    """world = None: the real frame. world = j: the dead frame j (covariate rows shifted within block, one anchor
    for all cells). plant = (column, deltaA, deltaB): add that many points of race rate in the column's top group,
    decided per signal bar and side so an event repeated across cells gets the same flip."""
    import calib_main as CM
    t0 = time.time(); cells = cells or registered_cells(); ax = axis_of(dump)
    D = {b: ax[b][1] - ax[b][0] + 1 for b in "AB"}
    names, T = [], {b: [] for b in "AB"}; nres = 0; wins = {b: [0.0, 0] for b in "AB"}; dropped = {}; units = {}
    hrs = {1: np.zeros(24), 2: np.zeros(24)}; csum = 0.0; nfin = {}; prof = {b: [] for b in "AB"}
    for cell in cells:
        r, c, Tt = core_v2.load_cell(cell, None)
        race = pd.read_parquet(os.path.join(dump, cell + ".parquet"), columns=["is_real", "race"])
        race = race.race[race.is_real == 1].to_numpy(); assert len(race) == len(r)
        res = np.isin(race, (1, -1)); r = r[res].reset_index(drop=True); hit = (race[res] == 1).astype(float)
        cols = core_v2.search_cols(r, None); blk = r.block.to_numpy(); day = r.day.to_numpy().astype(np.int64)
        nres += len(r)
        for b in "AB":
            wins[b][0] += hit[blk == b].sum(); wins[b][1] += int((blk == b).sum())
        if world is not None:
            r, _ = CM.shift_covariates(r, cols, CM.anchor_fracs(world), {b: (ax[b][0] + 10, ax[b][1] - 10) for b in "AB"}, day)
        if plant is not None and plant[0] in cols:
            u = (pd.util.hash_pandas_object(pd.DataFrame({"b": r.bar.to_numpy(), "s": np.sign(r.side.to_numpy())}), index=False).to_numpy() % 1000003) / 1000003.0
            mult = np.where(((day * 2654435761) % 1000003) / 1000003.0 < 0.5, 2.0, 0.0) if len(plant) > 3 and plant[3] else np.ones(len(day))
            for b, dl in (("A", plant[1]), ("B", plant[2])):
                m = blk == b; code = groups(r[plant[0]].to_numpy(float)[m]); idx = np.where(m)[0][code == 2]
                p = hit[idx].mean() if len(idx) else 0.5
                flip = (hit[idx] == 0) & (u[idx] < mult[idx] * dl / max(1 - p, 1e-9))
                hit[idx[flip]] = 1.0
        for col in cols:
            v = r[col].to_numpy(float)
            if col == "ct_hour":
                v = np.mod(v - 17, 24)
            if col not in names:
                names.append(col); dropped[col] = 0; units[col] = 0
                for b in "AB":
                    T[b].append(np.zeros((8, D[b]))); prof[b].append(np.zeros((5, 2)))
            j = names.index(col)
            for b in "AB":
                m = blk == b
                t, code = unit_tables(v[m], hit[m], day[m] - ax[b][0], D[b])
                units[col] += 1
                if t is None:
                    dropped[col] += 1; continue
                T[b][j] += t; prof[b][j] += fifths_profile(v[m], hit[m]); csum = max(csum, abs(float((hit[m][code > 0] - hit[m][code > 0].mean()).sum())))
                nfin[col] = nfin.get(col, 0) + int((code > 0).sum())
                if hours and col == "ct_hour":
                    for g in (1, 2):
                        hrs[g] += np.bincount(r.ct_hour.to_numpy()[m][code == g].astype(int), minlength=24)
    TA, hyp = stack_hyp(np.array(T["A"]), names); TB, _ = stack_hyp(np.array(T["B"]), names)
    return dict(names=np.array(hyp), cols=list(names), TA=TA, TB=TB, profA=np.array(prof["A"]), profB=np.array(prof["B"]), n=nres, rateA=100 * wins["A"][0] / wins["A"][1], rateB=100 * wins["B"][0] / wins["B"][1],
                secs=time.time() - t0, ncell=len(cells), dropped=dropped, units=units, hrs=hrs, csum=csum, nfin=nfin, ax=ax)


def known(d):
    ok = (d["ncell"] == 144) and (d["n"] == PUB["n"]) and abs(d["rateA"] - PUB["A"]) < 0.001 and abs(d["rateB"] - PUB["B"]) < 0.001 and d["csum"] < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] K0: {d['ncell']} cells; resolved setups {d['n']:,} (26.65: 3,260,501); race rate A {d['rateA']:.3f}% (54.815), B {d['rateB']:.3f}% (50.737); "
          f"largest |sum of centred hits| in a unit {d['csum']:.1e}")
    return ok


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest().upper()


def stamp(dump, extra=""):
    return (f"pooled extreme-groups race contrast | 144 registered cells, resolved setups, pooled sides | groups within (cell, block), min {MIN_G} rows | hits centred per unit on finite rows | "
            f"units = trade dates; iid, 28- and 56-calendar-day block bootstraps, B = {B_BOOT}; adjusted p = the largest of the three;Romano-Wolf step-down max-|t| over all block contrasts | "
            f"{dump} | pooled_condition_test.py {sha(__file__)} | {platform.node()} | python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__} | "
            f"ES train A+B, sealed untouched | raw, IN-SAMPLE, EXPLORATORY, not VWAP-specific{extra}")


def worlds():
    fs = sorted(glob.glob(os.path.join(WDIR, "world_*.csv")))
    if not fs:
        return None
    W = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    W = W[W.producer == sha(__file__)]
    return W if W[W.world < 29].world.nunique() == 29 else None


def factors(W):
    """Per column and block: f = max(1, SD of theta over the dead-frame worlds / mean SE used), no cutoff. Taken on the
    29 separated worlds and, when more worlds are on disk, also on all of them; the larger is used."""
    def ratio(X):
        g = X.groupby("col")
        return pd.DataFrame({"A": g.thA.std() / g.se0A.mean(), "B": g.thB.std() / g.se0B.mean(), "A_iid": g.thA.std() / g.se_iidA.mean(), "B_iid": g.thB.std() / g.se_iidB.mean(),
                             "zA": g.thA.mean() / (g.thA.std() / np.sqrt(g.thA.count())), "zB": g.thB.mean() / (g.thB.std() / np.sqrt(g.thB.count()))})
    r29 = ratio(W[W.world < 29]).dropna(subset=["A", "B"]); rall = ratio(W).reindex(r29.index) if W.world.nunique() > 29 else r29
    f = pd.DataFrame({"A": np.maximum(1.0, np.fmax(r29.A, rall.A)), "B": np.maximum(1.0, np.fmax(r29.B, rall.B))})
    return f, r29


def col_factors(names):
    W = worlds()
    if W is None:
        return None
    f, _ = factors(W); miss = [c for c in names if c not in f.index]
    fA = f.A.reindex(names).fillna(f.A.max()).to_numpy(); fB = f.B.reindex(names).fillna(f.B.max()).to_numpy()
    return fA, fB, miss, int(W.world.nunique())


def summarize():
    W = worlds()
    if W is None:
        print("[STOP] fewer than 29 separated dead-frame worlds written by THIS version of the script."); return 1
    f, r = factors(W); sep = W[W.world < 29]; both = np.r_[r.A, r.B]
    t = np.r_[sep.tA.dropna(), sep.tB.dropna()]
    anyrej = sep.groupby("world").apply(lambda x: bool(((x.adjA <= 0.05) | (x.adjB <= 0.05)).any()), include_groups=False); anypass = sep.groupby("world").passes.any()
    k = int(anyrej.sum()); n = len(anyrej); lo, hi = (beta.ppf(.025, k, n - k + 1) if k else 0.0), beta.ppf(.975, k + 1, n - k)
    print(f"DEAD-FRAME WORLDS (real outcomes, covariate rows shifted; 29 separated worlds, {W.world.nunique()} on disk, producer {sha(__file__)[:16]}). FACTOR 1 IN THESE WORLDS.")
    print(f"  SD of theta across worlds / mean SE used, per column and block: median {np.median(both):.2f} (A {r.A.median():.2f}, B {r.B.median():.2f}); p90 {np.quantile(both, .9):.2f}; max {both.max():.2f}; "
          f"against the iid-date SE alone: median {np.median(np.r_[r.A_iid, r.B_iid]):.2f}, max {np.r_[r.A_iid, r.B_iid].max():.2f}")
    print(f"  (an SD from 29 worlds has a relative error of about 13%, so ratios of 0.85-1.15 are what a correct SE gives)")
    print(f"  world means of theta: |mean / its SE| > 2 in {int((np.abs(np.r_[r.zA, r.zB]) > 2).sum())} of {2 * len(r)} contrasts (about {0.05 * 2 * len(r):.0f} expected if the dead frames are centred on 0)")
    print(f"  share of |t| > 1.96 over all contrasts and worlds: {100*(np.abs(t) > 1.96).mean():.1f}% (nominal 5%)")
    print(f"  worlds with ANY contrast rejected at the family-wise 5% bar: {k} of {n} [{100*lo:.1f}, {100*hi:.1f}]% (nominal 5%; 29 worlds cannot tell 5% from about 12%; NOT a trigger); worlds with any PASS: {int(anypass.sum())}")
    worst = r.assign(m=r[["A", "B"]].max(axis=1)).sort_values("m", ascending=False).head(8)
    print("  columns with the largest ratio: " + ", ".join(f"{i} {v:.2f}" for i, v in worst.m.items()))
    print(f"  RULE (fixed before any world ran): in --run each contrast's observed t is divided by f = max(1, its own ratio); no cutoff. f > 1.15 for {int((np.r_[f.A, f.B] > 1.15).sum())} of {2 * len(f)} contrasts; median f {np.median(np.r_[f.A, f.B]):.3f}.")
    return 0


def load_cache(dump):
    z = np.load(CACHE, allow_pickle=True)
    assert str(z["producer"]) == sha(__file__) and str(z["dump"]) == os.path.abspath(dump), "the cache was written by another version of this script or from another dump: re-run --check"
    return z


def main(argv):
    if not selftest():
        return 1
    dump = argv[0]; mode = argv[1] if len(argv) > 1 else "--check"
    part = next((a for a in argv if a.startswith("--part=")), None)
    if mode == "--check":
        d = build(dump, hours=True)
        if not known(d):
            print("[STOP] a known number failed; nothing written."); return 1
        np.savez_compressed(CACHE, names=d["names"], cols=np.array(d["cols"]), TA=d["TA"], TB=d["TB"], profA=d["profA"], profB=d["profB"], producer=sha(__file__), dump=os.path.abspath(dump))
        C = len(d["cols"]); print(f"\n{C} columns, {len(d['names'])} hypotheses (top-vs-bottom and range), built in {d['secs']:.0f}s. NO CONTRAST IS PRINTED IN THIS MODE.")
        print(f"day axis (day-ints, 10 days of margin each side): block A {d['ax']['A']}, block B {d['ax']['B']}; the 29 separated anchors were derived in 26.77 on A (14767, 17529), B (18995, 20671)")
        dr = pd.Series(d["dropped"]); un = pd.Series(d["units"])
        print(f"units (cell-blocks) per column: min {int(un.min())}, median {int(un.median())}, max {int(un.max())}; columns present in fewer than 288 units: " + (", ".join(f"{k} ({un[k]})" for k in un.index[un < 288]) or "none"))
        print(f"columns with any dropped unit: {int((dr > 0).sum())}; with more than half their units dropped: " + (", ".join(f"{k} ({dr[k]} of {un[k]})" for k in dr.index[dr > un / 2]) or "none"))
        nb = {b: (T[:C, 1].sum(1), T[:C, 3].sum(1)) for b, T in (("A", d["TA"]), ("B", d["TB"]))}
        nm = np.minimum(d["TA"][C:, 3].sum(1), d["TB"][C:, 3].sum(1)); tot = d["TA"][C:, 1].sum(1) + d["TB"][C:, 1].sum(1) + d["TA"][C:, 3].sum(1) + d["TB"][C:, 3].sum(1)
        ms = (d["TA"][C:, 3].sum(1) + d["TB"][C:, 3].sum(1)) / np.maximum(tot, 1)
        print(f"range contrast, middle group's share of the rows it uses: median {100*np.median(ms[nm > 0]):.0f}%; NOT a range contrast (middle under 10% or over 90%): " + (", ".join(f"{c} ({100*x:.0f}%)" for c, x, k in zip(d["cols"], ms, nm) if k > 0 and not 0.10 <= x <= 0.90) or "none"))
        print(f"range contrast (middle vs ends): defined for {int((nm > 0).sum())} of {C} columns; no middle group: " + (", ".join(c for c, x in zip(d["cols"], nm) if x == 0) or "none"))
        print(f"group sizes over columns (bottom / top rows): block A median {int(np.median(nb['A'][0])):,} / {int(np.median(nb['A'][1])):,}; block B median {int(np.median(nb['B'][0])):,} / {int(np.median(nb['B'][1])):,}; "
              f"smallest group {int(min(nb['A'][0].min(), nb['A'][1].min(), nb['B'][0].min(), nb['B'][1].min())):,}")
        nf = np.array([max(d["nfin"].get(c, 0), 1) for c in d["cols"]]); sb = (nb["A"][0] + nb["B"][0]) / nf; st = (nb["A"][1] + nb["B"][1]) / nf
        odd = [f"{c} ({100*a:.0f}% / {100*b:.0f}%)" for c, a, b in zip(d["cols"], sb, st) if not (0.10 <= a <= 0.35 and 0.10 <= b <= 0.35)]
        print("columns whose bottom / top group is NOT between 10% and 35% of the rows used (ties or few values; their contrast is not 'top fifth vs bottom fifth'): " + (", ".join(odd) or "none"))
        print("ct_hour groups (Chicago clock hours, share of rows): bottom = " + ", ".join(f"{h}h {100*x/d['hrs'][1].sum():.0f}%" for h, x in enumerate(d["hrs"][1]) if x / d["hrs"][1].sum() > 0.02)
              + " | top = " + ", ".join(f"{h}h {100*x/d['hrs'][2].sum():.0f}%" for h, x in enumerate(d["hrs"][2]) if x / d["hrs"][2].sum() > 0.02))
        print("SE of the contrast by bootstrap kind, as a ratio to the iid-date SE (median over columns; block A / block B). DESCRIPTIVE: no block length is chosen from this table; the rule uses iid, 28 and 56 together.")
        base = {}
        for kind in ("iid", 7, 14, 28, 56):
            se = []
            for bi, T in enumerate((d["TA"], d["TB"])):
                th, drw = boot(T, kind, (SEED if kind == "iid" else 0) + 1 + bi, 1500)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore"); se.append(np.nanstd(drw, 0))
            if kind == "iid":
                base = se; print(f"   iid dates: SE median {100*np.nanmedian(se[0]):.3f} pp (A), {100*np.nanmedian(se[1]):.3f} pp (B)")
            else:
                ra, rb = se[0] / base[0], se[1] / base[1]
                print(f"   blocks of {kind:>2} calendar days: median {np.nanmedian(ra):.2f} / {np.nanmedian(rb):.2f}; max {np.nanmax(ra):.2f} ({d['names'][np.nanargmax(ra)]}) / {np.nanmax(rb):.2f} ({d['names'][np.nanargmax(rb)]})")
        print(f"wrote {os.path.basename(CACHE)}")
        return 0
    if mode.startswith("--worlds"):
        n = int(mode.split("=")[1]); js = list(range(n))
        if part:
            k, m = part.split("=")[1].split("/"); js = [j for j in js if j % int(m) == int(k) - 1]
        os.makedirs(WDIR, exist_ok=True)
        for j in js:
            f = os.path.join(WDIR, f"world_{j:03d}.csv")
            if os.path.exists(f) and str(pd.read_csv(f, usecols=["producer"]).producer.iloc[0]) == sha(__file__):
                continue
            d = build(dump, world=j); r, drop, H = analyse(d["TA"], d["TB"])
            r.insert(0, "col", d["names"]); r["world"] = j; r["H"] = H; r["producer"] = sha(__file__); r.to_csv(f, index=False)
            print(f"world {j}: {d['secs']:.0f}s, family {H}, written (nothing printed)", flush=True)
        return 0
    if mode == "--summarize-worlds":
        return summarize()
    if mode.startswith("--recall"):
        j = int(mode.split("=")[1]) if "=" in mode else 0
        print(f"PLANTED EFFECTS ON DEAD FRAME {j} (real outcomes; flips of resolved misses in the column's top group). Counts only: NO sentence 'the test sees effects above X' may be written from this.")
        for col in ("speed", "relvol"):
            for lab, pl in (("+1.0 pp both periods, row flips", (col, 0.010, 0.010, False)), ("+2.0 pp both periods, row flips", (col, 0.020, 0.020, False)),
                            ("+2.0 pp older / +0.6 pp recent, whole-day flips", (col, 0.020, 0.006, True)), ("+2.0 pp older period only (must NOT pass)", (col, 0.020, 0.0, False))):
                d = build(dump, world=j, plant=pl); cf = col_factors(d["names"]); fA, fB = (cf[0], cf[1]) if cf else (None, None)
                r, _, H = analyse(d["TA"], d["TB"], fA=fA, fB=fB); i = list(d["names"]).index(col)
                print(f"  {col:<7} {lab:<48} thA {r.thA[i]:+.2f} (t {r.tA[i]:+.1f}), thB {r.thB[i]:+.2f} (t {r.tB[i]:+.1f}), adj {r.adjA[i]:.4f} / {r.adjB[i]:.4f} -> "
                      f"{'PASSES' if r.passes[i] else 'does not pass'}; its own range row passes: {bool(r.passes[i + len(d['cols'])])} (an echo if so); other hypotheses passing: {int(r.passes.sum()) - int(r.passes[i]) - int(r.passes[i + len(d['cols'])])}" + ("" if cf else "  [no world factors on disk: factor 1]"), flush=True)
        return 0
    if mode == "--run":
        z = load_cache(dump); names = list(z["names"]); cf = col_factors(names)
        if cf is None:
            print("[STOP] fewer than 29 separated dead-frame worlds from THIS version of the script: the real result is not computed."); return 1
        fA, fB, miss, nw = cf; r, drop, H = analyse(z["TA"], z["TB"], fA=fA, fB=fB)
        r.insert(0, "col", names); C = len(z["cols"]); mc = mid_class(z["TA"], z["TB"], C)
        r["kind"] = ["top-vs-bottom"] * C + ["range"] * C; r["mid_class"] = [""] * C + mc
        r["finding"] = r.passes & ((r.kind == "top-vs-bottom") | r.mid_class.isin(["middle-peak", "both-ends"]))
        st = stamp(dump, f" | dead-frame factors from {nw} worlds: f = max(1, ratio) per contrast, median {np.median(np.r_[fA, fB]):.3f}, max {max(fA.max(), fB.max()):.3f}")
        print("\n" + st + f"\nfamily = {H} block contrasts ({H // 2} conditions); dropped as undefined: {[names[i] for i in drop]}; columns absent from the worlds (given the largest factor): {miss}\n")
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
        r["ciA"] = r.apply(lambda x: f"[{x.thA - 1.96 * x.seA:+.2f}, {x.thA + 1.96 * x.seA:+.2f}]", axis=1); r["ciB"] = r.apply(lambda x: f"[{x.thB - 1.96 * x.seB:+.2f}, {x.thB + 1.96 * x.seB:+.2f}]", axis=1)
        show = r.assign(_m=np.minimum(r.tA.abs(), r.tB.abs())).sort_values("_m", ascending=False)
        print(show[["col", "thA", "ciA", "tA", "fA", "adjA", "thB", "ciB", "tB", "fB", "adjB", "passes", "mid_class", "nbotA", "ntopA", "nbotB", "ntopB"]].round(3).to_string(index=False))
        one = r[((r.adjA <= 0.05) ^ (r.adjB <= 0.05))]
        tb = r[(r.kind == "top-vs-bottom") & r.passes]; rg = r[(r.kind == "range") & r.finding]; ec = r[(r.kind == "range") & r.passes & ~r.finding]
        print(f"\nHEADLINE, top-vs-bottom: {len(tb)} of {C} conditions pass (both periods, same sign): " + ", ".join(tb.col))
        print(f"SECONDARY, range (added after Stage-1 shapes had been seen): {len(rg)} pass AND are classed middle-peak or both-ends: " + ", ".join(f"{c} ({m})" for c, m in zip(rg.col, rg.mid_class)))
        print(f"range rows that pass only as the ECHO of a one-tail effect (NOT a range finding, not listed for replication): {len(ec)}: " + ", ".join(ec.col))
        print(f"rejected in ONE period only (not a pass): {len(one)}: " + ", ".join(f"{c} ({'A' if a <= 0.05 else 'B'})" for c, a in zip(one.col, one.adjA)))
        print("CIs are 95% unadjusted intervals from the SE used (largest of iid / 28-day / 56-day blocks, times the dead-frame factor); the verdict uses the family-wise adjusted p. "
              "theta = percentage points of race rate among RESOLVED setups, top group minus bottom group. EXPLORATORY, IN-SAMPLE, NOT the registered A5(2); a non-pass is not 'no effect'.")
        print("\nFIVE-FIFTHS PROFILE, DESCRIPTIVE, NO VERDICT: mean centred race rate (pp) in each fifth of the condition, lowest fifth first; older period | recent period")
        with np.errstate(invalid="ignore", divide="ignore"):
            pa = 100 * z["profA"][:, :, 0] / z["profA"][:, :, 1]; pb = 100 * z["profB"][:, :, 0] / z["profB"][:, :, 1]
        for c, a, b in zip(z["cols"], pa, pb):
            if np.isfinite(a).all() and np.isfinite(b).all():
                print(f"  {str(c):<22}" + " ".join(f"{x:+6.2f}" for x in a) + "  | " + " ".join(f"{x:+6.2f}" for x in b))
        out = os.path.join(HERE, f"pooled_condition_result_{time.strftime('%Y-%m-%d')}.csv"); r.assign(stamp=st).to_csv(out, index=False)
        print(f"wrote {os.path.basename(out)}"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1:]))
