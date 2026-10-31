"""
gates.py -- the CODE gates. Every one reproduces a KNOWN answer (rule D3) on
SYNTHETIC data built on the spot, so they need no real data, run in ~1-2
minutes anywhere, and can run BEFORE a rebuild as well as after.

    python code\\gates.py            run every gate          -> PASS / FAIL each
    python code\\gates.py mutate     break each target on purpose; every gate
                                    must then FAIL (C113: a detector that
                                    passes a broken function tests nothing)

Known answers come from a SECOND, plain implementation written here (a loop,
a brute force, a direct sum) or from hand-checkable numbers -- never from the
production code checking itself.
"""
import os
import sys
import tempfile
import datetime as dt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_TMP = tempfile.mkdtemp(prefix="vwap_gates_")

import loader, spec, safeio, rearm, linearity, presignal, synthdata          # noqa: E402
import pipeline, walk_fast, controls_fast, add_covariates                      # noqa: E402
from rearm_reference import reference_cell                                     # noqa: E402

RESULTS = []


def chk(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<52} {detail}", flush=True)


def synth():
    p = os.path.join(_TMP, "outright.parquet")
    if not os.path.exists(p):
        synthdata.make_outright(p)
    return loader.load("train", raw=p)


# ------------------------------------------------------------------ G1 spec
def g_spec(S=spec):
    ok = (S.XS == [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
          and S.TAUS == [0.0, 0.10, 0.25, 0.50, 1.00]
          and S.ATR_EPS == [3.0, 4.0, 6.0, 8.0, 12.0] and S.SD_EPS == [1.0, 1.5, 2.0, 3.0]
          and S.NS == [60, 100, 160, 200, 240]
          and [S.t_stop("ATR", e) for e in S.ATR_EPS] == [54, 95, 214, 381, 857]
          and [S.t_stop("SD", e) for e in S.SD_EPS] == [30, 68, 120, 270])
    chk("G1 locked constants have their decided VALUES", ok, "XS 9-grid, TAUS, grids, T")


# ------------------------------------------------------------------ G2 time
def g_time(trade_day=loader.trade_day, bar_features=add_covariates.bar_features):
    cases = [  # (UTC, is_rth, trade date)
        ("2024-01-10 14:29", 0, "2024-01-10"), ("2024-01-10 14:30", 1, "2024-01-10"),
        ("2024-07-10 13:29", 0, "2024-07-10"), ("2024-07-10 13:30", 1, "2024-07-10"),
        ("2024-01-10 20:59", 1, "2024-01-10"), ("2024-01-10 21:00", 0, "2024-01-10"),
        ("2024-07-10 19:59", 1, "2024-07-10"), ("2024-07-10 20:00", 0, "2024-07-10"),
        ("2024-03-11 13:30", 1, "2024-03-11"), ("2024-11-04 13:30", 0, "2024-11-04"),
        ("2024-11-04 14:30", 1, "2024-11-04"), ("2024-01-13 15:00", 0, "2024-01-13"),
        ("2024-01-07 23:00", 0, "2024-01-08"), ("2024-01-08 23:00", 0, "2024-01-09"),
        ("2024-07-07 22:00", 0, "2024-07-08")]
    ts = pd.Series(pd.to_datetime([c[0] for c in cases]).tz_localize("UTC"))
    n = len(ts)
    d = pd.DataFrame(dict(ts=ts, close=100.0, high=101.0, low=99.0, atr=1.0, tr=1.0,
                          seg=1, exp_day=20000))
    bf = bar_features(d)
    td = trade_day(ts)
    want_td = np.array([(dt.date.fromisoformat(c[2]) - dt.date(1970, 1, 1)).days for c in cases])
    rth_ok = int((bf["is_rth"] == np.array([c[1] for c in cases])).sum())
    td_ok = int((td == want_td).sum())
    chk("G2 RTH (New York, both DST regimes) and CME trade date",
        rth_ok == n and td_ok == n, f"RTH {rth_ok}/{n}, trade date {td_ok}/{n}")


def g_minutes(minutes=loader.minutes_since_epoch):
    ts = pd.Series(pd.to_datetime(["2024-01-10 14:29", "2024-01-13 15:00"]).tz_localize("UTC"))
    us = ts.dt.as_unit("us"); ns = ts.dt.as_unit("ns")
    same = np.array_equal(minutes(us), minutes(ns)) and \
        minutes(ns)[0] == int(pd.Timestamp('2024-01-10 14:29', tz='UTC').timestamp() // 60)
    chk("G2b minutes are unit-proof (us == ns) (AUD-20)", same)


# ------------------------------------------------------------------ G3 expiry
def g_expiry(expiry_day=loader.expiry_day):
    cases = [("ESH4", "2023-12-11", "2024-03-15"), ("ESZ3", "2023-09-11", "2023-12-15"),
             ("ESM0", "2010-03-12", "2010-06-18"), ("ESU2", "2022-06-10", "2022-09-16"),
             ("ESH0", "2010-01-04", "2010-03-19"), ("ESZ7", "2017-09-11", "2017-12-15")]
    ok = sum((dt.date(1970, 1, 1) + dt.timedelta(days=expiry_day(s, dt.date.fromisoformat(f))))
             == dt.date.fromisoformat(w) for s, f, w in cases)
    chk("G3 contract expiry = 3rd Friday (R7)", ok == len(cases), f"{ok}/{len(cases)} known dates")


# ------------------------------------------------------------------ G4 anchors
def g_anchor(d, rae=loader.rolling_argextreme):
    seg = d.seg.to_numpy(); bad = 0; tot = 0
    for col, mx, N in (("high", True, 60), ("low", False, 60), ("volume", True, 240)):
        arr = d[col].to_numpy().astype(float)
        a = rae(arr, N, seg, mx)
        starts = np.r_[0, np.where(seg[1:] != seg[:-1])[0] + 1]
        s0 = starts[np.searchsorted(starts, np.arange(len(d)), "right") - 1]
        for i in np.random.default_rng(5).choice(len(d), 3000, replace=False):
            if i - s0[i] < N - 1:
                ref = -1
            else:
                w = arr[i - N + 1:i + 1][::-1]
                ref = i - int(np.argmax(w) if mx else np.argmin(w))
            bad += int(a[i] != ref); tot += 1
    chk("G4 rolling anchor == brute force (latest bar wins ties)", bad == 0, f"{tot - bad}/{tot}")


# ------------------------------------------------------------------ G5 VWAP / SD
def g_vwap(d, vs=pipeline.vwap_and_sd):
    a = loader.rolling_argextreme(d.high.to_numpy().astype(float), 240, d.seg.to_numpy(), True)
    vw, sd, *_ = vs(d, a)
    vp = d.vp.to_numpy().astype(float); v = d.volume.to_numpy().astype(float)
    wv = ws = 0.0; one = []
    for i in np.random.default_rng(2).choice(np.where(a >= 0)[0], 3000, replace=False):
        s_ = slice(a[i], i + 1)
        m = (vp[s_] * v[s_]).sum() / v[s_].sum()
        sdd = np.sqrt(max((v[s_] * (vp[s_] - m) ** 2).sum() / v[s_].sum(), 0.0))
        wv = max(wv, abs(vw[i] - m))
        if i > a[i]:
            ws = max(ws, abs(sd[i] - sdd))
        else:
            one.append(sd[i])
    chk("G5 PRODUCTION VWAP == direct sum (AUD-24)", wv < 1e-6, f"max dev {wv:.1e} pts")
    chk("G5b PRODUCTION SD == direct sum, lines of >=2 bars (AUD-38)", ws < 1e-6, f"max dev {ws:.1e} pts")
    one = np.array(one)
    chk("G5c one-bar line has SD exactly 0 (AUD-37)", len(one) > 0 and (one == 0).all(),
        f"{int((one > 0).sum())}/{len(one)} one-bar lines have sd>0, max {one.max() if len(one) else 0:.2g} pts")


def g_exact(d, vs=pipeline.vwap_and_sd):
    """The SD against Python's unlimited-precision integers -- the only
    reference that cannot round (AUD-38, decision B)."""
    from fractions import Fraction
    a = loader.rolling_argextreme(d.high.to_numpy().astype(float), 240, d.seg.to_numpy(), True)
    vw, sd, *_ = vs(d, a)
    q, v, ok_grid = pipeline.tick_ints(d)
    worst_v = worst_s = 0.0
    for i in np.random.default_rng(7).choice(np.where(a >= 0)[0], 400, replace=False):
        j = int(a[i])
        Q = [int(x) for x in q[j:i + 1]]; V = [int(x) for x in v[j:i + 1]]
        W = sum(V)
        if W <= 0:
            continue
        M1 = Fraction(sum(x * w for x, w in zip(Q, V)), W)          # exact
        M2 = Fraction(sum(x * x * w for x, w in zip(Q, V)), W)
        var = M2 - M1 * M1                                          # exact, in q^2 units
        worst_v = max(worst_v, abs(float(M1) / spec.PRICE_SCALE - vw[i]))
        worst_s = max(worst_s, abs(float(var) ** 0.5 / spec.PRICE_SCALE - sd[i]))
    chk("G5d VWAP and SD == exact rational arithmetic (AUD-38)",
        ok_grid and worst_v < 1e-9 and worst_s < 1e-9,
        f"on the 0.25 grid: {ok_grid} | max dev: vwap {worst_v:.1e}, sd {worst_s:.1e} pts")
    # a FLAT window must give exactly zero, not a crumb
    flat = d.iloc[:400].copy()
    for c_ in ("high", "low", "close"):
        flat[c_] = 4000.0
    flat["vp"] = 4000.0
    flat["seg"] = 1
    a2 = np.zeros(len(flat), dtype=np.int64)
    _, sd2, *_ = vs(flat, a2)
    chk("G5e a flat window has SD exactly 0", bool((sd2 == 0).all()),
        f"max {sd2.max():.2e} pts over {len(sd2)} bars")


# ------------------------------------------------------------------ G6 walk (setups + twins)
def _loop(close, hi, lo, seg, s, C0, G0, a0, T, h_rec):
    """Plain loop over bars 1..h_rec for ONE episode with a FROZEN level."""
    n = len(close); sg = np.sign(G0); V0 = C0 - G0; M0 = C0 + G0
    w0 = t0 = mir = -1; adv = {x: -1 for x in spec.XS}
    run_c = run_w = -np.inf; mae_c = mae_w = mae_b = np.nan; nvalid_T = 0
    for j in range(1, h_rec + 1):
        t = s + j
        if t >= n or seg[t] != seg[s]:
            break
        if j <= T:
            nvalid_T += 1
        g = close[t] - V0
        if w0 < 0 and ((lo[t] <= V0) if sg > 0 else (hi[t] >= V0)):
            w0 = j
            if j <= T:
                mae_c = max(run_c, sg * (close[t] - C0) / a0)
                mae_w = run_w if j >= 2 else np.nan
                mae_b = ((hi[t] - C0) if sg > 0 else (C0 - lo[t])) / a0
        if t0 < 0 and (np.sign(g) != sg or abs(g) <= 0.0):
            t0 = j
        if mir < 0 and ((hi[t] >= M0) if sg > 0 else (lo[t] <= M0)):
            mir = j
        if np.sign(g) == sg:
            for x in spec.XS:
                if adv[x] < 0 and (abs(g) - abs(G0)) / a0 >= x:
                    adv[x] = j
        if j <= T and w0 < 0:
            run_c = max(run_c, sg * (close[t] - C0) / a0)
            run_w = max(run_w, ((hi[t] - C0) if sg > 0 else (C0 - lo[t])) / a0)
    v = 1 <= w0 <= T; m = 1 <= mir <= T
    race = 9 if not (v or m) else (0 if v and m and w0 == mir else
                                   (1 if v and (not m or w0 < mir) else -1))
    return dict(w0=w0, t0=t0, mir=mir, race=race, adv=adv, mae_c=mae_c, mae_w=mae_w,
                mae_b=mae_b, cens=nvalid_T < T)


def g_walk(d, walk=walk_fast.walk, walk_static=controls_fast.walk_static):
    close = d.close.to_numpy().astype(float); hi = d.high.to_numpy().astype(float)
    lo = d.low.to_numpy().astype(float); seg = d.seg.to_numpy(); atr = d.atr.to_numpy()
    a = loader.rolling_argextreme(hi, 60, seg, True)
    vw, sd, W, cum, ref = pipeline.vwap_and_sd(d, a)
    ok_ = np.where((a >= 0) & (np.abs(close - vw) > 1.0 * atr) & (np.arange(len(d)) < len(d) - 700))[0]
    rb = np.where(seg[1:] != seg[:-1])[0]                     # last bars before rolls
    sig = np.unique(np.r_[ok_[::5], rb[rb > 1000] - 3])
    T, h_rec = 60, 180
    g0 = close[sig] - vw[sig]
    ew = walk(close, cum, ref, seg, sig, a[sig], atr[sig], g0, W[sig], h_rec, T=T, hi=hi, lo=lo)
    J = sig[::-1].copy(); G0j = np.sign(g0[::-1]) * 1.7 * atr[J]
    cw = walk_static(close, seg, J, atr[J], G0j, h_rec, T=T, hi=hi, lo=lo)
    for tag, bars, G0s, out, wk, tk, ak in (("setups", sig, g0, ew, "wf_0.0", "tf_0.0", "af_"),
                                           ("twins ", J, G0j, cw, "w_0.0", "tl_0.0", "al_")):
        bad = {}
        for q, s in enumerate(bars):
            r = _loop(close, hi, lo, seg, s, close[s], G0s[q], atr[s], T, h_rec)
            pairs = [("w", r["w0"], out[wk][q]), ("t", r["t0"], out[tk][q]),
                     ("mir", r["mir"], out["mir_w"][q]), ("race", r["race"], out["race"][q]),
                     ("cens", r["cens"], out["cens_T"][q])]
            pairs += [(f"adv{x}", r["adv"][x], out[f"{ak}{x}"][q]) for x in spec.XS]
            for nm in ("mae_c", "mae_w", "mae_b"):
                key = {"mae_c": "mae_fill_close", "mae_w": "mae_fill_wick", "mae_b": "mae_fillbar_wick"}[nm]
                x1, x2 = r[nm], out[key][q]
                if not ((np.isnan(x1) and np.isnan(x2)) or abs(x1 - x2) < 1e-9):
                    bad[nm] = bad.get(nm, 0) + 1
            for nm, x1, x2 in pairs:
                if x1 != x2:
                    bad[nm] = bad.get(nm, 0) + 1
        nf = int(((out[wk] >= 1) & (out[wk] <= T)).sum())
        chk(f"G6 walk == plain loop, {tag} (fill, race, mae, adv, cens)", not bad,
            f"{len(bars)} episodes, {nf} filled" + (f"  MISMATCH {bad}" if bad else ""))


# ------------------------------------------------------------------ G7 re-arm vs slow reference
def g_rearm(d, greedy=rearm.greedy_keep, elig_fn=rearm.eligible):
    cells = [(60, "hh", "ATR", 3.0), (60, "ll", "SD", 1.5), (240, "hv", "ATR", 4.0),
             (60, "session", "ATR", 6.0), (240, "hh", "SD", 1.0)]
    close = d.close.to_numpy(); hi = d.high.to_numpy().astype(float)
    lo = d.low.to_numpy().astype(float); seg = d.seg.to_numpy(); atr = d.atr.to_numpy()
    sess = pipeline.session_anchor(d)
    good = 0; detail = []
    for N, ak, tg, eps in cells:
        col, mx = {"hh": ("high", True), "ll": ("low", False), "hv": ("volume", True)}.get(ak, (None, None))
        a = sess if ak == "session" else loader.rolling_argextreme(d[col].to_numpy().astype(float), N, seg, mx)
        vw, sd, *_ = pipeline.vwap_and_sd(d, a)
        T = spec.t_stop(tg, eps); h_rec = spec.H_REC_MULT * T
        elig = elig_fn(d, a, vw, sd, atr, tg)
        scale = sd if tg == "SD" else atr
        dist = np.where(elig, np.abs(close - vw) / np.maximum(scale, 1e-9), np.nan)
        cand = rearm.candidates(dist, a, elig, eps, len(d), h_rec)
        f = rearm.first_fill(lo, hi, seg, cand, vw[cand], np.sign(close[cand] - vw[cand]), T)
        k = greedy(cand, f, T)
        ref = reference_cell(d, N, ak, tg, eps)
        same = np.array_equal(cand[k], ref["kept"]) and np.array_equal(f[k], ref["fills"])
        good += same
        detail.append(f"{ak}{N}/{tg}{eps}: {len(cand)}->{len(k)}" + ("" if same else f" vs ref {len(ref['kept'])}"))
    chk("G7 re-arm (fast) == slow independent reference", good == len(cells),
        f"{good}/{len(cells)} cells exact | " + "; ".join(detail[:3]))


# ------------------------------------------------------------------ G8 linearity
def g_lin(lin_of=linearity.lin_of):
    s0 = lin_of(np.linspace(100, 110, 41), 1.0)[0]
    V = np.linspace(100, 110, 41); V[17] += 2.5
    s1 = lin_of(V, 1.0)[0]
    s2 = lin_of(np.linspace(100, 110, 41) * 1.0, 2.0)[2]          # slope 0.25/bar / ATR 2
    chk("G8 linearity: straight line 0, hump h = h, chord slope", abs(s0) < 1e-12 and
        abs(s1 - 2.5) < 1e-12 and abs(s2 - 0.125) < 1e-12, f"{s0:.3g} | {s1:.6f} (h=2.5) | {s2}")


def g_linpath(d):
    a = loader.rolling_argextreme(d.high.to_numpy().astype(float), 60, d.seg.to_numpy(), True)
    vw, sd, W, cum, ref = pipeline.vwap_and_sd(d, a)
    worst = 0.0
    for i in np.random.default_rng(3).choice(np.where(a >= 0)[0], 300, replace=False):
        V = linearity.line_path(cum, ref, a[i], i)
        vp = d.vp.to_numpy().astype(float)[a[i]:i + 1]
        v = d.volume.to_numpy().astype(float)[a[i]:i + 1]      # float64: the REFERENCE must be exact
        worst = max(worst, float(np.max(np.abs(V - np.cumsum(vp * v) / np.cumsum(v)))))
    chk("G8b linearity's line path == direct running VWAP", worst < 1e-6, f"max dev {worst:.1e}")


# ------------------------------------------------------------------ G9 own-line crossings
def g_cross(d, lf=add_covariates.line_features):
    a = loader.rolling_argextreme(d.high.to_numpy().astype(float), 60, d.seg.to_numpy(), True)
    L_all, L_set, slope, since, pt = lf(d, a)
    close = d.close.to_numpy().astype(float); vp = d.vp.to_numpy().astype(float)
    v = d.volume.to_numpy().astype(float)                      # float64 reference
    bad = 0; expiry = 0; tot = 0
    for i in np.random.default_rng(0).choice(np.where((a >= 0) & (np.arange(len(d)) - a > 20))[0], 500, replace=False):
        V = np.cumsum(vp[a[i]:i + 1] * v[a[i]:i + 1]) / np.cumsum(v[a[i]:i + 1])
        s = np.sign(close[a[i]:i + 1] - V)
        cr = np.r_[False, (s[1:] != s[:-1]) & (s[1:] != 0) & (s[:-1] != 0)]
        last = np.where(cr)[0]
        ex_since = (len(s) - 1 - last[-1]) if len(last) else -1
        bad += int(since[i] != ex_since or pt[i] != cr.sum()); tot += 1
        expiry += int(a[i - 1] != a[i] or (a[a[i]:i + 1] != a[i]).any())
    chk("G9 bars_since_touch, prior_touches on the OWN line (AUD-21/22)", bad == 0,
        f"{tot - bad}/{tot} exact ({expiry} with an anchor switch inside)")


# ------------------------------------------------------------------ G10 day_pos
def g_daypos(d, bf=add_covariates.bar_features):
    B = bf(d)
    td = loader.trade_day(d.ts); seg = d.seg.to_numpy()
    key = pd.Series(td.astype(str)) + "|" + pd.Series(seg.astype(str))
    grp = (key != key.shift()).cumsum().to_numpy()
    hi = pd.Series(d.high.to_numpy()).groupby(grp).cummax().to_numpy()
    lo = pd.Series(d.low.to_numpy()).groupby(grp).cummin().to_numpy()
    ex = (d.close.to_numpy() - lo) / np.maximum(hi - lo, 1e-9)
    bad = int((np.abs(B["day_pos"] - ex) > 1e-4).sum())
    rolls = int((seg[1:] != seg[:-1]).sum())
    mid = int(sum(td[i] == td[i + 1] for i in np.where(seg[1:] != seg[:-1])[0]))
    chk("G10 day_pos per (trade date, contract) (AUD-23)", bad == 0,
        f"{bad} wrong bars | {mid}/{rolls} rolls fall INSIDE a trade date (the case that broke)")


# ------------------------------------------------------------------ G11 twins
def g_twins(d, draw=controls_fast.draw_controls):
    from control_matched import strata
    a = loader.rolling_argextreme(d.high.to_numpy().astype(float), 60, d.seg.to_numpy(), True)
    sig = np.where((a >= 0) & d.eligible.to_numpy())[0][::40]
    for mom in (False, True):
        lab = strata(d, with_momentum=mom)
        src, ctl, cs, cd = draw(d, sig, np.ones(len(sig)), np.full(len(sig), 3.0), lab, K=2, min_gap=500)
        td = loader.trade_day(d.ts); blk = d.block.to_numpy()
        v1 = int((td[ctl] == td[src]).sum()); v2 = int((np.abs(ctl - src) <= 500).sum())
        v3 = int((blk[ctl] != blk[src]).sum()); v4 = int((lab[ctl] != lab[src]).sum())
        chk(f"G11 twins {'M2 ' if mom else 'M1b'}: other trade date, >h_rec away, same block+stratum",
            len(ctl) > 100 and v1 + v2 + v3 + v4 == 0,
            f"{len(ctl)} twins | same day {v1}, within h_rec {v2}, cross-block {v3}, other stratum {v4}")


# ------------------------------------------------------------------ G12 pre-signal ER, safeio
def g_misc(build_pre=presignal.build_pre, load_npz=safeio.load_npz):
    c = np.array([100, 101, 102, 103, 100, 102, 100, 102.0]); seg = np.ones(8, int)
    PRE, cs = build_pre(np.r_[np.zeros(120), c], np.ones(128, int))
    # ER over the 3 bars ending at 123: 100,101,102,103 -> 1.0 ; at 127 (100,102,100,102) -> 1/3
    k3 = lambda i, k: abs(np.r_[np.zeros(120), c][i] - np.r_[np.zeros(120), c][i - k]) / \
        np.abs(np.diff(np.r_[np.zeros(120), c][i - k:i + 1])).sum()
    ok = abs(k3(123, 3) - 1.0) < 1e-12 and abs(k3(127, 3) - 1 / 3) < 1e-12 \
        and abs(PRE["er_pre15"][127] - k3(127, 15)) < 1e-6
    chk("G12 efficiency ratio known numbers (1.0 and 1/3)", ok)
    p = os.path.join(_TMP, "x.parquet"); safeio.write_parquet(pd.DataFrame({"a": [1, 2]}), p)
    q = os.path.join(_TMP, "c.npz"); safeio.save_npz(q, a=np.arange(3))
    old = os.path.join(_TMP, "old.npz")
    with open(old, "wb") as fh:
        np.savez_compressed(fh, a=np.arange(3))
    try:
        load_npz(old); refused = False
    except safeio.StaleCache:
        refused = True
    chk("G12b atomic writes; a cache from another spec is REFUSED",
        pd.read_parquet(p).a.sum() == 3 and load_npz(q)["a"].sum() == 3 and refused
        and not os.path.exists(p + ".tmp"))


def run_all_gates():
    print("building synthetic data ...", flush=True)
    d = synth()
    g_spec(); g_time(); g_minutes(); g_expiry(); g_anchor(d); g_vwap(d); g_exact(d); g_walk(d); g_rearm(d)
    g_lin(); g_linpath(d); g_cross(d); g_daypos(d); g_twins(d); g_misc()
    return RESULTS


def mutate():
    """Every gate, fed a deliberately BROKEN target, must FAIL.

    Only the TARGETED check counts: one call may print several lines (g_vwap
    prints three), and an untouched neighbour is expected to pass.
    """
    d = synth()
    RESULTS.clear()
    targets = []

    def T(prefix, fn, *a, **k):
        n0 = len(RESULTS)
        fn(*a, **k)
        hit = [(n, ok) for n, ok in RESULTS[n0:] if n.startswith(prefix)]
        assert hit, f"mutation for {prefix} produced no such check"
        targets.extend(hit)
    globals()["_T"] = T
    globals()["_TARGETS"] = targets
    import types
    S = types.SimpleNamespace(**{k: getattr(spec, k) for k in dir(spec) if not k.startswith("_")})
    S.XS = [1.0, 2.0, 4.0, 8.0]                                    # the AUD-30 regression
    _T("G1", g_spec, S)
    _T("G2 ", g_time, trade_day=lambda ts: loader.trade_day(ts) - 1)
    _T("G3", g_expiry, expiry_day=lambda s, f: loader.expiry_day(s, f) - 7)
    _T("G4", g_anchor, d, rae=lambda a, N, seg, mx: np.where(
        loader.rolling_argextreme(a, N, seg, mx) >= 0, np.arange(len(a)), -1))
    _T("G5 ", g_vwap, d, vs=lambda d_, a_: (lambda r: (r[0] + 3.0,) + r[1:])(pipeline.vwap_and_sd(d_, a_)))
    def bad_walk(*a, **k):
        o = walk_fast.walk(*a, **k); o["mae_fill_wick"] = o["mae_fill_close"]; return o
    def bad_static(*a, **k):                                       # race sides swapped
        o = controls_fast.walk_static(*a, **k); o["race"] = np.where(np.abs(o["race"]) == 1, -o["race"], o["race"]); return o
    _T("G6", g_walk, d, walk=bad_walk, walk_static=bad_static)
    def bad_greedy(cand, fill, T):                                 # ">=" instead of ">"
        keep, c_at = [], -1
        for k in range(len(cand)):
            if cand[k] >= c_at:
                keep.append(k); f = fill[k]; c_at = cand[k] + (f if 1 <= f <= T else T)
        return np.array(keep, dtype=np.int64)
    def worse_greedy(cand, fill, T):                               # ignores T cap
        return rearm.greedy_keep(cand, np.where(fill < 0, 3 * T, fill), 3 * T)
    _T("G7", g_rearm, d, greedy=worse_greedy)
    def wrong_minbars(d_, a_, vw_, sd_, atr_, tg):       # AUD-37 rule at the wrong value
        return rearm.eligible(d_, a_, vw_, sd_, atr_, tg) & ((np.arange(len(d_)) - a_) >= 4)
    _T("G7", g_rearm, d, elig_fn=wrong_minbars)
    def f32_sd(d_, a_):                                   # AUD-38 regression
        vw, sd, W, cum, ref = pipeline.vwap_and_sd(d_, a_)
        return vw, sd + 0.002, W, cum, ref
    _T("G5b", g_vwap, d, vs=f32_sd)
    def float_moments(d_, a_):                            # the pre-decision-B arithmetic
        vp = d_.vp.to_numpy().astype(float); v_ = d_.volume.to_numpy().astype(float)
        seg = d_.seg.to_numpy(); u = np.unique(seg)
        ref = np.array([dict(zip(u, vp[np.searchsorted(seg, u)]))[s_] for s_ in seg])
        x = vp - ref
        cV = np.r_[0.0, np.cumsum(v_)]; c1 = np.r_[0.0, np.cumsum(x * v_)]
        c2 = np.r_[0.0, np.cumsum(x * x * v_)]
        i_ = np.arange(len(d_)); safe = np.maximum(a_, 0); W = cV[i_ + 1] - cV[safe]
        m1 = np.where((a_ >= 0) & (W > 0), (c1[i_ + 1] - c1[safe]) / np.maximum(W, 1e-9), np.nan)
        m2 = np.where((a_ >= 0) & (W > 0), (c2[i_ + 1] - c2[safe]) / np.maximum(W, 1e-9), np.nan)
        return m1 + ref, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)), W, None, ref
    _T("G5d", g_exact, d, vs=float_moments)
    _T("G8 ", g_lin, lin_of=lambda V, a: (lambda r: (r[0] * 0.5,) + r[1:])(linearity.lin_of(V, a)))
    def old_lf(d_, a_):
        L, Ls, sl, since, pt = add_covariates.line_features(d_, a_)
        return L, Ls, sl, np.where(since > 5, since - 1, since), pt
    _T("G9", g_cross, d, lf=old_lf)
    def old_bf(d_):                                                # AUD-16 grouping, no contract split
        B = add_covariates.bar_features(d_)
        td = loader.trade_day(d_.ts); grp = np.cumsum(np.r_[True, td[1:] != td[:-1]])
        hi = pd.Series(d_.high.to_numpy()).groupby(grp).cummax().to_numpy()
        lo = pd.Series(d_.low.to_numpy()).groupby(grp).cummin().to_numpy()
        B["day_pos"] = ((d_.close.to_numpy() - lo) / np.maximum(hi - lo, 1e-9)).astype(np.float32)
        return B
    _T("G10", g_daypos, d, bf=old_bf)
    def bad_draw(d_, sig, sides, dists, lab, K=2, min_gap=0, **k):   # twin shares the path
        src, ctl, cs, cd = controls_fast.draw_controls(d_, sig, sides, dists, lab, K=K, min_gap=min_gap)
        return src, src + 1, cs, cd
    _T("G11", g_twins, d, draw=bad_draw)
    _T("G2b", g_minutes, minutes=lambda ts: (pd.Series(ts).astype("int64") // 60_000_000_000).to_numpy())
    def bad_pre(c, seg):
        P, cs = presignal.build_pre(c, seg); P["er_pre15"] = np.roll(P["er_pre15"], 1); return P, cs
    def lax_npz(path):
        z = np.load(path); return {k: z[k] for k in z.files if k != "_spec"}
    _T("G12", g_misc, build_pre=bad_pre, load_npz=lax_npz)
    return _TARGETS


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mutate":
        print("MUTATION TEST -- every line below must say FAIL")
        r = mutate()
        caught = sum(not ok for _, ok in r)
        print(f"\n{caught}/{len(r)} broken targets caught" +
              ("  -> ALL DETECTORS LIVE" if caught == len(r) else "  -> SOME DETECTOR IS BLIND"))
    else:
        r = run_all_gates()
        bad = [n for n, ok in r if not ok]
        print("\nALL GATES PASS" if not bad else f"\n{len(bad)} GATE(S) FAILED: {bad}")
