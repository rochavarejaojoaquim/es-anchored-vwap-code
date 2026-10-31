"""paper_yearly.py -- the effect by calendar year (TRAINING periods only) and two decay curves.

QUESTION. The paper reports claims 1 and 2 in three periods. Three points cannot show the shape of a
decline. This script splits the two TRAINING periods (2010-2017 and 2022-2026, data\\dump) into calendar
years, estimates claim 1 (M1b and M2 twins) and claim 2 for each year with the registered pooled
estimator and an iid day bootstrap (claim_checks.py functions, imported unchanged), and fits
    exponential  E(t) = E0 * exp(-lam * (t - 2010))          half-life ln2 / lam
    hyperbolic   E(t) = E0 / (1 + lam * (t - 2010))          (the "1/(1+lambda t)" form; E falls to
                                                              half of E0 after 1/lam years)
by weighted least squares (weights 1/SE^2), with a bootstrap interval for lam and the half-life.
The HELD-OUT period (2018-2021) is NOT re-read and NOT split: its already-published pooled estimate
(seal_use1_outputs\\claims_seal_2026-10-15.json) is used only as a single point to compare with what
the fit to the training years predicts. A held-out period read once for the two claims stays read once.

READING RULES, fixed here BEFORE any real run (they apply to every number printed):
  R1 DESCRIPTIVE. No verdict, no pass mark, nothing is tested. Training years only; the held-out point
     is a comparison, not an input to fit A.
  R2 lam and the half-life are printed with a 95% bootstrap interval. If the lower end of the lam interval
     is not above 0, the sentence is "a decline is not distinguished from zero"; if the share of
     bootstrap fits with lam <= 0.002 is above 2.5%, no finite half-life interval is quoted.
  R3 The held-out comparison prints, for each model, the curve's AVERAGE over 2018-2021 against the
     observed held-out estimate and its registered interval. "Closer" is a distance in pp, not a test.
  R4 The two models have the same number of parameters; the weighted sum of squares is printed for both.
     NO claim is made that either is the true shape.
  R5 Not causal. This is the decay of a price-touch difference against random matched levels, not an
     alpha, a return or a profit; whether it reflects crowding, the shift of distances in ticks or the
     twins' rate rising is NOT examined here (the paper's Section 3.2 does the decomposition).
  R6 Yearly estimates are post hoc and exploratory (the split was chosen after the three-period result
     was known). Years with fewer than 100 trade dates are printed and marked partial.

Usage:  python paper_yearly.py --selftest
        python paper_yearly.py data\\dump
Writes paper_yearly_<date>.json and paper_figure2_yearly_decay.svg (project root). Reads no sealed data.
"""
import datetime
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

T0 = 2010.0
LAMS = np.round(np.arange(-0.02, 1.0005, 0.001), 4)
HELD_OUT = (2018.0, 2022.0)       # 2018-01-01 .. 2021-12-31


def _as_days(day):
    """day-ints (days since 1970-01-01) or datetime.date / datetime64 objects -> datetime64[D]."""
    return np.asarray(day, dtype="datetime64[D]")


def day_to_year(day):
    return _as_days(day).astype("datetime64[Y]").astype(int) + 1970


def mid_time(days):
    d = _as_days(days).astype("int64").astype(float)
    return 1970.0 + (d.min() + d.max()) / 2.0 / 365.2425


# ---------------------------------------------------------------- the two curves
def basis(kind, lams, t):
    x = t - T0
    if kind == "exp":
        return np.exp(-np.outer(lams, x))
    if kind == "hyp":
        den = 1.0 + np.outer(lams, x)
        den = np.where(den <= 1e-9, np.nan, den)
        return 1.0 / den
    raise ValueError(kind)


def fit(kind, ymat, w, t):
    """ymat (B, K) or (K,), weights w (K,), times t (K,). Returns lam (B,), E0 (B,), sse (B,)."""
    y = np.atleast_2d(ymat)
    g = basis(kind, LAMS, t)                               # (G, K)
    swg2 = (w * g * g).sum(1)                              # (G,)
    swgy = (y * w) @ g.T                                   # (B, G)
    swy2 = (w * y * y).sum(1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        sse = swy2 - swgy ** 2 / swg2
    sse = np.where(np.isfinite(sse), sse, np.inf)
    j = sse.argmin(1)
    e0 = swgy[np.arange(len(y)), j] / swg2[j]
    return LAMS[j], e0, sse[np.arange(len(y)), j]


def halflife(kind, lam):
    lam = np.asarray(lam, float)
    out = np.full(lam.shape, np.inf)
    ok = lam > 0.002
    out[ok] = (np.log(2.0) / lam[ok]) if kind == "exp" else (1.0 / lam[ok])
    return out


def curve_mean(kind, lam, e0, a, b, n=400):
    t = np.linspace(a, b, n)
    return float(e0 * basis(kind, np.array([lam]), t)[0].mean())


def summarize(kind, y, ydraws, w, t):
    lam, e0, sse = fit(kind, y, w, t)
    lamb, _, _ = fit(kind, ydraws, w, t)
    hl = halflife(kind, lamb)
    fin = np.isfinite(hl)
    share_flat = float(1.0 - fin.mean())
    d = dict(kind=kind, lam=float(lam[0]), e0=float(e0[0]), sse=float(sse[0]),
             halflife=float(halflife(kind, lam)[0]),
             lam_lo=float(np.percentile(lamb, 2.5)), lam_hi=float(np.percentile(lamb, 97.5)),
             share_lam_le_0002=share_flat)
    if share_flat <= 0.025 and fin.any():
        d["halflife_lo"], d["halflife_hi"] = float(np.percentile(hl[fin], 2.5)), float(np.percentile(hl[fin], 97.5))
    else:
        d["halflife_lo"] = d["halflife_hi"] = None
    d["heldout_pred_pp"] = curve_mean(kind, d["lam"], d["e0"], *HELD_OUT)
    return d


# ---------------------------------------------------------------- per-year estimates
def yearly(df_arm, num_a, den_a, num_b, den_b, years, ratio_diff, day_codes, boot_draws):
    rows = []
    for yr in sorted(set(years)):
        s = df_arm[years == yr] if hasattr(years, "__len__") else None
        codes, D = day_codes(s.day)
        sums = {c: np.bincount(codes, weights=s[c].to_numpy(float), minlength=D)
                for c in {num_a, den_a, num_b, den_b}}
        est = 100 * ratio_diff(sums, num_a, den_a, num_b, den_b)
        dr = 100 * ratio_diff(sums, num_a, den_a, num_b, den_b, boot_draws(D, "iid"))
        dr = dr[np.isfinite(dr)]
        rows.append(dict(year=int(yr), n=int(len(s)), D=int(D), est=float(est),
                         lo=float(np.percentile(dr, 2.5)), hi=float(np.percentile(dr, 97.5)),
                         se=float(dr.std(ddof=1)), t=mid_time(s.day.to_numpy()), draws=dr))
    return rows


# ---------------------------------------------------------------- self-test
def selftest():
    ok_all = True

    def chk(lab, ok, det=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {lab}   {det}")

    print("SELF-TEST paper_yearly (toy data only, no dump)")
    # K1 date mapping: day 14610 is 2010-01-01 (days since 1970-01-01)
    chk("K1 day-to-year: 14610 -> 2010, 14609 -> 2009, 18993 -> 2022",
        list(day_to_year([14610, 14609, 18993])) == [2010, 2009, 2022])
    import datetime as _dt
    chk("K1c datetime.date objects (what the dump holds) map the same way as day-ints",
        list(day_to_year([_dt.date(2010, 1, 1), _dt.date(2009, 12, 31), _dt.date(2022, 1, 3)])) == [2010, 2009, 2022]
        and abs(mid_time([_dt.date(2012, 1, 1), _dt.date(2012, 12, 31)]) - mid_time([15340, 15705])) < 0.01)
    chk("K1b mid_time of the days of 2012 is about 2012.5",
        abs(mid_time([15340, 15705]) - 2012.5) < 0.05, f"{mid_time([15340, 15705]):.3f}")

    rng = np.random.default_rng(5)
    t = np.array([2010.4, 2011.5, 2012.5, 2013.5, 2014.5, 2015.5, 2016.5, 2017.5,
                  2022.5, 2023.5, 2024.5, 2025.5, 2026.4])
    se = np.full(len(t), 0.5)
    B = 1500

    def draws(truth):
        return truth + rng.normal(0, 0.5, size=(B, len(t)))

    # K2 exponential truth recovered
    lam0, e00 = 0.12, 5.0
    truth = e00 * np.exp(-lam0 * (t - T0))
    y = truth + rng.normal(0, 0.5, len(t))
    w = 1 / se ** 2
    d = summarize("exp", y, y[None, :] + rng.normal(0, 0.5, size=(B, len(t))), w, t)
    chk("K2 exponential decay lam 0.12 recovered, truth inside the 95% interval",
        abs(d["lam"] - lam0) < 0.04 and d["lam_lo"] < lam0 < d["lam_hi"],
        f"lam {d['lam']:.3f} [{d['lam_lo']:.3f}, {d['lam_hi']:.3f}], half-life {d['halflife']:.1f}")
    chk("K2b half-life = ln2/lam", abs(d["halflife"] - np.log(2) / d["lam"]) < 1e-9)
    # K3 hyperbolic truth recovered by the hyperbolic fit
    lam1, e01 = 0.30, 5.0
    truth_h = e01 / (1 + lam1 * (t - T0))
    yh = truth_h + rng.normal(0, 0.15, len(t))
    wh = np.full(len(t), 1 / 0.15 ** 2)
    dh = summarize("hyp", yh, yh[None, :] + rng.normal(0, 0.15, size=(B, len(t))), wh, t)
    chk("K3 hyperbolic lam 0.30 recovered", abs(dh["lam"] - lam1) < 0.1,
        f"lam {dh['lam']:.3f} [{dh['lam_lo']:.3f}, {dh['lam_hi']:.3f}], 1/lam {dh['halflife']:.1f}")
    # K4 held-out prediction is the curve mean over 2018-2022
    chk("K4 held-out prediction equals the curve's mean over 2018-2022",
        abs(d["heldout_pred_pp"] - curve_mean("exp", d["lam"], d["e0"], *HELD_OUT)) < 1e-9)
    # M1 mutation: no decay -> the lam interval must include 0 and NO finite half-life is quoted
    yf = 2.0 + rng.normal(0, 0.5, len(t))
    df_ = summarize("exp", yf, yf[None, :] + rng.normal(0, 0.5, size=(B, len(t))), w, t)
    chk("M1 flat truth: lam interval reaches 0 (no decline claimed)", df_["lam_lo"] <= 0.003,
        f"lam [{df_['lam_lo']:.3f}, {df_['lam_hi']:.3f}], share lam<=0.002 {df_['share_lam_le_0002']:.2f}")
    # M2 mutation: a rising truth gives lam <= 0
    yr = 1.0 + 0.3 * (t - T0) + rng.normal(0, 0.3, len(t))
    lr, _, _ = fit("exp", yr, w, t)
    chk("M2 rising truth: lam is not positive", lr[0] <= 0.0, f"lam {lr[0]:.3f}")
    # M3 weights matter: one tiny-SE point drags the fit
    w2 = w.copy(); w2[6] = 1e4
    y2 = y.copy(); y2[6] = 4.5
    l2, _, _ = fit("exp", y2, w2, t)
    l1, _, _ = fit("exp", y2, w, t)
    chk("M3 weights are used (an over-weighted point moves lam)", abs(l2[0] - l1[0]) > 0.01,
        f"{l1[0]:.3f} -> {l2[0]:.3f}")
    # K5 per-year estimator equals the registered pooled estimator on the same rows
    import pandas as pd
    import claim_checks as cc
    n_days, per = 200, 30
    day = np.repeat(np.arange(14610, 14610 + n_days), per)
    r = np.random.default_rng(3)
    toy = pd.DataFrame(dict(day=day, rf=(r.random(len(day)) < 0.55).astype(float),
                            tf=r.random(len(day)) * 0.5 + 0.05, w=1.0))
    pooled = cc.claim1_block(toy, "iid")
    yrs = day_to_year(toy.day.to_numpy())
    chk("K5 toy spans one calendar year", len(set(yrs)) == 1, str(sorted(set(yrs))))
    rows = yearly(toy, "rf", "w", "tf", "w", yrs, cc.ratio_diff, cc.day_codes, cc.boot_draws)
    chk("K5b per-year estimate equals claim1_block (same rows, same draws)",
        abs(rows[0]["est"] - pooled["est_pp"]) < 1e-9 and abs(rows[0]["lo"] - pooled["lo_pp"]) < 1e-9
        and abs(rows[0]["hi"] - pooled["hi_pp"]) < 1e-9,
        f"{rows[0]['est']:+.4f} [{rows[0]['lo']:+.4f}, {rows[0]['hi']:+.4f}]")
    print("  ALL DETECTORS ALIVE" if ok_all else "  SELF-TEST FAILED")
    return ok_all


# ---------------------------------------------------------------- figure
def svg(panels, held, out):
    W, H, L, R, T, Bm = 760, 300, 62, 20, 36, 40
    pw = W
    ph = H
    tot = 2 * ph + 20
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{tot}" viewBox="0 0 {W} {tot}" '
         f'font-family="sans-serif" font-size="12"><rect width="{W}" height="{tot}" fill="white"/>']
    for pi, (title, rows, fits, ho, ymin, ymax) in enumerate(panels):
        oy = pi * (ph + 20)

        def X(t):
            return L + (t - 2009.5) / (2027.0 - 2009.5) * (W - L - R)

        def Y(v):
            return oy + T + (ymax - v) / (ymax - ymin) * (ph - T - Bm)
        o.append(f'<text x="{W/2}" y="{oy+20}" text-anchor="middle" font-size="14">{title}</text>')
        v = int(np.ceil(ymin))
        while v <= ymax:
            o.append(f'<line x1="{L}" y1="{Y(v)}" x2="{W-R}" y2="{Y(v)}" stroke="{"#000" if v == 0 else "#ddd"}"/>')
            o.append(f'<text x="{L-8}" y="{Y(v)+4}" text-anchor="end">{v}</text>')
            v += 1
        for yr in range(2010, 2027, 2):
            o.append(f'<text x="{X(yr+0.5)}" y="{oy+ph-Bm+22}" text-anchor="middle">{yr}</text>')
        o.append(f'<rect x="{X(2018.0)}" y="{oy+T}" width="{X(2022.0)-X(2018.0)}" height="{ph-T-Bm}" fill="#f6e3e0" opacity="0.6"/>')
        o.append(f'<text x="{(X(2018.0)+X(2022.0))/2}" y="{oy+T+14}" text-anchor="middle" fill="#a33">held out</text>')
        for r in rows:
            x = X(r["t"])
            o.append(f'<line x1="{x}" y1="{Y(r["lo"])}" x2="{x}" y2="{Y(r["hi"])}" stroke="#555" stroke-width="1.5"/>')
            o.append(f'<circle cx="{x}" cy="{Y(r["est"])}" r="4" fill="#222"/>')
        hx = X(2020.0)
        o.append(f'<line x1="{hx}" y1="{Y(ho["lo_pp"])}" x2="{hx}" y2="{Y(ho["hi_pp"])}" stroke="#c0392b" stroke-width="2"/>')
        o.append(f'<rect x="{hx-5}" y="{Y(ho["est_pp"])-5}" width="10" height="10" fill="#c0392b"/>')
        ts = np.linspace(2010.0, 2026.7, 200)
        for kind, col, dash in (("exp", "#1f77b4", ""), ("hyp", "#2ca02c", ' stroke-dasharray="6 4"')):
            f = fits[kind]
            ys = f["e0"] * basis(kind, np.array([f["lam"]]), ts)[0]
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(ts, ys))
            o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        lx, ly = L + 12, oy + T + 36
        o.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="#222"/><text x="{lx+10}" y="{ly+4}">yearly estimate, 95% interval (training)</text>')
        o.append(f'<rect x="{lx-4}" y="{ly+12}" width="8" height="8" fill="#c0392b"/><text x="{lx+10}" y="{ly+20}">held-out period (one pooled estimate)</text>')
        o.append(f'<line x1="{lx-6}" y1="{ly+36}" x2="{lx+8}" y2="{ly+36}" stroke="#1f77b4" stroke-width="2"/><text x="{lx+14}" y="{ly+40}">exponential fit to the training years</text>')
        o.append(f'<line x1="{lx-6}" y1="{ly+54}" x2="{lx+8}" y2="{ly+54}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="6 4"/><text x="{lx+14}" y="{ly+58}">hyperbolic fit to the training years</text>')
    o.append("</svg>")
    open(out, "w", encoding="utf-8").write("\n".join(o))


# ---------------------------------------------------------------- main
def main(dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    import claim_checks as cc
    from power_registered_set import registered_cells
    import gate_receipts
    cells = registered_cells()
    df, bad_k, race_bad = cc.load(dump, cells)
    m1, m2 = cc.arm_frame(df, "M1b"), cc.arm_frame(df, "M2")
    # K0: the registered block estimates must be reproduced before anything yearly is printed
    pub = dict(c1_m1b=(4.376, 1.310), c1_m2=(4.178, 1.368), c2=(4.857, 0.780))
    got = {}
    for blk, i in (("A", 0), ("B", 1)):
        got[("c1_m1b", blk)] = cc.claim1_block(m1[m1.block == blk], "iid")["est_pp"]
        got[("c1_m2", blk)] = cc.claim1_block(m2[m2.block == blk], "iid")["est_pp"]
        got[("c2", blk)] = cc.claim2_block(m1[m1.block == blk], "iid")["est_pp"]
    checks = [(f"K0 block {b} {k} reproduces {pub[k][0 if b == 'A' else 1]:+.3f}",
               abs(got[(k, b)] - pub[k][0 if b == "A" else 1]) < 0.0015, f"{got[(k, b)]:+.4f}")
              for k in pub for b in ("A", "B")]
    nA, nB = int((df.block == "A").sum()), int((df.block == "B").sum())
    checks.append(("K0 setups per block 2,370,742 / 1,637,499", (nA, nB) == (2370742, 1637499), f"{nA:,} / {nB:,}"))
    checks.append(("K0 race consistent on every row", race_bad == 0, f"{race_bad}"))
    for lab, okk, det in checks:
        print(f"  [{'PASS' if okk else 'FAIL'}] {lab}   {det}")
    if not all(c[1] for c in checks):
        raise SystemExit("A known-number check FAILED -- no yearly estimate was read.")

    sl = json.load(open(os.path.join(HERE, "seal_use1_outputs", "claims_seal_2026-10-15.json")))["results"]["iid"]
    ho = dict(c1_m1b=sl["M1b"], c1_m2=sl["M2"], c2=sl["race"])
    spec = [("c1_m1b", "claim 1, M1b twins", m1, ("rf", "w", "tf", "w")),
            ("c1_m2", "claim 1, M2 twins", m2, ("rf", "w", "tf", "w")),
            ("c2", "claim 2, race (resolved setups)", m1, ("r_vf", "r_res", "t_vf", "t_res"))]
    result, panels = {}, []
    for key, lab, arm, cols in spec:
        yrs = day_to_year(arm.day.to_numpy())
        rows = yearly(arm, *cols, yrs, cc.ratio_diff, cc.day_codes, cc.boot_draws)
        t = np.array([r["t"] for r in rows]); y = np.array([r["est"] for r in rows])
        w = np.array([1.0 / r["se"] ** 2 for r in rows])
        dmat = np.column_stack([r["draws"][:2000] if len(r["draws"]) >= 2000 else
                                np.resize(r["draws"], 2000) for r in rows])
        fits = {k: summarize(k, y, dmat, w, t) for k in ("exp", "hyp")}
        print(f"\n{lab}: {len(rows)} calendar years of the training periods")
        for r in rows:
            part = "  (partial year)" if r["D"] < 100 else ""
            print(f"   {r['year']}  n={r['n']:>9,}  dates={r['D']:>4}  {r['est']:+7.2f} pp  [{r['lo']:+6.2f}, {r['hi']:+6.2f}]{part}")
        h = ho[key]
        for k, nm in (("exp", "exponential"), ("hyp", "hyperbolic")):
            f = fits[k]
            hl = (f"{f['halflife']:.1f} y [{f['halflife_lo']:.1f}, {f['halflife_hi']:.1f}]" if f["halflife_lo"] is not None
                  else f"{f['halflife']:.1f} y (no finite interval: {100*f['share_lam_le_0002']:.1f}% of bootstrap fits flat)")
            print(f"   {nm:<12} lam {f['lam']:+.3f}/y [{f['lam_lo']:+.3f}, {f['lam_hi']:+.3f}]  "
                  f"{'half-life' if k == 'exp' else '1/lam     '} {hl}  E0 {f['e0']:.2f}  wSSE {f['sse']:.1f}")
            print(f"                held-out 2018-2021: curve average {f['heldout_pred_pp']:+.2f} pp vs observed "
                  f"{h['est_pp']:+.2f} [{h['lo_pp']:+.2f}, {h['hi_pp']:+.2f}]   distance {abs(f['heldout_pred_pp']-h['est_pp']):.2f} pp")
        result[key] = dict(label=lab, years=[{k: v for k, v in r.items() if k != "draws"} for r in rows],
                           fits=fits, heldout=h)
        if key in ("c1_m1b", "c2"):
            panels.append((lab, rows, fits, h, -3.0 if key == "c2" else -2.0, 9.0 if key == "c2" else 8.0))
    svg(panels, None, os.path.join(HERE, "paper_figure2_yearly_decay.svg"))
    me = __import__("hashlib").sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    stamp = (f"yearly claims 1-2 and decay fits, DESCRIPTIVE | 144 registered cells, TRAINING periods only "
             f"({len(cells)} cells) | setup's own trade-date year, iid day bootstrap B={cc.B_BOOT} seed {cc.SEED} | "
             f"held-out = published pooled estimate, not re-read | dump {dump} | code-set {gate_receipts.code_set()} | "
             f"paper_yearly.py {me[:16]} | claim_checks.py imported unchanged | {platform.node()} | raw, no cost")
    print(f"\nSTAMP: {stamp}")
    out = os.path.join(HERE, f"paper_yearly_{datetime.date.today().isoformat()}.json")
    json.dump(dict(stamp=stamp, producer_sha256=me, results=result), open(out, "w", encoding="utf-8"), indent=1, default=float)
    print("wrote", os.path.basename(out), "and paper_figure2_yearly_decay.svg")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(0 if selftest() else 1)
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
