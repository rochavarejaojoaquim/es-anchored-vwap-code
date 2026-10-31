"""
paper_robustness.py -- the four measurements the reviewer's read of the full paper draft asked for
(M1, M3, M4, M5), on the TRAINING periods only (blocks A and B). The held-out period is NOT read here
and must not be re-run without a dated amendment (A15: registered for the claims run once).

All of it DESCRIPTIVE. Nothing here changes a registered statistic or verdict; every figure is a
robustness line beside claims 1 and 2 (claim_checks.py), which this script imports UNCHANGED
(load rules, fill(), day bootstrap draws, CI).

  R1  raw rates per period (setup fill, M1b twin fill, mirror fill; resolved / tie / open shares by arm)
      -- F3: raw return beside edge; M3: is the resolved share the same in both arms?
  R2  race without conditioning on 'resolved': opens scored 0.5 in BOTH arms, ties excluded; and
      the conditional claim-2 figure beside it (known number: reproduces 26.65).
  R3  claim 1 with a CLOSE-based touch (t_0.0, the frozen line, close crossing within T) instead of
      the wick touch: the optimistic-fill sensitivity.
  R4  claim 1 by tercile of distance in TICKS (dist_atr x atr0 / 0.25), terciles pooled over A and B,
      per period: does the A-to-B decay track the tick scale?
  R5  heterogeneity: per cell and period the claim-1 and claim-2 gap; share of the 144 cells positive,
      median, cell-equal-weighted mean, and the share of setups in the small-eps corner.
  R6  roll / end censoring (cens_T) share by arm if the column exists.

Known numbers first (any FAIL stops): claim 1 M1b A +4.376 / B +1.310, claim 2 A +4.857 / B +0.780,
to 0.001 pp; close-touch <= wick-touch on every row (a close crossing is a wick touch).
Usage:  python paper_robustness.py --selftest
        python paper_robustness.py data\\dump
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claim_checks as CC                                               # noqa: E402

TICK = 0.25
PUBLISHED = dict(c1_A=4.376, c1_B=1.310, c2_A=4.857, c2_B=0.780)
EXTRA = ["t_0.0", "dist_atr", "atr0", "cens_T"]


def _read(dump, name, cols):
    p = os.path.join(dump, name + ".parquet")
    have = set(pd.read_parquet(p, columns=None).columns[:0]) if False else None
    try:
        return pd.read_parquet(p, columns=cols)
    except Exception:
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(p).schema.names)
        return pd.read_parquet(p, columns=[c for c in cols if c in names])


def load(dump, cells):
    parts = []
    for c in cells:
        N, anchor, trig, eps = CC.CELL_RE.search(c).groups()
        x = _read(dump, c, CC.MAIN_COLS + EXTRA)
        T = int(x["T"].iloc[0])
        r = x[x.is_real == 1]
        fw = CC.fill(r["w_0.0"], T)
        rows = pd.DataFrame(dict(
            cell=c, N=int(N), anchor=anchor, trigger=trig, eps=float(eps), bar=r.bar.to_numpy(),
            day=r.day.to_numpy(), block=r.block.to_numpy(), rf=fw,
            r_res=np.isin(r.race, (1, -1)).astype(float), r_vf=(r.race.to_numpy() == 1).astype(float),
            r_tie=(r.race.to_numpy() == 0).astype(float), r_open=(r.race.to_numpy() == 9).astype(float),
            r_mf=CC.fill(r["mir_w"], T)))
        rows["rc"] = CC.fill(r["t_0.0"], T) if "t_0.0" in r else np.nan
        rows["ticks"] = (r["dist_atr"].to_numpy() * r["atr0"].to_numpy() / TICK
                         if "dist_atr" in r and "atr0" in r else np.nan)
        rows["cens"] = r["cens_T"].to_numpy(float) if "cens_T" in r else np.nan
        tw1 = x[x.is_real == 0]
        tw2 = _read(dump, c + ".m2", ["src_bar", "w_0.0", "mir_w", "race", "t_0.0"])
        if "cens_T" in tw1:      # twin-side censoring (R6 'by arm'), M1b twins averaged per setup
            ct = tw1.groupby("src_bar")["cens_T"].mean().rename("M1b_cens")
            rows = rows.join(ct, on="bar")
        else:
            rows["M1b_cens"] = np.nan
        for tag, tw in (("M1b", tw1), ("M2", tw2)):
            t = pd.DataFrame(dict(
                src_bar=tw.src_bar.to_numpy(), f=CC.fill(tw["w_0.0"], T), mf=CC.fill(tw["mir_w"], T),
                fc=(CC.fill(tw["t_0.0"], T) if "t_0.0" in tw else np.nan),
                res=np.isin(tw.race, (1, -1)).astype(float), vf=(tw.race.to_numpy() == 1).astype(float),
                tie=(tw.race.to_numpy() == 0).astype(float), opn=(tw.race.to_numpy() == 9).astype(float)))
            a = t.groupby("src_bar").agg(f=("f", "mean"), mf=("mf", "mean"), fc=("fc", "mean"),
                                         res=("res", "sum"), vf=("vf", "sum"), tie=("tie", "sum"),
                                         opn=("opn", "sum"), k=("f", "size"))
            a.columns = [f"{tag}_{k}" for k in a.columns]
            rows = rows.join(a, on="bar")
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def arm(df, tag):
    s = df[df[f"{tag}_k"].fillna(0) >= 1].copy()
    s["tf"] = s[f"{tag}_f"]; s["tfc"] = s[f"{tag}_fc"]; s["tmf"] = s[f"{tag}_mf"]
    s["t_res"] = s[f"{tag}_res"]; s["t_vf"] = s[f"{tag}_vf"]
    s["u_num"] = s.r_vf + 0.5 * s.r_open; s["u_den"] = 1.0 - s.r_tie
    s["v_num"] = s[f"{tag}_vf"] + 0.5 * s[f"{tag}_opn"]; s["v_den"] = s[f"{tag}_k"] - s[f"{tag}_tie"]
    s["w"] = 1.0
    return s


def gap(df, num_a, den_a, num_b, den_b, kind="iid"):
    codes, D = CC.day_codes(df.day)
    sm = CC._sums(df, sorted({num_a, den_a, num_b, den_b}), codes, D)
    est = CC.ratio_diff(sm, num_a, den_a, num_b, den_b)
    lo, hi, n = CC.ci(CC.ratio_diff(sm, num_a, den_a, num_b, den_b, CC.boot_draws(D, kind)))
    return dict(est=100 * est, lo=100 * lo, hi=100 * hi, D=D, n=len(df))


def fmt(r):
    return f"{r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  (n {r['n']:,}, dates {r['D']:,})"


def selftest():
    ok = True
    rng = np.random.default_rng(3)
    n = 4000
    df = pd.DataFrame(dict(day=rng.integers(0, 200, n), rf=(rng.random(n) < 0.6).astype(float),
                           tf=(rng.random(n) < 0.5).astype(float), w=1.0))
    g = gap(df, "rf", "w", "tf", "w")
    ok &= abs(g["est"] - 100 * (df.rf.mean() - df.tf.mean())) < 1e-9
    # opens scored 0.5 both arms, ties out: hand case
    s = pd.DataFrame(dict(day=[0, 0, 1], r_vf=[1.0, 0.0, 0.0], r_open=[0.0, 0.0, 1.0],
                          r_tie=[0.0, 1.0, 0.0], M1b_vf=[1.0, 0.0, 1.0], M1b_opn=[0.0, 1.0, 0.0],
                          M1b_tie=[0.0, 0.0, 0.0], M1b_k=[2, 2, 2], M1b_res=[1.0, 0, 1],
                          M1b_f=[0.5, 0.5, 0.5], M1b_fc=[0.5, 0.5, 0.5], M1b_mf=[0.5, 0.5, 0.5]))
    a = arm(s, "M1b")
    ok &= abs(a.u_num.sum() - 1.5) < 1e-12 and abs(a.u_den.sum() - 2.0) < 1e-12
    ok &= abs(a.v_num.sum() - 2.5) < 1e-12 and abs(a.v_den.sum() - 6.0) < 1e-12   # (1)+(0+.5)+(1)
    # close touch: gap on a hand case; tercile labels split a known vector into equal parts
    h = pd.DataFrame(dict(day=[0, 0, 1, 1], rc=[1.0, 0, 1, 1], tfc=[0.5, 0.5, 0.0, 0.5], w=1.0))
    ok &= abs(gap(h, "rc", "w", "tfc", "w")["est"] - 100 * (0.75 - 0.375)) < 1e-9
    v = np.arange(300.0)
    lab = np.digitize(v, np.nanpercentile(v, [100 / 3, 200 / 3]))
    ok &= sorted(np.bincount(lab).tolist()) == [100, 100, 100] or max(np.bincount(lab)) - min(np.bincount(lab)) <= 1
    # R5 binomial: 7 of 10 positive -> two-sided exact p = 0.34375
    from math import comb
    kk, nn2 = 7, 10
    pb = 2 * sum(comb(nn2, i) for i in range(0, min(kk, nn2 - kk) + 1)) / 2 ** nn2
    ok &= abs(pb - 0.34375) < 1e-12
    print(f"self-test: {'ALL DETECTORS ALIVE' if ok else 'FAILED'}")
    return bool(ok)


def per_block(df, f):
    return {b: f(df[df.block == b]) for b in sorted(df.block.unique())}


def main(dump):
    if not selftest():
        raise SystemExit("self-test failed")
    t0 = time.time()
    cells = CC.registered_cells()
    df = load(dump, cells)
    print(f"loaded {len(df):,} setups from {len(cells)} cells in {time.time()-t0:.0f}s\n")
    m1, m2 = arm(df, "M1b"), arm(df, "M2")
    out = dict(stamp=dict(dump=dump, n=len(df), machine=platform.node(),
                          script=hashlib.sha256(open(__file__, "rb").read()).hexdigest().upper(),
                          date=datetime.date.today().isoformat(),
                          note="TRAINING periods only; descriptive; claim_checks.py imported unchanged"))
    # known numbers
    c1 = per_block(m1, lambda d: gap(d, "rf", "w", "tf", "w"))
    c2 = per_block(m1, lambda d: gap(d, "r_vf", "r_res", "t_vf", "t_res"))
    blocks = sorted(df.block.unique())
    A, B = blocks[0], blocks[-1]
    chk = [("claim 1 M1b A", c1[A]["est"], PUBLISHED["c1_A"]), ("claim 1 M1b B", c1[B]["est"], PUBLISHED["c1_B"]),
           ("claim 2 A", c2[A]["est"], PUBLISHED["c2_A"]), ("claim 2 B", c2[B]["est"], PUBLISHED["c2_B"])]
    bad = False
    for lab, got, want in chk:
        okk = abs(got - want) < 0.001
        bad |= not okk
        print(f"  [{'PASS' if okk else 'FAIL'}] known number {lab}: {got:+.4f} vs {want:+.3f}")
    if "rc" in df and df.rc.notna().any():
        viol = int(((df.rc == 1) & (df.rf == 0)).sum())
        print(f"  [{'PASS' if viol == 0 else 'FAIL'}] close touch implies wick touch: {viol} violations")
        bad |= viol != 0
    if bad:
        raise SystemExit("known-number check FAILED -- nothing below is to be read")
    print()

    print("R1  RAW RATES PER PERIOD (training), % of setups / of twins")
    r1 = {}
    for b in (A, B):
        d, d1 = df[df.block == b], m1[m1.block == b]
        r1[b] = dict(n=len(d), setup_fill=100 * d.rf.mean(), twin_fill=100 * d1.tf.mean(),
                     setup_mirror=100 * d.r_mf.mean(), twin_mirror=100 * d1.tmf.mean(),
                     setup_resolved=100 * d.r_res.mean(), setup_tie=100 * d.r_tie.mean(),
                     setup_open=100 * d.r_open.mean(),
                     twin_resolved=100 * d1.M1b_res.sum() / d1.M1b_k.sum(),
                     twin_tie=100 * d1.M1b_tie.sum() / d1.M1b_k.sum(),
                     twin_open=100 * d1.M1b_opn.sum() / d1.M1b_k.sum())
        print(f"  block {b}: " + "; ".join(f"{k} {v:,.3f}" if k != "n" else f"n {v:,}" for k, v in r1[b].items()))
    out["R1"] = r1

    print("\nR2  RACE: conditional on resolved (claim 2) vs opens scored 0.5 in both arms, ties out (pp)")
    r2 = {}
    for b in (A, B):
        d = m1[m1.block == b]
        cond = gap(d, "r_vf", "r_res", "t_vf", "t_res")
        unc = gap(d, "u_num", "u_den", "v_num", "v_den")
        r2[b] = dict(conditional=cond, unconditional=unc)
        print(f"  block {b}: conditional {fmt(cond)}\n           unconditional {fmt(unc)}")
    out["R2"] = r2

    print("\nR3  CLAIM 1 WITH A CLOSE-BASED TOUCH (pp), M1b and M2")
    r3 = {}
    if df.rc.notna().any():
        for tag, fr in (("M1b", m1), ("M2", m2)):
            for b in (A, B):
                d = fr[fr.block == b]
                w = gap(d, "rf", "w", "tf", "w")
                c = gap(d.assign(rcc=d.rc), "rcc", "w", "tfc", "w")
                r3[f"{tag}_{b}"] = dict(wick=w, close=c)
                print(f"  {tag} block {b}: wick {fmt(w)}\n{' ' * 17}close {fmt(c)}")
    out["R3"] = r3

    print("\nR4  CLAIM 1 (M1b) BY TERCILE OF DISTANCE IN TICKS (terciles pooled over A and B), pp")
    r4 = {}
    if m1.ticks.notna().any():
        q = np.nanpercentile(m1.ticks, [100 / 3, 200 / 3])
        print(f"  tercile edges (ticks): {q[0]:.1f}, {q[1]:.1f}")
        lab = np.digitize(m1.ticks, q)
        for t in range(3):
            for b in (A, B):
                d = m1[(lab == t) & (m1.block == b)]
                if len(d) < 1000:
                    continue
                g = gap(d, "rf", "w", "tf", "w")
                r4[f"T{t + 1}_{b}"] = dict(g, share_of_block=100 * len(d) / (m1.block == b).sum())
                print(f"  tercile {t + 1} block {b}: {fmt(g)}  ({100 * len(d) / (m1.block == b).sum():.0f}% of the block)")
            ka, kb = f"T{t + 1}_{A}", f"T{t + 1}_{B}"
            if ka in r4 and kb in r4:      # older minus recent; periods have disjoint dates -> SEs add in quadrature
                sa = (r4[ka]["hi"] - r4[ka]["lo"]) / 3.92
                sb = (r4[kb]["hi"] - r4[kb]["lo"]) / 3.92
                dd = r4[ka]["est"] - r4[kb]["est"]
                se = (sa ** 2 + sb ** 2) ** 0.5
                r4[f"diff_T{t + 1}"] = dict(est=dd, lo=dd - 1.96 * se, hi=dd + 1.96 * se)
                print(f"  tercile {t + 1} older minus recent: {dd:+.3f} [{dd - 1.96 * se:+.3f}, {dd + 1.96 * se:+.3f}]"
                      f"  (normal approximation, disjoint dates)")
            else:
                print(f"  tercile {t + 1}: omitted from the older-minus-recent line (fewer than 1,000 setups in a period)")
    out["R4"] = r4

    print("\nR5  HETEROGENEITY ACROSS THE 144 CELLS (setup-weighted pooled figures hide this)")
    r5 = {}
    small = ((df.trigger == "ATR") & df.eps.isin([3.0, 4.0])) | ((df.trigger == "SD") & df.eps.isin([1.0, 1.5]))
    for b in (A, B):
        d = m1[m1.block == b]
        per = []
        for c, g in d.groupby("cell"):
            if len(g) < 200:
                continue
            e1 = 100 * (g.rf.mean() - g.tf.mean())
            den_s, den_t = g.r_res.sum(), g.t_res.sum()
            e2 = (100 * (g.r_vf.sum() / den_s - g.t_vf.sum() / den_t)) if den_s > 0 and den_t > 0 else np.nan
            per.append((c, len(g), e1, e2))
        pp = pd.DataFrame(per, columns=["cell", "n", "c1", "c2"])
        from math import comb
        kpos, nn = int((pp.c1 > 0).sum()), len(pp)       # exact two-sided binomial vs 0.5 (cells overlap: descriptive)
        pbin = min(1.0, 2 * sum(comb(nn, i) for i in range(0, min(kpos, nn - kpos) + 1)) / 2 ** nn)
        r5[b] = dict(cells=len(pp), cells_dropped_under_200_setups=len(d.cell.unique()) - len(pp),
                     c1_positive_cells=kpos, c1_binomial_p_vs_half=pbin,
                     c1_positive_share=100 * (pp.c1 > 0).mean(), c1_median=pp.c1.median(),
                     c1_equal_weight_mean=pp.c1.mean(), c2_positive_share=100 * (pp.c2.dropna() > 0).mean(),
                     c2_median=pp.c2.median(), c2_equal_weight_mean=pp.c2.mean(),
                     small_eps_share_of_setups=100 * small[df.block == b].mean())
        print(f"  block {b}: " + "; ".join(f"{k} {v:,.2f}" for k, v in r5[b].items()))
        print("    (c1_binomial_p_vs_half is of NO inferential use: cells overlap and a real effect makes the null wrong)")
    out["R5"] = r5

    print("\nR6  CENSORING (cens_T = roll / end-of-data truncated horizon), % of setups and of M1b twins")
    if df.cens.notna().any():
        for b in (A, B):
            d = df[df.block == b]
            tw = d.M1b_cens.mean() if d.M1b_cens.notna().any() else float("nan")
            print(f"  block {b}: setups {100 * d.cens.mean():.2f}%   M1b twins {100 * tw:.2f}%")
    out["R6"] = {str(b): dict(setups=(100 * df[df.block == b].cens.mean() if df.cens.notna().any() else None),
                              twins=(100 * df[df.block == b].M1b_cens.mean()
                                     if df.M1b_cens.notna().any() else None)) for b in (A, B)}

    path = os.path.join(HERE, f"paper_robustness_{datetime.date.today().isoformat()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {os.path.basename(path)}  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
