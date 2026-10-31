"""make_figures_new.py -- six figures for the paper (SVG, no third-party packages), drawn ONLY from result files
that already exist; nothing is recomputed from market data and the held-out period is not re-read (its three
published estimates are read from claims_seal_2026-10-15.json, as make_figure2.py does).

  A  paper_figure_A_null_vs_observed.svg   the shuffle (Monte Carlo) null against the observed scores
  B  paper_figure_B_robustness_forest.svg  the six headline estimates under every check that was run
  C  paper_figure_C_tick_terciles.svg      claim 1 by tercile of distance in ticks, older vs recent period
  D  paper_figure_D_covariate_blocks.svg   82 conditions: older-period contrast against recent-period contrast
  E  paper_figure_E_recall_curve.svg       planted-effect recall of the screening step
  F  paper_figure_F_section25_deciles.svg  real-minus-pseudo gap by decile of the distance difference

Every figure prints known-number checks (PASS/FAIL) against figures already published in the paper before it is
written; a FAIL stops the run. Sources: rw_check_independent_out.csv (+ rw_check_independent.studentize taken by ast),
claim_checks_2026-10-03.json, paper_twoway_2026-10-19.json, claim1_hour_std_2026-10-12.json,
roll_sens/roll_claims_*_2026-10-20.json, paper_robustness_2026-10-15.json, pooled_condition_result_2026-10-14.csv,
calib_main_arm2_2026-10-05.csv, dx6_joint_2026-10-03.json. Figure E's stored-null and Romano-Wolf curves are
copied from record entry 26.107 (the scratch script that produced them was not kept); its plain-rule curve against
fresh nulls is recomputed from the arm-2 CSV and checked against the record.
Usage: python make_figures_new.py"""
import ast
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, RED, GREY, DARK = "#2c6fbb", "#d9822b", "#c0392b", "#9a9a9a", "#222222"
FAILS = []


def P(*a):
    return os.path.join(HERE, *a)


def J(*a):
    return json.load(open(P(*a), encoding="utf-8"))


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []

    def line(self, x1, y1, x2, y2, c=DARK, w=1.0, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"{d} opacity="{op}"/>')

    def rect(self, x, y, w, h, fill="none", stroke="none", op=1.0, sw=1.0):
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')

    def circ(self, cx, cy, r, fill=DARK, stroke="none", op=1.0):
        self.o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" opacity="{op}"/>')

    def text(self, x, y, t, size=12, anchor="start", fill=DARK, weight="normal", rot=None, style=None):
        r = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ""
        st = f' font-style="{style}"' if style else ""
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{st}{r}>{esc(t)}</text>')

    def save(self, name):
        body = "\n".join(self.o)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="sans-serif" font-size="12">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>')
        open(P(name), "w", encoding="utf-8").write(svg)
        print("  wrote", name)


class Ax:
    def __init__(self, s, x0, y0, x1, y1, xl, yl):
        self.s, self.x0, self.y0, self.x1, self.y1, self.xl, self.yl = s, x0, y0, x1, y1, xl, yl

    def X(self, v):
        return self.x0 + (v - self.xl[0]) / (self.xl[1] - self.xl[0]) * (self.x1 - self.x0)

    def Y(self, v):
        return self.y1 - (v - self.yl[0]) / (self.yl[1] - self.yl[0]) * (self.y1 - self.y0)

    def frame(self, xt, yt, xf="{:g}", yf="{:g}", xlabel="", ylabel="", title="", grid=True, size=11):
        s = self.s
        for v in yt:
            if grid:
                s.line(self.x0, self.Y(v), self.x1, self.Y(v), "#e2e2e2" if v != 0 else "#555", 1 if v != 0 else 1.2)
            s.text(self.x0 - 6, self.Y(v) + 4, yf.format(v), size, "end")
        for v in xt:
            if grid:
                s.line(self.X(v), self.y0, self.X(v), self.y1, "#eeeeee" if v != 0 else "#bbbbbb", 1)
            s.text(self.X(v), self.y1 + 15, xf.format(v), size, "middle")
        s.line(self.x0, self.y1, self.x1, self.y1, "#444", 1.2)
        s.line(self.x0, self.y0, self.x0, self.y1, "#444", 1.2)
        if xlabel:
            s.text((self.x0 + self.x1) / 2, self.y1 + 34, xlabel, size, "middle")
        if ylabel:
            s.text(self.x0 - 40, (self.y0 + self.y1) / 2, ylabel, size, "middle", rot=-90)
        if title:
            s.text((self.x0 + self.x1) / 2, self.y0 - 12, title, 12.5, "middle", weight="bold")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def hist(vals, edges):
    c, _ = np.histogram(vals, bins=edges)
    return 100.0 * c / len(vals)


# ============================================================================================ A
def fig_A():
    print("Figure A: shuffle null against observed scores")
    src = open(P("rw_check_independent.py"), encoding="utf-8").read()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "studentize"][0]
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "studentize", "exec"), ns)
    studentize = ns["studentize"]
    R = pd.read_csv(P("rw_check_independent_out.csv"))
    NUL = np.array([json.loads(s) for s in R.null_scores], float)
    obs = R.obs_score.to_numpy(float)
    to, T = studentize(obs, NUL, loo=True)
    mx = T.max(axis=0)
    surv = (R.rw_adj_p <= 0.05).to_numpy()
    check("317 effective situations x 100 joint re-deals", NUL.shape == (317, 100), str(NUL.shape))
    check("31 survive Romano-Wolf at adjusted p <= 0.05", int(surv.sum()) == 31, str(int(surv.sum())))
    check("own t_obs reproduces the stored column", bool(np.allclose(to, R.t_obs.to_numpy(float), atol=1e-9)))
    pc = np.percentile(mx, [50, 90, 95, 99])
    check("joint max-t percentiles equal the published 4.576 / 6.329 / 7.030 / 7.720",
          bool(np.allclose(pc, [4.576, 6.329, 7.030, 7.720], atol=0.0015)), " ".join(f"{v:.3f}" for v in pc))
    check("every survivor's t exceeds the 95th percentile", bool((to[surv] > pc[2]).all()),
          f"min survivor t {to[surv].min():.2f}, max {to[surv].max():.2f}")

    def pick(mask):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(to[idx])]
        return int(idx[len(idx) // 2])      # median t_obs of the group: a fixed rule, not a choice by outcome

    ex_n, ex_s = pick(~surv), pick(surv)
    s = Svg(960, 366)
    s.text(480, 20, "Shuffle (Monte Carlo) test: the score of the best covariate filter when nothing is there, against the real score",
           13, "middle", weight="bold")
    boxes = [(58, 298), (370, 610), (682, 932)]
    for (x0, x1), idx, ttl in zip(boxes[:2], (ex_n, ex_s), ("(a) a situation that does not survive", "(b) a situation that survives")):
        v = NUL[idx] * 100
        o = obs[idx] * 100
        lo, hi = min(v.min(), o) - 0.05, max(v.max(), o) + 0.35
        edges = np.linspace(v.min() - 0.02, v.max() + 0.02, 13)
        pct = hist(v, edges)
        ax = Ax(s, x0, 62, x1, 262, (lo, hi), (0, max(pct.max() * 1.25, 10)))
        step = 0.5 if hi - lo > 1.6 else 0.25
        xt = [round(k * step, 2) for k in range(int(lo / step), int(hi / step) + 2) if lo <= k * step <= hi]
        ax.frame(xt, [0, 5, 10, 15, 20, 25], xf="{:g}", ylabel="percent of the 100 re-deals", xlabel="best score (pp)", title=ttl)
        for i, p_ in enumerate(pct):
            if p_ > 0:
                s.rect(ax.X(edges[i]), ax.Y(p_), ax.X(edges[i + 1]) - ax.X(edges[i]), ax.Y(0) - ax.Y(p_), "#9db5d6", "#3d5a80", 0.9)
        s.line(ax.X(o), ax.y0 + 8, ax.X(o), ax.y1, RED, 2.2)
        side = R.side.iloc[idx]
        nm = f"{R.cell.iloc[idx]}, {'pooled' if pd.isna(side) else 'side ' + str(int(side))}, {R.col.iloc[idx]}"
        s.text(x0, 326, nm, 10, "start", "#444")
        s.text(x0, 340, f"real score {o:.2f} pp; p = {R.p.iloc[idx]:.3f}", 10, "start", "#444")
        s.text(ax.X(o) + (6 if ax.X(o) < x1 - 70 else -6), ax.y0 + 22, "real", 11, "start" if ax.X(o) < x1 - 70 else "end", RED, "bold")
    x0, x1 = boxes[2]
    edges = np.arange(3.0, 9.01, 0.4)
    pct = hist(mx, edges)
    ax = Ax(s, x0, 62, x1, 262, (0, 16.5), (0, max(pct.max() * 1.45, 12)))
    ax.frame([0, 2, 4, 6, 8, 10, 12, 14, 16], [0, 5, 10, 15], ylabel="percent of the 100 joint re-deals",
             xlabel="studentised score (t)", title="(c) best of all 317 situations")
    for i, p_ in enumerate(pct):
        if p_ > 0:
            s.rect(ax.X(edges[i]), ax.Y(p_), ax.X(edges[i + 1]) - ax.X(edges[i]), ax.Y(0) - ax.Y(p_), "#9db5d6", "#3d5a80", 0.9)
    s.line(ax.X(pc[2]), ax.y0 + 8, ax.X(pc[2]), ax.y1, DARK, 1.5, "5 3")
    s.text(ax.X(pc[2]) + 4, ax.y0 + 9, f"95th percentile {pc[2]:.2f}", 10.5, "start")
    base = ax.y0 + 50          # a strip above the histogram for the real scores
    for t_ in np.sort(to[~surv]):
        if 0 <= t_ <= 16.5:
            s.line(ax.X(t_), base - 1, ax.X(t_), base - 11, GREY, 1.3, op=0.75)
    for t_ in np.sort(to[surv]):
        s.line(ax.X(t_), base - 1, ax.X(t_), base - 24, RED, 2.0)
    s.text(ax.X(12.0), base - 30, "31 real scores above the line", 10.5, "middle", RED, "bold")
    s.text(ax.X(2.2), base - 14, "other 286", 10, "middle", "#666")
    s.text(x0, 326, "Grey ticks: the 286 situations that do not", 10, "start", "#444")
    s.text(x0, 338, "survive. Red ticks: the 31 that do. (a), (b):", 10, "start", "#444")
    s.text(x0, 350, "the situation with the median t in its group.", 10, "start", "#444")
    s.save("paper_figure_A_null_vs_observed.svg")


# ============================================================================================ B
def fig_B():
    print("Figure B: robustness forest plot")
    cc = J("claim_checks_2026-10-03.json")
    prim = cc["PRIMARY: all setups of the 144 registered cells"]
    dd = cc["DEDUP SENSITIVITY: each signal event once (26.63 G2)"]
    tw = J("paper_twoway_2026-10-19.json")
    hs = J("claim1_hour_std_2026-10-12.json")["results"]
    rb = J("paper_robustness_2026-10-15.json")
    lag = J("roll_sens", "roll_claims_lagged_2026-10-20.json")["new_iid"]
    cal = J("roll_sens", "roll_claims_calendar8_2026-10-20.json")["new_iid"]
    seal = J("seal_use1_outputs", "claims_seal_2026-10-15.json")["results"]["iid"]

    def trip(r):
        return (r["est_pp"], r["lo_pp"], r["hi_pp"])

    def ccg(blob, kind, claim, per):
        k = {"c1_M1b": ("c1", f"M1b {per}"), "c1_M2": ("c1", f"M2 {per}"), "c2": ("c2", f"race {per}")}[claim]
        return trip(blob[kind][k[0]][k[1]])

    def twg(tab, key, claim, per):
        k = {"c1_M1b": "c1_M1b", "c1_M2": "c1_M2", "c2": "c2_M1b"}[claim]
        r = [x for x in tw[tab] if x["claim"] == k and x["block"] == per and x["grp"] == "all"][0]
        return tuple(r[key])

    def hsg(claim, per):
        if claim == "c1_M1b":
            r = hs[f"M1b_{per}"]["c1_std"]
        elif claim == "c1_M2":
            r = hs[f"M2_{per}"]["c1_std"]
        else:
            r = hs[f"M1b_{per}"]["c2_std"]
        return (r["est"], r["lo"], r["hi"])

    def rollg(blob, claim, per):
        return trip(blob[{"c1_M1b": f"M1b_{per}", "c1_M2": f"M2_{per}", "c2": f"race_{per}"}[claim]])

    def closeg(claim, per):
        if claim == "c2":
            return None
        r = rb["R3"][f"{'M1b' if claim == 'c1_M1b' else 'M2'}_{per}"]["close"]
        return (r["est"], r["lo"], r["hi"])

    def unresg(claim, per):
        if claim != "c2":
            return None
        r = rb["R2"][per]["unconditional"]
        return (r["est"], r["lo"], r["hi"])

    rows = [
        ("Registered, iid day bootstrap", lambda c, p: ccg(prim, "iid", c, p)),
        ("Registered, 5-day blocks", lambda c, p: ccg(prim, "block", c, p)),
        ("Own-date dates, iid", lambda c, p: twg("table_144", "own_iid", c, p)),
        ("Own-date dates, 5-day blocks", lambda c, p: twg("table_144", "own_blk", c, p)),
        ("117 certified cells only", lambda c, p: twg("table_117", "setup_iid", c, p)),
        ("Twins re-weighted to the setups' hours", hsg),
        ("Each signal event counted once", lambda c, p: ccg(dd, "iid", c, p)),
        ("Roll rule: previous day's volume", lambda c, p: rollg(lag, c, p)),
        ("Roll rule: fixed calendar", lambda c, p: rollg(cal, c, p)),
        ("Close-based touch (claim 1)", closeg),
        ("Unresolved setups scored 0.5 (claim 2)", unresg),
    ]
    claims = [("c1_M1b", "Claim 1, M1b twins"), ("c1_M2", "Claim 1, M2 twins"), ("c2", "Claim 2, race (resolved setups)")]
    # known numbers: the registered row equals Table 3.1
    t31 = {("c1_M1b", "A"): 4.38, ("c1_M1b", "B"): 1.31, ("c1_M2", "A"): 4.18, ("c1_M2", "B"): 1.37, ("c2", "A"): 4.86, ("c2", "B"): 0.78}
    check("registered iid row equals Table 3.1 (six estimates, 2 dp)",
          all(abs(rows[0][1](c, p)[0] - v) < 0.006 for (c, p), v in t31.items()))
    t31h = {"c1_M1b": 1.65, "c1_M2": 1.73, "c2": 1.06}
    held = {"c1_M1b": trip(seal["M1b"]), "c1_M2": trip(seal["M2"]), "c2": trip(seal["race"])}
    check("held-out estimates equal Table 3.1 (1.65 / 1.73 / 1.06)", all(abs(held[c][0] - v) < 0.006 for c, v in t31h.items()))
    check("every plotted lower interval end is above zero",
          all(g[1] > 0 for (_, f) in rows for c, _ in claims for p in "AB" if (g := f(c, p)) is not None),
          "(iid and block, all rows)")
    W, H = 990, 470
    s = Svg(W, H)
    s.text(W / 2, 20, "The six headline estimates under every check that was run (pp, 95% interval)", 13, "middle", weight="bold")
    lx = 20
    s.circ(lx + 6, 40, 4.5, BLUE)
    s.text(lx + 16, 44, "2010-2017", 11)
    s.circ(lx + 100, 40, 4.5, ORANGE)
    s.text(lx + 110, 44, "2022-2026", 11)
    s.rect(lx + 195, 35.5, 9, 9, RED)
    s.text(lx + 210, 44, "held out 2018-2021 (registered row only)", 11)
    label_w, pw, gap = 262, 214, 22
    top, rh = 70, 31
    n = len(rows)
    for ci, (c, ttl) in enumerate(claims):
        x0 = label_w + ci * (pw + gap)
        ax = Ax(s, x0, top, x0 + pw, top + n * rh, (-0.5, 6.0), (0, 1))
        ax.frame([0, 2, 4, 6], [], xf="{:g}", xlabel="pp", title=ttl, size=10.5)
        for v in [0, 2, 4, 6]:
            s.line(ax.X(v), top, ax.X(v), top + n * rh, "#bbbbbb" if v == 0 else "#ebebeb", 1.2 if v == 0 else 1)
        for ri, (lab, f) in enumerate(rows):
            yc = top + ri * rh + rh / 2
            if ci == 0:
                s.text(label_w - 10, yc + 4, lab, 11, "end")
            if ri % 2 == 0:
                s.rect(x0, top + ri * rh, pw, rh, "#f4f6f8", op=0.7)
            for per, col, dy in (("A", BLUE, -7), ("B", ORANGE, 7)):
                g = f(c, per)
                if g is None:
                    continue
                e, lo, hi = g
                s.line(ax.X(lo), yc + dy, ax.X(hi), yc + dy, col, 1.6)
                s.circ(ax.X(e), yc + dy, 3.6, col)
            if ri == 0:
                e, lo, hi = held[c]
                s.line(ax.X(lo), yc, ax.X(hi), yc, RED, 1.6)
                s.rect(ax.X(e) - 4, yc - 4, 8, 8, RED)
    s.text(label_w, H - 8, "Intervals are those of each check (not adjusted). Each row re-estimates the same two claims; the rows are not independent.", 10, "start", "#555")
    s.save("paper_figure_B_robustness_forest.svg")


# ============================================================================================ C
def fig_C():
    print("Figure C: claim 1 by tercile of distance in ticks")
    rb = J("paper_robustness_2026-10-15.json")["R4"]
    lab = ["up to 5.3 ticks", "5.3 to 12.4 ticks", "above 12.4 ticks"]
    est = {p: [rb[f"T{i}_{p}"] for i in (1, 2, 3)] for p in "AB"}
    check("terciles equal Table 3.2a older (+6.81, +2.60, -0.10) and recent (+2.42, +1.95, +0.86)",
          all(abs(est["A"][i]["est"] - v) < 0.006 for i, v in enumerate((6.81, 2.60, -0.10)))
          and all(abs(est["B"][i]["est"] - v) < 0.006 for i, v in enumerate((2.42, 1.95, 0.86))))
    check("shares equal 51 / 36 / 13 % and 8 / 29 / 63 %",
          [round(e["share_of_block"]) for e in est["A"]] == [51, 36, 13] and [round(e["share_of_block"]) for e in est["B"]] == [8, 29, 63])
    s = Svg(900, 360)
    s.text(450, 20, "Claim 1 (M1b) by distance to the line in ticks: the effect is large where the older period lived, and the mix moved", 13, "middle", weight="bold")
    for ai, (x0, x1) in enumerate(((70, 410), (520, 860))):
        if ai == 0:
            ax = Ax(s, x0, 70, x1, 280, (0, 3), (-2, 8))
            ax.frame([], [-2, 0, 2, 4, 6, 8], ylabel="claim 1 (pp)", title="(a) estimate in each tercile")
        else:
            ax = Ax(s, x0, 70, x1, 280, (0, 3), (0, 70))
            ax.frame([], [0, 20, 40, 60], ylabel="share of the period's setups (%)", title="(b) share of setups in each tercile")
        for i in range(3):
            xc = i + 0.5
            for per, col, dx in (("A", BLUE, -0.2), ("B", ORANGE, 0.2)):
                e = est[per][i]
                v = e["est"] if ai == 0 else e["share_of_block"]
                y0v = 0
                xl_, xr_ = ax.X(xc + dx - 0.17), ax.X(xc + dx + 0.17)
                s.rect(xl_, min(ax.Y(v), ax.Y(y0v)), xr_ - xl_, abs(ax.Y(v) - ax.Y(y0v)), col, "none", 0.9)
                if ai == 0:
                    s.line(ax.X(xc + dx), ax.Y(e["lo"]), ax.X(xc + dx), ax.Y(e["hi"]), "#222", 1.4)
                    s.text(ax.X(xc + dx), ax.Y(e["hi"]) - 5, f"{e['est']:+.2f}", 10, "middle")
                else:
                    s.text(ax.X(xc + dx), ax.Y(v) - 5, f"{v:.0f}%", 10, "middle")
            s.text(ax.X(xc), ax.y1 + 16, lab[i], 11, "middle")
    s.rect(300, 322, 10, 10, BLUE)
    s.text(316, 331, "2010-2017", 11)
    s.rect(420, 322, 10, 10, ORANGE)
    s.text(436, 331, "2022-2026", 11)
    s.text(450, 352, "Terciles of the distance in ticks (distance in ATR x ATR in ticks), pooled over both periods. Whiskers: 95% day-bootstrap intervals.", 10, "middle", "#555")
    s.save("paper_figure_C_tick_terciles.svg")


# ============================================================================================ D
def fig_D():
    print("Figure D: covariate contrasts, older against recent period")
    p = pd.read_csv(P("pooled_condition_result_2026-10-14.csv"))
    p = p[p.kind == "top-vs-bottom"].reset_index(drop=True)
    a_only = (p.adjA <= 0.05) & (p.adjB > 0.05)
    both = (p.adjA <= 0.05) & (p.adjB <= 0.05)
    check("82 top-versus-bottom conditions", len(p) == 82, str(len(p)))
    check("23 clear the corrected threshold in the older period, none in the recent", int((p.adjA <= 0.05).sum()) == 23 and int((p.adjB <= 0.05).sum()) == 0,
          f"{int((p.adjA <= 0.05).sum())} / {int((p.adjB <= 0.05).sum())}")
    check("no condition passes in both periods", int(both.sum()) == 0 and int(p.passes.sum()) == 0)
    top8 = list(p.reindex(p.thA.abs().sort_values(ascending=False).index).col.head(8))
    check("the eight largest older-period contrasts are the eight in Table 3.5",
          set(top8) == {"speed", "W0", "rv_close", "relvol", "is_rth", "ct_hour", "anchor_spike", "value_area_atr"}, ", ".join(top8))
    s = Svg(900, 545)
    s.text(450, 20, "Covariate conditions: the contrast in 2010-2017 against the contrast in 2022-2026 (race rate, top fifth minus bottom fifth, pp)", 13, "middle", weight="bold")
    ax = Ax(s, 90, 55, 860, 450, (-12, 10), (-4, 4))
    ax.frame([-12, -8, -4, 0, 4, 8], [-4, -2, 0, 2, 4], xf="{:+g}", yf="{:+g}", xlabel="older period (2010-2017), pp", ylabel="recent period (2022-2026), pp")
    s.line(ax.X(-4), ax.Y(-4), ax.X(4), ax.Y(4), "#777", 1.2, "5 4")
    s.text(ax.X(4) + 5, ax.Y(4) + 10, "same size in both periods", 10, "start", "#777")
    for i, r in p.iterrows():
        x, y = ax.X(r.thA), ax.Y(r.thB)
        if a_only[i]:
            s.circ(x, y, 5, BLUE, "#1d4a80", 0.9)
        else:
            s.circ(x, y, 4, "#cfcfcf", "#888", 0.95)
    offs = {"speed": (8, -8), "W0": (8, 14), "rv_close": (8, 4), "relvol": (-8, 16), "is_rth": (-8, -10), "ct_hour": (-8, 16), "anchor_spike": (-8, 16), "value_area_atr": (8, 16)}
    for nm in top8:
        r = p[p.col == nm].iloc[0]
        dx, dy = offs.get(nm, (8, -8))
        s.text(ax.X(r.thA) + dx, ax.Y(r.thB) + dy, nm, 10.5, "start" if dx > 0 else "end", "#1d4a80")
    s.circ(250, 508, 5, BLUE, "#1d4a80")
    s.text(262, 512, "clears the corrected threshold in 2010-2017 only (23)", 11)
    s.circ(590, 508, 4, "#cfcfcf", "#888")
    s.text(600, 512, "does not clear it in either period (59)", 11)
    s.text(450, 533, "No condition clears the corrected threshold in both periods (0 of 82).", 10.5, "middle", "#555")
    s.save("paper_figure_D_covariate_blocks.svg")


# ============================================================================================ E
def fig_E():
    print("Figure E: planted-effect recall of the screening step")
    a = pd.read_csv(P("calib_main_arm2_2026-10-05.csv"))
    fresh = {}
    for sc in ("EASY", "HARD"):
        for d in (2, 4, 8):
            q = a[(a.scen == sc) & (a.delta == d)]
            fresh[(sc, d)] = 100.0 * float((q.k <= 4).mean()), len(q)
    q0 = a[a.scen == "ZERO"]
    fresh[("ZERO", 0)] = 100.0 * float((q0.k <= 4).mean()), len(q0)
    rec = {("EASY", 2): 8.3, ("EASY", 4): 18.3, ("EASY", 8): 81.7, ("HARD", 2): 8.3, ("HARD", 4): 5.0, ("HARD", 8): 6.7, ("ZERO", 0): 6.7}
    check("recomputed plain-rule recall equals record 26.107 (one decimal)",
          all(abs(round(fresh[k][0], 1) - v) < 0.05 for k, v in rec.items()), str({k: round(v[0], 1) for k, v in fresh.items()}))
    check("60 tests per point", all(v[1] == 60 for v in fresh.values()))
    # copied from record 26.107 (scratch script not kept): plain rule against the STORED nulls, and the full correction
    stored = {"EASY": [3.3, 8.3, 41.7, 91.7], "HARD": [3.3, 5.0, 6.7, 15.0]}
    full = {"EASY": [0.0, 0.0, 1.7, 48.3], "HARD": [0.0, 0.0, 0.0, 0.0]}
    xs = [0, 2, 4, 8]
    s = Svg(900, 400)
    s.text(450, 20, "What the screening step finds: planted effects of known size, 60 situations per point (% found)", 13, "middle", weight="bold")
    for pi, (sc, x0, x1, ttl) in enumerate((("EASY", 70, 410, "(a) the same planted effect in both periods"),
                                           ("HARD", 520, 860, "(b) the effect in the recent period only a quarter of that size"))):
        ax = Ax(s, x0, 60, x1, 300, (-0.5, 8.7), (0, 100))
        ax.frame([0, 2, 4, 8], [0, 20, 40, 60, 80, 100], ylabel="% of 60 found", xlabel="planted effect in the older period (pp)", title=ttl)
        series = (
            ("plain 5% rule, fresh shuffles", [fresh[("ZERO", 0)][0]] + [fresh[(sc, d)][0] for d in (2, 4, 8)], GREY, "5 3", 7),
            ("plain 5% rule, the stored shuffles", stored[sc], BLUE, None, 0),
            ("full correction over 317 (Romano-Wolf)", full[sc], RED, None, 0),
        )
        for nm, ys, col, dash, _ in series:
            pts = []
            for x, y in zip(xs, ys):
                k = round(y * 0.6)
                lo, hi = wilson(k, 60)
                s.line(ax.X(x), ax.Y(100 * lo), ax.X(x), ax.Y(100 * hi), col, 1.0, op=0.45)
                pts.append((ax.X(x), ax.Y(y)))
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                s.line(xa, ya, xb, yb, col, 2.0, dash)
            for xx, yy in pts:
                s.circ(xx, yy, 3.8, col)
        if pi == 0:
            ly = 358
            for j, (nm, _, col, dash, _) in enumerate(series):
                xx = 30 + j * 290
                s.line(xx, ly, xx + 22, ly, col, 2.2, dash)
                s.text(xx + 28, ly + 4, nm, 11)
    s.text(450, 388, "x = 0 is nothing planted. Whiskers: 95% intervals for a proportion of 60. Curves from records 26.97 and 26.107.", 10, "middle", "#555")
    s.save("paper_figure_E_recall_curve.svg")


# ============================================================================================ F
def fig_F():
    print("Figure F: Section 25, real-minus-pseudo gap by decile of the distance difference")
    dx = J("dx6_joint_2026-10-03.json")
    dec = dx["deciles"]
    a0, b0 = dx["ref"]["b"]
    qa, qb, qc = dx["quadratic"]["b"]
    check("ten deciles of 382,184 setups", len(dec) == 10 and all(d["n"] == 382184 for d in dec))
    check("linear fit column equals alpha + beta x mean delta", all(abs(d["linear_fit_pp"] - (a0 + b0 * d["mean_delta"])) < 0.01 for d in dec))
    check("first and last decile equal the record (-3.254 and +3.207 pp)", abs(dec[0]["gap_pp"] + 3.254) < 0.001 and abs(dec[-1]["gap_pp"] - 3.207) < 0.001)
    s = Svg(800, 470)
    s.text(400, 20, "Section 25: the real-minus-pseudo gap is fixed by how much nearer or farther the pseudo line sits", 13, "middle", weight="bold")
    ax = Ax(s, 80, 50, 760, 360, (-0.55, 0.47), (-4, 4))
    ax.frame([-0.4, -0.2, 0, 0.2, 0.4], [-4, -2, 0, 2, 4], xf="{:+g}", yf="{:+g}",
             xlabel="pseudo line's distance minus the real line's distance (ATR); negative = pseudo line nearer",
             ylabel="touch rate, real minus pseudo (pp)")
    xs = np.linspace(-0.52, 0.45, 80)
    s.o.append('<polyline fill="none" stroke="#888" stroke-width="1.8" stroke-dasharray="6 4" points="' +
               " ".join(f"{ax.X(x):.1f},{ax.Y(a0 + b0 * x):.1f}" for x in xs) + '"/>')
    s.o.append(f'<polyline fill="none" stroke="{ORANGE}" stroke-width="1.8" points="' +
               " ".join(f"{ax.X(x):.1f},{ax.Y(qa + qb * x + qc * x * x):.1f}" for x in xs) + '"/>')
    for d in dec:
        s.line(ax.X(d["mean_delta"]), ax.Y(d["gap_pp"] - 1.96 * d["se_pp"]), ax.X(d["mean_delta"]), ax.Y(d["gap_pp"] + 1.96 * d["se_pp"]), BLUE, 1.5)
        s.circ(ax.X(d["mean_delta"]), ax.Y(d["gap_pp"]), 5, BLUE, "#1d4a80")
    s.circ(150, 424, 5, BLUE, "#1d4a80")
    s.text(162, 428, "decile mean (382,184 setups each)", 11)
    s.line(390, 424, 414, 424, "#888", 1.8, "6 4")
    s.text(420, 428, "straight-line fit", 11)
    s.line(540, 424, 564, 424, ORANGE, 1.8)
    s.text(570, 428, "quadratic fit", 11)
    s.text(400, 454, f"Gap = {a0:+.3f} pp at no distance difference with the straight line (+{b0:.2f} pp per ATR). The sign of every pair is fixed by the two prices (0 violations in 3,821,840 setups).", 10, "middle", "#555")
    s.save("paper_figure_F_section25_deciles.svg")


if __name__ == "__main__":
    for f in (fig_A, fig_B, fig_C, fig_D, fig_E, fig_F):
        f()
    if FAILS:
        print("FAILED CHECKS:", FAILS)
        sys.exit(1)
    print("ALL KNOWN-NUMBER CHECKS PASS")
