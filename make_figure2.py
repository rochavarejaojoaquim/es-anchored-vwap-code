"""make_figure2.py -- Figure 2 of the paper (SVG, no third-party packages): the yearly estimates of claim 1
(M1b twins) and claim 2 with 95% day-bootstrap intervals, the published pooled estimate of each training
period as a level line, and the published held-out estimate. Reads ONLY already-written result files
(paper_yearly_<date>.json, claim_checks_2026-10-03.json, seal_use1_outputs/claims_seal_2026-10-15.json);
recomputes nothing and draws no fitted curve."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
yj = json.load(open(sorted(glob.glob(os.path.join(HERE, "paper_yearly_2*.json")))[-1]))["results"]
tr = json.load(open(os.path.join(HERE, "claim_checks_2026-10-03.json")))["PRIMARY: all setups of the 144 registered cells"]["iid"]
sl = json.load(open(os.path.join(HERE, "seal_use1_outputs", "claims_seal_2026-10-15.json")))["results"]["iid"]
panels = [("Claim 1 (M1b twins): line touched vs matched level, pp", yj["c1_m1b"], tr["c1"]["M1b A"]["est_pp"], tr["c1"]["M1b B"]["est_pp"], sl["M1b"], -2.0, 8.0),
          ("Claim 2: line before mirror among resolved setups vs matched moment, pp", yj["c2"], tr["c2"]["race A"]["est_pp"], tr["c2"]["race B"]["est_pp"], sl["race"], -2.0, 9.0)]
W, PH, L, R, T, B = 760, 300, 62, 20, 40, 40
tot = 2 * PH + 10 + 62
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{tot}" viewBox="0 0 {W} {tot}" font-family="sans-serif" font-size="12">',
     f'<rect width="{W}" height="{tot}" fill="white"/>']
for pi, (title, res, mA, mB, ho, ymin, ymax) in enumerate(panels):
    oy = pi * (PH + 10)

    def X(t):
        return L + (t - 2009.5) / (2027.0 - 2009.5) * (W - L - R)

    def Y(v):
        return oy + T + (ymax - v) / (ymax - ymin) * (PH - T - B)
    o.append(f'<text x="{W/2}" y="{oy+22}" text-anchor="middle" font-size="13">{title}</text>')
    v = int(ymin)
    while v <= ymax:
        o.append(f'<line x1="{L}" y1="{Y(v)}" x2="{W-R}" y2="{Y(v)}" stroke="{"#000" if v == 0 else "#ddd"}"/>')
        o.append(f'<text x="{L-8}" y="{Y(v)+4}" text-anchor="end">{v}</text>')
        v += 2
    for yr in range(2010, 2027, 2):
        o.append(f'<text x="{X(yr+0.5)}" y="{oy+PH-B+22}" text-anchor="middle">{yr}</text>')
    o.append(f'<rect x="{X(2018.0)}" y="{oy+T}" width="{X(2022.0)-X(2018.0)}" height="{PH-T-B}" fill="#f6e3e0" opacity="0.6"/>')
    o.append(f'<text x="{(X(2018.0)+X(2022.0))/2}" y="{oy+T+14}" text-anchor="middle" fill="#a33">held out</text>')
    for a, b, m in ((2010.0, 2018.0, mA), (2022.0, 2026.7, mB)):
        o.append(f'<line x1="{X(a)}" y1="{Y(m)}" x2="{X(b)}" y2="{Y(m)}" stroke="#1f77b4" stroke-width="2" stroke-dasharray="6 4"/>')
    for r in res["years"]:
        x = X(r["t"])
        o.append(f'<line x1="{x}" y1="{Y(r["lo"])}" x2="{x}" y2="{Y(r["hi"])}" stroke="#555" stroke-width="1.5"/>')
        o.append(f'<circle cx="{x}" cy="{Y(r["est"])}" r="4" fill="#222"/>')
    hx = X(2020.0)
    o.append(f'<line x1="{hx}" y1="{Y(ho["lo_pp"])}" x2="{hx}" y2="{Y(ho["hi_pp"])}" stroke="#c0392b" stroke-width="2"/>')
    o.append(f'<rect x="{hx-5}" y="{Y(ho["est_pp"])-5}" width="10" height="10" fill="#c0392b"/>')
lx, ly = L + 4, 2 * PH + 10 + 18
o.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="#222"/><text x="{lx+10}" y="{ly+4}">yearly estimate, 95% interval (training years; 2010 and 2026 are partial)</text>')
o.append(f'<line x1="{lx+400}" y1="{ly}" x2="{lx+414}" y2="{ly}" stroke="#1f77b4" stroke-width="2" stroke-dasharray="6 4"/><text x="{lx+420}" y="{ly+4}">pooled estimate of the training period</text>')
o.append(f'<rect x="{lx-4}" y="{ly+16}" width="8" height="8" fill="#c0392b"/><text x="{lx+10}" y="{ly+24}">held-out period 2018-2021 (one pooled estimate, not split)</text>')
o.append("</svg>")
out = os.path.join(HERE, "paper_figure2_yearly.svg")
open(out, "w", encoding="utf-8").write("\n".join(o))
print("wrote", out)
