"""make_figure1.py -- Figure 1 of the paper (SVG, no third-party packages): claims 1 and 2 in the
three periods with 95% day-bootstrap intervals. Reads ONLY already-written result files
(claim_checks_2026-10-03.json for the training periods, seal_use1_outputs/claims_seal_2026-10-15.json
for the held-out period); recomputes nothing."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
tr = json.load(open(os.path.join(HERE, "claim_checks_2026-10-03.json")))
tr = tr["PRIMARY: all setups of the 144 registered cells"]["iid"]
sl = json.load(open(os.path.join(HERE, "seal_use1_outputs", "claims_seal_2026-10-15.json")))["results"]["iid"]

series = [  # label, marker, values for older / held out / recent
    ("claim 1, M1b twins", "circle", [tr["c1"]["M1b A"], sl["M1b"], tr["c1"]["M1b B"]]),
    ("claim 1, M2 twins", "ring", [tr["c1"]["M2 A"], sl["M2"], tr["c1"]["M2 B"]]),
    ("claim 2, race (resolved setups)", "square",[tr["c2"]["race A"], sl["race"], tr["c2"]["race B"]]),
]
periods = ["2010-2017", "2018-2021 (held out)", "2022-2026"]
W, H, L, R, T, B = 720, 400, 70, 20, 40, 60
ymin, ymax = -0.5, 5.5


def X(j, i):  # period j, series i
    return L + (j + 0.5) / 3 * (W - L - R) + (i - 1) * 28


def Y(v):
    return T + (ymax - v) / (ymax - ymin) * (H - T - B)


o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
     f'font-family="sans-serif" font-size="12">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14">Claims 1 and 2 by period: '
     f'setup minus matched random level, pp (95% day-bootstrap intervals)</text>']
for v in range(0, 6):
    y = Y(v)
    o.append(f'<line x1="{L}" y1="{y}" x2="{W-R}" y2="{y}" stroke="{"#000" if v == 0 else "#ddd"}"/>')
    o.append(f'<text x="{L-8}" y="{y+4}" text-anchor="end">{v}</text>')
for j, p in enumerate(periods):
    o.append(f'<text x="{L + (j + 0.5) / 3 * (W - L - R)}" y="{H-B+28}" text-anchor="middle">{p}</text>')
for i, (lab, mk, vals) in enumerate(series):
    for j, v in enumerate(vals):
        x, c = X(j, i), ("#c0392b" if j == 1 else "#333")
        o.append(f'<line x1="{x}" y1="{Y(v["lo_pp"])}" x2="{x}" y2="{Y(v["hi_pp"])}" stroke="{c}" stroke-width="2"/>')
        for e in (v["lo_pp"], v["hi_pp"]):
            o.append(f'<line x1="{x-4}" y1="{Y(e)}" x2="{x+4}" y2="{Y(e)}" stroke="{c}" stroke-width="2"/>')
        yy = Y(v["est_pp"])
        if mk == "circle":
            o.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{c}"/>')
        elif mk == "ring":
            o.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="white" stroke="{c}" stroke-width="2"/>')
        else:
            o.append(f'<rect x="{x-5}" y="{yy-5}" width="10" height="10" fill="{c}"/>')
lx = L + 10
for i, (lab, mk, _) in enumerate(series):
    y = T + 14 + i * 18
    if mk == "circle":
        o.append(f'<circle cx="{lx}" cy="{y}" r="5" fill="#333"/>')
    elif mk == "ring":
        o.append(f'<circle cx="{lx}" cy="{y}" r="5" fill="white" stroke="#333" stroke-width="2"/>')
    else:
        o.append(f'<rect x="{lx-5}" y="{y-5}" width="10" height="10" fill="#333"/>')
    o.append(f'<text x="{lx+12}" y="{y+4}">{lab}</text>')
o.append("</svg>")
out = os.path.join(HERE, "paper_figure1_claims_by_period.svg")
open(out, "w", encoding="utf-8").write("\n".join(o))
print("wrote", out)
for lab, _, vals in series:
    print(lab, [f"{v['est_pp']:+.2f} [{v['lo_pp']:+.2f}, {v['hi_pp']:+.2f}]" for v in vals])
