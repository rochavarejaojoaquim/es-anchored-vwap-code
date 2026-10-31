"""Race (mirror-null) SCORES of the 31 Stage-1 survivors, Stage-1 style, DESCRIPTIVE, train only, read-only.
For each core: its stored winner (col, cut, sense) from cores_v2; the cut's share of the side's finite setups;
known number K0: the fill-rate edges eA, eB recomputed with core_v2.edge_se on the stored mask equal the stored ones;
then the RACE frame (setups and their M1b twins with race in {1,-1}, hit := race == 1) and per block the race edge and
its 5-day block-bootstrap SE (core_v2.edge_se, nboot 500, seed 1). Score = min over blocks of (dir x edge - 2.6 SE),
dir = sign of the stored fill edge (+1 FADE, -1 CONTINUATION): the Stage-1 formula with the sign kept."""
import os, sys
import numpy as np, pandas as pd
ROOT = r"."
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "code")); os.chdir(ROOT)
import core_v2

rw = pd.read_csv("rw_check_independent_out.csv", dtype={"side": str}, keep_default_na=False)
s = rw[rw.rw_adj_p.astype(float) <= 0.05]
assert len(s) == 31
rows = []; k0bad = 0
for _, q in s.iterrows():
    side = None if q.side == "" else float(q.side)
    st = pd.read_parquet(f"cores_v2/{q.cell}_side{'None' if side is None else q.side}.parquet").iloc[0]
    r, c, T = core_v2.load_cell(q.cell, side)
    P = core_v2.prep(r, c)
    v = r[st["col"]].to_numpy(float)
    m = v <= st["cut"] if st["sense"] == "lo" else v > st["cut"]
    share = float(m[np.isfinite(v)].mean())
    cm = core_v2.twin_mask(m, P["src_idx"])
    fA = core_v2.edge_se(r, c, P, m, cm, "A", nboot=500); fB = core_v2.edge_se(r, c, P, m, cm, "B", nboot=500)
    ok = abs(fA[0] - float(q.eA)) < 1e-12 and abs(fB[0] - float(q.eB)) < 1e-12
    k0bad += not ok
    # race frame
    x = pd.read_parquet(f"{core_v2.DUMP}/{q.cell}.parquet", columns=["is_real", "side", "bar", "src_bar", "race"])
    xr = x[x.is_real == 1]
    if side is not None:
        xr = xr[np.sign(xr.side) == side]
    xc = x[x.is_real == 0]; xc = xc[xc.src_bar.isin(set(xr.bar))]
    assert (xr.bar.to_numpy() == r.bar.to_numpy()).all() and (xc.src_bar.to_numpy() == c.src_bar.to_numpy()).all()
    rr, cr = xr.race.to_numpy(), xc.race.to_numpy()
    kr = np.isin(rr, (1, -1)); r2 = r[kr].reset_index(drop=True); r2["hit"] = (rr[kr] == 1).astype(float)
    kc = np.isin(cr, (1, -1)) & c.src_bar.isin(set(r2.bar)).to_numpy()
    c2 = c[kc].reset_index(drop=True); c2["hit"] = (cr[kc] == 1).astype(float)
    P2 = core_v2.prep(r2, c2); m2 = m[kr]; cm2 = core_v2.twin_mask(m2, P2["src_idx"])
    eA = core_v2.edge_se(r2, c2, P2, m2, cm2, "A", nboot=500); eB = core_v2.edge_se(r2, c2, P2, m2, cm2, "B", nboot=500)
    d = 1.0 if float(q.eA) > 0 else -1.0
    sc = min(d * eA[0] - 2.6 * eA[1], d * eB[0] - 2.6 * eB[1]) if eA and eB else np.nan
    rows.append(dict(cell=q.cell, side="pooled" if side is None else ("+1" if side > 0 else "-1"), col=st["col"], sense=st["sense"],
                     cut=float(st["cut"]), share=share, dir="FADE" if d > 0 else "CONT", fill_score=float(q.obs_score),
                     rA=eA[0], seA=eA[1], rB=eB[0], seB=eB[1], race_score=sc, nA=eA[2], nB=eB[2]))
print(f"K0 stored fill-rate edges reproduced on {31 - k0bad} of 31: {'PASS' if k0bad == 0 else 'FAIL'}")
o = pd.DataFrame(rows).sort_values("race_score", ascending=False).reset_index(drop=True)
o.to_csv(os.path.join(os.path.dirname(__file__), "race_scores_31.csv"), index=False)
pd.set_option("display.width", 250)
f = {k: (lambda v: f"{100*v:+.2f}") for k in ("rA", "rB", "race_score", "fill_score")}
f.update(seA=lambda v: f"{100*v:.2f}", seB=lambda v: f"{100*v:.2f}", share=lambda v: f"{100*v:.0f}%", cut=lambda v: f"{v:.4g}")
print(o.to_string(formatters=f))
