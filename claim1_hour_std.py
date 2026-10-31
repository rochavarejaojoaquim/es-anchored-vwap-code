r"""claim1_hour_std.py -- claims 1 and 2 with HOUR-STANDARDISED twins, beside the registered form.
DESCRIPTIVE SENSITIVITY (HANDOFF 26.155, "O1+", confirmed by the author 2026-10-13). It changes no verdict:
the registered claims are 26.65's (claim_checks.py). Read-only; writes one JSON.

WHY. The twins are matched on a 4-hour Chicago bucket x ATR quintile x block (control_matched.py:40-41),
not on the hour. If setups sit at different hours than their twins inside a bucket, and the fill rate varies
by hour, part of "setup minus twin" is an hour difference. 26.155 measured that for FILTERED subsets.
This measures it for the UNFILTERED claims on the 144 registered cells.

WHAT. Inside every (cell, block), each twin gets the weight
    u = (1 / k_src) x n_setups(h) / m_twins(h),   h = the twin's OWN Chicago hour,
k_src = that setup's number of twins in the scheme, n_setups(h) = setups of the cell-block at hour h,
m_twins(h) = sum of 1/k_src over the cell-block's twins at hour h. So the twins are re-weighted to the
setups' hour distribution; with all ratios 1 the statistic IS the registered one. Setups at an hour with no
twin in their cell-block keep their outcome and lose their twin weight (share printed).
    claim 1: P(setup fill) - weighted P(twin fill), for M1b and for M2
    claim 2: P(VWAP first | resolved) on setups - the weighted same on the M1b twins
95% CI: iid day bootstrap, claim_checks.boot_draws (B = 2000, seed 20261003), days = the SETUP's trade date,
weights held fixed (not re-estimated per draw: stated simplification). NOT standardised on ATR quintile
(the quintile edges are not stored); hour only.

KNOWN NUMBERS, before any standardised value is printed (26.65): claim 1 M1b A +4.376 B +1.310; M2 A +4.178
B +1.368; claim 2 A +4.857 B +0.780 (to 0.001 pp). Any FAIL stops the run.

    python claim1_hour_std.py --selftest
    python claim1_hour_std.py data\dump        (about 2-4 min, single process; ESTIMATE, NOT MEASURED)
Not shardable: each number is one pooled statistic over the 144 cells.
"""
import os, sys, json, time, hashlib, platform
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import claim_checks as CC                                   # noqa: E402  (boot_draws, ratio_diff, ci, fill)
from power_registered_set import registered_cells           # noqa: E402

PUB = {("c1", "M1b", "A"): 4.376, ("c1", "M1b", "B"): 1.310, ("c1", "M2", "A"): 4.178, ("c1", "M2", "B"): 1.368,
       ("c2", "M1b", "A"): 4.857, ("c2", "M1b", "B"): 0.780}
COLS = ["is_real", "bar", "src_bar", "day", "block", "T", "w_0.0", "race", "ct_hour"]


def setup_rows(r, tw, T, tag):
    """One row per setup (>= 1 twin of the scheme): its outcomes, the registered twin terms and the
    hour-standardised twin terms. r: setups of ONE cell; tw: that scheme's twins (src_bar, w_0.0, race, ct_hour)."""
    blk = pd.Series(r.block.to_numpy(), index=r.bar.to_numpy())
    t = pd.DataFrame(dict(src=tw.src_bar.to_numpy(), h=tw.ct_hour.to_numpy(), f=CC.fill(tw["w_0.0"], T),
                          res=np.isin(tw.race, (1, -1)).astype(float), vf=(tw.race.to_numpy() == 1).astype(float)))
    t = t[t.src.isin(blk.index)]
    t["k"] = t.groupby("src").f.transform("size").astype(float)
    t["blk"] = blk.reindex(t.src).to_numpy()
    t["a"] = 1.0 / t.k                                           # as built: each setup's twin mean
    has = r.bar.isin(set(t.src)).to_numpy()
    rr = r[has]
    ns = rr.groupby(["block", "ct_hour"]).size()
    mt = t.groupby(["blk", "h"]).a.sum()
    ratio = (ns.reindex(mt.index).fillna(0.0) / mt)
    t["u"] = t.a.to_numpy() * ratio.reindex(pd.MultiIndex.from_arrays([t.blk, t.h])).to_numpy()
    lost = float(ns[~ns.index.isin(mt.index)].sum())             # setups at an hour with no twin
    agg = pd.DataFrame(dict(
        tf=(t.a * t.f).groupby(t.src).sum(), t_res=(t.a * t.res).groupby(t.src).sum(), t_vf=(t.a * t.vf).groupby(t.src).sum(),
        u=t.u.groupby(t.src).sum(), uf=(t.u * t.f).groupby(t.src).sum(),
        ures=(t.u * t.res).groupby(t.src).sum(), uvf=(t.u * t.vf).groupby(t.src).sum()))
    out = pd.DataFrame(dict(day=rr.day.to_numpy(), block=rr.block.to_numpy(), w=1.0, rf=CC.fill(rr["w_0.0"], T),
                            r_res=np.isin(rr.race, (1, -1)).astype(float), r_vf=(rr.race.to_numpy() == 1).astype(float)))
    a = agg.reindex(rr.bar.to_numpy())
    for c in agg.columns:
        out[c] = a[c].to_numpy()
    return out, lost


def stats(df):
    """Registered and hour-standardised claim values for one block's rows, with iid day-bootstrap CIs."""
    codes, D = CC.day_codes(df.day)
    sm = CC._sums(df, ["w", "rf", "r_res", "r_vf", "tf", "t_res", "t_vf", "u", "uf", "ures", "uvf"], codes, D)
    S = CC.boot_draws(D, "iid")
    def one(a, b, c, d):
        lo, hi, _ = CC.ci(CC.ratio_diff(sm, a, b, c, d, S))
        return dict(est=100 * CC.ratio_diff(sm, a, b, c, d), lo=100 * lo, hi=100 * hi)
    out = dict(c1_built=one("rf", "w", "tf", "w"), c1_std=one("rf", "w", "uf", "u"),
               c2_built=one("r_vf", "r_res", "t_vf", "t_res"), c2_std=one("r_vf", "r_res", "uvf", "ures"), D=D, n=len(df))
    d1 = (sm["tf"][S].sum(1) / sm["w"][S].sum(1)) - (sm["uf"][S].sum(1) / sm["u"][S].sum(1))
    lo, hi, _ = CC.ci(d1)
    out["c1_shift"] = dict(est=100 * (sm["tf"].sum() / sm["w"].sum() - sm["uf"].sum() / sm["u"].sum()), lo=100 * lo, hi=100 * hi)
    d2 = (sm["t_vf"][S].sum(1) / sm["t_res"][S].sum(1)) - (sm["uvf"][S].sum(1) / sm["ures"][S].sum(1))
    lo, hi, _ = CC.ci(d2)
    out["c2_shift"] = dict(est=100 * (sm["t_vf"].sum() / sm["t_res"].sum() - sm["uvf"].sum() / sm["ures"].sum()), lo=100 * lo, hi=100 * hi)
    return out


# ------------------------------------------------------------------ self-test
def _toy(seed, gap=0.0, hour_effect=0.15, tilt=0.8, n_days=600, per_day=40, uniform_w=False):
    """One cell, two blocks, bucket = hours {20,21,22,23}. Generic fill rate rises with the hour by hour_effect
    per hour. Setups sit at hours 22-23 with probability `tilt`; twins are uniform over the bucket (the build)."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day); block = np.where(day < n_days // 2, "A", "B")
    hs = np.where(rng.random(n) < tilt, rng.integers(22, 24, n), rng.integers(20, 22, n))
    base = lambda h: 0.30 + hour_effect * (h - 20)
    T = 100
    rfill = rng.random(n) < base(hs) + gap
    r = pd.DataFrame({"bar": np.arange(n), "day": day, "block": block, "ct_hour": hs,
                      "w_0.0": np.where(rfill, 5, -1), "race": np.where(rng.random(n) < 0.5 + gap, 1, -1)})
    ht = rng.integers(20, 24, 2 * n); tf = rng.random(2 * n) < base(ht)
    tw = pd.DataFrame({"src_bar": np.repeat(np.arange(n), 2), "ct_hour": ht, "w_0.0": np.where(tf, 5, -1),
                       "race": np.where(rng.random(2 * n) < 0.5, 1, -1)})
    rows, lost = setup_rows(r, tw, T, "M1b")
    if uniform_w:
        for a, b in (("u", "w"), ("uf", "tf"), ("ures", "t_res"), ("uvf", "t_vf")):
            rows[a] = rows[b]
    return {b: stats(rows[rows.block == b]) for b in "AB"}, lost


def selftest():
    ok = True
    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    print("SELF-TEST (toy fixtures, no data)")
    s, lost = _toy(1)
    a = s["A"]
    chk("K1 no VWAP effect, setups tilted to high-fill hours: the REGISTERED form shows a false gap",
        a["c1_built"]["lo"] > 3, f"built {a['c1_built']['est']:+.2f} pp [{a['c1_built']['lo']:+.2f}, {a['c1_built']['hi']:+.2f}]")
    chk("K2 ... and the hour-standardised form removes it", abs(a["c1_std"]["est"]) < 1.0 and a["c1_std"]["lo"] < 0 < a["c1_std"]["hi"],
        f"std {a['c1_std']['est']:+.2f} pp [{a['c1_std']['lo']:+.2f}, {a['c1_std']['hi']:+.2f}]")
    chk("K3 the shift (built twin rate minus standardised twin rate) is negative and its CI excludes 0",
        a["c1_shift"]["hi"] < 0, f"{a['c1_shift']['est']:+.2f} [{a['c1_shift']['lo']:+.2f}, {a['c1_shift']['hi']:+.2f}]")
    s, _ = _toy(2, gap=0.04)
    chk("K4 a planted +4 pp real gap is recovered by the standardised form", abs(s["B"]["c1_std"]["est"] - 4) < 1.2, f"{s['B']['c1_std']['est']:+.2f} pp")
    chk("K5 the race is flat by hour in the toy: built and standardised claim 2 agree and recover +4 pp",
        abs(s["B"]["c2_std"]["est"] - s["B"]["c2_built"]["est"]) < 0.8 and abs(s["B"]["c2_std"]["est"] - 4) < 1.5,
        f"built {s['B']['c2_built']['est']:+.2f}, std {s['B']['c2_std']['est']:+.2f}")
    s, _ = _toy(3, tilt=0.5)
    chk("K6 setups NOT tilted (uniform over the bucket): standardising changes nothing material",
        abs(s["A"]["c1_shift"]["est"]) < 0.5, f"shift {s['A']['c1_shift']['est']:+.3f} pp")
    s, _ = _toy(4, hour_effect=0.0)
    chk("K7 no hour effect in the fill rate: tilted setups, yet no false gap and no shift",
        abs(s["A"]["c1_built"]["est"]) < 1.0 and abs(s["A"]["c1_shift"]["est"]) < 0.5, f"built {s['A']['c1_built']['est']:+.2f}, shift {s['A']['c1_shift']['est']:+.3f}")
    s, _ = _toy(1, uniform_w=True)
    chk("M1 MUTATION weights forced to the as-built ones: the standardised form no longer removes the false gap",
        s["A"]["c1_std"]["lo"] > 3 and abs(s["A"]["c1_shift"]["est"]) < 1e-9, f"std {s['A']['c1_std']['est']:+.2f}")
    # weights sum to the number of setups per cell-block
    rng = np.random.default_rng(9); n = 4000
    r = pd.DataFrame({"bar": np.arange(n), "day": np.arange(n) // 20, "block": np.where(np.arange(n) < n // 2, "A", "B"),
                      "ct_hour": rng.integers(20, 24, n), "w_0.0": 5, "race": 1})
    tw = pd.DataFrame({"src_bar": np.repeat(np.arange(n), 2)[:-1], "ct_hour": rng.integers(20, 24, 2 * n - 1), "w_0.0": 5, "race": 1})
    rows, lost = setup_rows(r, tw, 100, "M1b")
    chk("K8 weights sum to the number of setups in each block (one setup has 1 twin)", all(abs(rows[rows.block == b].u.sum() - (rows.block == b).sum()) < 1e-6 for b in "AB") and lost == 0,
        f"A {rows[rows.block=='A'].u.sum():.3f} vs {(rows.block=='A').sum()}")
    print("ALL DETECTORS ALIVE" if ok else "SELF-TEST FAILED"); return ok


# ------------------------------------------------------------------ main
def main(dump):
    if not selftest():
        return 1
    t0 = time.time()
    cells = registered_cells()
    assert len(cells) == 144, len(cells)
    import pyarrow.parquet as pq
    m2n = pq.read_schema(os.path.join(dump, cells[0] + ".m2.parquet")).names
    if "ct_hour" not in m2n:
        print("[STOP] the .m2.parquet files carry no ct_hour column; M2 cannot be standardised. Nothing printed."); return 1
    parts = {"M1b": [], "M2": []}; lost = {"M1b": 0.0, "M2": 0.0}
    for c in cells:
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=COLS)
        T = int(x["T"].iloc[0]); r = x[x.is_real == 1]
        tws = {"M1b": x[x.is_real == 0][["src_bar", "w_0.0", "race", "ct_hour"]],
               "M2": pd.read_parquet(os.path.join(dump, c + ".m2.parquet"), columns=["src_bar", "w_0.0", "race", "ct_hour"])}
        for tag, tw in tws.items():
            rows, l = setup_rows(r, tw, T, tag); parts[tag].append(rows); lost[tag] += l
    res = {}
    for tag in ("M1b", "M2"):
        df = pd.concat(parts[tag], ignore_index=True)
        for b in "AB":
            res[(tag, b)] = stats(df[df.block == b])
        res[(tag, "n")] = len(df)
    print(f"\nloaded 144 cells in {time.time() - t0:.0f}s; setups M1b {res[('M1b', 'n')]:,}, M2 {res[('M2', 'n')]:,}")
    bad = 0
    for (cl, tag, b), pub in PUB.items():
        got = res[(tag, b)]["c1_built" if cl == "c1" else "c2_built"]["est"]
        okk = abs(got - pub) < 0.001 + 5e-4
        bad += not okk
        print(f"  [{'PASS' if okk else 'FAIL'}] K0 {cl} {tag} {b}: registered form {got:+.3f} pp vs 26.65's {pub:+.3f}")
    if bad:
        print("[STOP] a known number did not reproduce. No standardised value is printed."); return 1
    stamp = (f"claims 1-2 with hour-standardised twins (26.155 O1+), DESCRIPTIVE | 144 registered cells | clusters = setup trade date | "
             f"iid day bootstrap B={CC.B_BOOT} seed {CC.SEED}, weights fixed | hour only, not ATR quintile | tau 0, wick fill within T | {dump} | "
             f"code-set per verify_pc | claim1_hour_std.py {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]} | "
             f"claim_checks.py {hashlib.sha256(open(CC.__file__, 'rb').read()).hexdigest()[:16]} | {platform.node()} | ES train A+B, sealed untouched | raw, no cost")
    print("\n" + stamp)
    f = lambda d: f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    for tag in ("M1b", "M2"):
        print(f"  setups at an hour with no twin in their cell-block ({tag}): {int(lost[tag]):,} of {res[(tag, 'n')]:,}")
        for b in "AB":
            s = res[(tag, b)]
            print(f"  CLAIM 1 {tag} block {b}: registered {f(s['c1_built'])}   hour-standardised {f(s['c1_std'])}   twin-rate shift (built - std) {f(s['c1_shift'])} pp")
    for b in "AB":
        s = res[("M1b", b)]
        print(f"  CLAIM 2 M1b block {b}: registered {f(s['c2_built'])}   hour-standardised {f(s['c2_std'])}   twin-race shift {f(s['c2_shift'])} pp")
    out = os.path.join(HERE, f"claim1_hour_std_{time.strftime('%Y-%m-%d')}.json")
    json.dump(dict(stamp=stamp, lost=lost, results={f"{k[0]}_{k[1]}": v for k, v in res.items()}), open(out, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.basename(out)}   {time.time() - t0:.0f}s")
    print("READING (fixed here before the run): DESCRIPTIVE. The registered verdicts are 26.65's and do not change. "
          "If a standardised CI still lies above 0 the claim holds against same-hour controls; if it includes 0 that is REPORTED beside the claim, "
          "and any change of the claim's wording is the author's decision after a reviewer read.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1]))
