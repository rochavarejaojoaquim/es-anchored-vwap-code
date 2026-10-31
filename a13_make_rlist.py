r"""
a13_make_rlist.py -- the R-LIST of amendment A13 (AMENDMENT_A13_SECOND_B1_POPULATION_DRAFT_v4): STAGE-1 INFORMATION ONLY.
Reads rw_check_independent_out.csv (SHA-256 checked; the pooled side is the STRING "None") and the stored winners in cores_v2\ (col, cut, sense). Nothing from Stage 2, the seal or any dial setting.

THE GROUPING RULE (reviewer 26.112, E-1). Group = (anchor, trigger, eps, side); its representative = the highest Stage-1 t_obs among its members of the 195 (raw p <= 0.05), ties the smaller N.
A TRIPLE is (anchor, trigger, eps). If a triple has a pooled group P and sided groups: keep P alone if P's representative has a HIGHER Stage-1 t_obs than EACH sided group's; otherwise drop P and keep ALL its sided groups
(+1 and -1 do not share setups). Triples with sided groups only keep all of them. This is A13's OWN rule (Stage-1 t_obs, not A8's Stage-2 adjusted p).
HEADLINE = the REMAINDER: groups with NO member among the 31 Romano-Wolf survivors, the rule above applied WITHIN the remainder.

    python a13_make_rlist.py --selftest
    python a13_make_rlist.py            writes a13_rlist_2026-10-06.csv and prints its SHA-256 and the counts
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RW_CSV = os.path.join(HERE, "rw_check_independent_out.csv")
RW_SHA = "44E9D780FE0C8C6C1C2812340887CFDBF41DDB618A3BE4F99C0D4BC47DF4B0B0"
KEY = ["anchor", "trig", "eps", "side_s"]
TRIP = ["anchor", "trig", "eps"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest().upper()


def apply_rule(reps):
    """reps: one row per group with columns anchor, trig, eps, side_s, t_obs, N. Returns the kept groups."""
    out = []
    for _, g in reps.groupby(TRIP):
        pooled = g[g.side_s == "None"]
        sided = g[g.side_s != "None"]
        if len(pooled) and len(sided):
            P = pooled.iloc[0]
            if all(P.t_obs > s for s in sided.t_obs):
                out.append(pooled)
            else:
                out.append(sided)
        else:
            out.append(g)
    return pd.concat(out)


def load():
    if sha(RW_CSV) != RW_SHA:
        raise SystemExit("rw_check_independent_out.csv differs from the frozen hash")
    t = pd.read_csv(RW_CSV)
    t["side_s"] = t["side"].astype(object).where(t["side"].notna(), "None").astype(str)
    p = t.cell.str.split("_", expand=True)
    t["N"] = p[0].str[1:].astype(int)
    t["anchor"], t["trig"], t["eps"] = p[1], p[2], p[3]
    t["sgn"] = np.sign(t.eA)
    return t


def build(t):
    s = t[t.p <= 0.05].copy()
    assert len(s) == 195 and int(s.groupby(KEY).size().sum()) == 195, "member-sum checksum"
    s31 = t[t.rw_adj_p <= 0.05]
    g31 = set(map(tuple, s31[KEY].drop_duplicates().to_numpy()))
    reps = s.sort_values(["t_obs", "N"], ascending=[False, True]).groupby(KEY, as_index=False).first()
    reps["in31"] = [tuple(x) in g31 for x in reps[KEY].to_numpy()]
    reps["size"] = [int(((s.anchor == r.anchor) & (s.trig == r.trig) & (s.eps == r.eps) & (s.side_s == r.side_s)).sum()) for r in reps.itertuples()]
    assert len(reps) == 64 and int(reps["size"].sum()) == 195 and int(reps.in31.sum()) == 18
    all_kept = apply_rule(reps)
    rem = reps[~reps.in31]
    head = apply_rule(rem)
    return s, reps, all_kept, rem, head


def counts(reps, all_kept, rem, head, say=print):
    say(f"  groups: all {len(reps)} (18 with a member in the 31, {len(rem)} remainder) | member-sum {int(reps['size'].sum())} = 195")
    say(f"  the rule on all 64: {len(all_kept)} groups | WITHIN the remainder (the HEADLINE): {len(head)} groups, {int(head['size'].sum())} member cell+sides")
    say(f"  headline by anchor {head.anchor.value_counts().to_dict()} | pooled {int((head.side_s == 'None').sum())} sided {int((head.side_s != 'None').sum())} | sign {int((head.sgn > 0).sum())} pos / {int((head.sgn < 0).sum())} neg (mean {head.sgn.mean():+.3f})")
    # triples with a 31-group: pooled/sided pair (shared setups) vs +1/-1 pair (no shared setups)
    g31 = reps[reps.in31]
    shared_ps = shared_pm = 0
    for h in head.itertuples():
        sib = g31[(g31.anchor == h.anchor) & (g31.trig == h.trig) & (g31.eps == h.eps)]
        if not len(sib):
            continue
        for sg in sib.itertuples():
            if (h.side_s == "None") != (sg.side_s == "None"):
                shared_ps += 1                      # pooled vs sided: the setups are shared
            else:
                shared_pm += 1                      # +1 vs -1: disjoint
    n_share = int(head.apply(lambda h: ((g31.anchor == h.anchor) & (g31.trig == h.trig) & (g31.eps == h.eps)).any(), axis=1).sum())
    say(f"  headline groups sharing their triple with a 31-group: {n_share}; sibling pairs: pooled/sided (SHARED setups) {shared_ps}, +1/-1 (disjoint) {shared_pm}")
    return n_share, shared_ps, shared_pm


def selftest():
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else ""))
        ok[0] &= bool(cond)
    print("\na13_make_rlist.py SELF-TEST -- toy fixtures, then the real counts.\n")
    toy = pd.DataFrame(dict(anchor=["hh"] * 4 + ["ll"] * 3 + ["hv"] * 2, trig=["ATR"] * 9, eps=["3.0"] * 4 + ["3.0"] * 3 + ["4.0"] * 2,
                            side_s=["None", "1.0", "-1.0", "None", "None", "1.0", "-1.0", "1.0", "-1.0"], t_obs=[9, 8, 7, 1, 5, 9, 8, 6, 6.5], N=[60] * 9))
    toy = toy.iloc[[0, 1, 2, 4, 5, 6, 7, 8]]           # hh: P 9 > S+ 8, S- 7 ; ll: P 5 < S+ 9 ; hv: sided only
    k = apply_rule(toy)
    chk("K1 pooled kept ALONE when its t_obs beats EACH sided group", list(k[k.anchor == "hh"].side_s) == ["None"])
    chk("K2 pooled dropped and ALL sided kept when a sided group beats it", sorted(k[k.anchor == "ll"].side_s) == ["-1.0", "1.0"])
    chk("K3 sided-only triples keep both sides (+1 and -1 are disjoint)", sorted(k[k.anchor == "hv"].side_s) == ["-1.0", "1.0"])
    t = load()
    s, reps, all_kept, rem, head = build(t)
    chk("K4 member-sum 195 = 195; 64 groups; 18 with a member in the 31", int(reps["size"].sum()) == 195 and len(reps) == 64 and int(reps.in31.sum()) == 18)
    chk("K5 the pooled side is kept as the STRING 'None' (a naive group-by would drop it)", int((reps.side_s == "None").sum()) > 0 and int(s.groupby(KEY).size().sum()) == 195)
    chk("K6 the headline groups contain no member of the 31", int(head.in31.sum()) == 0)
    print("\n" + ("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED"))
    return ok[0], (t, s, reps, all_kept, rem, head)


def main():
    good, data = selftest()
    if not good:
        raise SystemExit(1)
    if "--selftest" in sys.argv:
        return
    t, s, reps, all_kept, rem, head = data
    counts(reps, all_kept, rem, head)
    rows = []
    for h in head.sort_values(["anchor", "trig", "eps", "side_s"]).itertuples():
        tag = f"{h.cell}_side{h.side_s}"
        st = pd.read_parquet(os.path.join(HERE, "cores_v2", tag + ".parquet")).iloc[0]
        rows.append(dict(cell=h.cell, side=h.side_s, N=h.N, anchor=h.anchor, trigger=h.trig, eps=h.eps, col=st["col"], cut=repr(float(st["cut"])), sense=st["sense"],
                         sign=int(h.sgn), stage1_t_obs=h.t_obs, group_size=int(h.size)))
    out = os.path.join(HERE, "a13_rlist_2026-10-06.csv")
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"\n  wrote {os.path.basename(out)}: {len(rows)} groups | SHA-256 {sha(out)} | rw_check_independent_out.csv {RW_SHA[:8]}... | Stage-1 information only")


if __name__ == "__main__":
    main()
