"""Three extra measurements for the roll-rule sensitivity (paper Section 3.7), requested by the author on 2026-10-21 after the reviewer read of v24
(HANDOFF 26.227). Training periods only. Imports roll_sensitivity.py UNCHANGED (its K3 and acceptance receipts name its hash) and writes
nothing under code\\.

  --selftest             toy checks (window exclusion, subset partition, change-date detection); no data
  --extras RULE          (i) claims on the setups that exist only under the production rule, only under RULE, and on the common setups,
                         by block; (ii) claims on the ROLL-FREE setups: every setup within +-W calendar days (W = 7, fixed here) of ANY contract
                         change under any of the three rules is removed together with its twins. ~2-5 min, single process.
  --seed S               twin-seed-only rebuild: rule SAME, lookback 60, spec.TWIN_SEED patched to S in memory (production value 17), into
                         roll_sens\\root_same_N60_seedS. Setups must equal the published ones; only the twin draw changes. ~25 min, one process.
  --seed-claims S        the six estimates for the 36 N=60 cells from that rebuild against the published dump, beside the lagged and calendar
                         rules restricted to the same 36 cells and the fresh-build floor. ~1 min.

READING RULES (fixed before any run; descriptive, nothing here changes a claim or a threshold):
  (i)  the three subsets are reported with their sizes; no cutoff. The setups that change are 1.2% (lagged) and 0.4% (calendar) of the total, so
       their own estimates have wide intervals; they say whether the changed setups behave like the rest, not more.
  (ii) a roll-free cell HOLDS iff its iid 95% lower end is above 0 in the published roll-free frame and in each rule's roll-free frame.
       Differences are printed in pp (rule minus published on the same roll-free setups; published roll-free minus published full) and flagged
       above 0.5 pp (the convention of Table 3.2a); read beside the 0.108 pp reference size of Section 3.7. W = 7 is a convention, not tuned,
       and no other W is run.
  seed: the spread of the six estimates across twin seeds (setups identical) is the like-for-like size of twin-draw noise at lookback 60;
       it is read beside the rule differences at lookback 60. Seeds 18 and 19 are the only ones run.
"""
import os, sys, json, time, platform, hashlib, runpy, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
sys.path.insert(0, HERE)
import roll_sensitivity as RS
SENS = RS.SENS
W_DAYS = 7
SEEDS_ALLOWED = (18, 19)


def sha(p):
    return RS.sha(p)


# ---------------------------------------------------------------------------------------------------- pure helpers (toy-testable)
def change_dates(chosen):
    """chosen: Series indexed by date -> instrument id. Dates on which the chosen contract differs from the previous date's."""
    c = chosen.sort_index()
    return list(c.index[1:][(c.values[1:] != c.values[:-1])])


def window_mask(dates, change_days, W):
    """True for dates (datetime64[D] ints as np.int64 days) within +-W days of any change day."""
    d = np.asarray(dates, dtype="int64")
    ch = np.sort(np.unique(np.asarray(change_days, dtype="int64")))
    if len(ch) == 0:
        return np.zeros(len(d), dtype=bool)
    i = np.searchsorted(ch, d)
    left = np.where(i > 0, d - ch[np.clip(i - 1, 0, len(ch) - 1)], 10**9)
    right = np.where(i < len(ch), ch[np.clip(i, 0, len(ch) - 1)] - d, 10**9)
    return np.minimum(left, right) <= W


def partition(kp, kn):
    common = set(kp) & set(kn)
    return common


def selftest():
    ok = True
    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]).date
    s = pd.Series([1, 1, 2, 2, 3], index=idx)
    cd = change_dates(s)
    chk("K1 change dates are the first date of each new contract", [str(x) for x in cd] == ["2020-01-03", "2020-01-05"])
    day = lambda x: int(np.datetime64(x, "D").astype("int64"))
    chg = [day("2020-01-10")]
    d = np.array([day("2020-01-02"), day("2020-01-03"), day("2020-01-10"), day("2020-01-17"), day("2020-01-18"), day("2019-12-01")])
    chk("K2 window +-7 days: edges included, 8 days excluded", list(window_mask(d, chg, 7)) == [False, True, True, True, False, False])
    chk("K3 empty change list excludes nothing", not window_mask(d, [], 7).any())
    # mutation: W = 6 must flip the edge cases
    chk("M1 W = 6 changes the edge cases (the test can fail)", list(window_mask(d, chg, 6)) != list(window_mask(d, chg, 7)))
    kp = pd.Series(["a|1|1", "b|2|1", "c|3|-1"]); kn = pd.Series(["a|1|1", "b|2|1", "d|4|1"])
    com = partition(kp, kn)
    chk("K4 partition: only-published + common = published, only-new + common = new",
        (~kp.isin(com)).sum() + kp.isin(com).sum() == len(kp) and (~kn.isin(com)).sum() + kn.isin(com).sum() == len(kn) and len(com) == 2)
    chk("K5 seeds allowed are exactly 18 and 19", SEEDS_ALLOWED == (18, 19))
    print("SELF-TEST", "ALL DETECTORS ALIVE" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------------------------------- data modes
def _chosen(rule):
    d = pd.read_parquet(os.path.join(SENS, f"raw_{rule}", "data", "outright.parquet"), columns=["ts", "instrument_id"])
    d["date"] = d.ts.dt.tz_convert("UTC").dt.date
    return d.groupby("date").instrument_id.agg(lambda s: s.value_counts().idxmax())


def _days(dates):
    return np.array([np.datetime64(x, "D").astype("int64") for x in dates], dtype="int64")


def est_str(r):
    return f"{r['est_pp']:+.3f} [{r['lo_pp']:+.3f},{r['hi_pp']:+.3f}] (n {r.get('n', 0):,})"


def extras(rule):
    import claim_checks as CC
    from power_registered_set import registered_cells
    dump = os.path.join(SENS, f"dump_{rule}")
    cells = registered_cells()
    pub, _, _ = CC.load(RS.REAL_DUMP, cells); new, _, _ = CC.load(dump, cells)
    pub["ts"] = RS._load_ts(RS.REAL_DUMP, cells); new["ts"] = RS._load_ts(dump, cells)
    p_iid = RS.six(CC, pub, "iid")
    if not all(abs(p_iid[k]["est_pp"] - RS.PUB[k]) < 0.0006 for k in RS.PUB):
        print("  [FAIL] KNOWN NUMBER: the published dump does not reproduce 26.65's six estimates"); return 1
    print(f"ROLL EXTRAS ({rule}) | {len(cells)} cells | published {len(pub):,}, new {len(new):,} setups | [PASS] KNOWN NUMBER (26.65's six estimates)")
    key = lambda d: d.cell.astype(str) + "|" + d.ts.astype(str) + "|" + d.side.astype(int).astype(str)
    kp, kn = key(pub), key(new); com = partition(kp, kn)
    out = dict(rule=rule, script_sha=sha(__file__), roll_sensitivity_sha=sha(RS.__file__), W_days=W_DAYS, finished=None)
    # (i) subsets
    subs = {"common (published rows)": pub[kp.isin(com)], "common (new rows)": new[kn.isin(com)],
            "only published": pub[~kp.isin(com)], "only new": new[~kn.isin(com)]}
    print(f"  (i) SETUP SUBSETS: common {len(com):,} | only published {len(pub)-len(com):,} | only new {len(new)-len(com):,}")
    out["subsets"] = {}
    for name, df in subs.items():
        blkn = df.block.value_counts().to_dict()
        r = {}
        for blk in ("A", "B"):                      # per block: an empty block (calendar rule, 2010-2017) must not stop the other block
            if blkn.get(blk, 0) < 1000:
                print(f"    {name}: block {blk} has {blkn.get(blk, 0):,} setups, not estimated")
                continue
            try:
                r.update(RS.six(CC, df[df.block == blk], "iid", (blk,)))
            except Exception as e:
                print(f"    {name}: block {blk} not estimable ({type(e).__name__}: {str(e)[:60]})")
        if not r:
            continue
        print(f"    {name} (setups A {blkn.get('A', 0):,}, B {blkn.get('B', 0):,})")
        out["subsets"][name] = {f"{k[0]}_{k[1]}": v for k, v in r.items()}
        for (tag, blk), x in r.items():
            nm = f"claim 1 {tag}" if tag != "race" else "claim 2 race"
            print(f"      {nm:<13} {blk}: {est_str(x)}")
    # (ii) roll-free
    ch = {}
    for rl in RS.RULES:
        ch[rl] = change_dates(_chosen(rl))
    allch = sorted(set(sum(ch.values(), [])))
    cdays = _days(allch)
    print(f"  (ii) ROLL-FREE: contract-change dates same {len(ch['same'])}, lagged {len(ch['lagged'])}, calendar8 {len(ch['calendar8'])}; union {len(allch)}; "
          f"W = +-{W_DAYS} days")
    def free(df):
        d = _days(pd.to_datetime(df.ts, utc=True).dt.date)
        return df[~window_mask(d, cdays, W_DAYS)]
    pf, nf = free(pub), free(new)
    print(f"    setups kept: published {len(pf):,} of {len(pub):,} ({100*len(pf)/len(pub):.1f}%), {rule} {len(nf):,} of {len(new):,} ({100*len(nf)/len(new):.1f}%)")
    pfr, nfr = RS.six(CC, pf, "iid"), RS.six(CC, nf, "iid")
    holds = True; flags = []
    print("    claim | block | published roll-free | rule roll-free | rule - published (roll-free) | published roll-free - published full")
    out["roll_free"] = {}
    for (tag, blk), x in pfr.items():
        y = nfr[(tag, blk)]
        holds &= (x["lo_pp"] > 0) and (y["lo_pp"] > 0)
        d1, d2 = y["est_pp"] - x["est_pp"], x["est_pp"] - p_iid[(tag, blk)]["est_pp"]
        flags += [(tag, blk)] if (abs(d1) > RS.FLAG_PP or abs(d2) > RS.FLAG_PP) else []
        nm = f"claim 1 {tag}" if tag != "race" else "claim 2 race"
        print(f"      {nm:<13} {blk}: {est_str(x)} | {est_str(y)} | {d1:+.3f} | {d2:+.3f}")
        out["roll_free"][f"{tag}_{blk}"] = dict(pub_free=x, new_free=y, d_rule=d1, d_pub_full=d2)
    print(f"  READING: all six lower ends above zero in the published roll-free frame and in the {rule} roll-free frame: {'YES' if holds else 'NO'}; "
          f"differences over {RS.FLAG_PP} pp: {flags if flags else 'none'}")
    out.update(holds=bool(holds), flags=[list(f) for f in flags], finished=time.strftime("%Y-%m-%d %H:%M:%S"), machine=platform.node())
    json.dump(out, open(os.path.join(SENS, f"roll_extras_{rule}_{time.strftime('%Y-%m-%d')}.json"), "w"), indent=1, default=float)
    return 0


def seed_build(S):
    if S not in SEEDS_ALLOWED:
        raise SystemExit(f"--seed must be one of {SEEDS_ALLOWED}")
    ar = os.path.join(SENS, "accept_same_receipt.json")
    if not (os.path.exists(ar) and json.load(open(ar)).get("pass_") and json.load(open(ar)).get("script_sha") == sha(RS.__file__)):
        raise SystemExit("REFUSED: the SAME acceptance has not passed for the current roll_sensitivity.py")
    if subprocess.call([sys.executable, __file__, "--selftest"]) != 0:
        return 1
    raw = os.path.join(SENS, "raw_same", "data", "outright.parquet")
    root = os.path.join(SENS, f"root_same_N60_seed{S}")
    if os.path.exists(os.path.join(root, "data", "dump")):
        raise SystemExit(f"REFUSED: {root} already holds a dump (rename it into stale\\ first)")
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    dst = os.path.join(root, "data", "outright.parquet")
    if not os.path.exists(dst):
        os.link(raw, dst)
    if os.path.getsize(dst) != os.path.getsize(raw) or sha(dst) != sha(raw):
        raise SystemExit("REFUSED: scratch outright differs from the SAME-filtered file")
    os.environ["VWAP_ROOT"] = root; os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, CODE)
    import spec
    old = spec.TWIN_SEED
    spec.TWIN_SEED = S                      # in memory only; dump_episodes does `from spec import ... TWIN_SEED` after this
    import loader, pipeline
    ok_paths = (os.path.normcase(os.path.realpath(loader.RAW)) == os.path.normcase(os.path.realpath(dst)) and
                os.path.normcase(os.path.realpath(pipeline.OUT)).startswith(os.path.normcase(os.path.realpath(root))))
    if not ok_paths:
        raise SystemExit("REFUSED: loader.RAW / pipeline.OUT do not point into the scratch root")
    t0 = time.time(); w = sha(__file__)
    sys.argv = [os.path.join(CODE, "dump_episodes.py"), "60"]
    g = runpy.run_path(os.path.join(CODE, "dump_episodes.py"), run_name="__main__")
    live = g.get("TWIN_SEED") == S
    print(f"TWIN_SEED in dump_episodes = {g.get('TWIN_SEED')} (production {old}): {'LIVE' if live else 'NOT LIVE'}")
    rec = dict(N=60, rule="same", twin_seed=S, production_twin_seed=old, seed_live=bool(live), root=root, wrapper_sha=w, roll_sensitivity_sha=sha(RS.__file__),
               raw_sha=sha(dst), dump_episodes_sha=sha(os.path.join(CODE, "dump_episodes.py")), spec_sha=sha(os.path.join(CODE, "spec.py")),
               wall_s=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
               machine=dict(node=platform.node(), cpu=os.cpu_count(), python=platform.python_version()))
    json.dump(rec, open(os.path.join(root, "build_receipt.json"), "w"), indent=1)
    print(f"BUILD DONE seed={S} N=60 {rec['wall_s']}s")
    return 0 if live else 1


def seed_claims(S):
    import claim_checks as CC
    from power_registered_set import registered_cells
    root = os.path.join(SENS, f"root_same_N60_seed{S}")
    rc = os.path.join(root, "build_receipt.json")
    if not (os.path.exists(rc) and json.load(open(rc)).get("seed_live") and json.load(open(rc)).get("twin_seed") == S):
        raise SystemExit("no live-seed build receipt")
    cells = [c for c in registered_cells() if c.startswith("N60_")]
    pub, _, _ = CC.load(RS.REAL_DUMP, cells)
    pub["ts"] = RS._load_ts(RS.REAL_DUMP, cells)
    seed, _, _ = CC.load(os.path.join(root, "data", "dump"), cells); seed["ts"] = RS._load_ts(os.path.join(root, "data", "dump"), cells)
    key = lambda d: d.cell.astype(str) + "|" + d.ts.astype(str) + "|" + d.side.astype(int).astype(str)
    same_setups = len(pub) == len(seed) and (key(pub).to_numpy() == key(seed).to_numpy()).all()
    print(f"TWIN-SEED REBUILD seed {S} (production 17) | {len(cells)} N=60 cells | setups {len(seed):,} vs published {len(pub):,}")
    print(f"  [{'PASS' if same_setups else 'FAIL'}] KNOWN NUMBER: the setup rows are identical to the published ones (cell, ts, side)")
    # twins must differ: compare the twin-bar columns of one tag
    tw = []
    for c in cells[:6]:
        a = pd.read_parquet(os.path.join(RS.REAL_DUMP, c + ".parquet"), columns=["is_real", "bar"]); b = pd.read_parquet(os.path.join(root, "data", "dump", c + ".parquet"), columns=["is_real", "bar"])
        a, b = a[a.is_real == 0].bar.to_numpy(), b[b.is_real == 0].bar.to_numpy()
        tw.append((len(a) == len(b)) and float((a != b).mean()) if len(a) == len(b) else 1.0)
    twin_differs = float(np.mean(tw)) > 0.5
    print(f"  [{'PASS' if twin_differs else 'FAIL'}] the twin bars differ from the published draw (share differing in 6 cells: {np.mean(tw):.2f})")
    if not (same_setups and twin_differs):
        return 1
    P, N = RS.six(CC, pub, "iid"), RS.six(CC, seed, "iid")
    out = {}
    print("  six estimates, 36 N=60 cells: published (dump) | seed rebuild | difference")
    for (tag, blk), x in N.items():
        d = x["est_pp"] - P[(tag, blk)]["est_pp"]; nm = f"claim 1 {tag}" if tag != "race" else "claim 2 race"
        print(f"    {nm:<13} {blk}: {P[(tag, blk)]['est_pp']:+.3f} | {x['est_pp']:+.3f} | {d:+.3f}")
        out[f"{tag}_{blk}"] = dict(pub=P[(tag, blk)]["est_pp"], seed=x["est_pp"], diff=d)
    print(f"  largest absolute difference {max(abs(v['diff']) for v in out.values()):.3f} pp (twin draw only; setups identical)")
    for rl in ("lagged", "calendar8"):
        dd = os.path.join(SENS, f"dump_{rl}")
        r, _, _ = CC.load(dd, cells); R = RS.six(CC, r, "iid")
        diffs = {f"{k[0]}_{k[1]}": R[k]["est_pp"] - P[k]["est_pp"] for k in R}
        print(f"  {rl} rule restricted to the same 36 cells: differences " + ", ".join(f"{k} {v:+.3f}" for k, v in diffs.items()) + f"  (largest {max(abs(v) for v in diffs.values()):.3f})")
        out[f"rule_{rl}_n60"] = diffs
    json.dump(dict(seed=S, script_sha=sha(__file__), finished=time.strftime("%Y-%m-%d %H:%M:%S"), results=out), open(os.path.join(SENS, f"roll_seed{S}_claims_{time.strftime('%Y-%m-%d')}.json"), "w"), indent=1, default=float)
    return 0


def main(a):
    if "--selftest" in a:
        return 0 if selftest() else 1
    if "--extras" in a:
        r = a[a.index("--extras") + 1]
        if r not in ("lagged", "calendar8"):
            raise SystemExit("--extras lagged|calendar8")
        if not selftest():
            return 1
        return extras(r)
    if "--seed" in a:
        return seed_build(int(a[a.index("--seed") + 1]))
    if "--seed-claims" in a:
        return seed_claims(int(a[a.index("--seed-claims") + 1]))
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
