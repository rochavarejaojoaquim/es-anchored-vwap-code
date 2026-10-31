"""Roll extras 3 (reviewer read of v26, objection 1): the claims on the setups whose OWN UTC date lies within +-7 calendar days of any contract change
(the complement of the setup-only roll-free frame), for one rule. Imports roll_extras.py and roll_sensitivity.py UNCHANGED. Training periods only.
READING RULES, fixed before this run: descriptive; a frame HOLDS iff its iid AND 5-day-block lower ends are above 0; differences over 0.5 pp are
flagged; the in-window sample is about 18% of setups and the intervals are wide in 2022-2026, so a hold is the thing to report and an interval that
contains zero is reported as such, not read as an absence. Nothing here changes a claim. Run: python roll_extras3.py lagged | calendar8
"""
import os, sys, json, time, platform, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import roll_extras as E, roll_sensitivity as RS


def selftest():
    ok = True
    def chk(n, c):
        nonlocal ok; ok &= bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    day = lambda s: int(np.datetime64(s, "D").astype("int64"))
    d = np.array([day("2020-01-02"), day("2020-01-03"), day("2020-01-10"), day("2020-01-18")])
    m = E.window_mask(d, [day("2020-01-10")], 7)
    chk("K1 in-window and out-of-window frames partition the setups exactly", int(m.sum()) + int((~m).sum()) == len(d) and not (m & ~m).any())
    chk("K2 in-window mask is the exact complement of the roll-free mask used by --extras2", list(m) == [False, True, True, False])
    print("SELF-TEST", "ALL DETECTORS ALIVE" if ok else "FAILED")
    return ok


def run(rule):
    import claim_checks as CC
    from power_registered_set import registered_cells
    dump = os.path.join(RS.SENS, f"dump_{rule}")
    cells = registered_cells()
    pub, _, _ = CC.load(RS.REAL_DUMP, cells); new, _, _ = CC.load(dump, cells)
    pub["ts"] = RS._load_ts(RS.REAL_DUMP, cells); new["ts"] = RS._load_ts(dump, cells)
    p_iid = RS.six(CC, pub, "iid")
    if not all(abs(p_iid[k]["est_pp"] - RS.PUB[k]) < 0.0006 for k in RS.PUB):
        print("  [FAIL] KNOWN NUMBER: the published dump does not reproduce 26.65's six estimates"); return 1
    print(f"ROLL EXTRAS 3 ({rule}) | published {len(pub):,}, new {len(new):,} setups | [PASS] KNOWN NUMBER")
    cdays = E._days(sorted(set(sum([E.change_dates(E._chosen(r)) for r in RS.RULES], []))))
    dayof = lambda df: df.ts.to_numpy().astype("int64") // 86_400_000_000_000
    pin, nin = E.window_mask(dayof(pub), cdays, E.W_DAYS), E.window_mask(dayof(new), cdays, E.W_DAYS)
    name = "setups whose own date is within a window"
    fp, fn = pub[pin], new[nin]
    print(f"  {name}: published {len(fp):,} ({100*len(fp)/len(pub):.1f}%), {rule} {len(fn):,} ({100*len(fn)/len(new):.1f}%)")
    res = {k: (RS.six(CC, fp, k), RS.six(CC, fn, k)) for k in ("iid", "block")}
    holds = True; flags = []
    for (tag, blk) in res["iid"][0]:
        xi, yi = res["iid"][0][(tag, blk)], res["iid"][1][(tag, blk)]
        xb, yb = res["block"][0][(tag, blk)], res["block"][1][(tag, blk)]
        holds &= min(xi["lo_pp"], yi["lo_pp"], xb["lo_pp"], yb["lo_pp"]) > 0
        d = yi["est_pp"] - xi["est_pp"]; flags += [(tag, blk)] if abs(d) > RS.FLAG_PP else []
        nm = f"claim 1 {tag}" if tag != "race" else "claim 2 race"
        print(f"      {nm:<13} {blk}: published {xi['est_pp']:+.3f} [{xi['lo_pp']:+.3f},{xi['hi_pp']:+.3f}] block lo {xb['lo_pp']:+.3f} | "
              f"{rule} {yi['est_pp']:+.3f} [{yi['lo_pp']:+.3f},{yi['hi_pp']:+.3f}] block lo {yb['lo_pp']:+.3f} | diff {d:+.3f}")
    print(f"      HOLDS (all four lower ends above zero, both frames): {'YES' if holds else 'NO'}; differences over {RS.FLAG_PP} pp: {flags if flags else 'none'}")
    out = dict(rule=rule, script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest().upper(), roll_extras_sha=E.sha(E.__file__),
               W_days=E.W_DAYS, n_pub=int(len(fp)), n_new=int(len(fn)), holds=bool(holds),
               iid_pub={f"{k[0]}_{k[1]}": v for k, v in res["iid"][0].items()}, iid_new={f"{k[0]}_{k[1]}": v for k, v in res["iid"][1].items()},
               blk_pub={f"{k[0]}_{k[1]}": v for k, v in res["block"][0].items()}, blk_new={f"{k[0]}_{k[1]}": v for k, v in res["block"][1].items()},
               finished=time.strftime("%Y-%m-%d %H:%M:%S"), machine=platform.node())
    json.dump(out, open(os.path.join(RS.SENS, f"roll_extras3_{rule}_{time.strftime('%Y-%m-%d')}.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    if not selftest():
        sys.exit(1)
    if len(sys.argv) != 2 or sys.argv[1] not in ("lagged", "calendar8"):
        raise SystemExit("usage: python roll_extras3.py lagged | calendar8")
    sys.exit(run(sys.argv[1]))
