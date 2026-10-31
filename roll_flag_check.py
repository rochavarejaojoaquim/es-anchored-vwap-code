"""Independent check of roll_extras.twin_window_flags (reviewer read of v26, objection 2) and a saved record of the contract-change dates
(objection 4). Training periods only; reads data\\dump and roll_sens\\raw_* ; writes one JSON in roll_sens\\. Imports roll_extras.py UNCHANGED.
  K1 toy dump on disk, the REAL twin_window_flags: a setup whose only M1b twin is in a window is flagged; one whose twins are out is not;
     one whose only M2 twin is in a window is flagged; M1 mutation W=1 changes the flags (the test can fail).
  K2 real cell N60_hh_ATR_3.0: flags recomputed by an independent plain loop (Timestamp.value, a Python loop over change days) equal
     twin_window_flags exactly; the shares flagged by own date, by a twin only, and either, are printed.
  K3 saves roll_sens\\roll_change_dates_2026-10-21.json: change dates per rule, counts, the union, and for each rule the largest distance in days
     from each of its changes to the nearest production change.
"""
import os, sys, json, hashlib, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import roll_extras as E, roll_sensitivity as RS

ok = True
def chk(n, c):
    global ok; ok &= bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n}")

W = 7
DAY_NS = 86_400_000_000_000
day = lambda s: int(np.datetime64(s, "D").astype("int64"))
NS = lambda s: pd.to_datetime(s, utc=True).as_unit("ns")

# K1 toy dump on disk -------------------------------------------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix="rollflag_")
main = pd.DataFrame({
    "is_real": [1, 1, 1, 0, 0, 0, 0],
    "bar":     [100, 200, 300, 900, 901, 902, 903],
    "src_bar": [100, 200, 300, 100, 200, 300, 300],
    "ts": NS(["2020-06-01", "2020-06-02", "2020-06-03",          # setups, all far from the change on 2020-03-10
              "2020-03-12",                                       # twin of setup 100: 2 days from the change -> flagged
              "2020-05-01",                                       # twin of setup 200: out
              "2020-05-02",                                       # M1b twin of setup 300: out
              "2020-05-03"])})                                    # second M1b twin of setup 300: out
m2 = pd.DataFrame({"src_bar": [200, 300], "ts": NS(["2020-05-04", "2020-03-14"])})   # M2 twin of 300 is 4 days from the change -> flagged
main.to_parquet(os.path.join(tmp, "toy.parquet")); m2.to_parquet(os.path.join(tmp, "toy.m2.parquet"))
cd = np.array([day("2020-03-10")], dtype="int64")
f = E.twin_window_flags(tmp, ["toy"], cd, W)
chk("K1 toy: setup 100 (M1b twin 2 days from a change) flagged, setup 200 (all twins out) not, setup 300 (only its M2 twin in) flagged",
    list(f) == [True, False, True])
f1 = E.twin_window_flags(tmp, ["toy"], cd, 1)
chk("M1 W=1 changes the flags (the test can fail): setup 100 and 300 both drop out", list(f1) == [False, False, False])

# K2 real cell ---------------------------------------------------------------------------------------------------------------------
cell = "N60_hh_ATR_3.0"
chs = {r: E.change_dates(E._chosen(r)) for r in RS.RULES}
union = sorted(set(sum(chs.values(), [])))
cdr = E._days(union)
cdl = [int(v) for v in cdr]
mine = E.twin_window_flags(RS.REAL_DUMP, [cell], cdr, W)
x = pd.read_parquet(os.path.join(RS.REAL_DUMP, cell + ".parquet"), columns=["is_real", "bar", "src_bar", "ts"])
m2r = pd.read_parquet(os.path.join(RS.REAL_DUMP, cell + ".m2.parquet"), columns=["src_bar", "ts"])
real = x[x.is_real == 1]
def ts_day(t):
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // DAY_NS)
def near(d):
    return any(abs(d - c) <= W for c in cdl)
cache = {}
def near_cached(t):
    d = ts_day(t)
    if d not in cache:
        cache[d] = near(d)
    return cache[d]
flag = {int(b): False for b in real.bar.to_numpy()}
for df in (x[x.is_real == 0], m2r):
    for s, t in zip(df.src_bar.to_numpy(), df.ts):
        if near_cached(t):
            flag[int(s)] = True
indep = np.array([flag[int(b)] for b in real.bar.to_numpy()])
chk(f"K2 twin_window_flags equals an independent loop on {cell} ({len(real):,} setups, {int(mine.sum()):,} flagged by a twin)",
    len(indep) == len(mine) and bool((indep == mine).all()))
sdays = np.array([ts_day(t) for t in real.ts])
own = np.array([near_cached(t) for t in real.ts])
print(f"  setups in a window themselves {own.mean():.3f}; flagged by a twin only {(mine & ~own).mean():.3f}; either {(mine | own).mean():.3f}")

# K3 saved dates -------------------------------------------------------------------------------------------------------------------
prod = E._days(chs["same"])
out = dict(window_days=W, script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest().upper(), n_union=len(union), rules={})
for r in RS.RULES:
    dist = [int(np.min(np.abs(E._days([c])[0] - prod))) for c in chs[r]]
    out["rules"][r] = dict(n=len(chs[r]), dates=[str(c) for c in chs[r]], max_days_to_nearest_production_change=max(dist))
out["union_dates"] = [str(c) for c in union]
json.dump(out, open(os.path.join(RS.SENS, "roll_change_dates_2026-10-21.json"), "w"), indent=1)
print("  change dates per rule", {r: out["rules"][r]["n"] for r in RS.RULES}, "union", len(union),
      "| max days from a rule's change to the nearest production change",
      {r: out["rules"][r]["max_days_to_nearest_production_change"] for r in RS.RULES})
print("RESULT", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
