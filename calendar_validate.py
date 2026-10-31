"""calendar_validate.py -- checks a release calendar AGAINST THE TRAIN BARS (section 28, E7).

NOT part of the registered rule set until a dated Calendar Amendment fixes its criteria.
Reads: the calendar CSV and loader.load("train") (the loader enforces the seal: blocks A and B
only, rule C3). Reads no outcome column, no fill, no edge. Writes ONE csv beside the calendar.

CALENDAR CSV columns: event, date (America/New_York calendar date, YYYY-MM-DD), time_et (HH:MM),
tier (1 or 2), source (url or note).

WHAT IT MEASURES (all per event TYPE and per block A / B, descriptive unless the amendment says so):
  R(d, k) = 1-minute range (high - low) of the bar that opens at listed_time + k minutes on day d,
            divided by the rolling median of the SAME ET minute-of-day over the 41 nearest trading
            days, with every day listed at that slot MASKED out of the baseline.
  ALIGNMENT: median R over the type's days for k = -5..+5. The peak must sit at k = 0.
  RECALL (missed-event scan): at each listed slot, among days with R >= R_MISSED, the share that
            the calendar lists. Unlisted spikes are printed (date, slot, R) -- never added silently.
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
ET = "America/New_York"
OFFSETS = list(range(-5, 6))
WIN = 41                 # trading days in the rolling baseline (convention, draft)
MIN_BASE = 15            # minimum baseline days
R_MISSED = 5.0           # "abnormal minute" for the missed-event scan (convention, draft)
MIN_N_ERA = 30           # fewer listed days in a block: reported, not judged


def _file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def bar_range_table(d):
    """-> DataFrame(date, mod, rng, block): one row per ET date and minute-of-day."""
    ts = d["ts"].dt.tz_convert(ET)
    out = pd.DataFrame({
        "date": ts.dt.normalize().dt.tz_localize(None).to_numpy(),
        "mod": (ts.dt.hour * 60 + ts.dt.minute).to_numpy(),
        "rng": (d["high"].to_numpy() - d["low"].to_numpy()),
        "block": d["block"].to_numpy(),
    })
    return out.groupby(["date", "mod"], as_index=False).agg(rng=("rng", "max"), block=("block", "first"))


def parse_calendar(df):
    need = {"event", "date", "time_et", "tier", "source"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"calendar CSV lacks columns {sorted(miss)}")
    c = df.copy()
    c["date"] = pd.to_datetime(c["date"]).dt.normalize()
    hm = c["time_et"].astype(str).str.split(":", expand=True)
    c["mod"] = hm[0].astype(int) * 60 + hm[1].astype(int)
    return c


def ratios(bars, cal, slot_mods):
    """For each slot minute, a Series of R indexed by date, baseline with listed days masked."""
    out = {}
    for m in slot_mods:
        s = bars[bars["mod"] == m].set_index("date")["rng"].sort_index()
        if s.empty:
            continue
        listed = pd.Series(False, index=s.index)
        near = cal[(cal["mod"] - m).abs() <= 3]["date"].unique()
        listed[listed.index.isin(near)] = True
        base = s.mask(listed).rolling(WIN, center=True, min_periods=MIN_BASE).median()
        out[m] = (s / base.replace(0, np.nan))
    return out


def validate(bars, cal):
    block_of = bars.drop_duplicates("date").set_index("date")["block"]
    slots = sorted(set(int(m) + k for m in cal["mod"].unique() for k in OFFSETS))
    R = ratios(bars, cal, slots)
    rows = []
    for ev, g in cal.groupby("event"):
        for blk in ("A", "B", "ALL"):
            gg = g if blk == "ALL" else g[g["date"].map(block_of) == blk]
            med = {}
            n_used = 0
            for k in OFFSETS:
                vals = []
                for dt, m in zip(gg["date"], gg["mod"]):
                    s = R.get(int(m) + k)
                    if s is not None and dt in s.index and np.isfinite(s.loc[dt]):
                        vals.append(s.loc[dt])
                med[k] = float(np.median(vals)) if vals else np.nan
                if k == 0:
                    n_used = len(vals)
            arr = np.array([med[k] for k in OFFSETS])
            peak = OFFSETS[int(np.nanargmax(arr))] if np.isfinite(arr).any() else None
            rows.append(dict(event=ev, tier=int(g["tier"].iloc[0]), block=blk, n_listed=len(gg),
                             n_with_bars=n_used, R0=med[0], peak_offset=peak,
                             judged=bool(n_used >= MIN_N_ERA and blk != "ALL"),
                             aligned=(peak == 0) if peak is not None else None,
                             **{f"R{k:+d}": med[k] for k in OFFSETS if k != 0}))
    al = pd.DataFrame(rows)
    # missed-event scan
    listed_by_slot = {}
    for m in cal["mod"].unique():
        listed_by_slot[int(m)] = set(cal.loc[(cal["mod"] - m).abs() <= 2, "date"])
    rec = []
    spikes = []
    for m, s in R.items():
        if m not in listed_by_slot:
            continue
        hot = s[s >= R_MISSED].dropna()
        lst = listed_by_slot[m]
        hit = [d for d in hot.index if d in lst]
        miss = [d for d in hot.index if d not in lst]
        rec.append(dict(slot=f"{m // 60:02d}:{m % 60:02d}", hot_days=len(hot), listed_hot=len(hit),
                        unlisted_hot=len(miss), recall=(len(hit) / len(hot)) if len(hot) else np.nan))
        spikes += [dict(slot=f"{m // 60:02d}:{m % 60:02d}", date=d, R=float(hot[d])) for d in miss]
    return al, pd.DataFrame(rec), pd.DataFrame(spikes)


# ------------------------------------------------------------------ self-test
def _toy(seed=1):
    rg = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", "2013-12-31")
    mods = sorted(set([8 * 60 + 25 + i for i in range(11)] + [10 * 60 + i - 5 for i in range(11)] +
                      [14 * 60 + i - 5 for i in range(11)]))
    d0 = np.repeat(days.to_numpy(), len(mods))
    m0 = np.tile(np.array(mods), len(days))
    rng = rg.lognormal(0.0, 0.25, len(d0))
    df = pd.DataFrame({"date": d0, "mod": m0, "rng": rng, "block": "A"})
    df["i"] = np.arange(len(df))
    key = {(a, b): i for a, b, i in zip(df["date"], df["mod"], df["i"])}
    cal = []

    def plant(ds, minute, mult, name, tier, list_minute=None, listed=True):
        for dt in ds:
            if (np.datetime64(dt), minute) in key:
                rng[key[(np.datetime64(dt), minute)]] *= mult
        if listed:
            lm = minute if list_minute is None else list_minute
            for dt in ds:
                cal.append(dict(event=name, date=pd.Timestamp(dt), time_et=f"{lm // 60:02d}:{lm % 60:02d}",
                                tier=tier, source="toy"))
    first_fri = [d for d in days if d.weekday() == 4 and d.day <= 7]
    first_bd = [d for d in days if d.day <= 3 and d.weekday() == 0]
    plant(first_fri, 8 * 60 + 30, 8.0, "NFP", 1)
    plant(first_bd, 10 * 60, 4.0, "ISM", 2)
    plant(first_fri, 8 * 60 + 30, 1.0, "BADTIME", 2, list_minute=8 * 60 + 33, listed=False)  # placeholder
    for dt in first_fri:                                   # BADTIME: effect at 08:30, listed 08:33
        cal.append(dict(event="BADTIME", date=pd.Timestamp(dt), time_et="08:33", tier=2, source="toy"))
    unl = list(rg.choice(np.array(days), 20, replace=False))
    plant(unl, 14 * 60, 8.0, "UNLISTED", 1, listed=False)  # spikes the calendar never lists
    plant(list(rg.choice(np.array(days), 25, replace=False)), 14 * 60, 8.0, "FOMC", 1)
    placebo = list(rg.choice(np.array(days), 60, replace=False))
    for dt in placebo:                                     # listed, no effect
        cal.append(dict(event="PLACEBO", date=pd.Timestamp(dt), time_et="10:00", tier=2, source="toy"))
    df["rng"] = rng
    return df.drop(columns="i"), pd.DataFrame(cal)


def selftest():
    bars, cal = _toy()
    cal = parse_calendar(cal)
    al, rec, spikes = validate(bars, cal)
    a = al[al.block == "ALL"].set_index("event")
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}")
        ok &= bool(cond)
    chk("K1 NFP peak at offset 0 and R0 > 5", a.loc["NFP", "peak_offset"] == 0 and a.loc["NFP", "R0"] > 5,
        f"peak {a.loc['NFP', 'peak_offset']} R0 {a.loc['NFP', 'R0']:.2f}")
    chk("K2 a wrong listed time (08:33, effect 08:30) peaks at offset -3",
        a.loc["BADTIME", "peak_offset"] == -3, f"peak {a.loc['BADTIME', 'peak_offset']}")
    chk("K3 a listed type with no effect has R0 near 1", abs(a.loc["PLACEBO", "R0"] - 1) < 0.35,
        f"R0 {a.loc['PLACEBO', 'R0']:.2f}")
    r14 = rec[rec.slot == "14:00"].iloc[0]
    chk("K4 the missed-event scan finds most of the 20 unlisted 14:00 spikes (noise lets a few fall under 5x)",
        r14.unlisted_hot >= 14 and r14.listed_hot >= 20, f"unlisted {r14.unlisted_hot} listed {r14.listed_hot}")
    chk("K5 recall at 14:00 is below 0.7 (calendar incomplete is detected)", r14.recall < 0.7,
        f"recall {r14.recall:.2f}")
    r10 = rec[rec.slot == "10:00"].iloc[0]
    chk("K6 a complete slot (10:00) has recall 1", r10.recall == 1.0 or np.isnan(r10.recall), f"{r10.recall}")
    shuf = cal.copy()
    nfp = shuf.event == "NFP"
    own = set(cal.loc[nfp, "date"])
    others = np.array([d for d in bars["date"].unique() if d not in own])
    shuf.loc[nfp, "date"] = np.random.default_rng(3).choice(others, int(nfp.sum()), replace=False)
    a2 = validate(bars, shuf)[0].pipe(lambda x: x[x.block == "ALL"].set_index("event"))
    chk("M1 shuffling the NFP dates destroys its alignment", a2.loc["NFP", "R0"] < 2.5,
        f"R0 {a2.loc['NFP', 'R0']:.2f}")
    print("ALL DETECTORS ALIVE" if ok else "SELF-TEST FAILED")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit(1)
    if len(sys.argv) < 2:
        print("usage: python calendar_validate.py calendar_events.csv")
        sys.exit(2)
    import loader
    path = sys.argv[1]
    cal = parse_calendar(pd.read_csv(path))
    d = loader.load("train")
    assert not ((d.ts >= pd.Timestamp("2018-01-01", tz="UTC")) & (d.ts < pd.Timestamp("2022-01-01", tz="UTC"))).any(), "SEAL BAR PRESENT"
    bars = bar_range_table(d)
    al, rec, spikes = validate(bars, cal)
    stem = os.path.splitext(path)[0]
    al.to_csv(stem + "_validation.csv", index=False)
    rec.to_csv(stem + "_recall.csv", index=False)
    spikes.to_csv(stem + "_unlisted_spikes.csv", index=False)
    print(f"calendar {_file_sha(path)[:16]} | {len(cal)} rows, {cal['event'].nunique()} types | train bars "
          f"{len(d):,} (seal enforced by loader) | baseline {WIN} days, R_MISSED {R_MISSED} | "
          f"calendar_validate.py {_file_sha(__file__)[:16]}")
    pd.set_option("display.width", 200)
    print(al[al.block != "ALL"][["event", "tier", "block", "n_listed", "n_with_bars", "R0", "peak_offset", "aligned"]].to_string(index=False))
    print(rec.to_string(index=False))
    print(f"unlisted spikes: {len(spikes)} (see {os.path.basename(stem)}_unlisted_spikes.csv)")


if __name__ == "__main__":
    main()

