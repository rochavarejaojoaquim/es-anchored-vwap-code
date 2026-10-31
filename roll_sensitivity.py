"""
roll_sensitivity.py (version 2) -- training-only sensitivity of claims 1 and 2 to the FRONT-CONTRACT RULE.
(design: ROLL_SENSITIVITY_DESIGN_v2_2026-10-20.md; version 1 is kept at stale\\roll_sensitivity_v1_before_redteam_2026-10-20.py
 and was never run on real data; HANDOFF 26.220-26.221)

The production loader keeps, on each UTC date, the contract with the highest volume ON THAT DATE (loader.py:132-139):
a look-ahead at an intraday signal time. Here the same pipeline is run on a FILTERED COPY of the raw file in which
each UTC date keeps ONE contract chosen by another rule; the loader's own volume rule then selects exactly that
contract. Nothing under code\\ is edited, the held-out years are not written into the copy, the code-set does not move.

Rules:  same      = the loader's own rule, re-implemented (known-number checks only)
        lagged    = the volume leader of the PREVIOUS date that has at least MIN_ROWS rows (a stub session such as the one-hour
                    Sunday evening is skipped), never moving to a contract with an earlier expiry than the one already in use
                    (monotone); the first date of each block uses its own volume leader; if the chosen contract has no rows on
                    the date the volume leader is used and counted as a FALLBACK
        calendar8 = per UTC date the traded contract with the smallest expiry that is >= 8 calendar days away (no volume)
Every rule is stated before any real-data result and none is tuned (K_DAYS = 8, MIN_ROWS = 500 are conventions).

Modes (nothing here reads the sealed years into any output):
  --selftest                              toy checks (no data read)
  --make-raw RULE                         write roll_sens\\raw_<rule>\\data\\outright.parquet + counts and profile
  --k3                                    loader.load('train', lean True and False) on the real file == on the 'same' file
  --build N --rule RULE                   dump for lookback N (own scratch root, hard-linked raw, unchanged dump_episodes)
  --accept-same                           root_same_N60's dump equals data\\dump EXACTLY (seal_build.compare_exact); writes the receipt
  --merge --rule RULE                     hard-link the five per-N dumps into roll_sens\\dump_<rule> and count files
  --noise                                 N=60 cells: claims on the fresh block-A-only / block-B-only builds against data\\dump (a floor)
  --claims DUMP --rule RULE               claims 1-2 on the 144 registered cells, blocks A and B, beside the published numbers

READING RULE (fixed in the design note before any real run): see ROLL_SENSITIVITY_DESIGN_v2_2026-10-20.md section 6.
"""
import contextlib, hashlib, io, json, os, platform, re, runpy, shutil, subprocess, sys, tempfile, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
REAL_RAW = os.path.join(HERE, "data", "outright.parquet")
REAL_DUMP = os.path.join(HERE, "data", "dump")
SENS = os.path.join(HERE, "roll_sens")
K_DAYS = 8
MIN_ROWS = 500          # a date with fewer rows (the Sunday evening hour has 60 per contract) is not a reference day
FALLBACK_MAX = 0.01     # a rule that falls back on more than 1% of dates is refused
FLAG_PP = 0.5
RULES = ("same", "lagged", "calendar8")
PUB = {("M1b", "A"): 4.376, ("M1b", "B"): 1.310, ("M2", "A"): 4.178, ("M2", "B"): 1.368,
       ("race", "A"): 4.857, ("race", "B"): 0.780}                        # 26.65, pp
EPOCH = pd.Timestamp("1970-01-01")
_NEED = ["ts", "instrument_id", "symbol", "volume"]


def sha(p, n=None):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest().upper()
    return h[:n] if n else h


def _loader():
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    import loader
    return loader


# ----------------------------------------------------------------------------------- the rules
def volume_leader(g):
    """g: (date, instrument_id, volume) -> one row per date, the loader's rule: rank desc by summed volume, ties by first."""
    g = g.copy()
    g["rk"] = g.groupby("date")["volume"].rank(ascending=False, method="first")
    return g.loc[g.rk == 1, ["date", "instrument_id"]]


def instrument_expiry(sub, expiry_fn):
    """One row per instrument_id with its expiry (days since 1970), keyed by instrument_id. Asserts one symbol per instrument_id and one
    instrument_id per symbol (training rows), expiry >= the first date the contract trades and within 800 days of it."""
    ns = sub.groupby("instrument_id", observed=True)["symbol"].nunique()
    if (ns != 1).any():
        raise SystemExit(f"REFUSED: instrument_ids with more than one symbol: {ns[ns != 1].index.tolist()[:5]}")
    t = sub.groupby("instrument_id", observed=True).agg(sym=("symbol", "first"), d0=("date", "min"), d1=("date", "max")).reset_index()
    t["sym"] = t.sym.astype(str)
    # ES symbols carry ONE year digit (ESH2 = March 2012 and March 2022), so a symbol may sit under two instrument_ids
    # provided the uses never overlap and are at least a year apart; expiry is keyed by instrument_id and its first date.
    reused = 0
    for s, g in t[t.sym.duplicated(keep=False)].groupby("sym"):
        g = g.sort_values("d0")
        for i in range(len(g) - 1):
            if not (g.d1.iloc[i] + pd.Timedelta(days=365) <= g.d0.iloc[i + 1]):
                raise SystemExit(f"REFUSED: symbol {s} under two instrument_ids with overlapping or near-adjacent dates")
        reused += 1
    t.attrs["reused"] = reused
    ex = []
    for r in t.itertuples(index=False):
        try:
            ex.append(float(expiry_fn(r.sym, r.d0.date())))
        except ValueError:
            ex.append(np.nan)
    t["expiry"] = ex
    t["d0n"] = ((t.d0.dt.tz_localize(None) - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
    ok = t.expiry.notna()
    if ok.any() and ((t.expiry[ok] < t.d0n[ok]) | (t.expiry[ok] - t.d0n[ok] > 800)).any():
        bad = t[ok & ((t.expiry < t.d0n) | (t.expiry - t.d0n > 800))]
        raise SystemExit(f"REFUSED: expiry outside [first date, first date + 800 d] for {bad.sym.tolist()[:5]}")
    return t


def _block_of(dates, blocks):
    out = np.array([""] * len(dates), dtype=object)
    for name, lo, hi in blocks:
        m = ((dates >= lo) & (dates < hi))
        out[np.asarray(m)] = name
    return out


def choose(sub, rule, expiry_fn=None, k=K_DAYS, min_rows=MIN_ROWS, blocks=None):
    """sub: rows (ts, instrument_id, symbol, volume), spreads removed, training rows only.
    Returns (chosen: DataFrame[date, instrument_id], info: dict with the profile)."""
    sub = sub.copy()
    sub["date"] = sub.ts.dt.floor("D")
    g = sub.groupby(["date", "instrument_id"], observed=True).agg(volume=("volume", "sum"), nrow=("volume", "size")).reset_index()
    lead = volume_leader(g)
    lead_by = dict(zip(lead.date, lead.instrument_id))
    dates = sorted(g.date.unique())
    info = dict(rule=rule, fallback=0, block_starts=0, unparsed_chosen=0)
    if rule == "same":
        chosen = lead.sort_values("date").reset_index(drop=True)
    else:
        inst = instrument_expiry(sub, expiry_fn)
        ex = dict(zip(inst.instrument_id, inst.expiry))
        info["unparsed_instruments"] = int(inst.expiry.isna().sum())
        present = set(zip(g.date, g.instrument_id))
        if rule == "calendar8":
            g2 = g.copy(); g2["expiry"] = g2.instrument_id.map(ex)
            g2["dn"] = ((g2.date.dt.tz_localize(None) - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
            ok = g2.expiry.notna(); g2 = g2[ok].copy(); g2["gap"] = g2.expiry.astype(np.int64) - g2.dn
            elig = g2[g2.gap >= k].sort_values(["date", "expiry"]).groupby("date").head(1)[["date", "instrument_id"]]
            miss = np.setdiff1d(lead.date.unique(), elig.date.unique())
            fb = lead[lead.date.isin(miss)]
            info["fallback"] = int(len(fb))
            chosen = pd.concat([elig, fb], ignore_index=True).sort_values("date").reset_index(drop=True)
        elif rule == "lagged":
            rows_by = g.groupby("date")["nrow"].sum()
            blk = dict(zip(dates, _block_of(pd.DatetimeIndex(dates), blocks) if blocks else [""] * len(dates)))
            ids, prev_ref, prev_choice, prev_ref_blk = [], None, None, None
            for D in dates:
                if prev_ref is None or blk[D] != prev_ref_blk:
                    c = lead_by[D]; info["block_starts"] += 1; prev_choice = None
                else:
                    c = lead_by[prev_ref]
                    if prev_choice is not None and ex.get(c, np.nan) < ex.get(prev_choice, np.nan):
                        c = prev_choice                                       # monotone: never to an earlier expiry
                    if (D, c) not in present:
                        c = lead_by[D]; info["fallback"] += 1
                if np.isnan(ex.get(c, np.nan)):
                    info["unparsed_chosen"] += 1
                ids.append(c); prev_choice = c
                if rows_by[D] >= min_rows:
                    prev_ref, prev_ref_blk = D, blk[D]
            chosen = pd.DataFrame(dict(date=dates, instrument_id=ids))
        else:
            raise ValueError(rule)
        if info["unparsed_chosen"]:
            raise SystemExit(f"REFUSED: {info['unparsed_chosen']} chosen dates carry a contract whose expiry could not be parsed")
        if info["fallback"] > FALLBACK_MAX * len(chosen):
            raise SystemExit(f"REFUSED: fallback on {info['fallback']} of {len(chosen)} dates (> {FALLBACK_MAX:.0%})")
    # ---- profile (printed, never thresholded)
    c = chosen.merge(g[["date", "instrument_id", "volume"]], on=["date", "instrument_id"], how="left")
    lv = g.groupby("date")["volume"].max().rename("lead_vol")
    c = c.merge(lv, left_on="date", right_index=True)
    ratio = (c.volume / c.lead_vol).to_numpy()
    sm = chosen.merge(lead.rename(columns={"instrument_id": "lead_id"}), on="date")
    chg = int((chosen.instrument_id.to_numpy()[1:] != chosen.instrument_id.to_numpy()[:-1]).sum())
    sb = None
    if rule != "same" or expiry_fn is not None:
        inst = instrument_expiry(sub, expiry_fn) if expiry_fn else None
        if inst is not None:
            info["reused_symbols"] = int(inst.attrs.get("reused", 0))
            e = chosen.instrument_id.map(dict(zip(inst.instrument_id, inst.expiry))).to_numpy(float)
            sb = int((np.diff(e) < 0).sum())                                 # switch-backs in expiry (includes the block change)
    info.update(n_dates=int(len(chosen)), dates_differ_from_volume=int((sm.instrument_id != sm.lead_id).sum()),
                contract_changes=chg, expiry_switchbacks=sb,
                liq_ratio_median=float(np.nanmedian(ratio)), liq_ratio_p10=float(np.nanpercentile(ratio, 10)),
                liq_ratio_min=float(np.nanmin(ratio)), dates_below_10pct_of_leader=int((ratio < 0.10).sum()))
    return chosen, info


def filter_raw(path_in, path_out, rule, blocks, expiry_fn=None, k=K_DAYS, min_rows=MIN_ROWS):
    """Write a copy of the raw file with, per UTC date, only the chosen contract's rows; spreads and everything outside the
    training blocks are dropped (the held-out years never enter the copy). Schema and dtypes are preserved."""
    import pyarrow as pa, pyarrow.parquet as pq
    t = pq.read_table(path_in)
    df = t.select(_NEED).to_pandas()
    sym = df.symbol.astype(str)
    keep = (~sym.str.contains("-")).to_numpy().copy()
    inblk = np.zeros(len(df), dtype=bool)
    for _, lo, hi in blocks:
        inblk |= ((df.ts >= lo) & (df.ts < hi)).to_numpy()
    keep &= inblk
    idx = np.where(keep)[0]
    sub = df.iloc[idx].copy()
    chosen, info = choose(sub, rule, expiry_fn, k, min_rows, blocks)
    sub["date"] = sub.ts.dt.floor("D")
    m = sub.merge(chosen.assign(_ok=True), on=["date", "instrument_id"], how="left")["_ok"].fillna(False).astype(bool).to_numpy()
    final = np.zeros(len(df), dtype=bool); final[idx[m]] = True
    out = t.filter(pa.array(final))
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    pq.write_table(out, path_out)
    info.update(rows_in=int(len(df)), rows_training=int(keep.sum()), rows_out=int(final.sum()),
                dates=int(chosen.date.nunique()), contracts_used=int(chosen.instrument_id.nunique()))
    return info, chosen


# ----------------------------------------------------------------------------------- toy data and self-test
def toy_raw(path, flipback=False, stub=True):
    """Two contracts ESH2 (expiry 2012-03-16) and ESM2, one hour of bars per UTC weekday 2012-02-27..2012-03-23; the volume leader
    flips to ESM2 on 2012-03-06 (flipback: and is ESH2 again on 2012-03-08); a huge calendar spread that must be ignored; a 20-row
    Sunday stub on 2012-03-04 where ESM2 leads. Returns the date list."""
    days = [d for d in pd.date_range("2012-02-27", "2012-03-23", freq="D", tz="UTC") if d.weekday() < 5]
    rows = []
    rng = np.random.default_rng(5)
    for d in days:
        ts = pd.date_range(d + pd.Timedelta(hours=14), periods=60, freq="min")
        lead_h = d < pd.Timestamp("2012-03-06", tz="UTC") or (flipback and d == pd.Timestamp("2012-03-08", tz="UTC"))
        for iid, sym, base in ((1, "ESH2", 100 if lead_h else 40), (2, "ESM2", 40 if lead_h else 100), (3, "ESH2-ESM2", 1000)):
            px = 1300 + rng.normal(0, 1, 60).cumsum()
            rows.append(pd.DataFrame(dict(ts=ts, instrument_id=iid, symbol=sym, high=px + 0.5, low=px - 0.5, close=px,
                                          volume=(base + rng.integers(0, 5, 60)).astype("int64"))))
    if stub:
        d = pd.Timestamp("2012-03-04", tz="UTC"); ts = pd.date_range(d + pd.Timedelta(hours=23), periods=20, freq="min")
        for iid, sym, base in ((1, "ESH2", 5), (2, "ESM2", 80)):
            px = 1300 + rng.normal(0, 1, 20).cumsum()
            rows.append(pd.DataFrame(dict(ts=ts, instrument_id=iid, symbol=sym, high=px + 0.5, low=px - 0.5, close=px,
                                          volume=(base + rng.integers(0, 3, 20)).astype("int64"))))
    df = pd.concat(rows, ignore_index=True).sort_values("ts").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return days


def selftest():
    ok = [True]

    def chk(lab, cond, det=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab}" + (f"   {det}" if det else "")); ok[0] &= bool(cond)

    L = _loader(); blocks = L.TRAIN_BLOCKS
    tmp = tempfile.mkdtemp(prefix="rollsens_")
    raw = os.path.join(tmp, "toy.parquet"); toy_raw(raw)
    T = lambda s: pd.Timestamp(s, tz="UTC")

    def run(rule, k=K_DAYS, mr=50, src=raw):
        f = os.path.join(tmp, f"{rule}_{k}_{mr}_{os.path.basename(src)}")
        i, c = filter_raw(src, f, rule, blocks, L.expiry_day, k=k, min_rows=mr)
        return f, i, c.set_index("date").instrument_id

    f_same, i_same, c_same = run("same"); f_cal, i_cal, c_cal = run("calendar8"); f_c9, _, c_c9 = run("calendar8", k=9)
    f_lag, i_lag, c_lag = run("lagged"); _, _, c_lag1 = run("lagged", mr=1)
    # K1: same-filtered file gives the identical loader frame (both lean modes)
    for lean in (True, False):
        a = L.load("train", raw=raw, lean=lean); b = L.load("train", raw=f_same, lean=lean)
        try:
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True); same_ok = True
        except AssertionError:
            same_ok = False
        chk(f"K1 SAME-filtered toy file gives a loader frame identical to the unfiltered file's (lean={lean})", same_ok and len(a) > 0, f"{len(a)} bars")
    chk("K1b the volume leader flips on 2012-03-06 and the spread (volume 1000) is never chosen",
        (c_same[c_same.index < T("2012-03-06")].drop(T("2012-03-04"), errors="ignore") == 1).all() and (c_same[c_same.index >= T("2012-03-06")] == 2).all())
    # K2: calendar8 flips when expiry - D < 8
    chk("K2 CALENDAR8 keeps ESH2 through 2012-03-08 and takes ESM2 from 2012-03-09",
        c_cal[c_cal == 1].index.max() == T("2012-03-08") and c_cal[c_cal == 2].index.min() == T("2012-03-09"))
    chk("K2m MUTATION k = 9 flips one day earlier (2012-03-07)", c_c9[c_c9 == 1].index.max() == T("2012-03-07"))
    chk("K3 CALENDAR8 is monotone", bool((np.diff(c_cal.to_numpy()) >= 0).all()) and i_cal["expiry_switchbacks"] == 0)
    d4 = sorted(c_same.index[c_same.to_numpy() != c_cal.reindex(c_same.index).to_numpy()])
    chk("K4 SAME and CALENDAR8 differ on exactly 03-04 (stub), 03-06, 03-07, 03-08",
        [d.strftime("%m-%d") for d in d4] == ["03-04", "03-06", "03-07", "03-08"], str([d.strftime("%m-%d") for d in d4]))
    chk("K5 no fallback on the toy and the profile is filled", i_cal["fallback"] == 0 and i_cal["n_dates"] == 21 and i_cal["liq_ratio_min"] > 0, str(i_cal)[:150])
    # K9-K12: the lagged rule
    chk("K9 LAGGED uses the previous date's leader: ESH2 on 03-06, ESM2 from 03-07", c_lag[T("2012-03-06")] == 1 and c_lag[T("2012-03-07")] == 2
        and c_lag[c_lag == 2].index.min() == T("2012-03-07"))
    chk("K10 LAGGED skips the 20-row Sunday stub as a reference day: Monday 03-05 is ESH2", c_lag[T("2012-03-05")] == 1)
    chk("K10m MUTATION min_rows = 1 lets the stub in: Monday 03-05 switches early to ESM2", c_lag1[T("2012-03-05")] == 2)
    chk("K9b LAGGED differs from SAME on exactly 03-04 (the stub date takes Friday's leader) and 03-06",
        [d.strftime("%m-%d") for d in sorted(c_same.index[c_same.to_numpy() != c_lag.reindex(c_same.index).to_numpy()])] == ["03-04", "03-06"])
    chk("K11 the first date of the block uses its own volume leader and is counted as a block start", i_lag["block_starts"] == 1 and i_lag["fallback"] == 0)
    # flip-back: SAME goes back to ESH2 on 03-08; LAGGED and CALENDAR8 never do
    raw_fb = os.path.join(tmp, "toyfb.parquet"); toy_raw(raw_fb, flipback=True)
    _, i_sfb, c_sfb = run("same", src=raw_fb); _, i_lfb, c_lfb = run("lagged", src=raw_fb); _, i_cfb, c_cfb = run("calendar8", src=raw_fb)
    chk("K12 a volume flip-back (ESM2 then ESH2 again): SAME goes back, LAGGED does not",
        i_sfb["expiry_switchbacks"] >= 1 and (np.diff(c_lfb.to_numpy()) >= 0).all() and i_lfb["expiry_switchbacks"] == 0
        and i_cfb["expiry_switchbacks"] == 0, f"same switchbacks {i_sfb['expiry_switchbacks']}")
    # K6: a date on which the eligible contract has no rows falls back and is counted (gate refuses above FALLBACK_MAX)
    d2 = pd.read_parquet(raw); d2 = d2[~((d2.symbol == "ESM2") & (d2.ts.dt.floor("D") == T("2012-03-12")))]
    raw2 = os.path.join(tmp, "toy2.parquet"); d2.to_parquet(raw2, index=False)
    refused = False
    try:
        filter_raw(raw2, os.path.join(tmp, "cal2.parquet"), "calendar8", blocks, L.expiry_day)
    except SystemExit as e:
        refused = "fallback on 1 of" in str(e)
    chk("K6 a date without the eligible contract is a counted fallback and 1 of 21 dates exceeds the 1% gate (refused)", refused)
    # K7: rows outside the training blocks never enter
    d3 = pd.read_parquet(raw); seal = d3.copy(); seal["ts"] = seal.ts + pd.DateOffset(years=7)
    raw3 = os.path.join(tmp, "toy3.parquet"); pd.concat([d3, seal], ignore_index=True).to_parquet(raw3, index=False)
    f3 = os.path.join(tmp, "same3.parquet"); filter_raw(raw3, f3, "same", blocks)
    y = pd.read_parquet(f3, columns=["ts"]).ts
    chk("K7 rows dated in the sealed window (2019) are not written to the filtered file", bool((y.dt.year == 2012).all()) and len(y) > 0, f"{len(y)} rows")
    # K13: instrument keys are asserted
    d4_ = pd.read_parquet(raw); d4_.loc[d4_.symbol == "ESM2", "instrument_id"] = 1
    raw4 = os.path.join(tmp, "toy4.parquet"); d4_.to_parquet(raw4, index=False)
    bad = False
    try:
        filter_raw(raw4, os.path.join(tmp, "bad.parquet"), "calendar8", blocks, L.expiry_day)
    except SystemExit as e:
        bad = "more than one symbol" in str(e)
    chk("K13 two symbols under one instrument_id are refused", bad)
    # K14: a one-digit-year symbol reused ten years later under a NEW instrument_id is accepted and rolls on its own expiry;
    # the same symbol reused with overlapping dates is refused
    d_ = pd.read_parquet(raw); d_ = d_[d_.instrument_id != 3]
    late = d_.copy(); late["ts"] = late.ts + pd.DateOffset(years=10); late["instrument_id"] = late.instrument_id + 10
    raw5 = os.path.join(tmp, "toy5.parquet"); pd.concat([d_, late], ignore_index=True).to_parquet(raw5, index=False)
    ok14 = False; det14 = ""
    try:
        _, i5, c5 = run("calendar8", src=raw5)
        last_h = c5[c5 == 11].index.max(); first_m = c5[c5 == 12].index.min()
        ok14 = (last_h == T("2022-03-09") and first_m == T("2022-03-12") and i5.get("reused_symbols") == 2)
        det14 = f"2022 ESH2 last {last_h.date()}, ESM2 first {first_m.date()}, reused symbols {i5.get('reused_symbols')}"
    except SystemExit as e:
        det14 = str(e)
    chk("K14 a symbol reused ten years later under a new instrument_id is accepted and rolls on its own expiry", ok14, det14)
    ov = d_.copy(); ov["instrument_id"] = ov.instrument_id + 20
    raw6 = os.path.join(tmp, "toy6.parquet"); pd.concat([d_, ov], ignore_index=True).to_parquet(raw6, index=False)
    ref6 = False
    try:
        run("calendar8", src=raw6)
    except SystemExit as e:
        ref6 = "overlapping" in str(e)
    chk("K14b the same symbol under two instrument_ids with overlapping dates is refused", ref6)
    # K8: comparator
    new = dict(PUB); new[("M1b", "A")] += 0.7; new[("race", "B")] += 0.2
    chk("K8 the difference flag fires at 0.7 pp and not at 0.2 pp", [k for k in PUB if abs(new[k] - PUB[k]) > FLAG_PP] == [("M1b", "A")])
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL DETECTORS ALIVE" if ok[0] else "SELF-TEST FAILED")
    return ok[0]


# ----------------------------------------------------------------------------------- real-data modes
def make_raw(rule):
    L = _loader()
    out_root = os.path.join(SENS, f"raw_{rule}")
    out = os.path.join(out_root, "data", "outright.parquet")
    if os.path.exists(out):
        raise SystemExit(f"REFUSED: {out} exists (never overwritten; rename it into stale\\ first)")
    t0 = time.time()
    info, chosen = filter_raw(REAL_RAW, out, rule, L.TRAIN_BLOCKS, L.expiry_day)
    info.update(k_days=K_DAYS, min_rows=MIN_ROWS, file=out, sha256=sha(out), wall_s=round(time.time() - t0, 1),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"), machine=platform.node(), script_sha=sha(__file__),
                raw_input_sha=sha(REAL_RAW))
    json.dump(info, open(os.path.join(out_root, "make_raw_info.json"), "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))
    return 0


def k3():
    same = os.path.join(SENS, "raw_same", "data", "outright.parquet")
    if not os.path.exists(same):
        raise SystemExit("run --make-raw same first")
    L = _loader(); res = []
    for lean in (True, False):
        A = L.load("train", raw=REAL_RAW, lean=lean); B = L.load("train", raw=same, lean=lean)
        try:
            pd.testing.assert_frame_equal(A, B, check_exact=True); good = True; why = ""
        except AssertionError as e:
            good = False; why = str(e)[:400]
        print(f"[{'PASS' if good else 'FAIL'}] K3 loader.load('train', lean={lean}) on the SAME-filtered file equals the real file's frame "
              f"({len(A):,} vs {len(B):,} rows, {A.shape[1]} columns, {int(A.seg.nunique()) if 'seg' in A.columns else 'n/a'} segments){'  ' + why if why else ''}")
        res.append(good); del A, B
    allgood = all(res)
    json.dump(dict(check="K3", pass_=allgood, lean_modes=[True, False], script_sha=sha(__file__), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                   same_raw_sha=sha(same), real_raw_sha=sha(REAL_RAW)), open(os.path.join(SENS, "k3_receipt.json"), "w"), indent=1)
    return 0 if allgood else 1


def build(N, rule):
    raw = os.path.join(SENS, f"raw_{rule}", "data", "outright.parquet")
    if not os.path.exists(raw):
        raise SystemExit(f"run --make-raw {rule} first")
    kr = os.path.join(SENS, "k3_receipt.json")
    if not (os.path.exists(kr) and json.load(open(kr)).get("pass_") and json.load(open(kr)).get("script_sha") == sha(__file__)):
        raise SystemExit("REFUSED: the K3 equivalence check has not passed for THIS version of the script (run --k3)")
    root =os.path.join(SENS, f"root_{rule}_N{N}")
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    dst = os.path.join(root, "data", "outright.parquet")
    if not os.path.exists(dst):
        os.link(raw, dst)
    if os.path.getsize(dst) != os.path.getsize(raw) or sha(dst) != sha(raw):
        raise SystemExit("REFUSED: scratch outright differs from the filtered file")
    os.environ["VWAP_ROOT"] = root; os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, CODE)
    import loader, pipeline                                              # first import in this process: they read VWAP_ROOT now
    same_file = os.path.normcase(os.path.realpath(loader.RAW)) == os.path.normcase(os.path.realpath(dst))
    out_ok = os.path.normcase(os.path.realpath(pipeline.OUT)).startswith(os.path.normcase(os.path.realpath(root)))
    if not (same_file and out_ok):
        raise SystemExit(f"REFUSED: loader.RAW={loader.RAW} pipeline.OUT={pipeline.OUT} do not point into {root}")
    t0 = time.time(); wsha = sha(__file__)
    sys.argv = [os.path.join(CODE, "dump_episodes.py"), str(N)]
    runpy.run_path(os.path.join(CODE, "dump_episodes.py"), run_name="__main__")
    rec = dict(N=N, root=root, rule=rule, wrapper_sha=wsha, raw_sha=sha(dst), loader_RAW=str(loader.RAW), pipeline_OUT=str(pipeline.OUT),
               dump_episodes_sha=sha(os.path.join(CODE, "dump_episodes.py")), loader_sha=sha(os.path.join(CODE, "loader.py")),
               control_matched_sha=sha(os.path.join(CODE, "control_matched.py")), wall_s=round(time.time() - t0, 1),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"),
               machine=dict(node=platform.node(), cpu=os.cpu_count(), python=platform.python_version()))
    json.dump(rec, open(os.path.join(root, "build_receipt.json"), "w"), indent=1)
    print(f"BUILD DONE rule={rule} N={N} {rec['wall_s']}s")
    return 0


def accept_same():
    sys.path.insert(0, HERE)
    import seal_build
    root = os.path.join(SENS, "root_same_N60")
    rc = os.path.join(root, "build_receipt.json")
    if not os.path.exists(rc) or json.load(open(rc)).get("rule") != "same":
        raise SystemExit("no SAME build receipt for N=60")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        rcode = seal_build.compare_exact(root)
    print(buf.getvalue().strip().splitlines()[-1])
    good = rcode == 0
    json.dump(dict(check="accept_same", pass_=good, build_receipt=json.load(open(rc)), compare_sha=sha(os.path.join(HERE, "seal_build.py")),
                   script_sha=sha(__file__), finished=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(SENS, "accept_same_receipt.json"), "w"), indent=1)
    return 0 if good else 1


def merge(rule):
    out = os.path.join(SENS, f"dump_{rule}")
    os.makedirs(out, exist_ok=True)
    n = 0
    for N in (60, 100, 160, 200, 240):
        base = os.path.join(SENS, f"root_{rule}_N{N}")
        src = os.path.join(base, "data", "dump")
        rc = os.path.join(base, "build_receipt.json")
        if not (os.path.isdir(src) and os.path.exists(rc)) or json.load(open(rc)).get("rule") != rule:
            raise SystemExit(f"missing or foreign build in {base}")
        for f in os.listdir(src):
            if f.startswith("N") and f.endswith((".parquet", ".json")) and not f.endswith(".tmp"):
                d = os.path.join(out, f)
                if os.path.exists(d):
                    raise SystemExit(f"REFUSED: name overlap {f}")
                os.link(os.path.join(src, f), d); n += 1
    pq_n = len([f for f in os.listdir(out) if f.endswith(".parquet") and ".m2." not in f])
    print(f"merged {n} files into {out}; cell files {pq_n} (180 expected)")
    return 0 if pq_n == 180 else 1


def _load_ts(dump, cells):
    """ts of the real setups in the order CC.load builds them (is_real == 1, file order)."""
    out = []
    for c in cells:
        x = pd.read_parquet(os.path.join(dump, c + ".parquet"), columns=["is_real", "ts"])
        out.append(pd.to_datetime(x.ts[x.is_real == 1], utc=True).astype("int64").to_numpy())
    return np.concatenate(out)


def six(CC, df, kind, blocks=("A", "B")):
    res = {}
    for blk in blocks:
        for tag in ("M1b", "M2"):
            s = CC.arm_frame(df, tag); res[(tag, blk)] = CC.claim1_block(s[s.block == blk], kind)
        s1 = CC.arm_frame(df, "M1b"); res[("race", blk)] = CC.claim2_block(s1[s1.block == blk], kind)
    return res


def claims(dump, rule):
    sys.path.insert(0, HERE)
    ar = os.path.join(SENS, "accept_same_receipt.json")
    if not (os.path.exists(ar) and json.load(open(ar)).get("pass_") and json.load(open(ar)).get("script_sha") == sha(__file__)):
        raise SystemExit("REFUSED: the end-to-end SAME build (N=60) has not been accepted by THIS version of the script "
                         "(run --build 60 --rule same, then --accept-same)")
    import claim_checks as CC
    from power_registered_set import registered_cells
    cells = registered_cells()
    have = set(f[:-8] for f in os.listdir(dump) if f.endswith(".parquet") and ".m2." not in f)
    miss = [c for c in cells if c not in have]
    if miss:
        raise SystemExit(f"REFUSED: {len(miss)} registered cells missing from {dump}")
    t0 = time.time()
    pub, _, _ = CC.load(REAL_DUMP, cells)
    new, bad_k, race_bad = CC.load(dump, cells)
    pub["ts"] = _load_ts(REAL_DUMP, cells); new["ts"] = _load_ts(dump, cells)
    print(f"ROLL SENSITIVITY ({rule}) | {len(cells)} registered cells | published {len(pub):,} setups, new {len(new):,} | {time.time()-t0:.0f}s | "
          f"race violations {race_bad}; setups without a twin M1b {bad_k['M1b']}, M2 {bad_k['M2']}")
    p_iid = six(CC, pub, "iid")
    kn = all(abs(p_iid[k]["est_pp"] - PUB[k]) < 0.0006 for k in PUB)
    print(f"  [{'PASS' if kn else 'FAIL'}] KNOWN NUMBER: the published dump reproduces 26.65's six estimates to 0.0006 pp on this code path")
    if not kn:
        return 1
    n_iid, n_blk, p_blk = six(CC, new, "iid"), six(CC, new, "block"), six(CC, pub, "block")
    key = lambda d: d.cell.astype(str) + "|" + d.ts.astype(str) + "|" + d.side.astype(int).astype(str)
    kp, kn_ = key(pub), key(new); common = set(kp) & set(kn_)
    pc, nc = pub[kp.isin(common)], new[kn_.isin(common)]
    print(f"  setups: common {len(common):,} | only published {len(pub) - len(common):,} | only new {len(new) - len(common):,}")
    c_pub, c_new = six(CC, pc, "iid"), six(CC, nc, "iid")
    print("  claim | block | new est [iid CI] [5-day-block CI] | published | diff | setup arm new/pub | twin arm new/pub | common-setups new-pub")
    flags, holds = [], True
    for (tag, blk), x in n_iid.items():
        d = x["est_pp"] - PUB[(tag, blk)]; fl = abs(d) > FLAG_PP; flags += [(tag, blk)] if fl else []
        b = n_blk[(tag, blk)]
        holds &= (x["lo_pp"] > 0) and (b["lo_pp"] > 0)
        if tag == "race":
            ra, rt = f"{100*x['rate_setup']:.3f}/{100*p_iid[(tag,blk)]['rate_setup']:.3f}", f"{100*x['rate_twin']:.3f}/{100*p_iid[(tag,blk)]['rate_twin']:.3f}"
        else:
            sn, sp = CC.arm_frame(new, tag), CC.arm_frame(pub, tag); sn, sp = sn[sn.block == blk], sp[sp.block == blk]
            ra, rt = f"{100*sn.rf.mean():.3f}/{100*sp.rf.mean():.3f}", f"{100*sn.tf.mean():.3f}/{100*sp.tf.mean():.3f}"
        cd = c_new[(tag, blk)]["est_pp"] - c_pub[(tag, blk)]["est_pp"]
        name = f"claim 1 {tag}" if tag != "race" else "claim 2 race"
        print(f"    {name:<14} {blk}: {CC.fmt(x)}  block[{b['lo_pp']:+.3f},{b['hi_pp']:+.3f}]  pub {PUB[(tag, blk)]:+.3f}  diff {d:+.3f}{' FLAG' if fl else ''}  "
              f"setup {ra}  twin {rt}  common {cd:+.3f}")
    print(f"  READING: all six lower ends above zero under BOTH the iid and the 5-day-block interval: {'YES' if holds else 'NO'}; "
          f"differences over {FLAG_PP} pp: {flags if flags else 'none'}")
    json.dump(dict(stamp="roll sensitivity", rule=rule, dump=dump, cells=len(cells), setups=[int(len(pub)), int(len(new))], common=int(len(common)),
                   claims_sha=sha(os.path.join(HERE, "claim_checks.py")), script_sha=sha(__file__), machine=platform.node(),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                   new_iid={f"{k[0]}_{k[1]}": v for k, v in n_iid.items()}, new_block={f"{k[0]}_{k[1]}": v for k, v in n_blk.items()},
                   pub_iid={f"{k[0]}_{k[1]}": v for k, v in p_iid.items()}, common_new={f"{k[0]}_{k[1]}": v for k, v in c_new.items()},
                   common_pub={f"{k[0]}_{k[1]}": v for k, v in c_pub.items()}, holds=bool(holds), flags=[list(f) for f in flags]),
              open(os.path.join(SENS, f"roll_claims_{rule}_{time.strftime('%Y-%m-%d')}.json"), "w"), indent=1, default=float)
    return 0


def noise():
    """A FLOOR for what a rebuild changes without any rule change: the fresh block-A-only and block-B-only N=60 builds (certified in
    the seal path, 26.183/26.185: same setups except the load edges, twins redrawn) against data\\dump on the same 36 cells."""
    sys.path.insert(0, HERE)
    import claim_checks as CC
    from power_registered_set import registered_cells
    cells = [c for c in registered_cells() if c.startswith("N60_")]
    pub, _, _ = CC.load(REAL_DUMP, cells)
    print(f"NOISE FLOOR (N=60, {len(cells)} registered cells; twin redraw + load edges only; no rule changed)")
    out = {}
    for blk, root in (("A", "seal_root_A60"), ("B", "seal_root_B60")):
        dd = os.path.join(HERE, root, "data", "dump")
        fresh, _, _ = CC.load(dd, cells)
        P, F = six(CC, pub, "iid", (blk,)), six(CC, fresh, "iid", (blk,))
        for tag in ("M1b", "M2", "race"):
            d = F[(tag, blk)]["est_pp"] - P[(tag, blk)]["est_pp"]
            out[f"{tag}_{blk}"] = dict(fresh=F[(tag, blk)]["est_pp"], dump=P[(tag, blk)]["est_pp"], diff=d)
            print(f"    {'claim 1 ' + tag if tag != 'race' else 'claim 2 race':<14} block {blk}: fresh {F[(tag, blk)]['est_pp']:+.3f}  data\\dump {P[(tag, blk)]['est_pp']:+.3f}  diff {d:+.3f} pp")
    json.dump(dict(stamp="noise floor", script_sha=sha(__file__), finished=time.strftime("%Y-%m-%d %H:%M:%S"), results=out),
              open(os.path.join(SENS, f"noise_floor_{time.strftime('%Y-%m-%d')}.json"), "w"), indent=1, default=float)
    return 0


def main(a):
    os.makedirs(SENS, exist_ok=True)
    rule = a[a.index("--rule") + 1] if "--rule" in a else None
    if rule is not None and rule not in RULES:
        raise SystemExit(f"--rule must be one of {RULES}")
    if "--selftest" in a:
        return 0 if selftest() else 1
    if "--make-raw" in a:
        r = a[a.index("--make-raw") + 1]
        if r not in RULES:
            raise SystemExit(f"rule must be one of {RULES}")
        if not selftest():
            return 1
        return make_raw(r)
    if "--k3" in a:
        return k3()
    if "--build" in a:
        if rule is None:
            raise SystemExit("--build N --rule RULE")
        if subprocess.call([sys.executable, __file__, "--selftest"]) != 0:     # a SEPARATE process: loader must not be imported before VWAP_ROOT is set
            return 1
        return build(int(a[a.index("--build") + 1]), rule)
    if "--accept-same" in a:
        return accept_same()
    if "--merge" in a:
        if rule is None:
            raise SystemExit("--merge --rule RULE")
        return merge(rule)
    if "--noise" in a:
        return noise()
    if "--claims" in a:
        if rule is None:
            raise SystemExit("--claims DUMP --rule RULE")
        return claims(a[a.index("--claims") + 1], rule)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
