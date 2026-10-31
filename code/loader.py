"""
loader.py -- the data constraints of the LOCKED SPEC (§2, currently v5)

Every constraint lives HERE and nowhere else (rule C3).
No analysis function may read raw files directly.

Constraints enforced:
  1. calendar spreads dropped
  2. front month = highest daily volume contract
  3. anchors RESET at each roll -> segment ids, no lookback across a roll
  4. SEAL: test split (2018-2022) refused unless unseal=True
  5. ATR = true range, period 14, Wilder smoothing
  6. session-reopen bar excluded from the ATR update
  7. 500-bar warm-up discarded after each roll reset
  8. prices (high/low/close/vp) are kept in full precision: the exact integer
     moments (AUD-38) need them exactly on the 0.25 tick grid
  9. exp_day: the front contract's expiry (3rd Friday of its month) as an
     integer day number, from the CONTRACT CALENDAR (R7/AUD-15, 2026-09-21).
     Was: the segment's last timestamp, which depends on future volume.

Time helpers (C116: never assume the storage unit of a timestamp):
  minutes_since_epoch(ts)  integer minutes, same answer for ns or us data
  trade_day(ts)            CME trade date as an integer day number
  trade_date(ts)           the same, as python dates (for consumers that
                           already use it; slower and heavier -- C100)
"""
import datetime as _dt
import re as _re
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
DATA_OUTRIGHT = _os.path.join(_ROOT, "data", "outright.parquet")
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- constants
# (restored to training in spec v4; unchanged since. Seal unchanged.)
TRAIN_BLOCKS = [("A", pd.Timestamp("2010-01-01", tz="UTC"),
                      pd.Timestamp("2018-01-01", tz="UTC")),
                ("B", pd.Timestamp("2022-01-01", tz="UTC"),
                      pd.Timestamp("2026-08-07", tz="UTC"))]
TEST_START  = pd.Timestamp("2018-01-01", tz="UTC")
TEST_END    = pd.Timestamp("2022-01-01", tz="UTC")

ATR_PERIOD   = 14
WARMUP_BARS  = 500          # discarded after each roll reset
PRICE_CONV   = "typical"    # (H+L+C)/3 for VWAP weighting

RAW = DATA_OUTRIGHT


# ----------------------------------------------------------------- utilities
def _wilder(tr, period, skip_mask):
    """Wilder smoothing. Bars where skip_mask is True do NOT update the ATR;
    the previous value is carried forward."""
    n = len(tr)
    out = np.full(n, np.nan)
    seed_end = period + 1
    if n <= seed_end:
        return out
    out[period] = np.nanmean(tr[1:seed_end])
    for i in range(period + 1, n):
        if skip_mask[i] or not np.isfinite(tr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = ((period - 1) * out[i - 1] + tr[i]) / period
    return out


def rolling_argextreme(a, N, seg, maximum=True):
    """Index of the extreme of `a` over the trailing N bars, never crossing a
    segment boundary. O(n) monotonic deque. Returns -1 where undefined."""
    n = len(a)
    out = np.full(n, -1, dtype=np.int64)
    dq = []
    seg_start = 0
    for i in range(n):
        if i > 0 and seg[i] != seg[i - 1]:
            dq = []
            seg_start = i
        while dq and ((a[dq[-1]] <= a[i]) if maximum else (a[dq[-1]] >= a[i])):
            dq.pop()
        dq.append(i)
        lo = max(i - N + 1, seg_start)
        while dq[0] < lo:
            dq.pop(0)
        if i - seg_start >= N - 1:
            out[i] = dq[0]
    return out


def reopen_mask(ts):
    """First bar of each Globex session (Chicago 17:00). ONE definition (AUD-9):
    the loader (ATR skip) and pipeline.session_anchor both use it."""
    chi_h = ts.dt.tz_convert("America/Chicago").dt.hour
    return ((chi_h == 17) & (chi_h.shift(1) != 17)).to_numpy()


# ----------------------------------------------------------------- the loader
def load(split="train", unseal=False, lean=True, raw=None):
    """raw: another outright file with the same columns (the synthetic gate
    data only). Every constraint below still applies to it."""
    if split == "test" and not unseal:
        raise PermissionError(
            "SEALED SET. 2018-2022 is the confirmation sample. "
            "Pass unseal=True only for the single pre-registered test."
        )

    df = pd.read_parquet(raw or RAW, columns=['ts','instrument_id','symbol',
                                       'high','low','close','volume'])

    # (1) drop calendar spreads
    df = df[~df["symbol"].astype(str).str.contains("-")].copy()

    # (4) seal
    if split == "train":
        keep = np.zeros(len(df), dtype=bool)
        blk = np.array([""] * len(df), dtype=object)
        for name, lo, hi in TRAIN_BLOCKS:
            m = (df.ts >= lo) & (df.ts < hi)
            keep |= m.to_numpy()
            blk[m.to_numpy()] = name
        df = df[keep].copy()
        df["block"] = blk[keep]
    elif split == "test":
        df = df[(df.ts >= TEST_START) & (df.ts < TEST_END)].copy()
        df["block"] = "SEALED"
    else:
        raise ValueError("split must be 'train' or 'test'")

    # (2) front month by daily volume
    df["date"] = df.ts.dt.floor("D")   # datetime64, NOT python date objects
    g = (df.groupby(["date", "instrument_id"], observed=True)["volume"]
           .sum().reset_index())
    g["rk"] = g.groupby("date")["volume"].rank(ascending=False, method="first")
    fm = g.loc[g.rk == 1, ["date", "instrument_id"]]
    d = (df.merge(fm, on=["date", "instrument_id"], how="inner")
           .sort_values("ts").reset_index(drop=True))
    # a training-block boundary is treated like a roll: no window may span it
    d["_blkchg"] = (d.block != d.block.shift()).cumsum()

    # (3) roll segments: a new segment each time the front-month contract changes
    d["seg"] = ((d.instrument_id != d.instrument_id.shift()) |
                (d.block != d.block.shift())).cumsum().astype(np.int64)

    # (6) session-reopen bars (first bar of the ETH session, Chicago 17:00)
    is_reopen = reopen_mask(d.ts)
    # a segment's first bar has no valid previous close either
    seg_first = (d.seg != d.seg.shift()).to_numpy()
    skip_atr = is_reopen | seg_first

    # (5) true range and Wilder ATR, computed within segments
    pc = np.array(d.close.shift(1).to_numpy(), dtype=float, copy=True)
    pc[seg_first] = np.nan
    tr = np.array(np.maximum(
        d.high.to_numpy() - d.low.to_numpy(),
        np.maximum(np.abs(d.high.to_numpy() - pc), np.abs(d.low.to_numpy() - pc)),
    ), dtype=float, copy=True)
    tr[seg_first] = (d.high.to_numpy() - d.low.to_numpy())[seg_first]

    atr = np.full(len(d), np.nan)
    for _, ix in d.groupby("seg", sort=True).indices.items():
        atr[ix] = _wilder(tr[ix], ATR_PERIOD, skip_atr[ix])

    d["tr"] = tr
    d["atr"] = atr
    d["is_reopen"] = is_reopen

    # (8) contract expiry per segment, from the calendar (R7)
    first = np.r_[0, np.where(d.seg.to_numpy()[1:] != d.seg.to_numpy()[:-1])[0] + 1]
    seg_ids = d.seg.to_numpy()[first]
    syms = d.symbol.astype(str).to_numpy()[first]
    fdays = d.ts.iloc[first].dt.date.to_numpy()
    emap = {s: expiry_day(sym, fd) for s, sym, fd in zip(seg_ids, syms, fdays)}
    d["exp_day"] = pd.Series(d.seg.to_numpy()).map(emap).to_numpy().astype(np.int32)

    # (7) warm-up: bars too close to the start of their segment are not eligible
    pos_in_seg = d.groupby("seg").cumcount().to_numpy()
    d["pos_in_seg"] = pos_in_seg
    d["eligible"] = (pos_in_seg >= WARMUP_BARS) & np.isfinite(atr)

    # VWAP weighting price
    d["vp"] = (d.high + d.low + d.close) / 3.0

    d["split"] = split
    if lean:
        for c in ("symbol", "instrument_id", "_blkchg", "is_reopen",
                  "pos_in_seg", "split"):
            if c in d.columns:
                d = d.drop(columns=c)
        # AUD-38 (decision B): high / low / close and the typical price stay in
        # full precision. Rounding them to float32 cost ~1.6e-4 points, which is
        # larger than the SD error we just removed, and the exact integer
        # moments need the prices exactly on the 0.25 grid. ~70 MB more RAM.
        for c in ("tr", "atr"):
            d[c] = d[c].astype(np.float32)
        d["volume"] = d["volume"].astype(np.float32)
    return d.reset_index(drop=True)


def trade_date(ts):
    """CME TRADE DATE (AUD-13, 2026-09-21). The Globex session that opens at
    17:00 Chicago belongs to the NEXT calendar day, so a UTC or local calendar
    date splits one session in two. Chicago time + 7h puts 17:00 at midnight.
    Accepts tz-naive (taken as UTC) or tz-aware timestamps."""
    t = pd.DatetimeIndex(pd.to_datetime(ts))
    if t.tz is None:
        t = t.tz_localize("UTC")
    return (t.tz_convert("America/Chicago") + pd.Timedelta(hours=7)).date


_MONTH = dict(F=1, G=2, H=3, J=4, K=5, M=6, N=7, Q=8, U=9, V=10, X=11, Z=12)
_EPOCH = _dt.date(1970, 1, 1)


def third_friday(y, m):
    first = _dt.date(y, m, 1)
    return first + _dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def expiry_day(symbol, first_day):
    """Expiry of `symbol` (e.g. 'ESH4') as days since 1970-01-01: the first
    3rd Friday of its month, on or after the day its front-month run began,
    whose year matches the symbol's year digit(s). Raises if none does --
    a wrong contract/date pairing must stop the build, not pass silently."""
    m = _re.match(r"^[A-Z]+?([FGHJKMNQUVXZ])(\d{1,2})$", symbol)
    if not m:
        raise ValueError(f"cannot parse contract symbol {symbol!r}")
    mon, ydig, nd = _MONTH[m.group(1)], int(m.group(2)), len(m.group(2))
    for y in range(first_day.year, first_day.year + 3):
        e = third_friday(y, mon)
        if y % (10 ** nd) == ydig and e >= first_day:
            return (e - _EPOCH).days
    raise ValueError(f"no expiry for {symbol} on/after {first_day}")


def minutes_since_epoch(ts):
    """Integer minutes since 1970-01-01 UTC. Unit-proof (C116, AUD-20): the old
    `astype('int64') // 60_000_000_000` assumed nanoseconds and finds no
    weekend at all when timestamps are stored in microseconds."""
    s = pd.Series(pd.to_datetime(ts))
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy(np.int64)


def trade_day(ts):
    """CME trade date as integer days since 1970-01-01 (Chicago + 7h, see
    trade_date). Integer arrays, no python date objects (C100)."""
    s = pd.Series(pd.to_datetime(ts))
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    loc = s.dt.tz_convert("America/Chicago").dt.tz_localize(None) + pd.Timedelta(hours=7)
    return ((loc - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).to_numpy(np.int64)


def cell_files(pat="N*", dump=None):
    """The dump's CELL files, and nothing else: never <cell>.m2.parquet (M2
    twins), never a legacy .tw2 file, never a .tmp. ONE definition (C3) --
    core_v2 once globbed the dump itself and would have searched .m2 files as
    cells (found 2026-09-21)."""
    import glob as _glob
    dump = dump or DUMP_DIR
    # AUD-44 (2026-09-24): accept a FULL cell name as well as a prefix. It used
    # to glue "_*" on, so "N60_hh_ATR_3.0" matched nothing and the caller did
    # nothing, in silence.
    hits = set(_glob.glob(_os.path.join(dump, f"{pat}_*.parquet")))
    hits |= set(_glob.glob(_os.path.join(dump, f"{pat}*.parquet")))
    return [f for f in sorted(hits) if ".m2." not in f and ".tw2." not in f]


def stamp(d, extra=""):
    """Provenance stamp (rule A1)."""
    blocks = "+".join(f"{b}:{n:,}" for b, n in d.block.value_counts().sort_index().items())
    return (f"train[{blocks}] | bars={len(d):,} | "
            f"eligible={int(d.eligible.sum()):,} | segments={d.seg.nunique()} | "
            f"{d.ts.min().date()}->{d.ts.max().date()} | "
            f"ATR=TR/14/Wilder/reopen-excl | warmup={WARMUP_BARS} | "
            f"vwap_price={PRICE_CONV}{(' | ' + extra) if extra else ''}")
