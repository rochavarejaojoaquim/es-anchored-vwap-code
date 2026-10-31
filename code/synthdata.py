"""
synthdata.py -- a small SYNTHETIC outright.parquet for the code gates (no real
data needed, so the gates run anywhere, including before any rebuild).

Two contracts per training block, real CME hours (daily 16:00-17:00 CT halt,
weekend), a DST switch, a quarterly roll (the next contract becomes the volume
leader and trades 20 points higher), occasional wide "spike" bars, and the
contract symbols the expiry calendar must parse. Random walk: it contains NO
VWAP effect, so it tests arithmetic, never findings.
"""
import numpy as np
import pandas as pd


def _block(start, days, sym_old, sym_new, iid_old, iid_new, roll, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    ct = idx.tz_convert("America/Chicago")
    h, dw = ct.hour, ct.dayofweek
    op = ~(h == 16) & ~((dw == 4) & (h >= 16)) & ~(dw == 5) & ~((dw == 6) & (h < 17))
    ts = idx[op]
    n = len(ts)
    c = 4000 + np.cumsum(rng.normal(0, 1.0, n) * np.where(rng.random(n) < 0.002, 8, 1))
    c = np.round(c / 0.25) * 0.25
    span = np.round(rng.uniform(0, 1.5, (2, n)) / 0.25) * 0.25
    hi = c + span[0]; lo = c - span[1]
    v = rng.integers(20, 800, n).astype(float)
    rows = []
    rollts = pd.Timestamp(roll, tz="UTC")
    for sym, iid, off, lead in ((sym_old, iid_old, 0.0, ts < rollts), (sym_new, iid_new, 20.0, ts >= rollts)):
        vol = np.where(lead, v, np.maximum(v // 10, 1))
        rows.append(pd.DataFrame(dict(ts=ts, instrument_id=iid, symbol=sym,
                                      open=c + off, high=hi + off, low=lo + off,
                                      close=c + off, volume=vol)))
    return pd.concat(rows)


def make_outright(path, seed=0):
    a = _block("2017-11-26 23:00", 26, "ESZ7", "ESH8", 101, 102, "2017-12-12 00:00", seed)
    b = _block("2024-02-25 23:00", 26, "ESH4", "ESM4", 201, 202, "2024-03-08 00:00", seed + 1)
    df = pd.concat([a, b]).sort_values(["ts", "instrument_id"]).reset_index(drop=True)
    df["ts"] = df.ts.dt.as_unit("ns")
    import safeio
    safeio.write_parquet(df, path)
    return len(df)
