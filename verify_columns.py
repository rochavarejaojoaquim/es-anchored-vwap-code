r"""
verify_columns.py -- §25.5 gate 1: adding outcome columns must not move a single
existing number, or the 409 floors are void again.

    python verify_columns.py OLD_DUMP_DIR NEW_DUMP_DIR CELL [CELL ...]
    python verify_columns.py OLD_DUMP_DIR NEW_DUMP_DIR ALL     every cell of OLD (FAILS
                                                               if NEW lacks any, or has extras)
    python verify_columns.py --selftest      known-number check + mutations

For each cell (its .parquet and its .m2.parquet) it checks, PASS/FAIL:
  1  same number of rows
  2  every OLD column is present in NEW, with the SAME dtype
  3  every OLD column is bit-identical (NaN == NaN; numeric compared as bytes)
  4  the OLD columns keep their relative ORDER (core_v2 keeps the FIRST of a tie)
  5  the NEW columns are exactly the expected set -- nothing more, nothing less
  6  core_v2.search_cols() and cell_columns() return the SAME list for OLD and
     NEW: the new outcome columns must not enter the covariate search
  7  pseudo columns: setups only; unmatched <-> (matched=0, anchor=-1, fill NaN);
     twins all NaN;  gap_at_T is NaN exactly where cens_T is True
Reads only.  Writes nothing.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

NEW_MAIN = ["mae_all_wick", "gap_at_T", "pseudo_w_0.0", "pseudo_matched", "pseudo_anchor_bar"]
NEW_M2 = ["mae_all_wick", "gap_at_T"]


def same_col(a, b):
    if a.dtype != b.dtype:
        return False
    if a.dtype.kind in "biuf":
        return a.to_numpy().tobytes() == b.to_numpy().tobytes()
    return bool(a.equals(b))


def compare(old, new, expected_new, search_cols, cell_columns, pseudo_rules=True):
    """-> list of (check_id, ok, message). old/new are DataFrames."""
    out = []

    def add(i, ok, msg):
        out.append((i, bool(ok), msg))
    add(1, len(old) == len(new), f"rows old {len(old):,} new {len(new):,}")
    miss = [c for c in old.columns if c not in new.columns]
    dt = [c for c in old.columns if c in new.columns and old[c].dtype != new[c].dtype]
    add(2, not miss and not dt, f"missing {miss[:3]} dtype-changed {dt[:3]}")
    bad = [c for c in old.columns if c in new.columns and len(old) == len(new)
           and not same_col(old[c], new[c])]
    add(3, not bad and not miss, f"{len(old.columns) - len(bad) - len(miss)} of {len(old.columns)} "
                                 f"old columns bit-identical; differing {bad[:4]}")
    order = [c for c in new.columns if c in old.columns]
    add(4, order == [c for c in old.columns if c in new.columns] and not miss,
        "old columns keep their relative order")
    extra = [c for c in new.columns if c not in old.columns]
    add(5, sorted(extra) == sorted(expected_new), f"new columns {extra} (expected {expected_new})")
    a, b = search_cols(old), search_cols(new)
    ca, cb = cell_columns(list(old.columns)), cell_columns(list(new.columns))
    add(6, a == b and ca == cb, f"search_cols {len(a)} vs {len(b)}, cell_columns {len(ca)} vs {len(cb)}")
    if pseudo_rules:
        real = new.is_real == 1
        ok = True
        msg = []
        if "pseudo_matched" in new.columns:
            s = new[real]
            um = s.pseudo_matched == 0
            ok &= bool((s.pseudo_anchor_bar[um] == -1).all() and s["pseudo_w_0.0"][um].isna().all()
                       and s["pseudo_w_0.0"][~um].notna().all() and (s.pseudo_anchor_bar[~um] >= 0).all())
            t = new[~real]
            ok &= bool(t[["pseudo_matched", "pseudo_anchor_bar", "pseudo_w_0.0"]].isna().all().all())
            msg.append(f"unmatched {int(um.sum()):,}/{len(s):,} setups")
        if "gap_at_T" in new.columns and "cens_T" in new.columns:
            ok &= bool((new.gap_at_T.isna() == new.cens_T.astype(bool)).all())
            msg.append("gap_at_T NaN == cens_T")
        add(7, ok, "; ".join(msg))
    return out


def run(old_dir, new_dir, cells):
    import glob
    import core_v2

    def names(d):
        return sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(d, "N*.parquet"))
                      if ".m2." not in f and ".tw2." not in f)
    if cells == ["ALL"]:                   # every cell of OLD must exist in NEW, and nothing else
        cells = names(old_dir)
        gone, extra = sorted(set(cells) - set(names(new_dir))), sorted(set(names(new_dir)) - set(cells))
        if gone or extra:
            print(f"  [FAIL] cell sets differ: {len(gone)} of {len(cells)} OLD cells missing in NEW "
                  f"{gone[:3]}, {len(extra)} extra in NEW {extra[:3]}")
            return False
        print(f"  [PASS] cell sets identical: {len(cells)} cells in both")
    allok = True
    tot_rows = 0
    for cell in cells:
        for suffix, expect, rules in ((".parquet", NEW_MAIN, True), (".m2.parquet", NEW_M2, False)):
            o = pd.read_parquet(os.path.join(old_dir, cell + suffix))
            n = pd.read_parquet(os.path.join(new_dir, cell + suffix))
            res = compare(o, n, expect, core_v2.search_cols, core_v2.cell_columns, rules)
            ok = all(r[1] for r in res)
            allok &= ok
            tot_rows += len(n)
            print(f"  [{'PASS' if ok else 'FAIL'}] {cell + suffix:<40} "
                  f"{len(o.columns)} -> {len(n.columns)} cols, {len(n):,} rows")
            for i, k, m in res:
                if not k or "-v" in sys.argv:
                    print(f"        check {i}: {'ok' if k else 'FAILED'}  {m}")
    print(f"\n  stamp: {len(cells)} cells x (main + m2) = {2*len(cells)} files, {tot_rows:,} rows, "
          f"old={old_dir}  new={new_dir}")
    print(f"  {'ALL IDENTICAL' if allok else 'A DIFFERENCE -- the 409 floors are void on this build. STOP.'}")
    return allok


# ------------------------------------------------------------------ self-test
def selftest():
    import core_v2
    r = np.random.default_rng(3)
    n = 200
    old = pd.DataFrame({
        "bar": np.arange(n), "is_real": np.r_[np.ones(100), np.zeros(100)].astype(int),
        "side": r.choice([-1.0, 1.0], n), "T": 45, "ts": pd.date_range("2020-01-01", periods=n, tz="UTC"),
        "w_0.0": r.integers(-1, 45, n), "speed": r.normal(size=n), "relvol": r.normal(size=n),
        "cens_T": r.random(n) < 0.1, "mae_fill_wick": r.normal(size=n)})
    new = old.copy()
    cens = old.cens_T.to_numpy()
    new["mae_all_wick"] = r.normal(size=n)
    new["gap_at_T"] = np.where(cens, np.nan, r.normal(size=n))
    real = old.is_real.to_numpy() == 1
    matched = real & (r.random(n) < 0.9)
    new["pseudo_w_0.0"] = np.where(matched, r.integers(-1, 45, n).astype(float), np.nan)
    new["pseudo_matched"] = np.where(real, matched.astype(float), np.nan)
    new["pseudo_anchor_bar"] = np.where(matched, 5.0, np.where(real, -1.0, np.nan))

    def go(o, nw, expect=NEW_MAIN):
        return all(k for _, k, _ in compare(o, nw, expect, core_v2.search_cols, core_v2.cell_columns))

    res = []

    def check(name, ok):
        res.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("SELF-TEST (known answers first, then each mutation must FAIL)\n")
    check("K1 old vs old + the 5 expected new columns -> identical", go(old, new))
    m = new.copy(); m.loc[7, "speed"] += 1e-12
    check("M1 one existing value moved by 1e-12 -> FAILS", not go(old, m))
    m = new.drop(columns=["relvol"])
    check("M2 an existing column dropped -> FAILS", not go(old, m))
    m = new[["speed"] + [c for c in new.columns if c != "speed"]]
    check("M3 existing columns reordered -> FAILS", not go(old, m))
    m = new.copy(); m["speed"] = m["speed"].astype("float32")
    check("M4 dtype changed float64 -> float32 -> FAILS", not go(old, m))
    m = new.copy(); m["extra_col"] = 1.0
    check("M5 an unexpected extra column -> FAILS", not go(old, m))
    m = new.drop(columns=["gap_at_T"])
    check("M6 an expected new column missing -> FAILS", not go(old, m))
    m = new.copy(); m["leaky_outcome"] = m["pseudo_w_0.0"]
    check("M7 a new column NOT in within_core.POST (would enter the search) -> FAILS",
          not go(old, m, NEW_MAIN + ["leaky_outcome"]))
    m = new.copy(); m.loc[real & (m.pseudo_matched == 0), "pseudo_anchor_bar"] = 3.0
    m.loc[np.where(real)[0][0], ["pseudo_matched", "pseudo_anchor_bar", "pseudo_w_0.0"]] = [0.0, 3.0, 1.0]
    check("M8 unmatched setup carrying an anchor -> FAILS", not go(old, m))
    m = new.copy(); m.loc[np.where(cens)[0][0], "gap_at_T"] = 0.5
    check("M9 gap_at_T finite where cens_T is True -> FAILS", not go(old, m))
    print(f"\n  {'ALL DETECTORS ALIVE' if all(res) else 'A DETECTOR IS DEAD'}")
    return all(res)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) < 3:
        raise SystemExit(__doc__)
    sys.exit(0 if run(args[0], args[1], args[2:]) else 1)
