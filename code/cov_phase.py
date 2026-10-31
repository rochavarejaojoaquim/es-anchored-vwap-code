"""
cov_phase.py -- the covariate pass, split so the 1.7GB data load and the
450MB cell frames never live in the same process (C100).

  phase A   load data -> compute arrays for ONE (N, anchor) group -> npz -> exit
  phase B   load npz (small) -> join into that group's cells -> exit

usage:  python cov_phase.py A <N> <anchor>
        python cov_phase.py B <N> <anchor>

2026-09-21 (AUD-27, C119): the npz carries SPEC_VERSION. Phase A SKIPS a cache
only if it was built under the CURRENT spec; a cache from the old definitions
STOPS the run with a message (rename the covcache folder) -- it is never
silently reused and never deleted. Writes are atomic (safeio).
Joined per setup (twins get NaN; context belongs to the setup):
  L_all L_settled bars_since_touch prior_touches slope_15 slope_60
  is_rth dow rv_close day_pos days_to_exp anchor_spike
"""
import os as _os
_ROOT = _os.environ.get("VWAP_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DUMP_DIR = _os.path.join(_ROOT, "data", "dump")
ROOT_DIR = _ROOT + "/"
import gc
import glob
import os
import sys
import numpy as np
import pandas as pd

import safeio
from spec import SLOPES, SPEC_VERSION

CACHE = ROOT_DIR + "covcache"
DUMP = DUMP_DIR
os.makedirs(CACHE, exist_ok=True)
JOIN = ["L_all", "L_settled", "bars_since_touch", "prior_touches",
        "slope_15", "slope_60", "is_rth", "dow", "rv_close", "day_pos", "days_to_exp"]


def cache_path(N, akey):
    return f"{CACHE}/N{N}_{akey}.npz"


def phase_a(N, akey, d=None):
    p = cache_path(N, akey)
    if os.path.exists(p):
        v = safeio.npz_version(p)
        if v == SPEC_VERSION:
            print(f"  phase A cached (spec {v}): N={N} {akey}", flush=True); return
        raise SystemExit(f"STOP: {p} was built under spec '{v}'. It would silently "
                         f"reuse OLD features. Rename the covcache folder (do not "
                         f"delete it) and run again.")
    from loader import load
    from pipeline import rolling_argextreme, session_anchor
    from add_covariates import line_features, bar_features
    if d is None:
        d = load("train")
    a = session_anchor(d) if akey == "session" else rolling_argextreme(
        d["high" if akey == "hh" else ("low" if akey == "ll" else "volume")]
        .to_numpy().astype(float), N, d.seg.to_numpy(), akey != "ll")
    L_all, L_set, slope, since, ptouch = line_features(d, a)
    BF = bar_features(d)
    safeio.save_npz(p, anchor=a.astype(np.int32),
                    L_all=L_all, L_settled=L_set, bars_since_touch=since,
                    prior_touches=ptouch,
                    **{f"slope_{k}": slope[k] for k in SLOPES},
                    **{k: v for k, v in BF.items()})
    print(f"  phase A done: N={N} {akey}", flush=True)


def phase_b(N, akey):
    z = safeio.load_npz(cache_path(N, akey))          # refuses a stale cache
    a = z["anchor"]
    from loader import cell_files
    files = cell_files(f"N{N}_{akey}", DUMP)
    for f in files:
        x = pd.read_parquet(f)
        if "L_all" in x.columns:
            del x; gc.collect(); continue
        idx = x.index[x.is_real == 1]
        b = x.loc[idx, "bar"].to_numpy().astype(np.int64)
        for k in JOIN:
            x.loc[idx, k] = z[k][b]
        x.loc[idx, "anchor_spike"] = z["spike"][a[b]]
        safeio.write_parquet(x, f)
        print(f"  {os.path.basename(f):<32} {len(x.columns)} cols", flush=True)
        del x; gc.collect()


if __name__ == "__main__":
    mode, N, akey = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    (phase_a if mode == "A" else phase_b)(N, akey)
