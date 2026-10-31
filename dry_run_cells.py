r"""
dry_run_cells.py -- build a few cells with the CURRENT code into a scratch
VWAP_ROOT (scratch_pseudo_2026-09-29\), never touching data\dump.  §25.5 gate 1
evidence + stopwatch for the rebuild estimate.

    python dry_run_cells.py CELL [CELL ...]          build + write the cells
    python dry_run_cells.py --forced CELL [CELL ...] §25.5 gate 5 on REAL data (no writes)

Does the same preparation as dump_episodes.py's __main__ block, then calls the
real build_cell().  Its own correctness is checked by verify_columns.py, which
compares every EXISTING column of each rebuilt cell with the stored cell.

Prints: matched fraction, draw counts, timings; with --forced, the collapse when
the real anchor is the only candidate.  It deliberately does NOT compute or
print the real-minus-pseudo fill gap in the normal mode: §25.4 requires the F5
power validation to print PASS/FAIL before any real number is read.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, "scratch_pseudo_2026-09-29")
os.environ["VWAP_ROOT"] = SCRATCH                    # this process only
os.environ["VWAP_CHUNK"] = "1000000"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(HERE, "code"))
sys.path.insert(1, HERE)     # pseudo_vwap.py until it is in code\

T_START = time.perf_counter()
import json                                          # noqa: E402
import numpy as np                                   # noqa: E402

import dump_episodes as DE                           # noqa: E402
import pseudo_vwap                                   # noqa: E402
from loader import load                              # noqa: E402

assert os.path.realpath(DE.OUT).startswith(os.path.realpath(SCRATCH)), DE.OUT


def peak_mb():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2 ** 20


def parse(cell):
    p = cell.split("_")
    return int(p[0][1:]), p[1], p[2], float(p[3])


if __name__ == "__main__":
    forced_only = sys.argv[1:2] == ["--forced"]
    cells = sys.argv[2:] if forced_only else sys.argv[1:]
    if not cells:
        raise SystemExit(__doc__)
    t = time.perf_counter()
    d = load("train")
    block = d.block.to_numpy()
    labs = {"M1b": DE.strata(d, with_momentum=False), "M2": DE.strata(d, with_momentum=True)}
    F = DE.VF.build(d)
    HTF = dict(t15=DE.htf_trend(d, 15, 16), t1h=DE.htf_trend(d, 60, 8))
    PRE = DE.presignal.build_pre(d.close.to_numpy(), d.seg.to_numpy())
    CT = d.ts.dt.tz_convert("America/Chicago").dt.hour.to_numpy()
    ctx = (d.close.to_numpy(), d.high.to_numpy().astype(float),
           d.low.to_numpy().astype(float), d.seg.to_numpy(), d.atr.to_numpy(),
           block, None, CT)
    sess = DE.session_anchor(d)
    prep_s = time.perf_counter() - t
    print(f"PREP (load + strata + HTF + pre-signal): {prep_s:.1f}s   "
          f"peak RSS so far {peak_mb():,.0f} MiB", flush=True)

    anchors = {k: (c, m) for k, c, m in DE.ANCHORS}
    caches = {}

    def get_cache(N, akey):
        if (N, akey) not in caches:
            t0 = time.perf_counter()
            col, mx = anchors[akey]
            a = sess if akey == "session" else DE.rolling_argextreme(
                d[col].to_numpy().astype(float), N, d.seg.to_numpy(), mx)
            vwap, sd, W, cum, ref = DE.vwap_and_sd(d, a)
            caches[(N, akey)] = (a, vwap, sd, W, cum, ref)
            print(f"  anchor + exact line for N={N} {akey}: {time.perf_counter()-t0:.1f}s", flush=True)
        return caches[(N, akey)]

    if forced_only:
        for cell in cells:
            N, akey, trig, eps = parse(cell)
            cache = get_cache(N, akey)
            real_anchor = cache[0]
            orig = pseudo_vwap.draw_candidate_anchor
            print(f"\nFORCED (§25.5 gate 5, real data) {cell}", flush=True)
            for label, draw in (("real anchor is the ONLY candidate", lambda lo, hi, ra, rng: ra),
                                ("MUTATED: anchor forced one bar off the real one",
                                 lambda lo, hi, ra, rng: ra - 1)):
                pseudo_vwap.draw_candidate_anchor = draw
                try:
                    r2, _, _ = DE.build_cell(d, cache, N, akey, trig, eps, labs, ctx, F, HTF, PRE)
                finally:
                    pseudo_vwap.draw_candidate_anchor = orig
                s2 = r2[r2.is_real == 1]
                w, pw = s2["w_0.0"].to_numpy(), s2["pseudo_w_0.0"].to_numpy()
                Tc = s2["T"].iloc[0]
                both = s2.pseudo_matched.to_numpy() == 1
                mism = int(((w != pw) & both).sum())
                gap = 100 * (((w >= 1) & (w <= Tc))[both].mean() - ((pw >= 1) & (pw <= Tc))[both].mean())
                on_real = int((s2.pseudo_anchor_bar.to_numpy()[both]
                               == real_anchor[s2.bar.to_numpy().astype(int)][both]).sum())
                print(f"  {label}:\n    {len(s2):,} setups, matched {int(both.sum()):,}; "
                      f"pseudo_w_0.0 != w_0.0 on {mism:,} matched setups; anchor == real on "
                      f"{on_real:,}; real-minus-pseudo fill gap on matched = {gap:+.4f} pp", flush=True)
        print(f"\nPEAK RSS {peak_mb():,.0f} MiB   total {time.perf_counter()-T_START:.1f}s")
        raise SystemExit(0)

    report = dict(prep_s=prep_s, cells={})
    for cell in cells:
        N, akey, trig, eps = parse(cell)
        assert DE.cell_name(N, akey, trig, eps) == cell, cell
        cache = get_cache(N, akey)
        t = time.perf_counter()
        r, m2, counts = DE.build_cell(d, cache, N, akey, trig, eps, labs, ctx, F, HTF, PRE)
        build_s = time.perf_counter() - t
        DE.safeio.write_parquet(m2, f"{DE.OUT}/{cell}.m2.parquet")
        DE.safeio.write_json(counts, f"{DE.OUT}/{cell}.counts.json")
        DE.safeio.write_parquet(r, f"{DE.OUT}/{cell}.parquet")
        ns = counts["n_setups"]
        nm, nu = counts["n_pseudo_matched"], counts["n_pseudo_unmatched"]
        tm, tw = counts["t_pseudo_match_s"], counts["t_pseudo_walk_s"]
        real = r[r.is_real == 1]
        # production target = the setup's own realized distance; compare with the
        # pre-registration's literal "within +-10% of eps" (ATR cells only)
        if trig == "ATR":
            inband = float((np.abs(real.dist_atr / eps - 1) <= 0.10).mean())
            a_, vwap_, sd_, W_, cum_, ref_ = cache
            sg_ = real.bar.to_numpy().astype(int)
            pa_l, _, _ = pseudo_vwap.pseudo_for_cell(
                cum_, ref_, sg_, a_[sg_],
                None if akey == "session" else pseudo_vwap.window_lo(len(d), N, d.seg.to_numpy()),
                akey == "session", ctx[0], real.atr0.to_numpy(), np.full(len(sg_), eps),
                real.side.to_numpy(), cell)
            lit = (f"  | literal-eps target: {100*(pa_l < 0).mean():5.1f}% unmatched, "
                   f"{100*inband:5.1f}% of setups already within +-10% of eps")
        else:
            q = np.percentile(real.dist_atr, [10, 50, 90]) if len(real) else [np.nan] * 3
            lit = f"  | SD cell: setup dist_atr p10/50/90 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}"
        print(f"  {cell:<22} setups {ns:>6,}  build {build_s:7.1f}s | pseudo matched {nm:,} "
              f"unmatched {nu:,} ({100*nu/max(ns,1):.1f}%) mean draws {counts['pseudo_mean_draws']:.1f}"
              f" | match {tm:.2f}s walk {tw:.2f}s "
              f"({1e6*tm/max(ns,1):.0f} us + {1e6*tw/max(nm,1):.0f} us per setup)"
              f"{lit}", flush=True)
        report["cells"][cell] = dict(setups=ns, build_s=build_s, matched=nm, unmatched=nu,
                                     t_match_s=tm, t_walk_s=tw,
                                     mean_draws=counts["pseudo_mean_draws"])
    report["total_s"] = time.perf_counter() - T_START
    report["peak_rss_mib"] = peak_mb()
    print(f"\nTOTAL {report['total_s']:.1f}s   PEAK RSS {report['peak_rss_mib']:,.0f} MiB "
          f"(one process: prep + these cells)")
    with open(os.path.join(SCRATCH, "dry_run_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
