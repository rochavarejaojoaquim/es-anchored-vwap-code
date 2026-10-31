"""
dump_episodes.py -- per-episode records for every cell: the setups, their M1b
twins (inside the cell file) and their M2 twins (<cell>.m2.parquet).

REBUILD 2026-09-21 (sec. 21.18). What changed from the old-rule dump:
  R1  re-arm: a setup fires only after the previous one in the cell CLOSED
      (rearm.py). The old rule's setup count is kept as n_candidates in
      <cell>.counts.json, so the burst rate is measured.
  R2  twins: different CME trade date AND |j - i| > h_rec; same block, stratum
  R3  strata / ct_hour on Chicago local time (AUD-17)
  R6  er_pre{15,30,60,120}, er_anchor (entry); er_ratio{k} (POST)
  R8  M2 twins (strata with momentum) in <cell>.m2.parquet -- every consumer
      already skips ".m2." files, so M2 can never mix into M1b
      claim-2 MIRROR race: mir_w, race (setups, M1b and M2 twins)
  R11 mae_fill_close / mae_fill_wick / mae_fillbar_wick (STORE ONLY)
  R12 lin_maxdev, lin_meandev, chord_slope (+ settled _s, + _xs)
  AUD-20 break detector unit-proof; bars_to_gap = 0 now gives the gap AT the
      signal instead of NaN
  AUD-25 every cell is written; underpowered = setups < MIN_SIGNALS
  AUD-26 atomic writes (safeio)
  AUD-30 adverse grid XS restored to {0.5..12}
  AUD-35 cens_T stored (store only)
  2026-09-29 (sec. 25): pseudo_w_0.0 / pseudo_matched / pseudo_anchor_bar (the
      pseudo-VWAP control, pseudo_vwap.py) and mae_all_wick / gap_at_T
      (_walkextra.py). Outcome columns only; all are in within_core.POST.

Rationale kept from the original: cell-level medians cannot show a
relationship that lives INSIDE a cell. Only per-episode records can.

Built-in check (D3): the full walk's tau=0 fill must equal the fill the re-arm
rule used, on every kept setup -- they are two different code paths.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from loader import load, stamp, trade_day, trade_date, minutes_since_epoch
from pipeline import (htf_trend, rolling_argextreme, session_anchor,
                      vwap_and_sd, profile_at, path_features, OUT)
from spec import (SPEC_VERSION, ATR_EPS, SD_EPS, NS, ANCHORS, TAUS, XS, PATH_BARS,
                  SD_MIN_AGE, SD_MIN_BAND, K_CTRL, TWIN_SEED, MIN_SIGNALS,
                  SETTLE, H_REC_MULT, t_stop)
from walk_fast import walk
from controls_fast import walk_static, draw_controls
from control_matched import strata
from _walkextra import EXTRA_KEYS
import volume_features as VF
import presignal
import linearity
import rearm
import safeio
import pseudo_vwap

os.makedirs(OUT, exist_ok=True)


def cell_name(N, akey, trig, eps):
    return f"N{N}_{akey}_{trig}_{eps}"


def twins(d, ctx, sig, side, dist, lab, h_rec, T, scheme):
    close, hi, lo, seg, atr, block, TD, CT = ctx
    src, ctl, cs, cd = draw_controls(d, sig, side, dist, lab, K=K_CTRL,
                                     seed=TWIN_SEED, min_gap=h_rec)
    catr = atr[ctl]
    cw = walk_static(close, seg, ctl, catr, cs * cd * catr, h_rec, T=T, hi=hi, lo=lo)
    C = dict(bar=ctl, ts=d.ts.to_numpy()[ctl], block=block[ctl],
             day=trade_date(d.ts.iloc[ctl]), is_real=0, scheme=scheme,
             side=cs, dist_atr=cd, atr0=catr, ct_hour=CT[ctl], src_bar=src)
    for t in TAUS:
        C[f"t_{t}"] = cw[f"tl_{t}"]
        C[f"w_{t}"] = cw[f"w_{t}"]
    for x in XS:
        C[f"adv_{x}"] = cw[f"al_{x}"]
    C["adv_T"] = cw["adv_T"]
    for k in EXTRA_KEYS:
        C[k] = cw[k]
    return pd.DataFrame(C)


def build_cell(d, cache, N, akey, trigger, eps, labs, ctx, F, HTF, PRE):
    close, hi, lo, seg, atr, block, TD, CT = ctx
    a, vwap, sd, W, cum, ref = cache
    n = len(d)
    vol = d.volume.to_numpy().astype(float); vp = d.vp.to_numpy()
    T = t_stop(trigger, eps); h_rec = H_REC_MULT * T

    elig = rearm.eligible(d, a, vwap, sd, atr, trigger)
    scale = sd if trigger == "SD" else atr
    dist = np.where(elig, np.abs(close - vwap) / np.maximum(scale, 1e-9), np.nan)

    # ---- R1: candidates (the OLD rule) -> primary fill -> greedy keep
    cand = rearm.candidates(dist, a, elig, eps, n, h_rec)
    cg0 = close[cand] - vwap[cand]
    cfill = rearm.first_fill(lo, hi, seg, cand, vwap[cand], np.sign(cg0), T)
    keep = rearm.greedy_keep(cand, cfill, T)
    sig = cand[keep]
    counts = dict(cell=cell_name(N, akey, trigger, eps), spec=SPEC_VERSION,
                  n_candidates=int(len(cand)), n_setups=int(len(sig)),
                  candidates_sha=rearm.bars_hash(cand), setups_sha=rearm.bars_hash(sig),
                  n_candidates_by_block={b: int((block[cand] == b).sum()) for b in "AB"},
                  n_setups_by_block={b: int((block[sig] == b).sum()) for b in "AB"})

    anc = a[sig]; atr0 = atr[sig]; g0 = close[sig] - vwap[sig]; W0 = W[sig]
    side = np.sign(g0)
    # --- anchor expiry: the trade is unaffected, but an ESCALATION measured
    #     after this bar is against a line that no longer exists.
    chg_idx = np.where(np.r_[False, a[1:] != a[:-1]])[0]
    k = np.searchsorted(chg_idx, sig + 1)
    nxt = np.where(k < len(chg_idx), chg_idx[np.minimum(k, len(chg_idx) - 1)], n + 1)
    anchor_expired_at = np.where(nxt <= sig + h_rec, nxt - sig, -1)
    # --- horizon (h_rec) truncated by a roll
    end = np.minimum(sig + h_rec, n - 1)
    censored = seg[end] != seg[sig]
    # --- weekend / multi-day break (AUD-20: integer minutes, unit-proof)
    mins = ctx_minutes(d)
    spans_gap = (mins[end] - mins[sig]).astype(float) > 1.8 * h_rec
    step = np.r_[0, np.diff(mins)]
    brk = np.where(step > 120)[0]                   # >2h with no bars = a break
    kb = np.searchsorted(brk, sig + 1)
    nb = np.where(kb < len(brk), brk[np.minimum(kb, len(brk) - 1)], n + 1)
    bars_to_gap = np.where(nb <= sig + h_rec, nb - 1 - sig, -1)
    lastbar = sig + np.maximum(bars_to_gap, 0)
    gap_at_break = np.where(bars_to_gap >= 0,       # AUD-20: 0 = break right after
                            (close[lastbar] - (close[sig] - g0)) / atr0, np.nan)

    ew = walk(close, cum, ref, seg, sig, anc, atr0, g0, W0, h_rec, T=T, hi=hi, lo=lo)
    # D3: two code paths, one answer
    wf = ew["wf_0.0"]; kf = cfill[keep]
    same = np.array_equal(np.where((wf >= 1) & (wf <= T), wf, -1), kf)
    if not same:
        raise SystemExit(f"FAIL {counts['cell']}: walk fill != re-arm fill "
                         f"on {int((np.where((wf >= 1) & (wf <= T), wf, -1) != kf).sum())} setups")

    # ---- sec. 25 pseudo-VWAP control: ONE pseudo anchor per setup (a redraw only
    # evaluates cumulative sums), then ONE walk for the whole cell, same function
    # and same code path as the real line (gate 4).
    if len(sig):                                     # D3: adapter == the real line
        d3 = float(np.abs(ref[sig] + cum.line(anc, sig) - vwap[sig]).max())
        if d3 > 1e-9:
            raise SystemExit(f"FAIL {counts['cell']}: pseudo adapter != real line "
                             f"at the signal, max diff {d3:.3g}")
    session = akey == "session"
    if not session and N not in _WLO:
        _WLO.clear()
        _WLO[N] = pseudo_vwap.window_lo(n, N, seg)
    t_ps = time.perf_counter()
    pa, pl, nd = pseudo_vwap.pseudo_for_cell(
        cum, ref, sig, anc, None if session else _WLO[N], session, close, atr0,
        np.abs(g0) / atr0, side, counts["cell"])
    t_match = time.perf_counter() - t_ps
    pm = pa >= 0
    pfill = np.full(len(sig), np.nan)
    t_ps = time.perf_counter()
    if pm.any():
        sm = sig[pm]
        pwk = walk(close, cum, ref, seg, sm, pa[pm], atr0[pm], close[sm] - pl[pm],
                   cum.vol(pa[pm], sm), h_rec, T=T, hi=hi, lo=lo)
        pfill[pm] = pwk["wf_0.0"]
    t_walk = time.perf_counter() - t_ps
    counts.update(n_pseudo_matched=int(pm.sum()), n_pseudo_unmatched=int((~pm).sum()),
                  pseudo_mean_draws=float(nd.mean()) if len(nd) else 0.0,
                  t_pseudo_match_s=round(t_match, 3), t_pseudo_walk_s=round(t_walk, 3))

    prof = profile_at(vp, vol, anc, sig, atr0, vwap[sig], close[sig])
    vf = VF.at_signals(F, sig, anc, atr0, vwap[sig], sd[sig])
    pf = path_features(close, seg, sig, atr0, g0, PATH_BARS)
    pre = presignal.at_signals(PRE[0], PRE[1], close, sig, anc, pf["er_p15"])
    lin = linearity.linearity_at(cum, ref, sig, anc, atr0, side, SETTLE)

    ep = dict(bar=sig, ts=d.ts.to_numpy()[sig], block=block[sig],
              day=trade_date(d.ts.iloc[sig]), is_real=1, scheme="setup",
              side=side, dist_atr=np.abs(g0) / atr0,
              dist_sd=np.abs(g0) / np.maximum(sd[sig], 1e-9),
              sd_in_atr=sd[sig] / atr0, age=(sig - anc).astype(float),
              speed=W0 / np.maximum(sig - anc, 1), W0=W0, atr0=atr0,
              ct_hour=CT[sig],
              anchor_expired_at=anchor_expired_at.astype(float),
              censored=censored, spans_gap=spans_gap,
              bars_to_gap=bars_to_gap.astype(float), gap_at_break=gap_at_break,
              underpowered=np.full(len(sig), len(sig) < MIN_SIGNALS),
              trend_15m=HTF["t15"][sig], trend_1h=HTF["t1h"][sig],
              anchor_px_dist=(np.abs(close[sig] - close[anc]) / atr0
                              if akey != "session" else np.full(len(sig), np.nan)))
    ep.update(prof); ep.update(vf); ep.update(pf); ep.update(pre); ep.update(lin)
    for t in TAUS:
        ep[f"t_{t}"] = ew[f"tf_{t}"]            # frozen line, CLOSE-based
        ep[f"w_{t}"] = ew[f"wf_{t}"]            # frozen line, WICK fill (PRIMARY)
        ep[f"tlive_{t}"] = ew[f"tl_{t}"]
        ep[f"wlive_{t}"] = ew[f"wl_{t}"]
    for x in XS:
        ep[f"adv_{x}"] = ew[f"af_{x}"]
    ep["adv_T"] = ew["adv_froz_T"]
    ep["min_T"] = ew["min_froz_T"]
    for kk in EXTRA_KEYS:
        ep[kk] = ew[kk]
    ep["pseudo_w_0.0"] = pfill                       # NaN = unmatched
    ep["pseudo_matched"] = pm.astype(np.int8)
    ep["pseudo_anchor_bar"] = pa                     # -1 = unmatched
    E = pd.DataFrame(ep)

    dA = np.abs(g0) / atr0
    M1 = twins(d, ctx, sig, side, dA, labs["M1b"], h_rec, T, "M1b")
    M2 = twins(d, ctx, sig, side, dA, labs["M2"], h_rec, T, "M2")
    counts["n_twins_M1b"] = int(len(M1)); counts["n_twins_M2"] = int(len(M2))

    out = pd.concat([E, M1], ignore_index=True)
    for f_ in (out, M2):
        f_["N"] = N; f_["anchor"] = akey; f_["trigger"] = trigger
        f_["eps"] = eps; f_["T"] = T
    return out, M2, counts


_MIN = {}
_WLO = {}


def ctx_minutes(d):
    if "m" not in _MIN:
        _MIN["m"] = minutes_since_epoch(d.ts)
    return _MIN["m"]


if __name__ == "__main__":
    only = [int(x) for x in sys.argv[1:]] or NS
    d = load("train")
    print("STAMP:", stamp(d, f"DUMP | 1m | wick | re-arm R1 | {SPEC_VERSION}"), flush=True)
    pseudo_vwap.write_manifest(OUT, os.path.dirname(os.path.abspath(__file__)))
    block = d.block.to_numpy()
    labs = {"M1b": strata(d, with_momentum=False), "M2": strata(d, with_momentum=True)}
    F = VF.build(d)
    HTF = dict(t15=htf_trend(d, 15, 16), t1h=htf_trend(d, 60, 8))
    PRE = presignal.build_pre(d.close.to_numpy(), d.seg.to_numpy())
    CT = d.ts.dt.tz_convert("America/Chicago").dt.hour.to_numpy()   # AUD-17
    ctx = (d.close.to_numpy(), d.high.to_numpy().astype(float),
           d.low.to_numpy().astype(float), d.seg.to_numpy(), d.atr.to_numpy(),
           block, None, CT)          # slot 6 unused (kept for the unpack)
    print("  twins strata, HTF trend, pre-signal ER built", flush=True)
    sess = session_anchor(d)
    t0 = time.time()
    for N in only:
        for akey, col, mx in ANCHORS:
            mk = f"{OUT}/N{N}_{akey}.done"
            if os.path.exists(mk):
                print(f"  skip N={N} {akey}", flush=True); continue
            a = sess if akey == "session" else rolling_argextreme(
                d[col].to_numpy().astype(float), N, d.seg.to_numpy(), mx)
            vwap, sd, W, cum, ref = vwap_and_sd(d, a)
            cache = (a, vwap, sd, W, cum, ref)
            nrows = 0
            for trig, grid in (("ATR", ATR_EPS), ("SD", SD_EPS)):
                for eps in grid:
                    name = cell_name(N, akey, trig, eps)
                    fn = f"{OUT}/{name}.parquet"
                    if os.path.exists(fn):          # per-CELL resume (atomic write)
                        continue
                    r, m2, counts = build_cell(d, cache, N, akey, trig, eps, labs,
                                               ctx, F, HTF, PRE)
                    safeio.write_parquet(m2, f"{OUT}/{name}.m2.parquet")
                    safeio.write_json(counts, f"{OUT}/{name}.counts.json")
                    safeio.write_parquet(r, fn)       # LAST: its existence = cell done
                    nrows += len(r)
                    print(f"    {name:<24} candidates {counts['n_candidates']:>7,} -> "
                          f"setups {counts['n_setups']:>7,}", flush=True)
            with open(mk + ".tmp", "w") as fh:
                fh.write(f"{nrows}\n")
            os.replace(mk + ".tmp", mk)
            del cache, vwap, sd, W, cum
            print(f"  done N={N} {akey}  {nrows:,} rows  ({time.time()-t0:.0f}s)", flush=True)
