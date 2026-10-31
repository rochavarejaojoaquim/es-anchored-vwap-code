"""
_walkextra.py -- per-episode outputs added in the 2026-09-21 rebuild, shared by
the setup walk (walk_fast.walk) and the twin walk (controls_fast.walk_static)
so both run the IDENTICAL rule (C65). Computed chunk by chunk.

Given, per row: side sg (+1 price above the level), start close C0, frozen
level V0 = C0 - G0, the MIRROR level M0 = C0 + G0 (same distance, other side),
wick arrays H, L, closes, a validity mask, and w0 = first wick fill of V0
(1-based, -1 if none within the walk).

  mae_fill_close   max over bars 1..w0 of the ADVERSE close excursion
                   side*(close_t - C0)/ATR0                      (sec. 21.20, R11)
  mae_fill_wick    max over bars 1..w0-1 of the adverse WICK (high for a short
                   fade, low for a long fade) - C0, / ATR0      (AUD-31)
  mae_fillbar_wick the fill bar's own adverse wick, / ATR0 -- inside that bar
                   we cannot know whether the stop or the target traded first
      all three: NaN unless 1 <= w0 <= T (the primary success), and may be
      NEGATIVE (price never went against the entry). Stored only; the
      stop-loss analysis is deferred (sec. 21.21).
  mir_w            first bar whose wick touches the MIRROR (1-based, -1 none)
  race             claim-2 race within T:  1 VWAP first | -1 mirror first |
                   0 both in the same bar (tie) | 9 neither (open)
  cens_T           the walk was cut by a roll / end of data before T bars
                   (store only -- AUD-35)
  mae_all_wick     2026-09-29 (sec. 23.8 item 22, sec. 25): the SAME adverse-wick
                   quantity as mae_fill_wick -- adv_w below, measured from C0,
                   the signal-bar close, in ATR0 -- but its running max over ALL
                   valid bars 1..T, whatever happened to the fill. Not NaN for
                   unfilled episodes. Includes the fill bar and later bars.
                   Partial (valid bars only) when the walk is cut; cens_T flags it.
  gap_at_T         2026-09-29 (R29, the T-bar mark): (close at bar T - V0) / ATR0,
                   V0 = C0 - G0 the frozen level, signed like gap_at_break (NOT
                   side-adjusted). NaN exactly when cens_T is True.
"""
import numpy as np


def extra(w0, H, L, C, C0, G0, sg, a0, valid, T):
    n = len(w0)
    TT = min(T, H.shape[1])
    Hs, Ls, Cs, vs = H[:, :TT], L[:, :TT], C[:, :TT], valid[:, :TT]
    adv_c = np.where(vs, sg * (Cs - C0) / a0, -np.inf)
    adv_w = np.where(vs, np.where(sg > 0, Hs - C0, C0 - Ls) / a0, -np.inf)
    cm_c = np.maximum.accumulate(adv_c, axis=1)
    cm_w = np.maximum.accumulate(adv_w, axis=1)
    ok = (w0 >= 1) & (w0 <= T)
    r = np.arange(n)
    k = np.clip(w0 - 1, 0, TT - 1)
    mae_c = np.where(ok, cm_c[r, k], np.nan)
    mae_w = np.where(ok & (w0 >= 2), cm_w[r, np.maximum(k - 1, 0)], np.nan)
    mae_b = np.where(ok, adv_w[r, k], np.nan)
    M0 = C0 + G0
    touch = np.where(sg > 0, H >= M0, L <= M0) & valid
    anym = touch.any(axis=1)
    mir = np.where(anym, np.argmax(touch, axis=1) + 1, -1)
    v = ok
    m = (mir >= 1) & (mir <= T)
    race = np.full(n, 9, dtype=np.int8)
    race[v & (~m | (w0 < mir))] = 1
    race[m & (~v | (mir < w0))] = -1
    race[v & m & (w0 == mir)] = 0
    cens = vs.sum(axis=1) < T
    last = TT - 1
    mae_all = np.where(np.isfinite(cm_w[:, last]), cm_w[:, last], np.nan)
    gap_T = np.where(~cens, ((Cs[:, last:last + 1] - (C0 - G0)) / a0)[:, 0], np.nan)
    return dict(mae_fill_close=mae_c, mae_fill_wick=mae_w, mae_fillbar_wick=mae_b,
                mir_w=mir, race=race, cens_T=cens, mae_all_wick=mae_all, gap_at_T=gap_T)


EXTRA_KEYS = ["mae_fill_close", "mae_fill_wick", "mae_fillbar_wick", "mir_w", "race", "cens_T",
              "mae_all_wick", "gap_at_T"]
