"""
spec.py -- every LOCKED constant of the study, in ONE place (rule C3, AUD-29).

Before 2026-09-21 these lived in episodes_v3.py (a legacy builder) and were
re-exported by pipeline.py. A "single source of truth" edit then silently
reverted the adverse grid XS from the DECIDED {0.5..12} to {1,2,4,8} (AUD-30,
C115). Values, not just names, are now checked by selfcheck against this file.

SPEC_VERSION is written into every cache file (covcache, HTF npz) and every
cell's counts sidecar. A cache built under a different version is REFUSED,
never silently reused (AUD-27, C119).
"""
SPEC_VERSION = "v5.1-rearm-2026-09-21"

# ---- trigger grid and horizon (unchanged since v5)
ATR_EPS = [3.0, 4.0, 6.0, 8.0, 12.0]
SD_EPS = [1.0, 1.5, 2.0, 3.0]
NS = [60, 100, 160, 200, 240]
ANCHORS = [("hh", "high", True), ("ll", "low", False),
           ("hv", "volume", True),          # highest-volume candle
           ("session", None, None)]
MIN_LINE_BARS = 2        # AUD-37: anchor bar + at least one more
SD_MIN_AGE = 30                             # R19
SD_MIN_BAND = 1.00                          # R19


def t_stop(trigger, eps):
    """Horizon T in bars. ATR: 45*(eps/2.75)^2.  SD: 30*eps^2."""
    return int(round(45 * (eps / 2.75) ** 2)) if trigger == "ATR" \
        else int(round(30 * eps ** 2))


H_REC_MULT = 3                              # walks record h_rec = 3*T bars

# ---- outcomes
TAUS = [0.0, 0.10, 0.25, 0.50, 1.00]        # 0.0 = the PRIMARY (turns 338-343)
XS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]   # DECIDED 2026-09-20 (AUD-30)
PATH_BARS = [5, 15, 30]

# ---- cells and twins
MIN_SIGNALS = 100        # a cell below this is FLAGGED underpowered, never dropped (AUD-25)
K_CTRL = 2               # twins per setup, per scheme
TWIN_SEED = 17
TICK = 0.25
# 4*(high+low+close) is a whole number on a 0.25 grid, and the typical price is
# that number / 12. PRICE_SCALE is that 12: the anchored sums are kept as EXACT
# integers in these units and converted to points only at the end (AUD-38).
PRICE_SCALE = 12

# ---- covariates
SETTLE = 10              # bars skipped as the line's violent birth (L_settled, lin_*_s)
SLOPES = [15, 60]
PRE_KS = [15, 30, 60, 120]                  # pre-signal efficiency ratio lookbacks (R6)
HTF = [(5, {"4h": 48, "1d": 288}), (15, {"4h": 16, "1d": 96})]
RULES = ["hh", "ll", "hv", "session"]

# columns that get a "_xs" copy = column x side (R10 / AUD-19).
# Positive _xs = pointing in the direction of the STRETCH (price's side of the
# line). For a slope that means "the line is moving TOWARD price".
XS_BASE = (["slope_15", "slope_60", "d_PC", "trend_15m", "trend_1h"]
           + [f"xa_{r}_{k}" for r in RULES for k in ("dPV", "dPC")]
           + [f"htf_{tf}m_{dur}_{k}" for tf, lb in HTF for dur in lb
              for k in ("dPV", "dPC")])
