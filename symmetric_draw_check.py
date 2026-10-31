r"""
symmetric_draw_check.py -- does section 25 amendment 7's symmetric draw actually
remove the bias it was written to remove?

The 8 pseudo gates all still pass after the change, and all 8 mutations still kill
their targets -- but NONE of them tests the thing that changed. This does, by
running the OLD matcher (kept at stale\pseudo_vwap_before_symmetric_2026-10-01.py)
and the NEW one side by side on the same synthetic setups.

THE SETUP MIRRORS THE REAL DEFECT. vwap_from_anchor is patched so a candidate's
distance is drawn from a distribution concentrated BELOW eps, which is the real
situation: the true anchor is the window extreme, so every other anchor gives a
line nearer to price and candidate distances pile up at the low end of the +-10%
acceptance band. Under "first acceptable" that must produce a negative mean delta.

PREDICTION, stated before running:
  OLD: mean delta clearly NEGATIVE (the measured dump had -0.02422 ATR, 53.16%
       of setups nearer)
  NEW: mean delta ~ 0 and the nearer-share ~ 50%, because the target half is a
       fair coin from each setup's own generator.
If the NEW draw does not centre, the rebuild must not run.

    python symmetric_draw_check.py

Written 2026-10-01. Project root; rotates nothing. Reads no dump.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import pseudo_vwap as NEW                                              # noqa: E402

OLD_PATH = os.path.join(HERE, "stale", "pseudo_vwap_before_symmetric_2026-10-01.py")
_s = importlib.util.spec_from_file_location("pseudo_vwap_old", OLD_PATH)
OLD = importlib.util.module_from_spec(_s)
_s.loader.exec_module(OLD)

EPS, ATR0, CLOSE, SIDE = 3.0, 1.0, 100.0, 1
NSET = 20000


def run(mod, skew=0.32, seed0=0):
    """Draw NSET setups through `mod`'s matcher. Returns
    (mean delta/eps, share nearer, unmatched, matched).

    THE CANDIDATE DISTANCE MUST HAVE A DECREASING DENSITY ACROSS THE BAND, or there
    is no bias for "first acceptable" to exploit. The first version of this fixture
    drew it UNIFORMLY over [0.84, 1.14]*eps, which is flat inside the +-10% band, so
    the OLD matcher came out unbiased (-0.0006) and K1 failed -- a defect in the
    FIXTURE, not evidence the old code was fine.
    A Beta(1,2) variate, v = 1 - sqrt(1-U), has density 2(1-v): decreasing. Mapped
    onto [0.84, 1.16]*eps the density across the acceptance band falls by about 4x
    from the near edge to the far one, which is the real geometry -- every anchor
    other than the window extreme gives a line NEARER to price.
    """
    deltas, fbs, unmatched = [], [], 0
    for i in range(NSET):
        rng = np.random.default_rng([12345, i])
        draw = np.random.default_rng([999, i])

        def fake_vwap(cum, ref, a, at, _d=draw):
            v = 1.0 - np.sqrt(1.0 - _d.random())      # Beta(1,2): density 2(1-v)
            dist = EPS * (0.84 + skew * v)
            return CLOSE - SIDE * dist * ATR0

        mod.vwap_from_anchor = fake_vwap
        out = mod.match_pseudo_anchor(None, None, 0, 200, 40, CLOSE, ATR0, EPS,
                                      SIDE, rng)
        a, V = out[0], out[1]
        fb = out[3] if len(out) > 3 else 0
        if a is None:
            unmatched += 1
            continue
        deltas.append(abs(CLOSE - V) / ATR0 - EPS)
        fbs.append(fb)
    d, f = np.array(deltas), np.array(fbs)
    run.fallback = float(f.mean()) if len(f) else 0.0
    run.mean_nofb = float(d[f == 0].mean() / EPS) if (f == 0).any() else float("nan")
    run.share_nofb = float((d[f == 0] < 0).mean()) if (f == 0).any() else float("nan")
    return (float(d.mean() / EPS), float((d < 0).mean()), unmatched, len(d))


def main():
    ok = True
    print("\nSYMMETRIC DRAW CHECK -- amendment 7, old matcher vs new, same setups")
    print(f"  {NSET:,} synthetic setups, eps {EPS}, band +-10%, cap "
          f"{NEW.REDRAW_CAP}\n")
    print(f"  {'draw':<8} {'mean delta/eps':>15} {'share nearer':>13} "
          f"{'unmatched':>10} {'matched':>9}")
    res = {}
    for name, mod in (("OLD", OLD), ("NEW", NEW)):
        m, share, unm, nm = run(mod, skew=0.30)
        res[name] = (m, share, unm, nm); print(f"           fallback {100*run.fallback:.2f}%  |  excluding fallbacks: mean {run.mean_nofb:+.5f}, nearer {100*run.share_nofb:.2f}%")
        print(f"  {name:<8} {m:>+15.5f} {100*share:>12.2f}% {unm:>10,} {nm:>9,}")

    mo, so, _, _ = res["OLD"]
    mn, sn, un, _ = res["NEW"]
    print()

    def chk(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}   {detail}")
        ok &= bool(cond)

    chk("K1 the OLD draw is biased LOW, reproducing the defect", mo < -0.005,
        f"mean delta/eps {mo:+.5f}, {100*so:.2f}% nearer (dump measured 53.16%)")
    chk("K2 the NEW draw is CENTRED (|mean delta/eps| <= 0.005, amendment 7 B1)",
        abs(mn) <= 0.005, f"mean delta/eps {mn:+.5f}")
    # B3's registered band is [49%, 51%] ON THE REBUILT DUMP, not on this fixture.
    # The fixture is a STRESS TEST about 3x harsher than reality -- its OLD arm is
    # 68.36% nearer where the real dump measured 53.16% -- so its residual is
    # expected to be correspondingly larger. The author confirmed option (a) on
    # 2026-10-01: judge B1-B3 where they were registered. Tuning code until a
    # harsher-than-registered sample passes would itself be a form of fitting.
    chk("K3 the NEW nearer-share is within the FIXTURE band [49%, 53%] "
        "(B3's [49%, 51%] is judged on the REBUILT DUMP, not here)",
        0.49 <= sn <= 0.53, f"{100*sn:.2f}%  -- on a fixture whose OLD arm is "
        f"{100*so:.2f}% against the real dump's 53.16%")
    chk("K4 the NEW draw does not inflate unmatched",
        un <= res["OLD"][2] + 0.01 * NSET,
        f"old {res['OLD'][2]:,} vs new {un:,} of {NSET:,}")
    chk("K5 the fix moved the bias by a material amount", abs(mo - mn) > 0.005,
        f"{mo:+.5f} -> {mn:+.5f}")

    if ok:
        print("\n  THE SYMMETRIC DRAW WORKS ON THE STRESS FIXTURE. The rebuild may")
        print("  run. B1-B5 are then judged ON THE REBUILT DUMP (amendment 7), and")
        print("  if B1-B3 fail there the rebuild is NOT accepted and the old dump")
        print("  stays in place -- that measurement is the real test, not this one.\n")
    else:
        print("\n  THE FIX DOES NOT CENTRE EVEN DIRECTIONALLY -- DO NOT REBUILD\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
