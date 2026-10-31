"""paper_yearly_summary.py -- arithmetic on the already-written paper_yearly_<date>.json (no dump read, no
bootstrap, nothing new estimated from market data). For each claim scheme it prints
  * the weighted least-squares slope of the yearly estimates INSIDE 2010-2017 and INSIDE 2022-2026,
    with the naive standard error and a dispersion-scaled one (the naive SE times sqrt(max(1, weighted
    residual sum of squares / (n-2))), the usual quasi-likelihood correction for overdispersion);
  * for each fitted curve, the dispersion (weighted SSE per degree of freedom) and the lam interval
    scaled by sqrt(dispersion).
Reading rules (fixed here before it is run): descriptive; a slope is "not distinguished from zero" when
its dispersion-scaled 95% interval (estimate +/- 1.96 x scaled SE) includes 0; no half-life is quoted from
any interval printed here; the curves are not claimed to be the true shape.
Usage: python paper_yearly_summary.py [paper_yearly_YYYY-MM-DD.json]   (default: the newest such file)"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def slope(t, e, se):
    w = 1.0 / np.asarray(se) ** 2
    X = np.c_[np.ones(len(t)), np.asarray(t) - np.mean(t)]
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    b = cov @ X.T @ W @ np.asarray(e)
    res = np.asarray(e) - X @ b
    disp = float((w * res ** 2).sum() / (len(t) - 2))
    se_n = float(np.sqrt(cov[1, 1]))
    return float(b[1]), se_n, se_n * np.sqrt(max(disp, 1.0)), disp


def main(path):
    j = json.load(open(path, encoding="utf-8"))
    out = {}
    print(f"SUMMARY of {os.path.basename(path)}\n{j['stamp']}\n")
    for key, v in j["results"].items():
        ys = v["years"]
        t = np.array([y["t"] for y in ys]); e = np.array([y["est"] for y in ys]); se = np.array([y["se"] for y in ys])
        d = dict(label=v["label"])
        print(v["label"])
        for nm, m in (("2010-2017", t < 2018), ("2022-2026", t > 2020)):
            b, sn, ss, disp = slope(t[m], e[m], se[m])
            d[nm] = dict(slope=b, se_naive=sn, se_scaled=ss, dispersion=disp, n=int(m.sum()),
                         lo=b - 1.96 * ss, hi=b + 1.96 * ss)
            print(f"   slope inside {nm}: {b:+.2f} pp/yr  naive SE {sn:.2f}  dispersion-scaled SE {ss:.2f}  "
                  f"95% [{b-1.96*ss:+.2f}, {b+1.96*ss:+.2f}]  (dispersion {disp:.1f}, {int(m.sum())} years)"
                  f"  -> {'not distinguished from zero' if b-1.96*ss <= 0 <= b+1.96*ss else 'excludes zero'}")
        K = len(ys)
        for kind in ("exp", "hyp"):
            f = v["fits"][kind]
            disp = f["sse"] / (K - 2)
            s = float(np.sqrt(max(disp, 1.0)))
            lo = f["lam"] - s * (f["lam"] - f["lam_lo"]); hi = f["lam"] + s * (f["lam_hi"] - f["lam"])
            d[kind] = dict(dispersion=disp, scale=s, lam=f["lam"], lam_scaled_lo=lo, lam_scaled_hi=hi,
                           heldout_pred=f["heldout_pred_pp"])
            print(f"   {kind}: dispersion {disp:.1f} (SD scale {s:.1f})  lam {f['lam']:.3f}  naive [{f['lam_lo']:.3f}, "
                  f"{f['lam_hi']:.3f}]  dispersion-scaled [{lo:.3f}, {hi:.3f}]  held-out curve average "
                  f"{f['heldout_pred_pp']:+.2f} vs observed {v['heldout']['est_pp']:+.2f}")
        out[key] = d
    o = os.path.join(HERE, "paper_yearly_summary.json")
    json.dump(out, open(o, "w", encoding="utf-8"), indent=1)
    print("\nwrote", os.path.basename(o))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(HERE, "paper_yearly_2*.json")))[-1]
    main(p)
