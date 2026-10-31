r"""
identity_full.py -- the nesting IDENTITY behind section 25, on the WHOLE registered set.

Section 25 compared two FROZEN, SAME-SIDE levels walked on the SAME price path. The nearer
level is touched whenever the farther one is, so per setup gap = r - p must satisfy
gap * delta >= 0 (delta = pseudo distance - real distance), and r == p when the levels
coincide. HANDOFF 26.68 measured it on 3 cells (125,470 setups, 0 violations). This runs
the same count on all 144 registered cells for the record.

    python identity_full.py --selftest
    python identity_full.py data\dump

Read-only; writes identity_full_<date>.json (stamped). Project root; rotates nothing.
"""
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PUB = dict(n=3_821_840, real=51.573, pseudo=51.757)       # 26.34


def count(gap, delta):
    gap, delta = np.asarray(gap), np.asarray(delta)
    out = dict(n=int(len(gap)), differ=int((gap != 0).sum()),
               violations=int((gap * delta < 0).sum()),
               neg_when_farther=int(((gap == -1) & (delta > 0)).sum()),
               pos_when_nearer=int(((gap == 1) & (delta < 0)).sum()))
    for w in (0.005, 0.02, 0.05):
        m = np.abs(delta) < w
        out[f"near_{w}"] = dict(n=int(m.sum()), differ=int((gap[m] != 0).sum()))
    return out


def selftest():
    print("\nidentity_full.py SELF-TEST\n")
    rng = np.random.default_rng(0)
    delta = rng.normal(0, 0.25, 10_000)
    gap = np.where(rng.random(10_000) < 0.2, np.sign(delta), 0).astype(int)   # obeys identity
    a = count(gap, delta)
    ok1 = a["violations"] == 0 and a["differ"] > 0
    print(f"  [{'PASS' if ok1 else 'FAIL'}] K1 a frame obeying the identity shows 0 violations   "
          f"{a['differ']} differ")
    bad = gap.copy(); bad[:7] = -np.sign(delta[:7]).astype(int)
    b = count(bad, delta)
    ok2 = b["violations"] == 7
    print(f"  [{'PASS' if ok2 else 'FAIL'}] M1 seven planted sign violations are counted exactly   "
          f"{b['violations']}")
    print(f"\n  {'ALL DETECTORS ALIVE' if ok1 and ok2 else 'A DETECTOR IS DEAD'}\n")
    return ok1 and ok2


def main(dump):
    if not selftest():
        raise SystemExit("SELF-TEST FAILED -- no dump was read.")
    from gate_receipts import code_set
    from power_registered_set import registered_cells
    from pseudo_dist import build
    t0 = time.time()
    d = build(dump, registered_cells(), verbose=False)
    real, pseudo = 100 * d.r.mean(), 100 * d.p.mean()
    k = [("K0a matched setups = 26.34's 3,821,840", len(d) == PUB["n"], f"{len(d):,}"),
         ("K0b real fill 51.573%", abs(real - PUB["real"]) < 0.0015, f"{real:.4f}%"),
         ("K0c pseudo fill 51.757%", abs(pseudo - PUB["pseudo"]) < 0.0015, f"{pseudo:.4f}%")]
    for lab, okk, det in k:
        print(f"  [{'PASS' if okk else 'FAIL'}] {lab}   {det}")
    if not all(x[1] for x in k):
        raise SystemExit("A known-number check FAILED.")
    c = count(d.r.to_numpy() - d.p.to_numpy(), d.delta.to_numpy())
    cs = code_set()
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    stamp = (f"section 25 nesting identity | 144 registered cells | {c['n']:,} matched setups | "
             f"tau 0, wick fill within T | {dump} | code-set {cs} | identity_full.py {me[:16]} | "
             f"{platform.node()} | ES train A+B, sealed untouched")
    print(f"\nSTAMP: {stamp}")
    print(f"  r != p                      : {c['differ']:,} ({c['differ']/c['n']:.3%})")
    print(f"  VIOLATIONS gap*delta < 0    : {c['violations']:,}   (identity says 0)")
    print(f"  gap -1 with pseudo farther  : {c['neg_when_farther']:,}")
    print(f"  gap +1 with pseudo nearer   : {c['pos_when_nearer']:,}")
    for w in (0.005, 0.02, 0.05):
        q = c[f"near_{w}"]
        print(f"  |delta| < {w:<5}: {q['n']:>9,} setups, r != p on {q['differ']:,}")
    verdict = "IDENTITY HOLDS" if c["violations"] == 0 else "IDENTITY VIOLATED -- investigate"
    print(f"\n  {verdict}   ({time.time()-t0:.0f}s)")
    path = os.path.join(HERE, f"identity_full_{datetime.date.today().isoformat()}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(stamp=stamp, code_set=cs, producer_sha256=me, verdict=verdict, **c), fh, indent=1)
    print(f"  wrote {os.path.basename(path)}")
    return 0 if c["violations"] == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    if not a:
        raise SystemExit(__doc__)
    sys.exit(main(a[0]))
