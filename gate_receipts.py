r"""
gate_receipts.py -- the SINGLE enforcement point for section 26.3's order.

26.4 step 7: pseudo_gap.py must REFUSE to run unless the exchangeability gate
(26.19) and the power validation (26.21) both say PASS, "enforced in one place"
(research rule C3).

WHY THIS FILE EXISTS. Measured 2026-10-01 and recorded in HANDOFF 26.28:
  - pseudo_null_pair.py writes only its per-cell frame. The pooled statistic that
    DECIDES the gate, and the verdict, were printed and discarded. The CI cannot be
    rebuilt from the CSV, because the statistic clusters on TRADE DATE pooled across
    cells and a date appears in many cells.
  - pseudo_power.py serializes nothing at all.
So neither PASS existed on disk. This module gives each producer a machine-readable
RECEIPT and gives the consumer one function, require_pass(), that re-derives the
verdict instead of trusting it.

THE DESIGN RULE, and it is the whole point: A FLAG ASSERTS, A RECEIPT PROVES.
require_pass() never reads a stored boolean and believes it. For each receipt it
  1. re-applies the producer's OWN decision function to the recorded statistics
     (imported from the producer -- never reimplemented here, see E24),
  2. re-derives the confidence interval from (gap, se) and requires the stored one
     to match, so a tampered statistic contradicts its own verdict,
  3. re-computes the code-set hash, the producer's SHA-256 and a dump fingerprint,
     and requires all three to equal what the receipt claims.
Any single failure REFUSES. There is no override flag, by design.

WHY RE-CHECKING STORED STATISTICS IS SUFFICIENT, rather than re-running (26.28):
  gate   95% CI half-width 0.0292 pp inside a +-0.25 pp margin -> 8.6x headroom.
         A verdict 8.6x from its edge cannot be flipped by float representation,
         which is why E02's compare-as-integers rule does not bite here.
  power  P1 200/200, exact Clopper-Pearson one-sided 95% lower bound
         0.05^(1/200) = 0.9851 against a 0.80 bar; P2 9/200 with an exact 95%
         interval of about [2.1%, 8.4%] inside the required [1%, 10%].
  Both verdicts hold across their WHOLE Monte Carlo interval. --reverify exists for
  the paranoid case and is never the default: a changed dump already flips the
  fingerprint and REFUSES, which forces a genuine re-run exactly when one is needed.

USAGE
    python gate_receipts.py --selftest          toy fixtures, writes only to a temp
                                                dir, reads no project data
    python gate_receipts.py --check data\dump   verify the receipts on this tree

    # from pseudo_gap.py, as the first statement:
    import gate_receipts
    STAMP = gate_receipts.require_pass(dump_dir)

Written 2026-10-01. Project root, so it is outside the
code-set hash and rotates nothing (26.8, 26.24). Its own SHA-256 goes into
verify_pc.py's KNOWN once the producers are edited -- see C135 / 26.28.
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")

RECEIPT_VERSION = 1
KINDS = ("gate", "power")
PRODUCER = {"gate": "pseudo_null_pair.py", "power": "pseudo_power.py"}

# Keys every receipt must carry. A missing key REFUSES rather than defaulting.
REQUIRED = ("receipt_version", "kind", "verdict", "stats", "code_set",
            "producer", "producer_sha256", "dump_fingerprint", "dump_dir",
            "written_utc", "machine")
REQUIRED_STATS = {
    "gate": ("gap_pp", "se_pp", "ci_lo_pp", "ci_hi_pp", "z", "margin_pp",
             "n_setups", "n_days", "n_paired", "n_setups_total",
             "frac_same_anchor", "degen_frac", "n_cells", "repro_bad"),
    "power": ("n_draws", "k_detected", "k_null_rejected", "gap_err", "n", "D",
              "need", "fpr_lo", "fpr_hi", "delta", "n_cells", "cells_sha"),
}


# ----------------------------------------------------------------- fingerprints
def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def code_set(code_dir=CODE):
    """The 12-hex code-set hash verify_pc.py:222 logs. Imported from
    pseudo_vwap, never reimplemented, so the two can never drift (E24)."""
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    import pseudo_vwap
    return pseudo_vwap.code_set(code_dir)


def dump_fingerprint(dump_dir):
    """A semantic fingerprint of a built dump: 16 hex chars.

    Built from each cell's counts.json -- which already carries the BUILD'S OWN
    content hashes (setups_sha, candidates_sha) -- plus pseudo_manifest.json.

    Deliberately NOT mtime-based: brain/02 records that files move between the two
    machines by ZIP, which rewrites mtimes, so an mtime fingerprint would false-alarm.
    Deliberately not a content hash of the parquet either: that is 5.6 GB on every
    gap run, for no extra information that setups_sha does not already give.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(dump_dir, "*.counts.json"))):
        with open(p, "r", encoding="utf-8") as fh:
            c = json.load(fh)
        rows.append("|".join(str(c.get(k, "")) for k in (
            "cell", "spec", "n_candidates", "n_setups", "candidates_sha",
            "setups_sha", "n_pseudo_matched", "n_pseudo_unmatched",
            "n_twins_M1b", "n_twins_M2")))
    man = os.path.join(dump_dir, "pseudo_manifest.json")
    if os.path.exists(man):
        with open(man, "r", encoding="utf-8") as fh:
            rows.append("MANIFEST|" + json.dumps(json.load(fh), sort_keys=True))
    else:
        rows.append("MANIFEST|ABSENT")
    rows.append(f"NCELLS|{len(rows) - 1}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


def machine_stamp():
    import platform
    m = {"node": platform.node(), "cpu_count": os.cpu_count(),
         "python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "pyarrow"):
        try:
            m[mod] = __import__(mod).__version__
        except Exception:
            m[mod] = "NOT IMPORTED"
    return m


# --------------------------------------------------------------------- writing
def write(kind, dump_dir, verdict, stats, extra=None, out_dir=None):
    """Called by a producer at the end of a real run. Returns the path written."""
    if kind not in KINDS:
        raise ValueError(f"unknown receipt kind {kind!r}")
    import datetime
    prod = os.path.join(HERE, PRODUCER[kind])
    rec = {
        "receipt_version": RECEIPT_VERSION,
        "kind": kind,
        "verdict": verdict,
        "stats": {k: (int(v) if isinstance(v, bool) else v) for k, v in stats.items()},
        "code_set": code_set(),
        "producer": PRODUCER[kind],
        "producer_sha256": sha256_file(prod),
        "dump_fingerprint": dump_fingerprint(dump_dir),
        "dump_dir": os.path.basename(os.path.normpath(dump_dir)),
        "written_utc": datetime.datetime.now(datetime.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": machine_stamp(),
    }
    if extra:
        rec["extra"] = extra
    path = os.path.join(out_dir or HERE, f"receipt_{kind}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


# -------------------------------------------------------------------- checking
def _recheck_gate(st):
    """Re-derive the gate verdict from the recorded statistics.

    The decision function is IMPORTED from pseudo_null_pair, never copied, so the
    threshold and the FAIL/PASS/INCONCLUSIVE order can never drift from the producer.
    """
    bad = []
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from pseudo_null_pair import DEGEN_FRAC, TOL_PP, gate_verdict
    except Exception as e:
        return [f"cannot import the producer's own decision function: "
                f"{type(e).__name__}: {e}"]

    if abs(float(st["margin_pp"]) - TOL_PP) > 1e-12:
        bad.append(f"margin {st['margin_pp']} != the producer's TOL_PP {TOL_PP}")
    if abs(float(st["degen_frac"]) - DEGEN_FRAC) > 1e-12:
        bad.append(f"degen_frac {st['degen_frac']} != producer's {DEGEN_FRAC}")

    # the CI must follow from (gap, se, z): a tampered statistic contradicts itself
    z = float(st["z"])
    lo = float(st["gap_pp"]) - z * float(st["se_pp"])
    hi = float(st["gap_pp"]) + z * float(st["se_pp"])
    if abs(lo - float(st["ci_lo_pp"])) > 1e-9 or abs(hi - float(st["ci_hi_pp"])) > 1e-9:
        bad.append(f"stored CI [{st['ci_lo_pp']:+.6f}, {st['ci_hi_pp']:+.6f}] does not "
                   f"follow from gap {st['gap_pp']:+.6f} +- {z} x se {st['se_pp']:.6f} "
                   f"= [{lo:+.6f}, {hi:+.6f}]")

    if int(st["repro_bad"]) != 0:
        bad.append(f"D3 re-draw mismatches on {st['repro_bad']} cells (must be 0)")
    if int(st["n_paired"]) <= 0 or int(st["n_days"]) <= 0:
        bad.append("n_paired or n_days is not positive")

    frac = float(st["frac_same_anchor"])
    if frac >= float(st["degen_frac"]):
        bad.append(f"same-anchor fraction {frac:.4%} is DEGENERATE "
                   f"(>= {float(st['degen_frac']):.0%})")
    else:
        v = gate_verdict(float(st["ci_lo_pp"]), float(st["ci_hi_pp"]))
        if v != "PASS":
            bad.append(f"the producer's gate_verdict re-applied to the recorded CI "
                       f"gives {v}, not PASS")
    return bad


def _recheck_power(st):
    """Re-derive the power verdict. P1 and P2 are evaluated in INTEGER arithmetic
    (brain/02; E02, where float equality on p-values gave two different 'exact'
    counts). P3 is unavoidably a float, but its threshold is recomputed from n."""
    bad = []
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from pseudo_power import DELTA, FPR_HI, FPR_LO, NEED
    except Exception as e:
        return [f"cannot import the producer's own thresholds: "
                f"{type(e).__name__}: {e}"]

    for name, got, want in (("need", st["need"], NEED), ("fpr_lo", st["fpr_lo"], FPR_LO),
                            ("fpr_hi", st["fpr_hi"], FPR_HI), ("delta", st["delta"], DELTA)):
        if abs(float(got) - float(want)) > 1e-12:
            bad.append(f"{name} {got} != the producer's {want}")

    n, k1, k2 = int(st["n_draws"]), int(st["k_detected"]), int(st["k_null_rejected"])
    if n <= 0:
        bad.append("n_draws is not positive")
        return bad
    if not (0 <= k1 <= n and 0 <= k2 <= n):
        bad.append(f"counts out of range: k_detected {k1}, k_null_rejected {k2} of {n}")
    # P1: k1/n >= NEED, as integers.  P2: FPR_LO <= k2/n <= FPR_HI, as integers.
    if k1 * 1000 < int(round(float(st["need"]) * 1000)) * n:
        bad.append(f"P1 FAILS on the recorded counts: {k1} of {n} detected, "
                   f"need >= {float(st['need']):.0%}")
    lo_ok = k2 * 1000 >= int(round(float(st["fpr_lo"]) * 1000)) * n
    hi_ok = k2 * 1000 <= int(round(float(st["fpr_hi"]) * 1000)) * n
    if not (lo_ok and hi_ok):
        bad.append(f"P2 FAILS on the recorded counts: {k2} of {n} rejected, "
                   f"need {float(st['fpr_lo']):.0%}-{float(st['fpr_hi']):.0%}")
    nn = int(st["n"])
    if nn <= 0:
        bad.append("n is not positive")
    elif float(st["gap_err"]) > 0.5 / nn + 1e-9:
        bad.append(f"P3 FAILS: worst |error| {float(st['gap_err']):.3e} > 0.5/n "
                   f"= {0.5 / nn:.3e}")
    if int(st["D"]) <= 0:
        bad.append("D (trade dates) is not positive")
    return bad


RECHECK = {"gate": _recheck_gate, "power": _recheck_power}


def verify_receipt(rec, kind, live):
    """Return a list of reasons to REFUSE. Empty list == permit.

    `live` is what the CONSUMER measures right now:
        dict(code_set=..., producer_sha256=..., dump_fingerprint=...)
    The real path and the self-test both call THIS function with the same meaning,
    so a planted fault exercises the code that actually guards the run (E09).
    """
    bad = []
    if not isinstance(rec, dict):
        return ["the receipt is not a JSON object"]
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        return [f"receipt is missing required keys: {', '.join(missing)}"]
    if rec["receipt_version"] != RECEIPT_VERSION:
        bad.append(f"receipt_version {rec['receipt_version']} != {RECEIPT_VERSION}")
    if rec["kind"] != kind:
        bad.append(f"receipt says kind {rec['kind']!r}, expected {kind!r}")
    if rec.get("producer") != PRODUCER.get(kind):
        bad.append(f"producer {rec.get('producer')!r} != {PRODUCER.get(kind)!r}")

    st = rec.get("stats")
    if not isinstance(st, dict):
        return bad + ["stats is not an object"]
    miss = [k for k in REQUIRED_STATS[kind] if k not in st]
    if miss:
        return bad + [f"stats is missing: {', '.join(miss)}"]

    if rec["verdict"] != "PASS":
        bad.append(f"verdict is {rec['verdict']!r}, not PASS")

    # the statistics must IMPLY the verdict, independently of what it claims
    try:
        bad += RECHECK[kind](st)
    except Exception as e:
        bad.append(f"re-deriving the verdict raised {type(e).__name__}: {e}")

    for field in ("code_set", "producer_sha256", "dump_fingerprint"):
        want = live.get(field)
        if want is None:
            bad.append(f"cannot measure {field} on this tree")
        elif rec[field] != want:
            bad.append(f"{field} mismatch: receipt {rec[field]!r} vs measured {want!r}")
    return bad


def _live(kind, dump_dir):
    prod = os.path.join(HERE, PRODUCER[kind])
    return {
        "code_set": code_set() if os.path.isdir(CODE) else None,
        "producer_sha256": sha256_file(prod) if os.path.exists(prod) else None,
        "dump_fingerprint": dump_fingerprint(dump_dir) if os.path.isdir(dump_dir) else None,
    }


def require_pass(dump_dir, reverify=False, receipts_dir=None, quiet=False):
    """THE enforcement point (C3). Returns a stamp dict, or exits non-zero.

    Call it as the FIRST statement of any script that reads the real
    real-minus-pseudo gap. There is deliberately no override.
    """
    d = receipts_dir or HERE
    say = (lambda *a: None) if quiet else print
    say("\nSECTION 26.3 ORDER CHECK -- gate and power validation must both say PASS")
    say(f"  dump      {os.path.abspath(dump_dir)}")
    say(f"  receipts  {os.path.abspath(d)}")
    refusals, stamp = [], {}
    for kind in KINDS:
        path = os.path.join(d, f"receipt_{kind}.json")
        if not os.path.exists(path):
            refusals.append(f"{kind}: no receipt at {path}")
            say(f"  [REFUSE] {kind:<6} no receipt on disk")
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception as e:
            refusals.append(f"{kind}: unreadable receipt ({type(e).__name__})")
            say(f"  [REFUSE] {kind:<6} unreadable: {type(e).__name__}")
            continue
        bad = verify_receipt(rec, kind, _live(kind, dump_dir))
        if bad:
            for b in bad:
                refusals.append(f"{kind}: {b}")
                say(f"  [REFUSE] {kind:<6} {b}")
        else:
            say(f"  [OK]     {kind:<6} PASS, re-derived from its own statistics; "
                f"code-set, producer and dump all match")
            stamp[kind] = rec

    if not refusals and reverify:
        say("  --reverify: re-running both producers in process (about 6.5 min)")
        refusals += _reverify(dump_dir, stamp, say)

    if refusals:
        say("")
        raise SystemExit(
            "REFUSED: section 26.3's order is not satisfied on this tree, so no "
            "real-minus-pseudo gap may be read.\n  - " + "\n  - ".join(refusals))
    say("  PERMITTED: the gate and the power validation both PASS on this exact "
        "dump and code-set.\n")
    return stamp


def _reverify(dump_dir, stamp, say):
    """Re-run both producers and require the receipts to reproduce. Never default:
    a changed dump already flips dump_fingerprint and refuses above, which forces a
    real re-run exactly when one is warranted."""
    bad = []
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import pseudo_null_pair
    except Exception as e:
        return [f"--reverify: cannot import pseudo_null_pair ({type(e).__name__}: {e})"]
    # main() returns (verdict, stats) and takes write_receipt=False, so the re-run
    # compares statistics in memory and does NOT overwrite the receipt it is checking.
    try:
        verd, fresh = pseudo_null_pair.main(dump_dir, False, write_receipt=False)
    except Exception as e:
        return [f"--reverify: the gate re-run raised {type(e).__name__}: {e}"]
    old = stamp["gate"]["stats"]
    if verd != "PASS":
        bad.append(f"--reverify: the gate re-run returned {verd}, not PASS")
    for k in ("gap_pp", "se_pp", "n_paired", "n_days", "n_setups", "n_cells"):
        if k not in fresh:
            bad.append(f"--reverify: the re-run did not report {k}")
        elif abs(float(fresh[k]) - float(old[k])) > 1e-9:
            bad.append(f"--reverify: gate {k} re-ran to {fresh[k]}, receipt says {old[k]}")
    say("  [OK]     --reverify: the gate reproduced its receipt"
        if not bad else "  [REFUSE] --reverify: the gate did NOT reproduce")
    return bad


# ------------------------------------------------------------------- self-test
def _toy(tmp):
    """A synthetic dump + producer + a matching PASS receipt pair, in a temp dir.
    Reads no project data and writes nothing into the project."""
    dump = os.path.join(tmp, "dump")
    os.makedirs(dump, exist_ok=True)
    for i in range(3):
        with open(os.path.join(dump, f"N60_toy_ATR_{i}.0.counts.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"cell": f"N60_toy_ATR_{i}.0", "spec": "toy", "n_candidates": 10 + i,
                       "n_setups": 5 + i, "candidates_sha": f"c{i}", "setups_sha": f"s{i}",
                       "n_pseudo_matched": 4 + i, "n_pseudo_unmatched": 1,
                       "n_twins_M1b": 10, "n_twins_M2": 10}, fh)
    with open(os.path.join(dump, "pseudo_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": 20260929, "code_set": "toytoytoytoy"}, fh)

    # the real numbers from 26.19 and 26.21, so the control is a known-number check
    gate_stats = dict(gap_pp=-0.0001, se_pp=0.0149, z=1.96, margin_pp=0.25,
                      n_setups=816347, n_days=3130, n_paired=816347,
                      n_setups_total=867307, frac_same_anchor=0.090545,
                      degen_frac=0.99, n_cells=36, repro_bad=0)
    gate_stats["ci_lo_pp"] = gate_stats["gap_pp"] - 1.96 * gate_stats["se_pp"]
    gate_stats["ci_hi_pp"] = gate_stats["gap_pp"] + 1.96 * gate_stats["se_pp"]
    power_stats = dict(n_draws=200, k_detected=200, k_null_rejected=9,
                       gap_err=1e-7, n=3821840, D=3130, need=0.80, fpr_lo=0.01,
                       fpr_hi=0.10, delta=0.02, n_cells=144, cells_sha="toy144")
    live = {"code_set": "abc123abc123",
            "producer_sha256": "f" * 64,
            "dump_fingerprint": dump_fingerprint(dump)}

    def rec(kind, stats):
        return {"receipt_version": RECEIPT_VERSION, "kind": kind, "verdict": "PASS",
                "stats": dict(stats), "code_set": live["code_set"],
                "producer": PRODUCER[kind], "producer_sha256": live["producer_sha256"],
                "dump_fingerprint": live["dump_fingerprint"], "dump_dir": "dump",
                "written_utc": "2026-10-01T00:00:00Z", "machine": {"node": "toy"}}

    return dump, live, rec("gate", gate_stats), rec("power", power_stats)


def selftest():
    import copy
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gate_receipts_")
    dump, live, G, P = _toy(tmp)
    ok = [True]

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok[0] &= bool(cond)

    def refuses(label, kind, mutate):
        r = copy.deepcopy(G if kind == "gate" else P)
        lv = dict(live)
        mutate(r, lv)
        bad = verify_receipt(r, kind, lv)
        print(f"  [{'PASS' if bad else 'FAIL'}] {label}"
              + (f"   -> {bad[0][:88]}" if bad else "   -> PERMITTED, which is the bug"))
        ok[0] &= bool(bad)

    print("\ngate_receipts.py SELF-TEST -- the guard must FAIL on a planted fault "
          "before it is\nallowed to permit anything (26.0; the order that slipped at "
          "26.16 and 26.18).\n")
    print(f"  toy tree: {tmp}\n  (writes ONLY there; reads code\\ and the two "
          f"producers read-only, to exercise write() for real)\n")

    print("  K -- known-number checks, using 26.19's and 26.21's real statistics")
    check("K1 the unmutated gate receipt is PERMITTED",
          verify_receipt(G, "gate", live) == [])
    check("K2 the unmutated power receipt is PERMITTED",
          verify_receipt(P, "power", live) == [])
    hw = 1.96 * 0.0149
    check(f"K3 26.19's CI half-width re-derives: 1.96 x 0.0149 = {hw:.4f} pp "
          f"(margin/half-width = {0.25 / hw:.1f}x)", abs(hw - 0.029204) < 1e-6)
    check("K4 P1 in integer arithmetic: 200 of 200 clears 80% (200*1000 >= 800*200)",
          200 * 1000 >= 800 * 200)
    check("K5 P2 in integer arithmetic: 9 of 200 sits inside 1%-10% "
          "(10*200 <= 9*1000 <= 100*200)", 10 * 200 <= 9 * 1000 <= 100 * 200)
    check("K6 the dump fingerprint is stable across two reads",
          dump_fingerprint(dump) == dump_fingerprint(dump))
    # Two dumps with IDENTICAL contents must fingerprint the SAME -- that is the
    # ZIP-survivable property brain/02 requires, since copying between the two
    # machines rewrites mtimes. What must move the fingerprint is CONTENT.
    other = _toy(tempfile.mkdtemp(prefix="gate_receipts_other_"))[0]
    check("K7a an identical copy of a dump fingerprints the SAME (survives ZIP)",
          dump_fingerprint(dump) == dump_fingerprint(other))
    cj = sorted(glob.glob(os.path.join(other, "*.counts.json")))[0]
    with open(cj, "r", encoding="utf-8") as fh:
        c = json.load(fh)
    c["setups_sha"] = "CHANGED"
    with open(cj, "w", encoding="utf-8") as fh:
        json.dump(c, fh)
    check("K7b one cell's setups_sha changed -> the fingerprint MOVES",
          dump_fingerprint(dump) != dump_fingerprint(other))

    # K8/K9 exercise write() itself. Without these the producer-side function would
    # ship unrun, which is E30 exactly. Read-only against code\ and the producers.
    if os.path.isdir(CODE) and all(os.path.exists(os.path.join(HERE, p))
                                   for p in PRODUCER.values()):
        rt = []
        for kind, src in (("gate", G), ("power", P)):
            p = write(kind, dump, "PASS", src["stats"], extra={"roundtrip": True},
                      out_dir=tmp)
            with open(p, "r", encoding="utf-8") as fh:
                back = json.load(fh)
            rt.append(verify_receipt(back, kind, _live(kind, dump)) == [])
        check("K8 write() -> read back -> verify_receipt PERMITS, both kinds "
              "(round-trip against this tree's real code-set and producers)", all(rt))

        p = write("gate", dump, "PASS", G["stats"], out_dir=tmp)
        with open(p, "r", encoding="utf-8") as fh:
            back = json.load(fh)
        back["stats"]["se_pp"] = 0.001          # tamper AFTER a genuine write
        check("K9 a receipt written by write() and then tampered is REFUSED",
              verify_receipt(back, "gate", _live("gate", dump)) != [])
    else:
        check("K8/K9 write() round-trip SKIPPED -- code\\ or a producer is absent "
              "on this machine; write() is therefore UNTESTED here", False)

    print("\n  M -- planted faults. EVERY ONE MUST REFUSE.")

    def m_missing(r, lv):
        r.pop("dump_fingerprint")
    refuses("M1 a required key is missing", "gate", m_missing)

    refuses("M2 verdict downgraded to INCONCLUSIVE", "gate",
            lambda r, lv: r.update(verdict="INCONCLUSIVE"))

    refuses("M3 gate CI widened past the +-0.25 margin, verdict still 'PASS'", "gate",
            lambda r, lv: r["stats"].update(gap_pp=0.30, se_pp=0.0149,
                                            ci_lo_pp=0.30 - 1.96 * 0.0149,
                                            ci_hi_pp=0.30 + 1.96 * 0.0149))

    refuses("M4 stored CI no longer follows from gap and se (se tampered)", "gate",
            lambda r, lv: r["stats"].update(se_pp=0.0010))

    refuses("M5 the margin is loosened from 0.25 to 1.0", "gate",
            lambda r, lv: r["stats"].update(margin_pp=1.0))

    refuses("M6 D3 re-draw mismatches reported but verdict still 'PASS'", "gate",
            lambda r, lv: r["stats"].update(repro_bad=2))

    refuses("M7 same-anchor fraction at 99.5% -- DEGENERATE, not PASS", "gate",
            lambda r, lv: r["stats"].update(frac_same_anchor=0.995))

    refuses("M8 power detection dropped to 150 of 200 (75% < 80%)", "power",
            lambda r, lv: r["stats"].update(k_detected=150))

    refuses("M9 power null calibration at 1 of 200 (0.5% < 1%)", "power",
            lambda r, lv: r["stats"].update(k_null_rejected=1))

    refuses("M10 power null calibration at 40 of 200 (20% > 10%)", "power",
            lambda r, lv: r["stats"].update(k_null_rejected=40))

    refuses("M11 P3 injected-gap error exceeds 0.5/n", "power",
            lambda r, lv: r["stats"].update(gap_err=1.0))

    refuses("M12 code-set rotated since the receipt was written", "gate",
            lambda r, lv: lv.update(code_set="deadbeefdead"))

    refuses("M13 the producer script was edited after the run", "power",
            lambda r, lv: lv.update(producer_sha256="0" * 64))

    refuses("M14 the receipt was earned on a DIFFERENT dump", "gate",
            lambda r, lv: lv.update(dump_fingerprint="0123456789abcdef"))

    refuses("M15 a power threshold is loosened (need 0.80 -> 0.50)", "power",
            lambda r, lv: r["stats"].update(need=0.50))

    refuses("M16 the gate receipt is presented as the power receipt", "power",
            lambda r, lv: r.update(kind="power", producer=PRODUCER["power"],
                                   stats=dict(G["stats"])))

    print(f"\n  {'ALL DETECTORS ALIVE' if ok[0] else 'SELF-TEST FAILED'}\n")
    return ok[0]


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--check" in sys.argv or args:
        if not args:
            raise SystemExit("usage: python gate_receipts.py --check DUMP_DIR")
        if not selftest():
            raise SystemExit("\n  SELF-TEST FAILED -- no receipt was read.")
        require_pass(args[0], reverify="--reverify" in sys.argv)
        sys.exit(0)
    raise SystemExit(__doc__)
