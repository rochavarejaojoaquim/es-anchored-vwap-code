"""
logs_fingerprint.py -- the ONE method for fingerprinting a Stage-2 sealed-logs folder (Stage 2 reading
amendment S2-9). Reads file BYTES to hash them; prints only a file count and one hash; opens nothing for
display. Writes nothing.

    python logs_fingerprint.py                          fingerprint logs_stage2_sealed\\ next to this file
    python logs_fingerprint.py FOLDER                   another folder
    python logs_fingerprint.py FOLDER --expect HEX      also print PASS/FAIL against a recorded value
    python logs_fingerprint.py --selftest               known-number check + mutations, touches only a temp dir

THE METHOD (exact):
  files   = sorted(glob.glob(FOLDER + '/*')), Python's default string order (code-point, case-sensitive).
            A subfolder is refused (the method has no rule for one).
  h_f     = lowercase hex SHA-256 of the file's bytes.
  folder  = SHA-256 of the UTF-8 concatenation, over the files in that order, of
            basename + ':' + h_f + LF   (LF = one byte, \\n)
  printed = len(files), then the folder hash as UPPER-CASE hex.
It is the same computation as the one-line command that produced the 2026-10-09 values
(PC 204 / EAF9612E...A045, laptop 115 / 5D5EEEE9...5AF6).
"""
import glob
import hashlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "logs_stage2_sealed")


def fingerprint(folder):
    files = sorted(glob.glob(folder + "/*"))
    for f in files:
        if os.path.isdir(f):
            raise SystemExit(f"REFUSED: {f} is a subfolder; the method has no rule for one")
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            hf = hashlib.sha256(fh.read()).hexdigest()
        h.update((os.path.basename(f) + ":" + hf + "\n").encode("utf-8"))
    return len(files), h.hexdigest().upper()


def selftest():
    ok = True

    def chk(name, cond, det=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {det}")

    with tempfile.TemporaryDirectory() as d:
        names = {"N60_a.log": b"alpha", "N100_b.log": b"beta", "Preflight.log": b"gamma", "alpha.log": b"delta"}
        for n, b in names.items():
            with open(os.path.join(d, n), "wb") as fh:
                fh.write(b)
        cnt, hx = fingerprint(d)
        # independent re-computation, written differently (os.listdir, explicit order)
        order = sorted(os.listdir(d))
        lines = b"".join(
            n.encode("utf-8") + b":" + hashlib.sha256(names[n]).hexdigest().encode("ascii") + b"\n" for n in order)
        want = hashlib.sha256(lines).hexdigest().upper()
        chk("K1 the folder hash equals an independent recomputation", cnt == 4 and hx == want, f"{cnt} files {hx[:16]}")
        chk("K2 the order is code-point and case-sensitive (N100 < N60 < Preflight < alpha)",
            order == ["N100_b.log", "N60_a.log", "Preflight.log", "alpha.log"], str(order))
        # mutations: each must CHANGE the hash
        def h_of(ordered, lower_hex=True, sep=b"\n"):
            parts = []
            for n in ordered:
                x = hashlib.sha256(names[n]).hexdigest()
                x = x if lower_hex else x.upper()
                parts.append(n.encode() + b":" + x.encode() + sep)
            return hashlib.sha256(b"".join(parts)).hexdigest().upper()
        chk("M1 a case-insensitive sort gives a different hash", h_of(sorted(names, key=str.lower)) != want)
        chk("M2 a natural (numeric) sort gives a different hash",
            h_of(sorted(names, key=lambda s: (s.lower().replace("n100", "n0100").replace("n60", "n0060")))) != want)
        chk("M3 upper-case per-file hex gives a different hash", h_of(order, lower_hex=False) != want)
        chk("M4 CRLF instead of LF gives a different hash", h_of(order, sep=b"\r\n") != want)
        with open(os.path.join(d, "N60_a.log"), "ab") as fh:
            fh.write(b"x")
        chk("M5 one appended byte changes the folder hash", fingerprint(d)[1] != want)
        os.mkdir(os.path.join(d, "sub"))
        try:
            fingerprint(d)
            chk("M6 a subfolder is refused", False)
        except SystemExit:
            chk("M6 a subfolder is refused", True)
    print("  ALL DETECTORS ALIVE" if ok else "  SELF-TEST FAILED")
    return ok


def main(argv):
    if "--selftest" in argv:
        sys.exit(0 if selftest() else 1)
    expect = None
    if "--expect" in argv:
        i = argv.index("--expect")
        expect = argv[i + 1].upper()
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    folder = args[0] if args else DEFAULT
    cnt, hx = fingerprint(folder)
    print(cnt, hx)
    if expect is not None:
        print(f"  [{'PASS' if hx == expect else 'FAIL'}] equals the recorded value {expect[:16]}...")
        sys.exit(0 if hx == expect else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
