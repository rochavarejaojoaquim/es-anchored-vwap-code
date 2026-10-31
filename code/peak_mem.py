"""
peak_mem.py -- measure what core_v2 ACTUALLY costs in RAM, per cell.

VWAP_CHUNK does not govern core_v2 (it never calls walk()). Its footprint is
one cell parquet in memory plus the search arrays -- never measured until now.

    python peak_mem.py            biggest cell
    python peak_mem.py <cell>     a named cell

Works on Windows AND Linux. The platform-specific import happens INSIDE
peak_mb(), so importing this module never fails (selfcheck imports every file).

D3 SANITY CHECK, run first in a separate process so it cannot inflate the
cell's reading: allocate a KNOWN 400 MiB array and confirm the meter sees it.
PASS window 360-460 MiB (allocator overhead and page rounding add a little;
a meter reading in the wrong UNITS would be off by 1024x and fail loudly).
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(os.path.dirname(HERE), "data", "dump")
KNOWN_MIB = 400


def peak_mb():
    """Peak RAM of THIS process so far, in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return pmc.PeakWorkingSetSize / 2**20
    import resource                                   # Linux / macOS only
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20


def selftest():
    import numpy as np
    base = peak_mb()
    a = np.ones(KNOWN_MIB * 2**20 // 8)               # 400 MiB of float64
    a.sum()                                           # touch every page
    delta = peak_mb() - base
    ok = 360 <= delta <= 460
    print(f"  [{'PASS' if ok else 'FAIL'}]  meter sanity check: allocated "
          f"{KNOWN_MIB} MiB, meter saw +{delta:.0f} MiB (window 360-460)")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)

    # 1. D3 check in a CHILD process, so its 400 MiB never touches our reading
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest"]).returncode
    if rc != 0:
        print("  meter FAILED its known-number check -- do not trust the numbers below")
        sys.exit(1)

    # 2. the real measurement
    import numpy as np
    import pandas as pd
    sys.path.insert(0, HERE)
    if len(sys.argv) > 1:
        cell = sys.argv[1]
    else:
        sys.path.insert(0, HERE)
        from loader import cell_files
        fs = cell_files("N*", DUMP)
        if not fs:
            print(f"!! no cells found in {DUMP}"); sys.exit(2)
        cell = os.path.basename(max(fs, key=os.path.getsize))[:-8]
    size = os.path.getsize(os.path.join(DUMP, cell + ".parquet")) / 2**20
    base = peak_mb()
    from core_v2 import load_cell, prep, search
    from within_core import META, POST
    print(f"cell: {cell}   ({size:.0f} MiB on disk)")
    print(f"  baseline (python + libraries) : {base:7.0f} MiB")
    r, c, T = load_cell(cell, None)
    print(f"  after load_cell               : {peak_mb():7.0f} MiB   "
          f"({len(r):,} setups, {len(c):,} twins)")
    cols = [x for x in r.columns
            if x not in META and not x.startswith(("t_", "w_", "tlive_", "wlive_"))
            and not any(x.startswith(p) or x == p for p in POST)]
    P = prep(r, c)
    search(r, c, P, cols, nboot=20)
    pk = peak_mb()
    print(f"  PEAK during search            : {pk:7.0f} MiB   "
          f"({len(cols)} covariates searched)")
    print()
    print(f"  processes that fit = (free RAM in Task Manager, in MiB) / {pk:.0f}")
