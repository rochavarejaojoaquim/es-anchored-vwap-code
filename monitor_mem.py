r"""
monitor_mem.py -- standalone memory backstop for a long multi-process run.

It is its OWN process.  It never starts work and never stops work for good: it
watches physical memory every INTERVAL seconds, logs it, and

    SUSPENDS every listed worker process when available memory < --suspend-below-gib
    RESUMES  them              when available memory >= --resume-above-gib

(two different numbers on purpose, so it cannot flap around one threshold).
On ANY exit -- stop file, Ctrl+C, end of the run, a crash -- it resumes everything
it suspended, so it can never leave a worker frozen.

    python monitor_mem.py --pids-file FILE --log LOG [--stop-file F]
                          [--suspend-below-gib 2.0] [--resume-above-gib 3.0]
                          [--interval 5] [--watch-pid PID] [--pids N N ...]

  --pids-file   JSON {"pids": [...], "done": false}, rewritten by whoever starts
                the workers.  Re-read every cycle, so workers may come and go.
  --stop-file   the monitor exits when this file exists
  --watch-pid   also exit once this process is gone AND no listed worker is alive
  --fake-free-file / --max-seconds   TEST HOOKS ONLY: read "available GiB" from a
                file instead of the system / exit after N seconds.

HONEST LIMIT: suspending a worker stops it growing but does NOT give its memory
back.  If the workers themselves hold the memory, "recovery" needs something else
to release RAM (close an app).  The log says so every cycle while suspended.
"""
import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes

GIB = float(2 ** 30)
STILL_ACTIVE = 259
P_SUSPEND_RESUME = 0x0800
P_QUERY_LIMITED = 0x1000
P_QUERY = 0x0400
P_VM_READ = 0x0010

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_nt = ctypes.WinDLL("ntdll")
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_nt.NtSuspendProcess.argtypes = [wintypes.HANDLE]
_nt.NtResumeProcess.argtypes = [wintypes.HANDLE]


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]


def avail_gib():
    """Physical memory a process could be given right now (incl. the standby cache),
    the same quantity runner.py's free_gb() reads."""
    m = _MEMSTATUS()
    m.dwLength = ctypes.sizeof(m)
    _k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / GIB


def total_gib():
    m = _MEMSTATUS()
    m.dwLength = ctypes.sizeof(m)
    _k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / GIB


def alive(pid):
    h = _k32.OpenProcess(P_QUERY_LIMITED, False, int(pid))
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        return bool(_k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        _k32.CloseHandle(h)


def working_set_gib(pid):
    h = _k32.OpenProcess(P_QUERY_LIMITED | P_QUERY | P_VM_READ, False, int(pid))
    if not h:
        return 0.0
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / GIB
        return 0.0
    finally:
        _k32.CloseHandle(h)


def _signal(pid, fn):
    h = _k32.OpenProcess(P_SUSPEND_RESUME, False, int(pid))
    if not h:
        return False
    try:
        return fn(h) == 0
    finally:
        _k32.CloseHandle(h)


def suspend(pid):
    return _signal(pid, _nt.NtSuspendProcess)


def resume(pid):
    return _signal(pid, _nt.NtResumeProcess)


def read_pids(path, extra):
    pids, done = list(extra), False
    if path and os.path.exists(path):
        try:
            with open(path) as fh:
                j = json.load(fh)
            pids += [int(p) for p in j.get("pids", [])]
            done = bool(j.get("done", False))
        except (OSError, ValueError):
            pass                                   # mid-rewrite: use the next cycle
    return sorted(set(pids)), done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids-file")
    ap.add_argument("--pids", type=int, nargs="*", default=[])
    ap.add_argument("--log")
    ap.add_argument("--stop-file")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--suspend-below-gib", type=float, default=2.0)
    ap.add_argument("--resume-above-gib", type=float, default=3.0)
    ap.add_argument("--watch-pid", type=int)
    ap.add_argument("--fake-free-file")
    ap.add_argument("--max-seconds", type=float)
    a = ap.parse_args()
    if a.resume_above_gib < a.suspend_below_gib:
        raise SystemExit("--resume-above-gib must be >= --suspend-below-gib")
    if not (a.pids_file or a.pids):
        raise SystemExit("give --pids-file or --pids")

    logfh = open(a.log, "a", buffering=1) if a.log else None

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")

    def free_now():
        if a.fake_free_file:
            try:
                with open(a.fake_free_file) as fh:
                    return float(fh.read().strip()), True
            except (OSError, ValueError):
                pass
        return avail_gib(), False

    log(f"monitor start | pid {os.getpid()} | total RAM {total_gib():.1f} GiB | suspend < "
        f"{a.suspend_below_gib} GiB, resume >= {a.resume_above_gib} GiB | every {a.interval}s"
        + (" | FAKE FREE-MEMORY SOURCE (test)" if a.fake_free_file else ""))
    t0 = time.time()
    suspended, state, since = set(), "RUN", None
    lowest, peak_workers = 1e9, 0.0
    try:
        while True:
            if a.stop_file and os.path.exists(a.stop_file):
                log("stop file found")
                break
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                log("max-seconds reached")
                break
            free, fake = free_now()
            lowest = min(lowest, free)
            pids, done = read_pids(a.pids_file, a.pids)
            pids = [p for p in pids if alive(p)]
            suspended &= set(pids)
            ws = {p: working_set_gib(p) for p in pids}
            tot = sum(ws.values())
            peak_workers = max(peak_workers, tot)
            if state == "RUN" and free < a.suspend_below_gib and pids:
                ok = [p for p in pids if suspend(p)]
                suspended |= set(ok)
                state, since = "SUSPENDED", time.time()
                log(f"*** SUSPENDED {len(ok)} of {len(pids)} workers: available {free:.2f} GiB < "
                    f"{a.suspend_below_gib} GiB ***")
            elif state == "SUSPENDED" and free >= a.resume_above_gib:
                for p in list(suspended):
                    resume(p)
                suspended.clear()
                log(f"*** RESUMED: available {free:.2f} GiB >= {a.resume_above_gib} GiB "
                    f"(was suspended {time.time() - since:.0f}s) ***")
                state, since = "RUN", None
            elif state == "SUSPENDED":
                for p in pids:                     # a worker that started while suspended
                    if p not in suspended and suspend(p):
                        suspended.add(p)
            note = ""
            if state == "SUSPENDED":
                note = (f" | SUSPENDED {time.time() - since:.0f}s -- workers still hold "
                        f"{tot:.2f} GiB; need {a.resume_above_gib} GiB available to resume")
            log(f"avail {free:5.2f} GiB | workers {len(pids)} hold {tot:5.2f} GiB | {state}{note}"
                + (" | " + " ".join(f"{p}:{w:.2f}" for p, w in ws.items()) if ws else ""))
            if done and not pids:
                log("run marked done and no workers alive")
                break
            if a.watch_pid and not alive(a.watch_pid) and not pids:
                log("watched launcher is gone and no workers alive")
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        log("Ctrl+C")
    finally:
        for p in list(suspended):
            resume(p)
        if suspended:
            log(f"resumed {len(suspended)} suspended worker(s) on exit")
        log(f"monitor exit | lowest available seen {lowest:.2f} GiB | peak workers total {peak_workers:.2f} GiB")
        if logfh:
            logfh.close()


if __name__ == "__main__":
    main()
