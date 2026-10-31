r"""
todo.py -- which core searches are done, and what to run next.

The files in cores_v2\ ARE the record: the search skips any cell+side whose
result file already exists, so this is exactly what is left to do. Run it from
<project folder>:   python todo.py
Reads nothing else; writes nothing.
"""
import glob
import os

NS = [60, 100, 160, 200, 240]
AN = ["hh", "ll", "hv", "session"]
EPS = [("ATR", e) for e in (3.0, 4.0, 6.0, 8.0, 12.0)] + [("SD", e) for e in (1.0, 1.5, 2.0, 3.0)]

have = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join("cores_v2", "*.parquet"))}
print(f"  {'shard':<17} {'done':>7}   command to finish it")
left = 0
for N in NS:
    for a in AN:
        want = [f"N{N}_{a}_{t}_{e}_side{s}" for t, e in EPS for s in ("None", "1.0", "-1.0")]
        d = sum(w in have for w in want)
        left += 27 - d
        print(f"  N{N}_{a:<12} {d:>4} / 27   " +
              ("-- finished" if d == 27 else f"python code\\core_v2.py N{N}_{a}"))
print(f"\n  {len(have)} / 540 done, {left} left")
print("  Pick any shard that is NOT finished and that no other window is running.")
