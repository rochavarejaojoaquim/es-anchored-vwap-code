"""
safeio.py -- the ONLY place that writes data files (C3, AUD-26, C117).

Every write goes to <path>.tmp and is then moved over <path> with os.replace,
which is atomic on Windows and Linux: a crash leaves either the old file or
the new one, never a half-written file that resume logic would trust.

npz caches carry the SPEC_VERSION they were built under; load_npz refuses a
cache from another version instead of silently reusing it (AUD-27, C119).
selfcheck fails if any other module calls to_parquet / savez directly.
"""
import json
import os
import numpy as np

from spec import SPEC_VERSION


def write_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:                     # a file handle, so numpy
        np.savez_compressed(fh, _spec=np.array(SPEC_VERSION), **arrays)  # adds no ".npz"
    os.replace(tmp, path)


class StaleCache(Exception):
    pass


def npz_version(path):
    with np.load(path) as z:
        return str(z["_spec"]) if "_spec" in z.files else "(none -- built before 2026-09-21)"


def load_npz(path):
    """Load a cache, refusing one built under another SPEC_VERSION."""
    v = npz_version(path)
    if v != SPEC_VERSION:
        raise StaleCache(f"{path} was built under spec '{v}', current is "
                         f"'{SPEC_VERSION}'. Rename its folder; do not reuse it.")
    z = np.load(path)
    return {k: z[k] for k in z.files if k != "_spec"}
