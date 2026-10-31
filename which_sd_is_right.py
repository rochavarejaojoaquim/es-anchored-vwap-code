"""which_sd_is_right.py -- one-off: where production and the slow reference
disagree about sd > 0, which one does EXACT integer arithmetic side with?
Run from <project folder>:  python which_sd_is_right.py   (~1 minute)
Reads only data\\outright.parquet; writes nothing."""
import sys; sys.path.insert(0, "code")
import numpy as np
from fractions import Fraction
from loader import load, rolling_argextreme
from pipeline import vwap_and_sd, tick_ints
from sd_precision import exact_sd
d = load("train")
a = rolling_argextreme(d.high.to_numpy().astype(float), 60, d.seg.to_numpy(), True)
vw, sd_fast, *_ = vwap_and_sd(d, a)
sd_slow = exact_sd(d, a)
q, v, _ = tick_ints(d)
ok = np.isfinite(sd_fast) & (a >= 0) & (a != np.arange(len(d)))
bad = np.where(ok & ((sd_fast > 0) != (sd_slow > 0)))[0]
print("disagreeing bars:", len(bad))
prod_right = ref_right = 0
for i in bad[:300]:
    j = int(a[i]); Q = [int(x) for x in q[j:i+1]]; V = [int(x) for x in v[j:i+1]]
    W = sum(V); M1 = Fraction(sum(x*w for x, w in zip(Q, V)), W)
    truth = (Fraction(sum(x*x*w for x, w in zip(Q, V)), W) - M1*M1) > 0
    prod_right += (sd_fast[i] > 0) == truth
    ref_right += (sd_slow[i] > 0) == truth
print(f"of {min(300, len(bad))} checked with exact integers: production right {prod_right}, slow reference right {ref_right}")
