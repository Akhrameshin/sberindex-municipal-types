import sys; sys.path.insert(0, "src")
import numpy as np
from features import build
from pipeline import *
from methods2 import pooled_kmeans_viterbi
from icvi import silhouette
d = build(); Xt, _ = month_features(d, trim=2)
for lam in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3):
    p, _ = pooled_kmeans_viterbi(Xt, 6, lam)
    ch = np.mean([(p[t] != p[t+1]).mean() for t in range(len(p)-1)])
    print(f"λ={lam:4}  churn {ch:.3f}  SW {np.mean([silhouette(Xt[t], p[t]) for t in (0, 10, 21)]):.3f}  K_first {len(set(p[0]))} K_last {len(set(p[-1]))}  смен за период {np.mean(p[0]!=p[-1]):.3f}")
