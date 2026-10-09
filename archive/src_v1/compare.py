import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from features import build
from pipeline import *
from icvi import all_indices
from evalkit import row, align, churn
from config import CFG
K = CFG["compare"]["k"]
d = build(); Xt, months = month_features(d, trim=2); reg = d["meta"].region_name.fillna("?").values
A = [graph(x) for x in Xt]; G = [to_ig(a) for a in A]; mid = len(G) // 2
gam = calibrate(G[mid], K)
def spectral(x, a): return SpectralClustering(K, affinity="precomputed", random_state=0, assign_labels="cluster_qr").fit_predict(a)
M = {"k-means": lambda x, a, g: KMeans(K, n_init=5, random_state=0).fit_predict(x),
     "Ward": lambda x, a, g: AgglomerativeClustering(K).fit_predict(x),
     "GMM": lambda x, a, g: GaussianMixture(K, random_state=0, covariance_type="diag").fit_predict(x),
     "спектральный": lambda x, a, g: spectral(x, a),
     "Leiden (срезы)": lambda x, a, g: snap(g, gam)}
def noisy(seed, s=0.15):
    r = np.random.default_rng(seed); xs = [x + r.normal(0, s, x.shape) for x in Xt]
    As = [graph(x) for x in xs]; return xs, As, [to_ig(a) for a in As]
noise = [noisy(s) for s in (1, 2, 3)]
rows = []
for name, f in M.items():
    t0 = time.time(); L = [f(Xt[t], A[t], G[t]) for t in range(len(G))]
    L = [L[0]] + [None] * (len(L) - 1)
    for t in range(1, len(G)): L[t] = align(L[t - 1], f(Xt[t], A[t], G[t]))
    stab = np.mean([ARI(L[-1], f(xs[-1], As[-1], Gs[-1])) for xs, As, Gs in noise])
    rows.append(row(name, L, stab, Xt, A, reg))
    print(name, f"{time.time()-t0:.0f}s", flush=True)
t0 = time.time(); L = temporal(G, CFG["temporal"]["omega"], gam)
stab = np.mean([ARI(L[-1], temporal(Gs, CFG["temporal"]["omega"], gam, seed=i)[-1]) for i, (xs, As, Gs) in enumerate(noise)])
rows.append(row(f"Leiden мультислой ω={CFG['temporal']['omega']}", list(L), stab, Xt, A, reg))
pd.DataFrame(rows).to_csv("data/processed/compare.csv", index=False)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
