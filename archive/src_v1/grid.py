import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from features import build
from pipeline import *
from icvi import all_indices, perm_z

d = build(); Xt, months = month_features(d, trim=2)
A = [graph(x) for x in Xt]; G = [to_ig(a) for a in A]
reg = d["meta"].region_name.fillna("?").values
mid = len(G) // 2

def drop(a, p, seed):
    r = np.random.default_rng(seed); c = sp.triu(a).tocoo(); m = r.random(c.nnz) > p
    b = sp.coo_matrix((c.data[m], (c.row[m], c.col[m])), shape=a.shape); return (b + b.T).tocsr()

def churn(L): return np.mean([(L[t] != L[t + 1]).mean() for t in range(len(L) - 1)])

def cal(g, target):
    lo, hi = 0.01, 4.0
    for _ in range(12):
        m = np.sqrt(lo * hi); k = len(set(snap(g, m))); lo, hi = (m, hi) if k < target else (lo, m)
    return np.sqrt(lo * hi)

rows = []
for target in (5, 7, 9):
    gam = cal(G[mid], target)
    for om in (0.05, 0.2, 0.5):
        t0 = time.time(); L = temporal(G, om, gam)
        zs = [perm_z(Xt[t][:, :], A[t], L[t], n_perm=20) for t in (0, mid, len(G) - 1)]
        z = {k: np.mean([x[k] for x in zs]) for k in zs[0]}
        raw = all_indices(Xt[-1], A[-1], L[-1])
        Gd = [to_ig(drop(a, 0.1, s)) for s in (1, 2) for a in A]
        stab = np.mean([ARI(L[-1], temporal(Gd[i * len(G):(i + 1) * len(G)], om, gam, seed=i + 1)[-1]) for i in range(2)])
        rows.append(dict(target=target, gamma=round(gam, 3), omega=om, K_first=len(set(L[0])), K_last=len(set(L[-1])), churn=churn(L),
                         NMI_region=NMI(reg, L[-1]), stab_ARI=stab, **{k: raw[k] for k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ")},
                         **{"z_" + k: v for k, v in z.items() if k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI")}))
        np.save(f"data/processed/L_k{target}_om{om}.npy", L)
        pd.DataFrame(rows).to_csv("data/processed/grid.csv", index=False)
        print(target, om, f"{time.time()-t0:.0f}s", flush=True)
