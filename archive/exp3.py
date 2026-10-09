import sys, time; sys.path.insert(0, "src")
import numpy as np, igraph as ig, leidenalg as la, scipy.sparse as sp, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import kneighbors_graph
from features import build, static_features, load
from icvi import silhouette

d = build(); T = d["tensor"]; ids = d["ids"]; N = len(ids); months = d["months"]
st = d["X"].iloc[:, 6:].values                                   # статич. Росстат (z)
sm = pd.DataFrame(T.reshape(len(months), -1)).rolling(3, min_periods=1).mean().values.reshape(T.shape)  # сглаживание 3 мес
Xt = []
for t in range(len(months)):
    z = (sm[t] - sm[t].mean(0)) / sm[t].std(0)                   # убираем сезонность: z по МО внутри месяца
    Xt.append(np.c_[z, st])
def graph(Z, k=15):
    A = kneighbors_graph(Z, k, mode="distance", metric="cosine"); M = A.minimum(A.T)
    n3 = kneighbors_graph(Z, 3, mode="distance", metric="cosine"); n3 = n3.maximum(n3.T)
    B = M.maximum(n3).tocsr(); B.data = 1 / (B.data + 1e-3); B.data /= B.data.mean(); return B
def to_ig(A):
    c = sp.triu(A).tocoo(); g = ig.Graph(n=A.shape[0], edges=list(zip(c.row, c.col))); g.es["w"] = c.data; g.vs["id"] = list(range(A.shape[0])); return g
G = [to_ig(graph(x)) for x in Xt]
RB = la.RBConfigurationVertexPartition
def snap(g, gam): return np.array(la.find_partition(g, RB, weights="w", resolution_parameter=gam, seed=0).membership)
lo, hi = 0.01, 3.0
for _ in range(12):                                              # γ под K≈7 на среднем месяце
    mid = np.sqrt(lo * hi); k = len(set(snap(G[12], mid))); lo, hi = (mid, hi) if k < 7 else (lo, mid)
gam = np.sqrt(lo * hi); print("γ =", round(gam, 3), "K(m12) =", len(set(snap(G[12], gam))))

def align(prev, cur):
    P, C = sorted(set(prev)), sorted(set(cur)); M = np.zeros((len(C), len(P)))
    for i, c in enumerate(C):
        for j, p in enumerate(P): M[i, j] = ((cur == c) & (prev == p)).sum()
    r, c = linear_sum_assignment(-M); mp = {C[i]: P[j] for i, j in zip(r, c)}; nxt = max(P) + 1
    return np.array([mp.get(x, nxt + 1000 + x) for x in cur])
def churn(L): return np.mean([(L[t] != L[t + 1]).mean() for t in range(len(L) - 1)])
def report(name, L):
    sw = np.mean([silhouette(Xt[t], L[t]) for t in range(0, len(L), 4)])
    q = np.mean([G[t].modularity(L[t], weights="w") for t in range(0, len(L), 4)])
    print(f"{name:14}  K/мес {np.mean([len(set(l)) for l in L]):5.1f}  churn {churn(L):.3f}  SW {sw:.3f}  Q {q:.3f}", flush=True)

L0 = [snap(G[0], gam)]
for t in range(1, len(G)): L0.append(align(L0[-1], snap(G[t], gam)))
report("независимые", L0)
for om in (0.05, 0.2, 0.5, 1.0):
    t0 = time.time()
    mem, _ = la.find_partition_temporal(G, RB, interslice_weight=om, resolution_parameter=gam, weights="w", seed=0)
    L = [np.array(m) for m in mem]; report(f"мультислой ω={om}", L)
    if om == 0.2: np.save("data/processed/labels_om0.2.npy", np.array(L))
