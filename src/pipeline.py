"""Динамический пайплайн: месячные графы (косинус, взаимный kNN + 3 ближайших) → мультислойный Leiden."""
import numpy as np, igraph as ig, leidenalg as la, scipy.sparse as sp, pandas as pd
from sklearn.neighbors import kneighbors_graph
from features import build, static_block
from config import CFG

RB = la.RBConfigurationVertexPartition


def month_features(d, trim=CFG['features']['trim_months'], win=CFG['features']['smooth_window'], lens="combined"):
    """Месячные матрицы признаков: расходы (log_total + CLR), сглаженные и нормированные внутри месяца (убирает сезонность),
    плюс статический блок по режиму lens (spend | behavioral | combined)."""
    T = d["tensor"]; st = static_block(d, lens)
    sm = pd.DataFrame(T.reshape(len(T), -1)).rolling(win, min_periods=win).mean().values.reshape(T.shape)
    Xt = []
    for t in range(trim, len(T)):                      # первые win-1 месяцев — неполное окно, отбрасываем
        z = np.clip((sm[t] - sm[t].mean(0)) / sm[t].std(0), -CFG['features']['clip_z'], CFG['features']['clip_z'])
        Xt.append(np.c_[z, st] if st.shape[1] else z)
    return Xt, d["months"][trim:]


def graph(Z, k=CFG['graph']['k']):
    from graph_support import weighted_knn
    return weighted_knn(Z, "cosine", k=k, minimum=CFG['graph']['k_min_neighbors'])


def to_ig(A):
    c = sp.triu(A).tocoo(); g = ig.Graph(n=A.shape[0], edges=list(zip(c.row, c.col)))
    g.es["w"] = c.data; g.vs["id"] = list(range(A.shape[0])); return g


def snap(g, gam, seed=0):
    return np.array(la.find_partition(g, RB, weights="w", resolution_parameter=gam, seed=seed).membership)


def calibrate(g, target=7):
    lo, hi = 0.01, 3.0
    for _ in range(12):
        mid = np.sqrt(lo * hi); k = len(set(snap(g, mid))); lo, hi = (mid, hi) if k < target else (lo, mid)
    return np.sqrt(lo * hi)


def temporal(G, omega, gam, seed=0):
    mem, _ = la.find_partition_temporal(G, RB, interslice_weight=omega, resolution_parameter=gam, weights="w", seed=seed)
    return np.array(mem)
