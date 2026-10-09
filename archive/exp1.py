import sys; sys.path.insert(0, "src")
import numpy as np, igraph as ig, leidenalg as la
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score as SW, calinski_harabasz_score as CH, normalized_mutual_info_score as NMI
from features import build

d = build(); X = d["X"].values; reg = d["meta"].region_name.fillna("?").values

def graph(X, k=15):
    A = kneighbors_graph(X, k, mode="distance", metric="cosine")
    A = A.minimum(A.T)                      # взаимный kNN
    A.data = 1 / (A.data + 1e-3)
    A = A.tocoo()
    g = ig.Graph(n=X.shape[0], edges=list(zip(A.row, A.col)), directed=False)
    g.es["w"] = A.data
    return g.simplify(combine_edges="max") if False else g

def leiden(g, gamma):
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="w", resolution_parameter=gamma, seed=0)
    return np.array(p.membership), p.modularity

g = graph(X); g = g.simplify(combine_edges="max") if False else g
print("граф: рёбер", g.ecount(), "изолированных", sum(1 for x in g.degree() if x == 0))
res = []
for k in (5, 7, 9):
    for name, lab in [("kmeans", KMeans(k, n_init=10, random_state=0).fit_predict(X)),
                      ("ward", AgglomerativeClustering(k).fit_predict(X)),
                      ("gmm", GaussianMixture(k, random_state=0).fit_predict(X))]:
        res.append((name, k, len(set(lab)), SW(X, lab), CH(X, lab), g.modularity(lab, weights="w"), NMI(reg, lab)))
for gam in (0.3, 0.6, 1.0):
    lab, q = leiden(g, gam)
    res.append(("leiden", gam, len(set(lab)), SW(X, lab) if len(set(lab)) > 1 else np.nan, CH(X, lab), g.modularity(lab, weights="w"), NMI(reg, lab)))
print(f"{'метод':8}{'k/γ':>5}{'#кл':>5}{'SW':>7}{'CH':>8}{'MQ':>7}{'NMI_регион':>11}")
for r in res: print(f"{r[0]:8}{r[1]:>5}{r[2]:>5}{r[3]:7.3f}{r[4]:8.0f}{r[5]:7.3f}{r[6]:11.3f}")
