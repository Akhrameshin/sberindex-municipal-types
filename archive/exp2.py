import sys; sys.path.insert(0, "src")
import numpy as np, igraph as ig, leidenalg as la, scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from features import build
from icvi import all_indices

d = build(); X = d["X"].values; reg = d["meta"].region_name.fillna("?").values
T = d["tensor"]                                            # T×N×F
dyn = (T - T.mean(0)); dyn = dyn.transpose(1, 0, 2).reshape(T.shape[1], -1)   # динамика, центр. по МО
geo = d["meta"][["municipal_district_center_lat", "municipal_district_center_lon"]].fillna(d["meta"].median(numeric_only=True)).values
geo = np.c_[geo[:, 0], geo[:, 1] * np.cos(np.radians(geo[:, 0]))] + np.random.default_rng(0).normal(0, 1e-4, (len(geo), 2))
print('МО без координат:', int(d['meta'].municipal_district_center_lat.isna().sum()))
K = 15
def mknn(Z, metric, w="inv"):
    A = kneighbors_graph(Z, K, mode="distance", metric=metric)
    M = A.minimum(A.T)                                   # взаимный kNN
    near = kneighbors_graph(Z, 3, mode="distance", metric=metric); near = near.maximum(near.T)   # страховка от изолятов
    B = M.maximum(near).tocsr(); B.data = 1 / (B.data + 1e-3); return B
rules = {"cosine(признаки)": mknn(X, "cosine"), "евклид(признаки)": mknn(X, "euclidean"),
         "корр.динамики": mknn(dyn, "cosine"), "география(кНН)": mknn(geo, "euclidean")}

def to_ig(A):
    c = sp.triu(A).tocoo(); g = ig.Graph(n=A.shape[0], edges=list(zip(c.row, c.col))); g.es["w"] = c.data; return g

def part(g, target=7, seed=0):
    lo, hi, best = 0.002, 3.0, None
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        lab = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights="w", resolution_parameter=mid, seed=seed).membership)
        k = len(set(lab)); best = lab if best is None or abs(k - target) < abs(len(set(best)) - target) else best
        lo, hi = (mid, hi) if k < target else (lo, mid)
    return best

def drop(A, p=0.1, seed=1):
    r = np.random.default_rng(seed); c = sp.triu(A).tocoo(); m = r.random(c.nnz) > p
    B = sp.coo_matrix((c.data[m], (c.row[m], c.col[m])), shape=A.shape); return (B + B.T).tocsr()

D = None
print(f"{'правило':18}{'K':>3}{'SW':>7}{'CH':>6}{'S_Dbw':>7}{'AVI':>7}{'MQ':>6}{'ANUI':>7}{'NMI_рег':>8}{'ARI_уст':>8}{'изол':>5}")
for name, A in rules.items():
    g = to_ig(A); lab = part(g)
    r = all_indices(X, A, lab)
    stab = np.mean([ARI(lab, part(to_ig(drop(A, 0.1, s)), len(set(lab)))) for s in (1, 2, 3)])
    iso = int((np.asarray(A.sum(1)).ravel() == 0).sum())
    print(f"{name:18}{len(set(lab)):>3}{r['SW']:7.3f}{r['CH']:6.0f}{r['S_Dbw']:7.2f}{r['AVI']:7.3f}{r['MQ']:6.2f}{r['ANUI']:7.3f}{NMI(reg, lab):8.3f}{stab:8.3f}{iso:5}")
