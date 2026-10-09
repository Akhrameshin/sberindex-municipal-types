"""Значимость внутренних индексов против трёх нулевых моделей (заголовочное разбиение, 3 месяца, число нулей — significance.n_null):
  1) labels  — случайная перестановка меток (размеры кластеров те же);
  2) region  — метки переставлены ВНУТРИ регионов (сохраняется региональный состав типов): структура сильнее «просто регионов»?
  3) config  — конфигурационная модель: рёбра графа переставлены с сохранением степеней (только графовые индексы AVI, AVU, ANUI, MQ, Q).
Знак z: «больше — лучше» для всех (для S_Dbw и AVU знак инвертирован). p — односторонний эмпирический, пол = 1/(n_null+1).
Признаковые индексы (SW, CH, S_Dbw) под нулём config не меняются по построению — там стоит NaN, это не пропуск расчёта."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, igraph as ig, scipy.sparse as sp
from scipy.spatial.distance import cdist
from config import CFG, SEED
from features import build
from pipeline import *
from icvi import all_indices, graph_indices, DIRECTION
import random as _random
# У igraph свой внутренний генератор: rewire() не подчиняется numpy. Без этого засева
# конфигурационная нулевая модель не воспроизводится между запусками.
ig.set_random_number_generator(_random.Random(SEED))
rng = np.random.default_rng(SEED)
d = build(); Xt, months = month_features(d); A = [graph(x) for x in Xt]
from headline import labels
KK = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["headline"]["Ks"][-1]; L = labels(KK)
reg = pd.Series(d["meta"].region_name.fillna("?").values); groups = [np.array(ix) for ix in reg.groupby(reg).groups.values()]
N_NULL = CFG["significance"]["n_null"]; GR = ("AVI", "AVU", "ANUI", "MQ", "Q")

def within_region(lab):
    out = lab.copy()
    for ix in groups: out[ix] = rng.permutation(lab[ix])
    return out

def rewire(a):
    c = sp.triu(a).tocoo(); g = ig.Graph(n=a.shape[0], edges=list(zip(c.row, c.col))); w = rng.permutation(c.data)
    g.rewire(n=10 * g.ecount()); e = np.array(g.get_edgelist())
    m = sp.coo_matrix((w[:len(e)], (e[:, 0], e[:, 1])), shape=a.shape); return (m + m.T).tocsr()

rows = []
for t in (0, len(Xt) // 2, len(Xt) - 1):
    D = cdist(Xt[t], Xt[t])              # матрица расстояний считается один раз на месяц, а не на каждый нуль
    obs = all_indices(Xt[t], A[t], L[t], D); null = {"labels": {k: [] for k in obs}, "region": {k: [] for k in obs}, "config": {k: [] for k in GR}}
    for _ in range(N_NULL):
        for nm, lab in (("labels", rng.permutation(L[t])), ("region", within_region(L[t]))):
            r = all_indices(Xt[t], A[t], lab, D)
            for k in r: null[nm][k].append(r[k])
        r = graph_indices(rewire(A[t]), L[t])
        for k in GR: null["config"][k].append(r[k])
    for nm, dct in null.items():
        for k, v in dct.items():
            v = np.array(v); s = DIRECTION[k]; z = s * (obs[k] - v.mean()) / (v.std() + 1e-12); p = (1 + np.sum(s * v >= s * obs[k])) / (len(v) + 1)
            rows.append(dict(месяц=months[t], нулевая=nm, индекс=k, значение=obs[k], z=z, p=p))
r = pd.DataFrame(rows); r.to_csv(f"data/processed/significance_K{KK}.csv", index=False)
piv = r.groupby(["индекс", "нулевая"]).agg(значение=("значение", "mean"), z=("z", "mean"), p_max=("p", "max")).round(3).reset_index()
pd.set_option("display.width", 200); print(piv.pivot(index="индекс", columns="нулевая", values=["z", "p_max"]).to_string())
