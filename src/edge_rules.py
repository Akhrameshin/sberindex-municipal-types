"""Сравнение правил ребра: чем именно соединять муниципалитеты и важен ли этот выбор.

Правило ребра — это содержательное решение, а не деталь реализации: оно задаёт, какое отношение между МО
мы считаем «похожестью». Сравниваются восемь правил из пяти семейств плюс один вариант разрежения:

  cos_feat      косинус месячного профиля признаков (правило из конфига)      — похожесть структуры и уровня трат
  euclid_feat   евклид того же профиля                                        — то же семейство, другая метрика
  corr_growth   косинус вектора месячных приростов                            — со-движение во времени
  shape_level   корреляция траектории относительного уровня                   — форма траектории
  lag_corr      макс. корреляция рядов расходов по лагам до 3 мес.            — один регион опережает другой
  dtw           многомерный DTW с полосой Сакое–Тиба (3 мес.)                  — сдвиг и деформация во времени
  road_gravity  exp(−d/d₀) по дорожным расстояниям (шоссе), kNN               — география
  fusion        усреднение трёх нормированных: профиль + со-движение + дороги — комбинация
  cos_feat_nm   то же, что cos_feat, но НЕвзаимный kNN                        — чувствительность к разрежению

Для каждой сети: структура (рёбра, изоляты, доля рёбер внутри типа и внутри региона), пересечение рёбер
с базовым правилом по Жаккару, затем заголовочный метод НА ЭТОЙ сети — индексы, churn и покластерный Жаккар
по Хеннигу на том же наборе подвыборок, что везде. Плюс значимость AVI против конфигурационной модели:
выбор сети тоже проверяется на то, что он не случаен.
"""
import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, scipy.sparse as sp, igraph as ig
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score
from config import CFG, SEED
from features import build
from pipeline import month_features
from headline import fit
from icvi import all_indices, graph_indices
from icvi_reference import graph_indices_pattern
from evalkit import subsamples, jaccard_per_cluster

import random as _random
# У igraph свой внутренний генератор: rewire() не подчиняется numpy. Без этого засева
# конфигурационная нулевая модель не воспроизводится между запусками.
ig.set_random_number_generator(_random.Random(SEED))
K_GRAPH = CFG["graph"]["k"]; K_MIN = CFG["graph"]["k_min_neighbors"]
RULES = ["cos_feat", "euclid_feat", "corr_growth", "shape_level", "lag_corr", "dtw", "road_gravity", "fusion", "cos_feat_nm"]
VARIANTS = [f"{r}@{v}" for r in ("cos_feat", "dtw", "lag_corr") for v in ("nm", "k8", "k30", "eps")]   # разрежение × три ведущих правила
MAX_LAG = 3      # лаг до квартала: на 22 месяцах больший лаг оставляет слишком короткое перекрытие
DTW_BAND = 3     # полоса Сакое–Тиба: допустимый сдвиг фаз между рядами, в месяцах
N_SPEND = 7      # колонки расходов в месячной матрице: log_total + 6 CLR; дальше статический блок


SPARS = {"": dict(), "nm": dict(mutual=False), "k8": dict(k=8), "k30": dict(k=30), "eps": dict(eps=True)}   # способы разрежения: суффикс правила после «@»


def knn(Z, metric, k=K_GRAPH, mutual=True, eps=False):
    """Взвешенная kNN-сеть. mutual=True — только взаимные соседи, с возвратом K_MIN сильнейших рёбер изолятам.
    eps=True — ε-сеть: рёбра между парами с расстоянием ниже порога, подобранного так, чтобы средняя степень была k; изолятам возвращаются K_MIN ближайших."""
    from graph_support import weighted_knn
    return weighted_knn(Z, metric, k=k, minimum=K_MIN, mutual=mutual, eps=eps)


def road_matrix(ids):
    """Матрица дорожных расстояний (шоссе) между МО: inf там, где связи нет; симметрична, диагональ 0."""
    N = len(ids); ix = {int(t): i for i, t in enumerate(ids)}
    c = pd.read_parquet("data/raw/hack/hackathonlicence/connection.parquet"); c = c[c.type == "highway"]
    c = c[c.territory_id_x.isin(ix) & c.territory_id_y.isin(ix) & (c.territory_id_x != c.territory_id_y)]
    D = np.full((N, N), np.inf); D[c.territory_id_x.map(ix).values, c.territory_id_y.map(ix).values] = c.distance.values
    D = np.minimum(D, D.T); np.fill_diagonal(D, 0.0); return D


def road_graph(D, k=K_GRAPH):
    """Гравитация по дорожной сети: w = exp(−d/d₀), d₀ — медиана расстояния до k-го соседа; затем kNN-разрежение."""
    n = D.shape[0]; Df = np.where(np.isfinite(D), D, np.inf)
    kth = np.partition(Df, min(k, n - 1), axis=1)[:, min(k, n - 1)]
    d0 = np.median(kth[np.isfinite(kth)])
    W = np.exp(-Df / max(d0, 1e-9)); np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -np.inf)
    ix = np.argsort(-W, axis=1, kind="stable")[:, :min(k, n - 1)]
    row = np.repeat(np.arange(n), ix.shape[1]); col = ix.ravel()
    val = np.maximum(W[row, col], 0)
    M = sp.csr_matrix((val, (row, col)), shape=(n, n))
    M = M.maximum(M.T); M.setdiag(0); M.eliminate_zeros()
    if M.data.size: M.data /= M.data.mean()
    return M


def _growth(Xt):
    G = np.stack([Xt[t] - Xt[t - 1] for t in range(1, len(Xt))], 0)   # T−1 × N × F
    return G.transpose(1, 0, 2).reshape(G.shape[1], -1)


def _level_shape(Xt):
    L = np.stack([x[:, 0] for x in Xt], 1)                            # N × T, относительный уровень
    return (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)


def _series(Xt):
    return np.stack([x[:, :N_SPEND] for x in Xt], 1)                  # N × T × F, относительные уровни расходов


def lag_corr_dist(Xt, max_lag=MAX_LAG):
    """Расстояние 1 − max_l corr_l(i, j): лучшая корреляция рядов расходов при сдвиге одного на l ≤ max_lag месяцев
    (один регион опережает другой). Ряды стандартизованы по времени; корреляция усреднена по признакам.
    Оговорка: максимум по лагам смещён вверх (чем больше лаг, тем короче перекрытие), поэтому лаг ограничен."""
    S = _series(Xt); N, T, F = S.shape
    best = np.full((N, N), -np.inf); lagstat = np.zeros((N, N), dtype=np.int8)
    for l in range(0, max_lag + 1):
        a = S[:, l:]; b = S[:, :T - l]
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        a = np.divide(a, a.std(1, keepdims=True), out=np.zeros_like(a), where=a.std(1, keepdims=True) > 1e-12)
        b = np.divide(b, b.std(1, keepdims=True), out=np.zeros_like(b), where=b.std(1, keepdims=True) > 1e-12)
        a, b = a.reshape(N, -1), b.reshape(N, -1)
        C = np.clip(a @ b.T / ((T - l) * F), -1, 1)
        for M, sg in ((C, l), (C.T, -l)) if l else ((C, 0),):
            up = M > best; best = np.where(up, M, best); lagstat = np.where(up, sg, lagstat).astype(np.int8)
    D = np.clip(1 - best, 0, None); np.fill_diagonal(D, 0.0)
    return D, lagstat


def dtw_dist(Xt, band=DTW_BAND):
    """Многомерный DTW с полосой Сакое–Тиба между рядами расходов (стоимость — евклид по признакам), нормирован на длину ряда.
    Допускает локальную деформацию времени: один регион проходит ту же траекторию быстрее или медленнее."""
    S = _series(Xt).astype(np.float64); N, T, F = S.shape; D = np.zeros((N, N)); big = 1e18
    for i in range(N):
        C = ((S[i][None, :, None, :] - S.transpose(0, 1, 2)[:, None, :, :]) ** 2).sum(-1)   # N × T(a) × T(b)
        acc = np.full((N, T, T), big); acc[:, 0, 0] = C[:, 0, 0]
        for a in range(T):
            for b in range(max(0, a - band), min(T, a + band + 1)):
                if a == 0 and b == 0: continue
                prev = np.full(N, big)
                if a > 0: prev = np.minimum(prev, acc[:, a - 1, b])
                if b > 0: prev = np.minimum(prev, acc[:, a, b - 1])
                if a > 0 and b > 0: prev = np.minimum(prev, acc[:, a - 1, b - 1])
                acc[:, a, b] = C[:, a, b] + prev
        D[i] = np.sqrt(acc[:, T - 1, T - 1] / (2 * T))
    D = np.minimum(D, D.T); np.fill_diagonal(D, 0.0); return D


_DCACHE = {}


def _dist(name, Xt):
    import hashlib
    values = np.ascontiguousarray(np.stack(Xt), dtype=np.float64)
    key = (name, values.shape, hashlib.sha256(values.tobytes()).hexdigest(), MAX_LAG, DTW_BAND)
    if key not in _DCACHE:
        _DCACHE[key] = lag_corr_dist(Xt)[0] if name == "lag_corr" else dtw_dist(Xt)
    return _DCACHE[key]


def make_gfun(name, Xt, idx, D_road):
    """Возвращает gfun(x) → смежность. Правила, не зависящие от месяца, считаются один раз и срезаются по idx."""
    name, _, var = name.partition("@"); kw = SPARS[var]
    if name == "cos_feat":    return lambda x: knn(x, "cosine", **kw)
    if name == "cos_feat_nm": return lambda x: knn(x, "cosine", mutual=False)
    if name == "euclid_feat": return lambda x: knn(x, "euclidean", **kw)
    if name == "corr_growth":
        G = _growth(Xt)[idx]; A = knn(G, "cosine", **kw); return lambda x: A
    if name == "shape_level":
        S = _level_shape(Xt)[idx]; A = knn(S, "correlation", **kw); return lambda x: A
    if name in ("lag_corr", "dtw"):
        A = knn(_dist(name, Xt)[np.ix_(idx, idx)], "precomputed", **kw); return lambda x: A
    if name == "road_gravity":
        A = road_graph(D_road[np.ix_(idx, idx)]); return lambda x: A
    if name == "fusion":
        G = _growth(Xt)[idx]; Ag = knn(G, "cosine"); Ar = road_graph(D_road[np.ix_(idx, idx)])
        def f(x):
            Ap = knn(x, "cosine")
            S = (Ap / max(Ap.data.mean(), 1e-9) + Ag / max(Ag.data.mean(), 1e-9) + Ar / max(Ar.data.mean(), 1e-9)) / 3
            S = S.tocsr(); S.setdiag(0); S.eliminate_zeros(); return S
        return f
    raise ValueError(name)


def edge_set(A):
    c = sp.triu(sp.csr_matrix(A), 1).tocoo()
    return set(zip(c.row.tolist(), c.col.tolist()))


def rewire_z(A, lab, n=100, seed=SEED):
    """z индекса AVI против конфигурационной модели (перестановка рёбер с сохранением степеней)."""
    rng = np.random.default_rng(seed); obs = graph_indices(A, lab)["AVI"]; nul = []
    c = sp.triu(sp.csr_matrix(A), 1).tocoo()
    for _ in range(n):
        g = ig.Graph(n=A.shape[0], edges=list(zip(c.row.tolist(), c.col.tolist())))
        g.rewire(n=10 * g.ecount()); e = np.array(g.get_edgelist()); w = rng.permutation(c.data)
        m = sp.coo_matrix((w[:len(e)], (e[:, 0], e[:, 1])), shape=A.shape)
        nul.append(graph_indices((m + m.T).tocsr(), lab)["AVI"])
    return float((obs - np.mean(nul)) / (np.std(nul) + 1e-12))


if __name__ == "__main__":
    d = build(); Xt, months = month_features(d); N = Xt[0].shape[0]; T = len(Xt)
    ids = np.array(d["ids"]); reg = pd.Series(d["meta"].region_name.fillna("?").values).values
    # дорожные расстояния (шоссе) в плотную матрицу
    c = pd.read_parquet("data/raw/hack/hackathonlicence/connection.parquet"); c = c[c.type == "highway"]
    ix = {int(t): i for i, t in enumerate(ids)}
    c = c[c.territory_id_x.isin(ix) & c.territory_id_y.isin(ix) & (c.territory_id_x != c.territory_id_y)]
    D = np.full((N, N), np.inf)
    D[c.territory_id_x.map(ix).values, c.territory_id_y.map(ix).values] = c.distance.values
    D = np.minimum(D, D.T); np.fill_diagonal(D, 0.0)
    print(f"дорожная сеть: пар с конечным расстоянием {np.isfinite(D).mean():.1%}", flush=True)

    full = np.arange(N); SUB = subsamples(N, reps=20, frac=0.8)
    base_edges = None; rows = []; base_L = {}; LEAD = ("cos_feat", "dtw", "lag_corr")      # Хеннинг на K=5 — только для ведущих правил и их базовых вариантов
    for name in RULES + VARIANTS:
        t0 = time.time(); gf = make_gfun(name, Xt, full, D)
        A = [gf(x) for x in Xt]; Alast = A[-1]
        es = edge_set(Alast)
        if base_edges is None: base_edges = es
        jac_base = len(es & base_edges) / max(len(es | base_edges), 1)
        iso = int((np.asarray(Alast.sum(1)).ravel() == 0).sum())
        rec = dict(правило=name, рёбер=len(es), изолятов=iso, Жаккар_с_базовым=jac_base)
        for K in CFG["headline"]["Ks"]:
            L, _ = fit(Xt, K, gfun=gf)
            if K not in base_L: base_L[K] = L[-1]                  # базовое правило идёт первым (cos_feat)
            rec[f"K{K}_ARI_с_базовым"] = float(adjusted_rand_score(base_L[K], L[-1]))
            ev = [all_indices(Xt[t], A[t], L[t]) for t in (0, T // 2, T - 1)]
            ref = [graph_indices_pattern(A[t], L[t]) for t in (0, T // 2, T - 1)]
            ch = float(np.mean([(L[t] != L[t + 1]).mean() for t in range(T - 1)]))
            same_type = float(np.mean([L[-1][i] == L[-1][j] for i, j in es])) if es else np.nan
            same_reg = float(np.mean([reg[i] == reg[j] for i, j in es])) if es else np.nan
            rec |= {f"K{K}_churn": ch, f"K{K}_внутри_типа": same_type, f"K{K}_внутри_региона": same_reg,
                    **{f"K{K}_{k}": float(np.mean([e[k] for e in ev])) for k in ("SW", "AVI", "AVU", "Q")},
                    f"K{K}_AVI_ref": float(np.mean([r["AVI_ref"] for r in ref])),
                    f"K{K}_z_AVI_конфиг": rewire_z(Alast, L[-1])}
            if K == CFG["headline"]["Ks"][0] or name in LEAD:        # Хеннинг на головном уровне; для ведущих правил и на подробном
                J = {cl: [] for cl in sorted(set(L[-1]))}
                for r, keep in enumerate(SUB):
                    gfs = make_gfun(name, Xt, keep, D)
                    lab = fit([x[keep] for x in Xt], K, seed=r, gfun=gfs)[0][-1]
                    for cl, v in jaccard_per_cluster(L[-1], lab, keep).items(): J[cl].append(v)
                j = np.array([np.nanmean(v) for v in J.values()])
                rec |= {f"K{K}_J_по_кластерам": j.mean(), f"K{K}_J_худший": j.min(),
                        f"K{K}_устойчивых": f"{int((j > .75).sum())}/{K}"}
        rows.append(rec); print(f"{name}: {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv("data/processed/edge_rules.csv", index=False)

    # попарное пересечение рёбер между правилами (на последнем месяце)
    sets = {n: edge_set(make_gfun(n, Xt, full, D)(Xt[-1])) for n in RULES + VARIANTS}
    Jm = pd.DataFrame({a: {b: len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1) for b in RULES + VARIANTS} for a in RULES + VARIANTS})
    Jm.round(3).to_csv("data/processed/edge_rules_jaccard.csv")
    pd.set_option("display.width", 260)
    r = pd.DataFrame(rows)
    print("\n=== Правила ребра"); print(r.round(3).to_string(index=False))
    print("\n=== Попарное пересечение рёбер (Жаккар)"); print(Jm.round(3).to_string())
