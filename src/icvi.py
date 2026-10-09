"""Внутренние индексы качества кластеризации: SW, CH, S_Dbw (по признакам); AVI, AVU, ANUI, MQ, Q (по графу).

Обозначения. A — симметричная матрица весов без петель; w_i(c) — суммарный вес рёбер вершины i в кластер c;
d_i = Σ_c w_i(c) — взвешенная степень; c(i) — кластер вершины i; S_ab — вес рёбер между кластерами
(S_aa = 2·внутренний вес, S_ab = вес между a и b); out_a = Σ_{b≠a} S_ab.

ВЕРШИННЫЕ индексы (усреднение по вершинам, изолированные вершины исключены):
  AVI  = mean_i w_i(c(i)) / d_i                        доля веса вершины, идущая «к своим»   (больше — лучше)
  AVU  = mean_i max_{b≠c(i)} w_i(b) / d_i              доля веса в самый притягательный чужой кластер (меньше — лучше)
  ANUI = 1 / (AVU + 1/AVI)                             свёртка двух предыдущих            (больше — лучше)

КЛАСТЕРНЫЙ индекс:
  MQ   = Σ_a S_aa / (S_aa + out_a)                     TurboMQ (Mancoridis et al. 1998), максимум = K (больше — лучше)
  Q    = модульность Ньюмана                                                               (больше — лучше)

Связь AVI и MQ: оба опираются на долю внутреннего веса, но AVI взвешивает вершины одинаково, а MQ — кластеры.
На сбалансированных кластерах MQ ≈ K·AVI, на несбалансированных расходятся. Это зависимость, а не тождество;
в отчёте индексы подаются как одно семейство, а не как независимые свидетельства.

История правки: до 2026-10-03 AVI и AVU считались на уровне ПАР кластеров, и AVU в той форме сводился к
(K−1)/(2K−3), то есть зависел только от K и не отличал планированное разбиение от случайного
(tests/tests_avu.py фиксирует это как регрессионный тест). Вершинные определения выше это исправляют.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist


def _onehot(lab):
    u, inv = np.unique(lab, return_inverse=True)
    return sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), inv)), shape=(len(lab), len(u))), inv, len(u)


def graph_indices(A, lab):
    A = sp.csr_matrix(A)
    A = A.maximum(A.T)                      # симметризуем
    A.setdiag(0); A.eliminate_zeros()       # петли не участвуют ни в одном индексе
    H, inv, K = _onehot(lab)
    n = len(lab)
    W = (A @ H).toarray()                   # N×K: вес рёбер вершины i в кластер c
    deg = W.sum(1)
    ok = deg > 0                            # изолированные вершины не определяют доли
    own = W[np.arange(n), inv]
    avi = float((own[ok] / deg[ok]).mean()) if ok.any() else 0.0
    if K > 1 and ok.any():
        oth = W.copy(); oth[np.arange(n), inv] = -np.inf
        avu = float((oth[ok].max(1) / deg[ok]).mean())
    else:
        avu = 0.0                           # при K=1 чужих кластеров нет
    S = (H.T @ A @ H).toarray()
    tot = S.sum(1)
    m2 = S.sum()
    q = float(np.sum(np.diag(S) / m2 - (tot / m2) ** 2)) if m2 > 0 else 0.0
    mq = float(np.sum(np.diag(S) / np.maximum(tot, 1e-12)))
    anui = 1 / (avu + 1 / avi) if avi > 0 else 0.0
    return dict(AVI=avi, AVU=avu, ANUI=anui, MQ=mq, Q=q)


def silhouette(X, lab, D=None):
    D = cdist(X, X) if D is None else D
    u, inv = np.unique(lab, return_inverse=True)
    K, n = len(u), len(lab)
    M = np.stack([D[:, inv == k].sum(1) for k in range(K)], 1)
    cnt = np.bincount(inv, minlength=K)
    own = M[np.arange(n), inv]
    a = own / np.maximum(cnt[inv] - 1, 1)
    other = M / cnt
    other[np.arange(n), inv] = np.inf
    b = other.min(1)
    s = np.where(cnt[inv] > 1, (b - a) / np.maximum(a, b), 0)
    return s.mean()


def calinski_harabasz(X, lab):
    u, inv = np.unique(lab, return_inverse=True)
    K, n = len(u), len(lab)
    mu = X.mean(0)
    C = np.stack([X[inv == k].mean(0) for k in range(K)])
    cnt = np.bincount(inv)
    B = (cnt * ((C - mu) ** 2).sum(1)).sum()
    W = ((X - C[inv]) ** 2).sum()
    return (B / (K - 1)) / (W / (n - K))


def s_dbw(X, lab):
    """Halkidi & Vazirgiannis (2001): Scat + Dens_bw, меньше — лучше."""
    u, inv = np.unique(lab, return_inverse=True)
    K = len(u)
    sig_x = np.linalg.norm(X.var(0))
    C = np.stack([X[inv == k].mean(0) for k in range(K)])
    sig_c = np.array([np.linalg.norm(X[inv == k].var(0)) for k in range(K)])
    scat = sig_c.mean() / sig_x
    std = np.sqrt(sig_c.sum()) / K
    dens_c = np.array([(np.linalg.norm(X[inv == k] - C[k], axis=1) <= std).sum() for k in range(K)])
    tot = 0.0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            Xij = X[(inv == i) | (inv == j)]
            mid = (C[i] + C[j]) / 2
            d_mid = (np.linalg.norm(Xij - mid, axis=1) <= std).sum()
            tot += d_mid / max(dens_c[i], dens_c[j], 1)
    return scat + tot / (K * (K - 1))


def all_indices(X, A, lab, D=None):
    valid = 1 < len(np.unique(lab)) < len(lab)
    r = dict(SW=silhouette(X, lab, D), CH=calinski_harabasz(X, lab), S_Dbw=s_dbw(X, lab)) if valid else dict(SW=np.nan,CH=np.nan,S_Dbw=np.nan)
    r.update(graph_indices(A, lab))
    return r


DIRECTION = dict(SW=1, CH=1, S_Dbw=-1, AVI=1, AVU=-1, ANUI=1, MQ=1, Q=1)


def perm_z(X, A, lab, n_perm=100, seed=0, D=None):
    """z-оценка каждого индекса относительно перестановок меток (знак учитывает направление: больше — лучше)."""
    rng = np.random.default_rng(seed)
    obs = all_indices(X, A, lab, D)
    null = {k: [] for k in obs}
    for _ in range(n_perm):
        r = all_indices(X, A, rng.permutation(lab), D)
        for k in r:
            null[k].append(r[k])
    return {k: DIRECTION[k] * (obs[k] - np.mean(null[k])) / (np.std(null[k]) + 1e-12) for k in obs}
