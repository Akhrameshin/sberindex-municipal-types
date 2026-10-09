"""Заголовочный метод GS-TKM: графовое сглаживание признаков (SGC) → общие центроиды на все месяцы → Витерби по траекториям МО."""
import numpy as np, scipy.sparse as sp
from pathlib import Path
from config import CFG, SEED
from features import build
from pipeline import month_features, graph
from methods2 import pooled_kmeans_viterbi

HC = CFG["headline"]; CACHE = Path("data/processed")


def smooth(x, s=HC["smooth_steps"], gfun=graph):
    """X̃ = Â^s X, Â — симметрично нормированная смежность с петлями. gfun задаёт правило ребра
    (по умолчанию — косинусный взаимный kNN из pipeline.graph; альтернативы см. src/edge_rules.py)."""
    if s == 0: return x
    n = x.shape[0]; A = gfun(x); A = A.maximum(A.T) + sp.eye(n); dg = np.asarray(A.sum(1)).ravel() ** -.5; Ah = sp.diags(dg) @ A @ sp.diags(dg)
    for _ in range(s): x = Ah @ x
    return x


def fit(Xt, K, seed=SEED, s=HC["smooth_steps"], lam=HC["lam"], gfun=graph):
    """Возвращает (T×N метки, центроиды). Метки упорядочены по убыванию размера (детерминированно)."""
    path, km = pooled_kmeans_viterbi([smooth(x, s, gfun) for x in Xt], K, lam, seed=seed)
    order = np.argsort(-np.bincount(path.ravel(), minlength=K)); remap = np.empty(K, int); remap[order] = np.arange(K)
    return remap[path], km


def centroids(Xt, K, seed=SEED):
    """Центры типов в том же порядке, что и заголовочные метки (по убыванию размера), и сами метки T×N. Для назначения новых МО по ближайшему центру."""
    from methods2 import pooled_kmeans_viterbi
    path, km = pooled_kmeans_viterbi([smooth(x) for x in Xt], K, HC["lam"], seed=seed)
    order = np.argsort(-np.bincount(path.ravel(), minlength=K)); remap = np.empty(K, int); remap[order] = np.arange(K)
    C = np.empty_like(km.cluster_centers_); C[remap] = km.cluster_centers_; return C, remap[path]


def _cache_key():
    """Кэш меток привязан к конфигу и к панели: иначе после смены данных или параметров labels() молча вернул бы устаревший файл."""
    import hashlib, json
    h = hashlib.sha256(json.dumps(CFG, sort_keys=True, default=str).encode())
    h.update((CACHE / "panel.parquet").read_bytes())
    for name in ("features.py", "pipeline.py", "methods2.py", "headline.py", "graph_support.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:10]


def labels(K, lens="combined", d=None, force=False):
    f = CACHE / f"headline_K{K}_{lens}_{_cache_key()}.npy"
    if f.exists() and not force: return np.load(f)
    d = d or build(); Xt, _ = month_features(d, lens=lens); L, _ = fit(Xt, K); np.save(f, L); return L


if __name__ == "__main__":
    d = build()
    for lens in ("combined", "behavioral"):      # обе линзы строятся здесь, а не лениво в validate.py
        for K in HC["Ks"]:
            L = labels(K, lens=lens, d=d, force=True)
            print(f"{lens:11} K={K}: размеры (последний месяц) {np.bincount(L[-1]).tolist()} | смен за период {(L[0] != L[-1]).mean():.3f}")
