"""Сверка наших реализаций индексов с эталонными (scikit-learn, igraph) на реальной матрице признаков.
Проверки, а не печать: расхождение с эталоном должно быть в пределах машинной точности."""
import sys; sys.path.insert(0, "src")
import numpy as np, igraph as ig
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from features import build
from icvi import silhouette, calinski_harabasz, s_dbw, graph_indices, perm_z, DIRECTION

X = build()["X"].values
A = kneighbors_graph(X, 15, mode="connectivity")
A = A.maximum(A.T)
lab = KMeans(5, n_init=5, random_state=0).fit_predict(X)
c = A.tocoo()
g = ig.Graph(n=len(X), edges=[(i, j) for i, j in zip(c.row, c.col) if i < j])


def test_silhouette_matches_sklearn():
    ours, ref = silhouette(X, lab), silhouette_score(X, lab)
    assert abs(ours - ref) < 1e-9, f"SW: наш {ours:.10f} против sklearn {ref:.10f}"


def test_ch_matches_sklearn():
    ours, ref = calinski_harabasz(X, lab), calinski_harabasz_score(X, lab)
    assert abs(ours - ref) / ref < 1e-9, f"CH: наш {ours:.6f} против sklearn {ref:.6f}"


def test_modularity_matches_igraph():
    ours, ref = graph_indices(A, lab)["Q"], g.modularity(lab)
    assert abs(ours - ref) < 1e-9, f"Q: наш {ours:.10f} против igraph {ref:.10f}"


def test_sdbw_positive_and_lower_for_better_partition():
    """S_Dbw меньше — лучше: разбиение k-means должно быть не хуже случайных меток."""
    rng = np.random.default_rng(0)
    good = s_dbw(X, lab); bad = np.mean([s_dbw(X, rng.permutation(lab)) for _ in range(3)])
    assert good > 0 and good < bad, f"S_Dbw: k-means {good:.4f} не лучше случайных {bad:.4f}"


def test_vertex_indices_in_range():
    r = graph_indices(A, lab)
    assert 0 <= r["AVI"] <= 1 and 0 <= r["AVU"] <= 1, f"вершинные доли вне [0,1]: {r}"
    assert r["MQ"] <= len(set(lab)) + 1e-9, f"MQ={r['MQ']:.4f} превышает K"


def test_perm_z_positive_for_real_partition():
    """На реальном разбиении z относительно перестановки меток должен быть положительным по всем индексам."""
    z = perm_z(X, A, lab, n_perm=30)
    bad = {k: round(v, 2) for k, v in z.items() if v <= 0}
    assert not bad, f"индексы не отличают разбиение от случайного: {bad}"


if __name__ == "__main__":
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_"):
            try:
                fn(); print(f"OK   {nm}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {nm}: {e}")
    print("\nz относительно перестановки меток:", {k: round(v, 1) for k, v in perm_z(X, A, lab, n_perm=30).items()})
    sys.exit(1 if fails else 0)
