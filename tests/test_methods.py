"""Проверки методов на игрушках с известным ответом: каждое утверждение здесь можно проверить без данных конкурса."""
import itertools
import numpy as np, pandas as pd
import pytest


def test_viterbi_is_optimal_against_brute_force():
    """Путь Витерби должен давать минимум суммарной стоимости (расстояния до центров + λ за смену) среди ВСЕХ траекторий."""
    from methods2 import pooled_kmeans_viterbi
    rng = np.random.default_rng(0); K, T, N = 3, 4, 5
    Xt = [rng.normal(size=(N, 2)) + rng.integers(0, K, size=(N, 1)) * 2.0 for _ in range(T)]
    lam = 0.7; path, km = pooled_kmeans_viterbi(Xt, K, lam, seed=0)
    D = np.stack([((x[:, None, :] - km.cluster_centers_[None]) ** 2).sum(2) for x in Xt]); pen = lam * np.median(D.min(2))
    cost = lambda p, i: sum(D[t, i, p[t]] for t in range(T)) + pen * sum(p[t] != p[t + 1] for t in range(T - 1))
    for i in range(N):
        best = min(cost(p, i) for p in itertools.product(range(K), repeat=T))
        assert cost(path[:, i], i) == pytest.approx(best)


def test_viterbi_penalty_monotone():
    """Чем больше штраф λ, тем меньше смен типа."""
    from methods2 import pooled_kmeans_viterbi
    rng = np.random.default_rng(1); Xt = [rng.normal(size=(200, 3)) for _ in range(12)]
    ch = [np.mean([(p[t] != p[t + 1]).mean() for t in range(11)]) for p in (pooled_kmeans_viterbi(Xt, 4, lam)[0] for lam in (0.0, 0.5, 3.0))]
    assert ch[0] >= ch[1] >= ch[2] and ch[0] > ch[2]


def _series(N=6, T=14, seed=0):
    r = np.random.default_rng(seed); return [r.normal(size=(N, 7)) for _ in range(T)]


def test_lag_corr_finds_shifted_copy():
    """Копия ряда, сдвинутая на 2 месяца, ближе к оригиналу при учёте лагов, чем без них."""
    from edge_rules import lag_corr_dist
    r = np.random.default_rng(0); base = np.cumsum(r.normal(size=(20, 7)), 0)
    S = np.stack([base[:14], base[2:16], np.cumsum(r.normal(size=(20, 7)), 0)[:14]], 1)      # T × N × F
    Xt = [S[t] for t in range(14)]
    D, lag = lag_corr_dist(Xt, max_lag=3)
    assert D[0, 1] < D[0, 2] and abs(int(lag[0, 1])) == 2
    D0, _ = lag_corr_dist(Xt, max_lag=0); assert D[0, 1] < D0[0, 1]
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)


def _naive_dtw(a, b, band):
    T = len(a); acc = np.full((T, T), np.inf); acc[0, 0] = ((a[0] - b[0]) ** 2).sum()
    for i in range(T):
        for j in range(max(0, i - band), min(T, i + band + 1)):
            if i == j == 0: continue
            prev = min(acc[i - 1, j] if i else np.inf, acc[i, j - 1] if j else np.inf, acc[i - 1, j - 1] if i and j else np.inf)
            acc[i, j] = ((a[i] - b[j]) ** 2).sum() + prev
    return np.sqrt(acc[-1, -1] / (2 * T))


def test_dtw_matches_naive_and_is_metric_like():
    from edge_rules import dtw_dist
    Xt = _series(N=5, T=10); D = dtw_dist(Xt, band=3)
    S = np.stack([x[:, :7] for x in Xt], 1)
    for i, j in itertools.combinations(range(5), 2):
        assert D[i, j] == pytest.approx(_naive_dtw(S[i], S[j], 3), rel=1e-9)
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0) and (D >= 0).all()


def test_dtw_tolerates_time_shift_better_than_euclid():
    from edge_rules import dtw_dist
    t = np.linspace(0, 3, 16); a = np.sin(t)[:, None] * np.ones((1, 7)); b = np.sin(t - 0.6)[:, None] * np.ones((1, 7))
    Xt = [np.stack([a[k], b[k]]) for k in range(16)]
    D = dtw_dist(Xt, band=3); eu = np.sqrt(((a - b) ** 2).sum() / (2 * 16))
    assert D[0, 1] < eu


def test_pattern_clustering_is_order_invariant():
    """Паттерн-кластеризация не меняется при любом монотонном преобразовании долей (её ключевое свойство)."""
    from methods2 import pattern_labels
    r = np.random.default_rng(0); sh = r.dirichlet(np.ones(6), size=(4, 120))
    a = pattern_labels(sh); b = pattern_labels(sh ** 3 * 17 + 2)
    assert (a == b).all()
    assert len(set(pattern_labels(sh, adjacent=True).ravel())) <= len(set(a.ravel()))


def test_threshold_rank_toy():
    import rank_aggregate as ra
    c = pd.DataFrame({"SW": [3, 2, 1], "CH": [3, 2, 1], "S_Dbw": [1, 2, 3], "AVI": [3, 1, 2], "AVU": [1, 2, 3], "MQ": [3, 2, 1]}, index=list("ABC"))
    r, G = ra.threshold_rank(c, ra.SETS["6 индексов"])
    assert r["A"] == 1 and r["C"] == 3 and (G.loc["C"] == 3).sum() >= 4


def test_complete_only_filters_short_series():
    from features import complete_only
    p = pd.DataFrame({"tid": [1] * 24 + [2] * 5, "month": list(range(24)) + list(range(5))})
    assert set(complete_only(p, 24).tid) == {1} and set(complete_only(p, 3).tid) == {1, 2}


def test_subsamples_are_shared_and_jaccard_identity():
    from evalkit import subsamples, jaccard_per_cluster
    a, b = subsamples(100, 5, 0.8), subsamples(100, 5, 0.8)
    assert all((x == y).all() for x, y in zip(a, b))
    lab = np.repeat([0, 1, 2], 20); keep = np.arange(60)
    assert all(v == 1.0 for v in jaccard_per_cluster(lab, lab, keep).values())


def test_cache_key_depends_on_config(monkeypatch):
    import headline
    k1 = headline._cache_key(); monkeypatch.setitem(headline.CFG["headline"], "lam", 123.0)
    assert headline._cache_key() != k1


def test_eps_and_nm_graphs_are_symmetric_without_isolates():
    from edge_rules import knn
    Z = np.random.default_rng(0).normal(size=(120, 5))
    for kw in (dict(), dict(mutual=False), dict(eps=True), dict(k=8)):
        A = knn(Z, "cosine", **kw); assert abs(A - A.T).max() < 1e-12 and (np.asarray(A.sum(1)).ravel() > 0).all()


def test_comp_graph_carries_more_community_information_than_knn_graph():
    """Смысл двух режимов синтетики: граф из того же шумного сигнала (knn_graph) знает о сообществах не больше самих признаков,
    граф из независимого дополняющего сигнала (comp_graph) — больше. Иначе сравнение режимов бессмысленно."""
    import synthetic as sy
    def within(A, z):
        c = A.tocoo(); m = c.row < c.col; return float((z[c.row[m]] == z[c.col[m]]).mean())
    _, A_knn, z = sy.generate(0, N=400, T=3, **sy.REGIMES["knn_graph"]); _, A_cmp, z2 = sy.generate(0, N=400, T=3, **sy.REGIMES["comp_graph"])
    assert (z == z2).all() and within(A_cmp[0], z2[0]) > within(A_knn[0], z[0]) + 0.05


def test_graph_smoothing_helps_only_with_complementary_graph():
    """Главный вывод пункта «вклад сети»: сглаживание помогает при дополняющем графе и не помогает при графе из тех же признаков."""
    import synthetic as sy
    from sklearn.metrics import normalized_mutual_info_score as NMI
    gain = {}
    for reg in ("knn_graph", "comp_graph"):
        d = []
        for s in range(3):
            Y, A, z = sy.generate(100 + s, N=600, T=6, **sy.REGIMES[reg])      # N=600 как в synthetic.py: при N=300 выигрыш графа ещё не виден
            with_g = sy.m_gs_tkm(Y, A, 5, s, s=1); no_g = sy.m_gs_tkm(Y, A, 5, s, s=0)
            d.append(np.mean([NMI(z[t], with_g[t]) for t in range(6)]) - np.mean([NMI(z[t], no_g[t]) for t in range(6)]))
        gain[reg] = np.mean(d)
    assert gain["comp_graph"] > 0.03 and gain["knn_graph"] < 0.0 and gain["comp_graph"] > gain["knn_graph"] + 0.05
