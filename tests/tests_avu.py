"""Регрессионные тесты графовых индексов на планированных разбиениях (SBM с известной истиной).

Главный тест — AVU. До 2026-10-03 AVU считался на уровне пар кластеров и сводился к (K−1)/(2K−3),
то есть зависел только от K: на SBM истинное разбиение и случайные метки давали 0.571 против 0.572.
Тест 2 падает на той реализации и проходит на вершинной.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, scipy.sparse as sp
from icvi import graph_indices, DIRECTION

rng = np.random.default_rng(0)
GR = ("AVI", "AVU", "ANUI", "MQ", "Q")


def sbm(sizes, pin=0.15, pout=0.01):
    lab = np.repeat(np.arange(len(sizes)), sizes); n = len(lab)
    P = np.where(lab[:, None] == lab[None, :], pin, pout)
    A = np.triu(rng.random((n, n)) < P, 1)
    return sp.csr_matrix((A | A.T).astype(float)), lab


def test_truth_beats_random():
    """Истинное разбиение должно превосходить случайные метки по каждому индексу в его направлении."""
    for sizes in ([100] * 5, [300, 200, 100, 50, 50]):
        A, lab = sbm(sizes)
        t = graph_indices(A, lab)
        r = [graph_indices(A, rng.permutation(lab)) for _ in range(10)]
        for k in GR:
            s = DIRECTION[k]; nul = np.mean([x[k] for x in r])
            assert s * t[k] > s * nul + 1e-6, f"{k}: истина {t[k]:.4f} не лучше случайных {nul:.4f} (sizes={sizes})"


def test_avu_discriminates():
    """РЕГРЕССИЯ: AVU обязан заметно отличать истину от случайных меток, а не зависеть только от K."""
    A, lab = sbm([100] * 5)
    t = graph_indices(A, lab)["AVU"]
    nul = np.mean([graph_indices(A, rng.permutation(lab))["AVU"] for _ in range(10)])
    assert t < 0.6 * nul, f"AVU не различает разбиения: истина {t:.4f}, случайные {nul:.4f}"
    assert abs(t - (len(set(lab)) - 1) / (2 * len(set(lab)) - 3)) > 0.1, "AVU совпал с (K−1)/(2K−3): индекс определяется одним K"


def test_avu_not_function_of_k_alone():
    """При одном и том же K более качественное разбиение должно давать меньший AVU."""
    A, lab = sbm([100] * 4)
    bad = lab.copy(); ix = rng.choice(len(lab), len(lab) // 3, replace=False); bad[ix] = rng.integers(0, 4, len(ix))
    good, worse = graph_indices(A, lab)["AVU"], graph_indices(A, bad)["AVU"]
    assert good < worse - 1e-6, f"AVU не реагирует на порчу разбиения: {good:.4f} против {worse:.4f}"


def test_mq_bounded_by_k():
    """TurboMQ ограничен числом кластеров и достигает максимума на почти несвязанных кластерах."""
    A, lab = sbm([80] * 4, pin=0.3, pout=0.0005); K = 4
    mq = graph_indices(A, lab)["MQ"]
    assert 0 <= mq <= K + 1e-9, f"MQ={mq:.4f} вне [0, {K}]"
    assert mq > 0.9 * K, f"MQ={mq:.4f} слишком мал для почти несвязанных кластеров"


def test_degenerate():
    """K=1 и изолированные вершины не ломают индексы."""
    A, lab = sbm([50] * 2)
    r = graph_indices(A, np.zeros(len(lab), int))
    assert r["AVU"] == 0.0 and abs(r["AVI"] - 1.0) < 1e-9, f"K=1: {r}"
    B = sp.lil_matrix(A.shape); B[:len(lab) - 5, :len(lab) - 5] = A[:len(lab) - 5, :len(lab) - 5]
    r = graph_indices(sp.csr_matrix(B), lab)             # последние 5 вершин изолированы
    assert all(np.isfinite(list(r.values()))), f"изоляты дали нечисловой результат: {r}"


def test_reference_family_is_reproduced_and_insensitive():
    """Эталон Pattern воспроизводится и документируется как нечувствительный.

    Это не наш баг, а свойство референсной формулы: AVU_ref определяется одним K, а MQ_ref ≡ K·AVI_ref.
    Мы приводим эталон для сопоставимости, но выводы делаем по вершинному семейству."""
    from icvi_reference import graph_indices_pattern
    for K in (4, 5, 6):
        A, lab = sbm([100] * K)
        t = graph_indices_pattern(A, lab)
        nul = np.mean([graph_indices_pattern(A, rng.permutation(lab))["AVU_ref"] for _ in range(5)])
        assert abs(t["AVU_ref"] - (K - 1) / (2 * K - 3)) < 0.01, f"K={K}: эталонный AVU разошёлся с (K-1)/(2K-3)"
        assert abs(t["AVU_ref"] / nul - 1) < 0.02, f"K={K}: эталонный AVU внезапно стал различать разбиения"
        assert abs(t["MQ_ref"] - K * t["AVI_ref"]) < 1e-9, f"K={K}: MQ_ref больше не равен K*AVI_ref"
        v = graph_indices(A, lab)["AVU"]
        assert v < 0.6 * nul, f"K={K}: вершинный AVU не лучше эталонного"


if __name__ == "__main__":
    fails = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_"):
            try:
                fn(); print(f"OK   {nm}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {nm}: {e}")
    A, lab = sbm([100] * 5); t = graph_indices(A, lab); r = graph_indices(A, rng.permutation(lab))
    print("\nSBM [100]*5 — истина:  " + "  ".join(f"{k} {t[k]:.3f}" for k in GR))
    print("SBM [100]*5 — случайно:" + "  ".join(f"{k} {r[k]:.3f}" for k in GR))
    sys.exit(1 if fails else 0)
