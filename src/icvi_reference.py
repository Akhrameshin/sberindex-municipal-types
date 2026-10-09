"""Эталонные графовые индексы (реализация библиотеки Pattern) рядом с нашими вершинными — и доказательство,
что эталонный AVU не отличает разбиение от случайного.

Зачем. AVI/AVU/ANUI/MQ в списке критериев конкурса, и в открытых решениях они считаются по формулам
библиотеки Pattern, где AVU и AVI определены на уровне ПАР КЛАСТЕРОВ:
    AVU_ref = (1/K) Σ_i Σ_{j≠i} S_ij / (out_i + in_j − S_ij),      AVI_ref = mean_i S_ii / Σ_j S_ij
При сбалансированных внедиагональных блоках AVU_ref сводится к (K−1)/(2K−3), то есть определяется одним K
и не зависит от качества разбиения. Кроме того MQ_ref ≡ K·AVI_ref — это перемасштабирование, а не отдельный индекс.

Что делаем. Приводим ОБА семейства: эталонное (для сопоставимости с чужими таблицами и с библиотекой,
на которую ориентируется конкурс) и наше вершинное из icvi.py (для содержательных выводов). Демонстрация на
стохастической блочной модели с известной истиной показывает, какое из них вообще несёт сигнал.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, scipy.sparse as sp
from config import CFG, SEED
from icvi import graph_indices


def graph_indices_pattern(A, lab):
    """Дословно формулы Pattern (AdjacencyClusteringMetrics.get_metric): уровень пар кластеров."""
    A = sp.csr_matrix(A); A = A.maximum(A.T); A.setdiag(0); A.eliminate_zeros()
    u, inv = np.unique(lab, return_inverse=True); K = len(u)
    H = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), K))
    S = (H.T @ A @ H).toarray()
    avu = 0.0
    for i in range(K):
        out_i = S[i].sum() - S[i, i]; s = 0.0
        for j in range(K):
            if j == i: continue
            in_j = S[:, j].sum() - S[j, j]; den = out_i + in_j - S[i, j]
            s += S[i, j] / den if den != 0 else 0.0
        avu += s / K
    avi = float(np.mean([S[i, i] / S[i].sum() if S[i].sum() else 0.0 for i in range(K)]))
    anui = 1.0 / (avu + 1.0 / avi) if avi else 0.0
    mq = float(sum((S[i, i] / 2) / ((S[i, i] / 2) + 0.5 * (S[i].sum() - S[i, i]))
                   if S[i].sum() else 0.0 for i in range(K)))
    return dict(AVI_ref=avi, AVU_ref=avu, ANUI_ref=anui, MQ_ref=mq)


def sbm(sizes, pin=0.15, pout=0.01, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(len(sizes)), sizes); n = len(lab)
    P = np.where(lab[:, None] == lab[None, :], pin, pout)
    M = np.triu(rng.random((n, n)) < P, 1)
    return sp.csr_matrix((M | M.T).astype(float)), lab


if __name__ == "__main__":
    # --- 1) демонстрация на SBM: отличает ли индекс планированное разбиение от случайных меток
    rows = []
    for K in (4, 5, 6):
        A, lab = sbm([100] * K)
        rng = np.random.default_rng(SEED)
        t = {**graph_indices_pattern(A, lab), **graph_indices(A, lab)}
        nul = [{**graph_indices_pattern(A, p), **graph_indices(A, p)} for p in (rng.permutation(lab) for _ in range(20))]
        for k in t:
            m = float(np.mean([x[k] for x in nul])); s = float(np.std([x[k] for x in nul]))
            rows.append(dict(K=K, индекс=k, семейство="эталон Pattern" if k.endswith("_ref") else "вершинный",
                             истина=t[k], случайно_среднее=m,
                             отношение=t[k] / m if m else np.nan,
                             z=(t[k] - m) / s if s > 1e-12 else np.nan))
    demo = pd.DataFrame(rows)
    demo["аналитика_(K-1)/(2K-3)"] = np.where(demo.индекс == "AVU_ref", (demo.K - 1) / (2 * demo.K - 3), np.nan)
    demo.round(4).to_csv("data/processed/icvi_reference_demo.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== SBM с известной истиной: что различает индекс")
    print(demo.round(3).to_string(index=False))
    a = demo[(demo.индекс == "AVU_ref") & (demo.K == 5)].iloc[0]
    print(f"\nAVU по эталону при K=5: истина {a.истина:.4f}, случайные метки {a.случайно_среднее:.4f}, "
          f"аналитическое (K−1)/(2K−3) = {4/7:.4f} — индекс определяется одним K.")
    b = demo[(demo.индекс == "AVU") & (demo.K == 5)].iloc[0]
    print(f"AVU вершинный при K=5: истина {b.истина:.4f}, случайные {b.случайно_среднее:.4f} "
          f"(в {1/b.отношение:.1f} раза меньше у истины — индекс несёт сигнал).")
    mq = demo[(demo.индекс == "MQ_ref") & (demo.K == 5)].iloc[0]; avi = demo[(demo.индекс == "AVI_ref") & (demo.K == 5)].iloc[0]
    print(f"MQ_ref / AVI_ref при K=5 = {mq.истина / avi.истина:.4f} = K, то есть MQ_ref ≡ K·AVI_ref.")

    # --- 2) оба семейства на наших итоговых разбиениях
    from features import build
    from pipeline import month_features, graph
    from headline import labels
    d = build(); out = []
    for lens in ("combined", "behavioral"):
        Xt, months = month_features(d, lens=lens); A = [graph(x) for x in Xt]
        for K in CFG["headline"]["Ks"]:
            L = labels(K, lens=lens, d=d)
            ev = [{**graph_indices_pattern(A[t], L[t]), **graph_indices(A[t], L[t])} for t in (0, len(Xt) // 2, len(Xt) - 1)]
            out.append(dict(линза=lens, K=K, **{k: float(np.mean([e[k] for e in ev])) for k in ev[0]}))
    res = pd.DataFrame(out); res.round(4).to_csv("data/processed/icvi_reference.csv", index=False)
    print("\n=== Итоговые разбиения: эталонные и вершинные индексы рядом")
    print(res.round(3).to_string(index=False))
