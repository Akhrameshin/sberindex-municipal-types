"""Проверка на синтетических динамических сетях с известной истиной: находит ли метод НАСТОЯЩИЕ смены типа.

Внутренние индексы и устойчивость к подвыборкам не отвечают на главный вопрос динамической типологии:
отличает ли метод реальную смену типа от шума. Ответить можно только там, где истина известна.

Генератор. N узлов, K посаженных сообществ, T месяцев, F признаков:
    y_it = μ_{k(i,t)} + u_i + ε_it,   сеть — SBM по ТЕКУЩЕМУ составу с персистентностью рёбер ρ.
В месяц t_event доля `mig` узлов меняет сообщество; плюс фоновая смена 1% в месяц.

Режимы. Ключевое отличие от калибровки под один метод: мы проверяем себя и в тех режимах, где наши
допущения не выполняются, и сообщаем, где метод проигрывает.
    fe         устойчивые эффекты узлов, слабый шум — как в реальных данных (σ_u=0.66, σ_ε=0.28)
    no_fe      эффектов узлов нет: методы, которые их моделируют, теряют преимущество
    highnoise  шум ×3: проверка инерционности
    drift      центры сообществ дрейфуют: нестационарные центроиды
    knn_graph  сеть = kNN по тем же шумным признакам (как в реальном конвейере), а не независимый SBM
    comp_graph сеть = kNN по другому, независимому шумному сигналу о тех же сообществах (граф «знает» то, чего нет в признаках)

Метрики. NMI к истине; доля настоящих мигрантов, распознанных через 0/1/2 месяца; доля ложных смен
(узел-месяц, где метод сменил метку, а истина не менялась). Хорош тот метод, у которого высокая
распознаваемость при низкой доле ложных смен — одной стабильностью это не достигается.
"""
import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, scipy.sparse as sp, igraph as ig, leidenalg as la
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from config import SEED
from methods2 import pooled_kmeans_viterbi

REGIMES = {"fe":        dict(sig_c=0.78, sig_u=0.66, sig_e=0.28, drift=0.0),
           "no_fe":     dict(sig_c=0.78, sig_u=0.00, sig_e=0.28, drift=0.0),
           "highnoise": dict(sig_c=0.78, sig_u=0.66, sig_e=0.84, drift=0.0),
           "drift":     dict(sig_c=0.78, sig_u=0.66, sig_e=0.28, drift=0.15),
           # сеть строится из тех же шумных признаков (kNN), как в реальном конвейере, а не как независимый SBM по истинным сообществам:
           # в SBM-режимах граф несёт информацию о типах отдельно от признаков и тем самым помогает графовым методам по построению
           "knn_graph": dict(sig_c=0.78, sig_u=0.66, sig_e=0.28, drift=0.0, graph="knn"),
           # сеть строится из ДРУГОГО сигнала о тех же сообществах (независимый шум, без эффектов узлов): так выглядит DTW- или дорожный граф,
           # который знает про типы то, чего нет в самих признаках. Единственный режим, где сети есть что добавить к атрибутам, кроме их повторения
           "comp_graph": dict(sig_c=0.78, sig_u=0.66, sig_e=0.28, drift=0.0, graph="comp", sig_z=0.5)}


def generate(seed, N=600, K=5, T=12, F=7, mig=0.15, t_event=6, bg=0.01, p_in=0.10, p_out=0.01, rho=0.7, **reg):   # reg: sig_c, sig_u, sig_e, drift, graph
    rng = np.random.default_rng(seed)
    C = rng.normal(0, reg["sig_c"], (K, F)); U = rng.normal(0, reg["sig_u"], (N, F))
    z = np.zeros((T, N), int); z[0] = rng.integers(0, K, N)
    for t in range(1, T):
        z[t] = z[t - 1].copy()
        frac = mig if t == t_event else bg
        mv = rng.choice(N, max(1, int(frac * N)), replace=False)
        z[t, mv] = (z[t - 1, mv] + rng.integers(1, K, len(mv))) % K
    Y, A, prev = [], [], None
    for t in range(T):
        Ct = C + reg["drift"] * t * rng.normal(0, 1, (K, F)) if reg["drift"] else C
        Y.append(Ct[z[t]] + U + rng.normal(0, reg["sig_e"], (N, F)))
        P = np.where(z[t][:, None] == z[t][None, :], p_in, p_out)
        new = np.triu(rng.random((N, N)) < P, 1)
        if prev is not None:                                   # персистентность рёбер
            keep = np.triu(rng.random((N, N)) < rho, 1)
            new = np.where(keep, prev, new)
        prev = new
        M = sp.csr_matrix((new | new.T).astype(float)); M.setdiag(0); M.eliminate_zeros(); A.append(M)
    if reg.get("graph") in ("knn", "comp"):
        from sklearn.neighbors import kneighbors_graph
        A = []
        if reg["graph"] == "comp":                       # второй, независимый сигнал: свои центры сообществ, свой шум, без эффектов узлов
            C2 = rng.normal(0, reg["sig_c"], (K, F)); Zs = [C2[z[t]] + rng.normal(0, reg["sig_z"], (N, F)) for t in range(T)]
        for Yt in (Zs if reg["graph"] == "comp" else Y):
            B = kneighbors_graph(Yt, 15, mode="connectivity", metric="cosine"); B = B.minimum(B.T)
            n3 = kneighbors_graph(Yt, 3, mode="connectivity", metric="cosine"); B = B.maximum(n3.maximum(n3.T)).tocsr(); B.setdiag(0); B.eliminate_zeros(); A.append(B)
    return Y, A, z


def _align(truth, cur):
    ut, uc = np.unique(truth), np.unique(cur)
    M = np.zeros((len(uc), len(ut)))
    for i, a in enumerate(uc):
        for j, b in enumerate(ut): M[i, j] = ((cur == a) & (truth == b)).sum()
    r, c = linear_sum_assignment(-M); mp = {uc[i]: ut[j] for i, j in zip(r, c)}
    return np.array([mp.get(v, -1) for v in cur])


def _norm_adj(A):
    n = A.shape[0]; B = A.maximum(A.T) + sp.eye(n)
    dg = np.asarray(B.sum(1)).ravel() ** -.5
    return sp.diags(dg) @ B @ sp.diags(dg)


def m_gs_tkm(Y, A, K, seed, lam=0.5, s=1):
    Xs = [(_norm_adj(A[t]) @ Y[t]) if s else Y[t] for t in range(len(Y))]
    for _ in range(s - 1): Xs = [_norm_adj(A[t]) @ Xs[t] for t in range(len(Y))]
    return pooled_kmeans_viterbi(Xs, K, lam, seed=seed)[0]


def m_pooled_viterbi(Y, A, K, seed, lam=0.5):
    return pooled_kmeans_viterbi(Y, K, lam, seed=seed)[0]


def m_kmeans_hungarian(Y, A, K, seed):
    out = [KMeans(K, n_init=10, random_state=seed).fit_predict(Y[0])]
    for t in range(1, len(Y)):
        out.append(_align(out[-1], KMeans(K, n_init=10, random_state=seed + t).fit_predict(Y[t])))
    return np.array(out)


def _seq_smooth(Y, K, seed, alpha):
    """Последовательное сглаживание с тёплым стартом (эволюционная кластеризация, Chakrabarti 2006).
    alpha=None — вес истории оценивается по данным shrinkage-оценкой AFFECT (Xu, Kliger, Hero 2014)."""
    S = Y[0].copy(); lab = KMeans(K, n_init=10, random_state=seed).fit_predict(S); out = [lab]
    for t in range(1, len(Y)):
        if alpha is None:
            C = np.stack([S[lab == c].mean(0) if (lab == c).any() else S.mean(0) for c in range(K)])
            res = Y[t] - C[lab]; var = res.var(0, ddof=1) + 1e-12
            num = len(Y[t]) * var.sum(); den = ((S - C[lab]) ** 2).sum() + num
            a = float(np.clip(num / den, 0, 1))
        else:
            a = alpha
        S = a * S + (1 - a) * Y[t]
        km = KMeans(K, n_init=1, init=np.stack([S[lab == c].mean(0) if (lab == c).any() else S.mean(0)
                                                for c in range(K)]), random_state=seed).fit(S)
        lab = km.labels_; out.append(lab)
    return np.array(out)


def m_seq_fixed(Y, A, K, seed):  return _seq_smooth(Y, K, seed, 0.5)
def m_seq_affect(Y, A, K, seed): return _seq_smooth(Y, K, seed, None)


def m_temporal_leiden(Y, A, K, seed, omega=1.0):
    gs = []
    for M in A:
        c = sp.triu(M, 1).tocoo(); g = ig.Graph(n=M.shape[0], edges=list(zip(c.row.tolist(), c.col.tolist())))
        g.es["weight"] = c.data.tolist(); g.vs["id"] = list(range(M.shape[0])); gs.append(g)
    mem, _ = la.find_partition_temporal(gs, la.RBConfigurationVertexPartition, interslice_weight=omega,
                                        resolution_parameter=1.0, weights="weight", seed=seed, vertex_id_attr="id")
    return np.asarray(mem)


def m_gs_tkm_lam0(Y, A, K, seed):  return m_gs_tkm(Y, A, K, seed, lam=0.0, s=1)   # без штрафа за смену
def m_gs_tkm_s0(Y, A, K, seed):    return m_gs_tkm(Y, A, K, seed, lam=0.5, s=0)   # без графового сглаживания


METHODS = {"GS-TKM (наш)": m_gs_tkm, "GS-TKM λ=0 (без штрафа)": m_gs_tkm_lam0,
           "GS-TKM s=0 (без графа)": m_gs_tkm_s0, "общий k-means + Витерби": m_pooled_viterbi,
           "k-means по месяцам + Hungarian": m_kmeans_hungarian,
           "послед. сглаживание α=0.5": m_seq_fixed, "послед. сглаживание AFFECT": m_seq_affect,
           "temporal Leiden (только сеть)": m_temporal_leiden}


def score(L, z, t_event):
    """NMI, распознавание мигрантов через 0/1/2 мес., доля ложных смен."""
    T = len(z); La = np.stack([_align(z[t], L[t]) for t in range(T)])
    nmi = float(np.mean([NMI(z[t], L[t]) for t in range(T)]))
    mig = np.where(z[t_event] != z[t_event - 1])[0]
    det = [float(np.mean([La[min(t_event + d, T - 1), i] == z[t_event, i] for i in mig])) for d in (0, 1, 2)]
    false = tot = 0
    for t in range(1, T):
        ch = La[t] != La[t - 1]; tru = z[t] != z[t - 1]
        false += int((ch & ~tru).sum()); tot += len(z[t])
    return dict(NMI=nmi, мигранты_0=det[0], мигранты_1=det[1], мигранты_2=det[2], ложных_смен=false / tot)


if __name__ == "__main__":
    K, T_EV, SEEDS = 5, 6, [SEED + i for i in range(5)]
    rows = []
    for reg, par in REGIMES.items():
        for nm, fn in METHODS.items():
            t0 = time.time(); sc = []
            for s in SEEDS:
                Y, A, z = generate(s, K=K, t_event=T_EV, **par)
                try: sc.append(score(fn(Y, A, K, s), z, T_EV))
                except Exception as e: print(f"  {nm} / {reg}: сбой {e}", flush=True)
            if sc:
                rows.append(dict(режим=reg, метод=nm, **{k: float(np.mean([x[k] for x in sc])) for k in sc[0]}))
                print(f"{reg:10} {nm:32} NMI {rows[-1]['NMI']:.3f}  "
                      f"мигранты {rows[-1]['мигранты_0']:.2f}/{rows[-1]['мигранты_1']:.2f}/{rows[-1]['мигранты_2']:.2f}  "
                      f"ложных {rows[-1]['ложных_смен']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv("results/v2/synthetic.csv", index=False)
    r = pd.DataFrame(rows); pd.set_option("display.width", 240)
    print("\n=== Синтетика с известной истиной (среднее по 5 сеансам генератора)")
    print(r.round(3).to_string(index=False))
    print("\nЧитать так: высокая распознаваемость мигрантов ПРИ низкой доле ложных смен. "
          "Метод, который просто запрещает смены, даёт ложных смен около нуля и мигрантов около нуля.")
