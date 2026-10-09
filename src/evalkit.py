"""Общая обвязка сравнения методов: сопоставление меток, смены, возмущения, строка метрик, подвыборки для Хеннига."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from config import SEED
from icvi import all_indices


def subsamples(N, reps=20, frac=0.8, seed=SEED):
    """ОДИН И ТОТ ЖЕ набор подвыборок для всех сравниваемых вариантов (K, λ, s, метод).

    Важно: если брать подвыборки из общего потока случайных чисел по ходу вложенных циклов,
    каждый вариант получает свои подвыборки, и разница в Жаккаре между вариантами включает
    шум выборки, а не только разницу вариантов. Здесь поток фиксирован и зависит только от
    (N, reps, frac, seed), поэтому варианты сравниваются на одних и тех же подвыборках
    и результат не зависит от порядка и состава вызовов в одном запуске."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(N, int(frac * N), replace=False)) for _ in range(reps)]


def jaccard_per_cluster(base, lab, keep):
    """Для каждого кластера базового разбиения — максимальный Жаккар с кластерами разбиения на подвыборке."""
    out = {}
    for c in sorted(set(base[keep])):
        A = set(keep[base[keep] == c])
        out[c] = max(len(A & set(keep[lab == s])) / len(A | set(keep[lab == s])) for s in set(lab)) if A else np.nan
    return out

def align(prev, cur):
    P, C = sorted(set(prev)), sorted(set(cur)); Mx = np.array([[((cur == c) & (prev == p)).sum() for p in P] for c in C])
    r, c = linear_sum_assignment(-Mx); mp = {C[i]: P[j] for i, j in zip(r, c)}; return np.array([mp.get(v, 1000 + v) for v in cur])

def churn(L): return np.mean([(L[t] != L[t + 1]).mean() for t in range(len(L) - 1)])

def row(name, L, stab, Xt, A, reg, step=7):
    ev = [all_indices(Xt[t], A[t], L[t]) for t in range(0, len(L), step)]
    return dict(метод=name, K=np.mean([len(set(l)) for l in L]), churn=churn(L), stab=stab, NMI_reg=NMI(reg, L[-1]),
                **{k: np.mean([e[k] for e in ev]) for k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "Q")})
