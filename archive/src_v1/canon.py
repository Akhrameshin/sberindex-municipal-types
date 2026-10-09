import numpy as np
from scipy.optimize import linear_sum_assignment

def canonicalize(L, min_overlap=0.3):
    """Сквозная идентичность кластеров: метка месяца t сопоставляется метке t−1 по максимуму перекрытия (венгерский),
    пары с долей перекрытия (Jaccard) < min_overlap считаются новыми кластерами."""
    out = [L[0].copy()]; nxt = L[0].max() + 1
    for t in range(1, len(L)):
        prev, cur = out[-1], L[t]; P, C = sorted(set(prev)), sorted(set(cur))
        J = np.array([[((cur == c) & (prev == p)).sum() / ((cur == c) | (prev == p)).sum() for p in P] for c in C])
        r, c = linear_sum_assignment(-J); mp = {}
        for i, j in zip(r, c):
            if J[i, j] >= min_overlap: mp[C[i]] = P[j]
        new = cur.copy()
        for cc in C:
            if cc not in mp: mp[cc] = nxt; nxt += 1
        out.append(np.array([mp[v] for v in cur]))
    return np.array(out)
