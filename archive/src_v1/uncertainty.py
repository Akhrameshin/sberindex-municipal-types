"""Неопределённость типа каждого МО. Жёсткая метка скрывает, какие МО лежат на границе типов: именно они делают нестабильными малые типы
(Хеннинг: Север и ДВ). Для каждого МО считаются:
  доля_месяцев_в_типе   — доля месяцев, проведённых в модальном типе (по траектории Витерби);
  запас                 — относительный запас расстояния до центра своего типа над ближайшим чужим, (d₂−d₁)/(d₁+d₂), среднее по месяцам (0 — на границе, 1 — в центре);
  второй_тип            — ближайший чужой тип в последнем месяце;
  уверенность_подвыборки — доля 80%-подвыборок МО (общие подвыборки evalkit), в которых МО попадает в ТОТ ЖЕ тип после полной переподгонки метода
                           (кластеры подвыборки сопоставляются с базовыми венгерским алгоритмом);
  пограничное           — уверенность_подвыборки < BOUND.
Центроиды переупорядочены так же, как метки (по убыванию размера), иначе расстояния считались бы до не того центра."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from config import CFG, SEED
from features import build
from pipeline import month_features
from headline import smooth, fit, labels
from methods2 import pooled_kmeans_viterbi
from evalkit import subsamples
from names import names

BOUND = 0.75
d = build(); ids = d["ids"]; meta = d["meta"]; N = len(ids); Xt, months = month_features(d); SUB = subsamples(N, reps=20, frac=0.8)
HC = CFG["headline"]; summ = []
for K in HC["Ks"]:
    Xs = [smooth(x) for x in Xt]; path, km = pooled_kmeans_viterbi(Xs, K, HC["lam"], seed=SEED)
    order = np.argsort(-np.bincount(path.ravel(), minlength=K)); remap = np.empty(K, int); remap[order] = np.arange(K)
    L = remap[path]; assert (L == labels(K)).all(), "перенумерация не совпала с заголовочными метками"
    C = np.empty_like(km.cluster_centers_); C[remap] = km.cluster_centers_
    D = np.stack([((x[:, None, :] - C[None]) ** 2).sum(2) for x in Xs])                  # T×N×K, центры в порядке меток
    own = np.take_along_axis(D, L[:, :, None], axis=2)[:, :, 0]
    alt = D.copy(); np.put_along_axis(alt, L[:, :, None], np.inf, axis=2)
    other = alt.min(2); margin = ((other - own) / (other + own + 1e-12)).mean(0)
    modal = np.array([np.bincount(L[:, i], minlength=K).argmax() for i in range(N)]); share = np.array([(L[:, i] == modal[i]).mean() for i in range(N)])
    Dl = D[-1].copy(); Dl[np.arange(N), L[-1]] = np.inf; second = Dl.argmin(1)
    hits = np.zeros(N); cnt = np.zeros(N)
    for r, keep in enumerate(SUB):
        sub = fit([x[keep] for x in Xt], K, seed=r)[0][-1]; base = L[-1][keep]
        M = np.array([[((sub == a) & (base == b)).sum() for b in range(K)] for a in range(K)]); ra, ca = linear_sum_assignment(-M); mp = dict(zip(ra, ca))
        mapped = np.array([mp.get(v, -1) for v in sub]); hits[keep] += (mapped == base); cnt[keep] += 1
    conf = hits / np.maximum(cnt, 1); nm = names(K)
    out = pd.DataFrame({"territory_id": ids, "МО": meta.municipal_district_name.values, "регион": meta.region_name.values, "тип": L[-1], "название_типа": [nm[t] for t in L[-1]],
                        "модальный_тип": modal, "доля_месяцев_в_типе": share, "запас": margin, "второй_тип": second, "название_второго_типа": [nm[t] for t in second],
                        "уверенность_подвыборки": conf, "пограничное": conf < BOUND})
    out.round(4).to_csv(f"data/processed/mo_types_K{K}.csv", index=False)
    g = out.groupby("название_типа").agg(n=("тип", "size"), доля_пограничных=("пограничное", "mean"), уверенность_средняя=("уверенность_подвыборки", "mean"),
                                          запас_средний=("запас", "mean"), доля_месяцев_в_типе=("доля_месяцев_в_типе", "mean")).reset_index().assign(K=K)
    summ.append(g); print(f"\nK={K}: пограничных МО {int(out.пограничное.sum())} из {N} ({out.пограничное.mean():.1%})"); print(g.round(3).to_string(index=False))
    pairs = out[out.пограничное].groupby(["название_типа", "название_второго_типа"]).size().sort_values(ascending=False).head(5); print("главные границы между типами у пограничных МО:"); print(pairs.to_string())
pd.concat(summ).round(4).to_csv("data/processed/uncertainty_summary.csv", index=False)
