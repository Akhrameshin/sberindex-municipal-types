"""Добавляет заголовочный метод GS-TKM (граф-сглаженный временной k-means) в общую таблицу сравнения (тот же протокол, K = compare.k)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from config import CFG
from features import build
from pipeline import *
from evalkit import row
from headline import fit
K = CFG["compare"]["k"]; d = build(); Xt, months = month_features(d); reg = d["meta"].region_name.fillna("?").values; A = [graph(x) for x in Xt]
L, _ = fit(Xt, K); L = list(L)
st = np.mean([ARI(L[-1], fit([x + np.random.default_rng(s).normal(0, CFG["compare"]["noise_sigma"], x.shape) for x in Xt], K, seed=s)[0][-1]) for s in (1, 2, 3)])
r = row(f"GS-TKM (наш, K={K})", L, st, Xt, A, reg)
base = pd.read_csv("data/processed/compare_all.csv"); base = base[~base.метод.str.startswith("GS-TKM")]
out = pd.concat([base, pd.DataFrame([r])]); out.to_csv("data/processed/compare_all.csv", index=False)
c = out[~out.метод.str.contains("λ=0.5|λ=5")].drop_duplicates("метод").set_index("метод")

# Два критерия, объявленные ДО таблицы, потому что они отвечают на разные вопросы.
# ICVI измеряют компактность и связность ОДНОГО среза. MQ и ANUI — производные от AVI (см. docstring icvi.py),
# поэтому в свёртку входит только AVI как представитель семейства: иначе одно свидетельство считалось бы трижды.
ICVI = {"SW": 1, "CH": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "Q": 1}
DYN = {"churn": -1, "stab": 1}                      # устойчивость типологии во времени и к шуму признаков
R = pd.DataFrame({k: (c[k] * s).rank(ascending=False) for k, s in ICVI.items()}); c["Борда_ICVI"] = R.sum(axis=1)
Rd = pd.DataFrame({k: (c[k] * s).rank(ascending=False) for k, s in DYN.items()}); c["Борда_динамика"] = Rd.sum(axis=1)

# Фронты Парето. Метод доминируется, если другой не хуже по всем осям и строго лучше хотя бы по одной.
# Две оси (SW ↑, churn ↓) — компактность среза против устойчивости во времени.
# Три оси (+ AVI ↑) — тот же набор критериев, по которому выбирались гиперпараметры в sgc_select/lam_select/
# pooled_select, то есть он объявлен до этой таблицы, а не подобран под неё.
def pareto(M):      # M: столбцы уже приведены к «больше — лучше»
    return [not any(all(M[j] >= M[i]) and any(M[j] > M[i]) for j in range(len(M)) if j != i) for i in range(len(M))]
M2 = np.c_[c["SW"].values, -c["churn"].values]
M3 = np.c_[c["SW"].values, -c["churn"].values, c["AVI"].values]
c["Парето_SW_churn"] = pareto(M2); c["Парето_SW_churn_AVI"] = pareto(M3)
pd.set_option("display.width", 260)
cols = ["K", "churn", "stab", "SW", "CH", "S_Dbw", "AVI", "AVU", "Q", "Борда_ICVI", "Борда_динамика", "Парето_SW_churn", "Парето_SW_churn_AVI"]
print(c[cols].round(3).sort_values("Борда_ICVI").to_string())
print("\nФронт по двум осям (силуэт, churn):", ", ".join(c.index[c["Парето_SW_churn"]]))
print("Фронт по трём осям (силуэт, churn, согласие с сетью AVI):", ", ".join(c.index[c["Парето_SW_churn_AVI"]]))
mine = [i for i in c.index if i.startswith("GS-TKM")]
if mine:
    m = mine[0]
    print(f"\nЧестно о нашем методе: по двум осям он {'НА фронте' if c.loc[m, 'Парето_SW_churn'] else 'НЕ на фронте'}, "
          f"по трём — {'НА фронте' if c.loc[m, 'Парето_SW_churn_AVI'] else 'НЕ на фронте'}. "
          f"Графовое сглаживание снижает силуэт и повышает churn относительно того же метода без сглаживания, "
          f"но поднимает согласие разбиения с сетью (см. sgc_select.csv). Это объявленный компромисс, а не победа по всем осям.")
c.to_csv("data/processed/compare_ranked.csv")
