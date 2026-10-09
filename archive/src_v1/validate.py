"""Внешняя валидация без утечки. Две линзы:
  поведенческая (расходы + их динамика)         → проверяем на Росстате и market_access, которых в признаках нет;
  комбинированная (расходы + динамика + Росстат) → проверяем на market_access и индексе мобильности (в т.ч. срез 2025-11, вне периода построения).
Для каждого показателя: ω² типов, ω² регионов, и эффект типа ВНУТРИ региона (прирост скорректированного R² сверх региональных
фиксированных эффектов) с перестановочным p-значением (метки типов переставляются внутри регионов)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy import stats
from config import CFG, SEED
from features import build, load
from pipeline import month_features

rng = np.random.default_rng(SEED)
d = build(); ids = d["ids"]; meta = d["meta"]; N = len(ids)
reg = pd.Series(meta.region_name.fillna("?").values, index=ids)
mode = lambda L: pd.Series([np.bincount(L[:, i]).argmax() for i in range(L.shape[1])], index=ids)

# --- типы заголовочного метода (последний месяц): комбинированная и поведенческая линзы
from headline import labels
from sklearn.metrics import adjusted_rand_score
def design(g):
    return pd.get_dummies(g, drop_first=True).values.astype(float)


def r2adj(y, X):
    X = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b
    n, k = X.shape; return 1 - (res @ res / (n - k)) / (((y - y.mean()) ** 2).sum() / (n - 1))


def omega2(y, g):
    k = g.nunique(); n = len(y); gm = y.mean()
    ssb = sum((g == a).sum() * (y[g == a].mean() - gm) ** 2 for a in g.unique()); sst = ((y - gm) ** 2).sum(); msw = (sst - ssb) / (n - k)
    return max(0.0, (ssb - (k - 1) * msw) / (sst + msw))


def within_region(y, types, regions, n_perm=CFG["validation"]["n_perm"]):
    """Прирост adj.R² от типов сверх региональных эффектов + p-значение перестановок типов внутри регионов."""
    R = design(regions); base = r2adj(y, R); full = r2adj(y, np.c_[R, design(types)]); obs = full - base
    null = []
    for _ in range(n_perm):
        tp = types.copy()
        for r, idx in regions.groupby(regions).groups.items():
            tp.loc[idx] = rng.permutation(types.loc[idx].values)
        null.append(r2adj(y, np.c_[R, design(tp)]) - base)
    return obs, (1 + sum(x >= obs for x in null)) / (n_perm + 1)


def table(lens, types, vars_):
    rows = []
    for name, y in vars_.items():
        m = y.notna(); yy, tt, rr = y[m].values, types[m], reg[m]
        w, p = within_region(yy, tt, rr)
        rows.append(dict(линза=lens, показатель=name, n=int(m.sum()), **{"ω² типы": omega2(yy, tt), "ω² регионы": omega2(yy, rr), "Δadj.R² тип|регион": w, "p": p}))
    return pd.DataFrame(rows)


p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
ros = {"log зарплата": np.log(p["wage"]), "log население": np.log(p["pop"]), "доля занятых: торговля": p["emp_trade"], "доля занятых: обработка": p["emp_manuf"],
       "доля занятых: добыча": p["emp_mining"], "доля занятых: с/х": p["emp_agri"], "доля занятых: госуправл.": p["emp_gov"], "доля занятых: ИКТ": p["emp_ict"]}
ma = {"market_access": meta.market_access}

# мобильность: сопоставление по однозначному названию
mob = pd.read_parquet("data/raw/indeks-mobilnosti.parquet")
nm = meta.municipal_district_name.reset_index(); nm.columns = ["tid", "name"]; uniq = nm[~nm.name.duplicated(keep=False)].set_index("name").tid
mob = mob[mob.ref_area.isin(uniq.index)].assign(tid=lambda t: t.ref_area.map(uniq))
mv = {f"мобильность {pd.Timestamp(per).strftime('%Y-%m')}": mob[mob.period == per].set_index("tid").value.reindex(ids) for per in sorted(mob.period.unique())}
print("мобильность: сопоставлено МО", {k: int(v.notna().sum()) for k, v in mv.items()})

outs = []
for K in CFG["headline"]["Ks"]:
    tc = pd.Series(labels(K, "combined")[-1], index=ids); tb = pd.Series(labels(K, "behavioral")[-1], index=ids)
    print(f"K={K}: ARI между поведенческой и комбинированной линзами = {adjusted_rand_score(tb, tc):.3f}")
    outs.append(table("поведенческая", tb, {**ros, **ma}).assign(K=K)); outs.append(table("комбинированная", tc, {**ma, **mv}).assign(K=K))
out = pd.concat(outs)
pd.set_option("display.width", 250); print(out.round(3).to_string(index=False)); out.to_csv("data/processed/validation.csv", index=False)
a = [k for k in mv if k.endswith("2024-12")][0]; b = [k for k in mv if k.endswith("2025-11")][0]
both = mv[a].notna() & mv[b].notna(); print("корреляция мобильности 2024-12 и 2025-11:", round(stats.spearmanr(mv[a][both], mv[b][both])[0], 3), "n =", int(both.sum()))
