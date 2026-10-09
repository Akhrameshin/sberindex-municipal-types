"""Falsifiable-тест. Спецификация зафиксирована заранее:
  типы замораживаются по расходам ТОЛЬКО до дек. 2023 (режим spend, без динамических дескрипторов и Росстата — без утечки из 2024);
  цели: прирост 2024 к 2023 для (a) log расходов на душу, (b) доли маркетплейсов;
  модели (OLS): регионы | типы | уровень+регионы | уровень+типы | регионы+типы; метрика — OOS MAE, повторная 5-кратная CV × 10.
  Гипотеза H1: типы предсказывают прирост не хуже регионов при в ~15 раз меньшем числе параметров."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold
from config import CFG, SEED
from features import build, load, spend_tensor
from pipeline import month_features
d = build(); ids = d["ids"]; N = len(ids)
d23 = dict(d); d23["tensor"] = d["tensor"][:12]; d23["months"] = d["months"][:12]
from headline import fit
KK = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["headline"]["Ks"][-1]
Xt, mo = month_features(d23, lens="spend"); L, _ = fit(Xt, KK)
types = pd.Series(L[-1], index=ids); print("замороженная типология (до дек. 2023): K =", types.nunique(), "| размеры", types.value_counts().to_dict())
T = d["tensor"]; sh = d["shares"]
y = {"прирост log расходов": T[12:, :, 0].mean(0) - T[:12, :, 0].mean(0), "прирост доли маркетплейсов": sh[12:, :, 3].mean(0) - sh[:12, :, 3].mean(0)}
level = T[:12, :, 0].mean(0); reg = pd.Series(d["meta"].region_name.fillna("?").values, index=ids)
dm = lambda s: pd.get_dummies(s, drop_first=True).values.astype(float)
Rg, Ty = dm(reg), dm(types); Lv = ((level - level.mean()) / level.std())[:, None]
specs = {"только среднее": np.zeros((N, 0)), "регионы": Rg, "типы": Ty, "уровень": Lv, "уровень + регионы": np.c_[Lv, Rg], "уровень + типы": np.c_[Lv, Ty], "регионы + типы": np.c_[Rg, Ty]}
def oos(X, t):
    rk = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED); out = []
    for tr, te in rk.split(X):
        A = np.c_[np.ones(len(tr)), X[tr]]; b = np.linalg.lstsq(A, t[tr], rcond=None)[0]; out.append(np.abs(t[te] - np.c_[np.ones(len(te)), X[te]] @ b).mean())
    return np.array(out)
rows = []
for nm, t in y.items():
    base = oos(specs["только среднее"], t)
    for sn, X in specs.items():
        m = oos(X, t); rows.append(dict(цель=nm, модель=sn, параметров=X.shape[1] + 1, MAE=m.mean(), **{"MAE/среднее": m.mean() / base.mean()}))
r = pd.DataFrame(rows); r.to_csv(f"data/processed/outoftime_K{KK}.csv", index=False); pd.set_option("display.width", 200); print(r.round(4).to_string(index=False))

# --- вердикт по предзарегистрированной H1: он печатается здесь, а не формулируется позже при написании отчёта
ver = []
for nm in y:
    g = r[r.цель == nm].set_index("модель")
    ty, rg = g.loc["типы", "MAE"], g.loc["регионы", "MAE"]
    both, худ = g.loc["регионы + типы", "MAE"], (ty - rg) / rg
    ver.append(dict(цель=nm, MAE_типы=ty, MAE_регионы=rg, типы_хуже_на=худ,
                    H1_не_хуже_регионов=bool(ty <= rg),
                    параметров_типы=int(g.loc["типы", "параметров"]), параметров_регионы=int(g.loc["регионы", "параметров"]),
                    MAE_регионы_и_типы=both, прирост_сверх_регионов=(rg - both) / rg,
                    типы_добавляют_сверх_регионов=bool(both < rg)))
v = pd.DataFrame(ver); v.to_csv(f"data/processed/outoftime_verdict_K{KK}.csv", index=False)
print("\nВердикт по H1 («типы не хуже регионов при ~15× меньшем числе параметров»):")
for _, x in v.iterrows():
    itog = "H1 ПОДТВЕРЖДЕНА" if x["H1_не_хуже_регионов"] else "H1 НЕ ПОДТВЕРЖДЕНА, типы хуже на {:.1%}".format(x["типы_хуже_на"])
    print("  {}: типы {:.5f} против регионов {:.5f} ({} против {} параметров) → {}".format(
        x["цель"], x["MAE_типы"], x["MAE_регионы"], x["параметров_типы"], x["параметров_регионы"], itog))
print("  Независимо от H1: добавление типов к региональным эффектам улучшает прогноз — " +
      "; ".join(f"{x['цель']} на {x['прирост_сверх_регионов']:.1%}" for _, x in v.iterrows()) +
      " (то есть типы несут информацию, которой нет в регионе).")
