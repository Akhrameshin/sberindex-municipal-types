"""Интерпретация типов: (1) контрасты и интервальные описания (в духе Миркина); (2) IMM-дерево (Dasgupta et al. 2020) —
дерево порогов с K листьями, воспроизводящее типы; точность листа против метки и против обычного CART."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from config import CFG
from features import build, load, PARTS_ALL, PARTS_RU
d = build(); ids = d["ids"]; T = d["tensor"]; sh = d["shares"]; meta = d["meta"]
p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
import yaml
from headline import labels
KK = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["headline"]["Ks"][-1]; lab = labels(KK)[-1]
from names import names
NAMES = names(KK)
F = pd.DataFrame(index=ids)
for i, c in enumerate(PARTS_ALL): F[f"доля расходов: {PARTS_RU[c]}"] = sh[:, :, i].mean(0)
F["расходы на душу, руб/мес"] = np.expm1(T[:, :, 0]).mean(0)
F["население, чел"] = p["pop"]; F["зарплата, руб"] = p["wage"]
for g, nm in (("agri", "с/х"), ("mining", "добыча"), ("manuf", "обработка"), ("trade", "торговля"), ("gov", "госуправл."), ("edu", "образование"), ("ict", "ИКТ")): F[f"занятость: {nm}"] = p["emp_" + g]
F["рост доли маркетплейсов за период"] = sh[-3:, :, 3].mean(0) - sh[:3, :, 3].mean(0)
F["волатильность расходов"] = d["X"]["volatility"]; F = F.fillna(F.median())
types = sorted(set(lab)); Z = (F - F.mean()) / F.std()
# --- (1) контрасты и интервалы
con = pd.DataFrame({t: Z[lab == t].mean() for t in types}); iv = {}
for t in types:
    top = con[t].abs().sort_values(ascending=False).index[:4]; m = lab == t
    iv[t] = "; ".join(f"{c} {'↑' if con.loc[c, t] > 0 else '↓'} [{F.loc[m, c].quantile(.1):.3g}…{F.loc[m, c].quantile(.9):.3g}]" for c in top)
# Названия типов заданы вручную (configs/type_names.yaml) и привязаны к номерам меток, а номера упорядочены по размеру.
# Размеры меняются при смене данных, поэтому каждое название проверяется по своему профилю: несовпадение — остановка, а не молчаливая ошибка.
CHECKS = {"Север": ("занятость: добыча", "max"), "Столичные": ("расходы на душу, руб/мес", "max"), "Индустриальные": ("занятость: обработка", "max"),
          "сельские": ("расходы на душу, руб/мес", "min")}
for t in types:
    for key, (feat, how) in CHECKS.items():
        if key in NAMES.get(t, ""):
            want = con.loc[feat, types].idxmax() if how == "max" else con.loc[feat, types].idxmin()
            if want != t: sys.exit(f"название типа {t} «{NAMES[t]}» не соответствует профилю: по признаку «{feat}» {how} у типа {want}. Проверьте configs/type_names.yaml")
con.columns = [f"{t}: {NAMES.get(t, '?')}" for t in types]; con.round(2).to_csv(f"data/processed/contrasts_K{KK}.csv")
pd.Series({f"{t}: {NAMES.get(t,'?')} (n={int((lab==t).sum())})": iv[t] for t in types}).to_csv(f"data/processed/interval_descriptions_K{KK}.csv", header=["описание"])
for t in types: print(f"тип {t} {NAMES.get(t,'?')} (n={(lab==t).sum()}): {iv[t]}")

# --- (1б) траектории типов: сдвиг структуры трат и рост уровня между годами (сюжет про маркетплейсы)
mk = PARTS_ALL.index("market")
y23, y24 = slice(0, 12), slice(12, 24)
tr = pd.DataFrame([dict(тип=NAMES.get(t, "?"), n=int((lab == t).sum()),
                        **{"маркетплейсы 2023, %": 100 * sh[y23][:, lab == t, mk].mean(),
                           "маркетплейсы 2024, %": 100 * sh[y24][:, lab == t, mk].mean(),
                           "рост расходов 24/23, %": 100 * (np.expm1(T[y24][:, lab == t, 0]).mean() / np.expm1(T[y23][:, lab == t, 0]).mean() - 1)})
                   for t in types])
tr.round(1).to_csv(f"data/processed/type_trajectories_K{KK}.csv", index=False)
print(); print(tr.round(1).to_string(index=False))

# --- (2) IMM
cent = np.stack([F[lab == t].mean().values for t in types]); X = F.values; y = np.searchsorted(types, lab)
def best_split(idx, cs):
    best = (np.inf, None, None)
    for j in range(X.shape[1]):
        cv = cent[cs, j]; lo, hi = cv.min(), cv.max()
        if hi - lo < 1e-12: continue
        for th in np.unique(np.quantile(X[idx, j], np.linspace(.02, .98, 97))):
            if not (lo < th <= hi): continue
            side_pt = X[idx, j] <= th; side_c = cent[y[idx], j] <= th                      # точка и её центр по разные стороны → ошибка
            if (cent[cs, j] <= th).all() or (cent[cs, j] > th).all(): continue
            err = (side_pt != side_c).sum()
            if err < best[0]: best = (err, j, th)
    return best
def grow(idx, cs):
    if len(cs) == 1: return ("leaf", cs[0])
    _, j, th = best_split(idx, cs); left = X[idx, j] <= th
    cl, cr = [c for c in cs if cent[c, j] <= th], [c for c in cs if cent[c, j] > th]
    return ("node", j, th, grow(idx[left], cl), grow(idx[~left], cr))
tree = grow(np.arange(len(X)), list(range(len(types))))
def predict(node, x):
    while node[0] == "node": node = node[3] if x[node[1]] <= node[2] else node[4]
    return node[1]
pred = np.array([predict(tree, x) for x in X]); acc = (pred == y).mean()
def show(node, ind=0):
    if node[0] == "leaf": print("  " * ind + f"→ тип {types[node[1]]}: {NAMES.get(types[node[1]],'?')}"); return
    print("  " * ind + f"если {F.columns[node[1]]} ≤ {node[2]:.4g}:"); show(node[3], ind + 1); print("  " * ind + "иначе:"); show(node[4], ind + 1)
print(f"\nIMM-дерево ({len(types)} листьев): точность воспроизведения типов {acc:.1%}"); show(tree)
cart = DecisionTreeClassifier(max_leaf_nodes=len(types), random_state=CFG["seed"]).fit(X, y); print(f"CART с тем же числом листьев (верхняя граница для дерева): {cart.score(X, y):.1%}")
pd.DataFrame({"IMM": [acc], "CART": [cart.score(X, y)], "листьев": [len(types)]}).to_csv(f"data/processed/imm_accuracy_K{KK}.csv", index=False)
