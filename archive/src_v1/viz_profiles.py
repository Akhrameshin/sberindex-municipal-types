import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CFG
from features import build, load
from headline import labels
from names import names
KK = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["headline"]["Ks"][-1]
NAMES = names(KK)
d = build(); ids = d["ids"]; T = d["tensor"]; cols = d["cols"]; L = labels(KK)
p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
typ = pd.Series(L[-1], index=ids)
SH = d['shares'][2:].mean(0); avg = pd.DataFrame({'log_total': T[2:, :, 0].mean(0), **{'sh_' + c: SH[:, i] for i, c in enumerate(['health', 'catering', 'food', 'market', 'transport', 'other'])}}, index=ids)
F = pd.DataFrame({"Расходы на душу (лог)": avg.log_total, "Продовольствие": avg.sh_food, "Общепит": avg.sh_catering, "Маркетплейсы": avg.sh_market,
                  "Здоровье": avg.sh_health, "Транспорт": avg.sh_transport, "Население (лог)": np.log(p["pop"]), "Зарплата (лог)": np.log(p["wage"]),
                  "Занятость: с/х": p.emp_agri, "Занятость: добыча": p.emp_mining, "Занятость: обработка": p.emp_manuf, "Занятость: торговля": p.emp_trade,
                  "Занятость: госуправл.": p.emp_gov, "Занятость: образование": p.emp_edu}, index=ids)
emp = [c for c in F if c.startswith("Занятость")]
med = F.groupby(typ).median(); med[emp] = F[emp].groupby(typ).mean(); med = med.T          # доли занятости скошены (много нулей) — берём среднее
base = F.median(); base[emp] = F[emp].mean(); z = med.sub(base, axis=0).div(F.std(), axis=0)
fig, ax = plt.subplots(figsize=(2.2 * KK + 1.6, 6.2)); im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(KK)); ax.set_xticklabels([f"{chr(10).join(__import__("textwrap").wrap(NAMES[k], 16))}\n(n={(typ==k).sum()})" for k in z.columns], fontsize=9); ax.set_yticks(range(len(z))); ax.set_yticklabels(z.index, fontsize=9)
for i in range(z.shape[0]):
    for j in range(z.shape[1]): ax.text(j, i, f"{z.values[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(z.values[i, j]) > 0.9 else "black")
ax.set_title("Профили типов: отклонение типа от всех МО, в σ (медиана; для долей занятости среднее)", fontsize=11); fig.colorbar(im, shrink=0.7, label="σ")
fig.tight_layout(); fig.savefig(f"outputs/profiles_K{KK}.png", dpi=160); print("ok"); print(z.round(2).to_string())
