import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from features import build
from pipeline import *
from icvi import all_indices
from evalkit import row, align, churn
from methods2 import *
from config import CFG
K = CFG["compare"]["k"]
d = build(); Xt, months = month_features(d, trim=2); reg = d["meta"].region_name.fillna("?").values
A = [graph(x) for x in Xt]
def noisy(seed, s=0.15):
    r = np.random.default_rng(seed); xs = [x + r.normal(0, s, x.shape) for x in Xt]; return xs, [graph(x) for x in xs]
noise = [noisy(s) for s in (1, 2, 3)]
rows = []
t0 = time.time()
Ld = [dmon(Xt[0], A[0], K)]
for t in range(1, len(Xt)): Ld.append(align(Ld[-1], dmon(Xt[t], A[t], K)))
st = np.mean([ARI(Ld[-1], dmon(xs[-1], As[-1], K, seed=i)) for i, (xs, As) in enumerate(noise)])
rows.append(row("DMoN (срезы)", Ld, st, Xt, A, reg)); print("DMoN", f"{time.time()-t0:.0f}s", flush=True)
for lam in (0.5, 2, 5):
    path, km = pooled_kmeans_viterbi(Xt, K, lam)
    st = np.mean([ARI(path[-1], pooled_kmeans_viterbi([x + np.random.default_rng(i).normal(0, 0.15, x.shape) for x in Xt], K, lam)[0][-1]) for i in range(3)])
    rows.append(row(f"k-means общий + Витерби λ={lam}", list(path), st, Xt, A, reg)); print("hyb", lam, flush=True)
# паттерн-кластеризация (метод жюри Алескерова–Мячина): число кластеров не задаётся, поэтому K в таблице — число различных паттернов
SH = d["shares"][CFG["features"]["trim_months"]:]
for nm_, adj in (("паттерны (полный порядок)", False), ("паттерны (соседние оси)", True)):
    Lp = pattern_labels(SH, adjacent=adj); st = np.mean([ARI(Lp[-1], pattern_labels(SH, adjacent=adj, noise=0.15, seed=i)[-1]) for i in range(3)])
    rows.append(row(nm_, list(Lp), st, Xt, A, reg)); print(nm_, "K =", len(set(Lp.ravel())), flush=True)
base = pd.read_csv("data/processed/compare.csv")
out = pd.concat([base, pd.DataFrame(rows)]); out.to_csv("data/processed/compare_all.csv", index=False)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
