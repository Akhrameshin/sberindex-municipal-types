import sys, time; sys.path.insert(0, "src"); sys.path.insert(0, "external/KEFRiN")
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from features import build
from pipeline import *
from evalkit import *
import kefrin, logging; logging.disable(logging.CRITICAL)
from config import CFG, SEED
K = CFG["compare"]["k"]
d = build(); Xt, months = month_features(d, trim=2); reg = d["meta"].region_name.fillna("?").values
A = [graph(x) for x in Xt]
P = lambda a: (a.toarray() > 0).astype(float)                      # бинарная смежность графа
def noisy(seed, s=0.15):
    r = np.random.default_rng(seed); xs = [x + r.normal(0, s, x.shape) for x in Xt]; return xs, [graph(x) for x in xs]
noise = [noisy(s) for s in (1, 2, 3)]

# Вызываем класс KEFRiN напрямую, а не обёртки kefrin.KEFRiNe/KEFRiNc/KEFRiNm: в конфиге авторов есть
# random_state, но обёртки его не пропускают, и тогда kefrin.py работает на ГЛОБАЛЬНОМ состоянии numpy
# (засев в их коде стоит под `if config.random_state is not None`). Без него числа KEFRiN зависели от
# того, что считалось в процессе до него: между двумя прогонами stab у KEFRiNc расходился на 0,286.
# Это параметр самих авторов, мы лишь задаём его явно, как задаём сид своим методам.
DM = kefrin.DistanceMetric
def kef(metric, pp, rho=1.0, xi=1.0):
    def run(y, p_, n_clusters):
        cfg = kefrin.KEFRiNConfig(n_clusters=n_clusters, rho=rho, xi=xi, distance_metric=metric,
                                  kmeans_plus_plus=True, random_state=SEED,
                                  preprocessing_y=kefrin.PreprocessingMethod("z_score"),
                                  preprocessing_p=kefrin.PreprocessingMethod(pp))
        return kefrin.KEFRiN(cfg).fit_predict(y, p_)
    return run

cfgs = [("KEFRiNe", kef(DM.EUCLIDEAN, "none"), None), ("KEFRiNc", kef(DM.COSINE, "none"), None),
        ("KEFRiNm", kef(DM.MANHATTAN, "none"), None),
        ("KEFRiNe (сеть z)", kef(DM.EUCLIDEAN, "z_score"), None),
        ("KEFRiNc (сеть z)", kef(DM.COSINE, "z_score"), None)]
import yaml; from pathlib import Path
if Path("configs/baselines_tuned.yaml").exists():       # подобранный вариант (tune_baselines.py): тот же метод с параметрами, лучшими на валидационном месяце
    t_ = yaml.safe_load(open("configs/baselines_tuned.yaml")).get("KEFRiN")
    if t_: cfgs.append((f"KEFRiN (подобран: {t_['конфигурация']})", kef(DM(t_["metric"]), "none", t_.get("rho", 1.0), t_.get("xi", 1.0)), None))
rows = []
for name, f, pp in cfgs:
    t0 = time.time()
    run = lambda ys, As_, t: f(ys[t], P(As_[t]), K)
    L = [run(Xt, A, 0)]
    for t in range(1, len(Xt)): L.append(align(L[-1], run(Xt, A, t)))
    stab = np.mean([ARI(L[-1], run(xs, As_, len(Xt) - 1)) for xs, As_ in noise])
    rows.append(row(name, L, stab, Xt, A, reg)); print(name, f"{time.time()-t0:.0f}s", flush=True)
    base = pd.read_csv("data/processed/compare_all.csv"); base = base[~base.метод.isin([r["метод"] for r in rows])]
    pd.concat([base, pd.DataFrame(rows)]).to_csv("data/processed/compare_all.csv", index=False)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(3).to_string(index=False))
