"""CANUS (Shalileh, 2025, IEEE Access) — метод жюри: k-means-подобный метод для сетей с атрибутами с градиентным обновлением центров
и фильтрацией вкладов по норме градиента (правило vanilla_filtered, как в статье). Код авторов (external/CANUS/canus.py, закреплён по SHA-256).
Протокол тот же, что у KEFRiN в run_kefrin.py: те же признаки, тот же граф (бинарная смежность), K из config, согласование меток по месяцам,
устойчивость — ARI последнего месяца при шуме 0.15σ в признаках. Параметры по умолчанию авторов: 300 эпох, шаг 1e-3, τ=1, ρ=ζ=1, косинус по признакам и по строкам сети."""
import sys, time; sys.path.insert(0, "src"); sys.path.insert(0, "external/CANUS")
import numpy as np, pandas as pd, torch, yaml
from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ARI
from features import build
from pipeline import month_features, graph
from evalkit import row, align
from config import CFG, SEED
from canus import CANUSClusterer

K = CFG["compare"]["k"]; d = build(); Xt, months = month_features(d, trim=2); reg = d["meta"].region_name.fillna("?").values
A = [graph(x) for x in Xt]; P = lambda a: (a.toarray() > 0).astype(float)


DEFAULT = dict(lr=1e-3, epochs=300, tau=1.0, rho=1.0, zeta=1.0, rule="vanilla_filtered")
VARIANTS = [("CANUS", DEFAULT)]
if Path("configs/baselines_tuned.yaml").exists():       # подобранный вариант (tune_baselines.py)
    t_ = yaml.safe_load(open("configs/baselines_tuned.yaml")).get("CANUS")
    if t_: VARIANTS.append((f"CANUS (подобран: {t_['конфигурация']})", {**DEFAULT, **{k: t_[k] for k in DEFAULT if k in t_}, "epochs": int(t_.get("epochs", 300))}))


def run(x, a, c, seed=SEED):
    # Параметр seed авторов задаёт только инициализацию центров. Бутстрэп полосы фильтрации у них идёт через torch.randint БЕЗ генератора,
    # то есть на глобальном состоянии torch: два запуска с одним seed давали разные разбиения. Файл авторов закреплён по SHA-256 и не правится,
    # поэтому глобальное состояние мы засеваем сами перед каждым запуском (так же, как в run_kefrin.py для KEFRiN).
    torch.manual_seed(seed)
    return CANUSClusterer(n_clusters=K, seed=seed, epochs=c["epochs"], learning_rate=c["lr"], tau=c["tau"], rho=c["rho"], zeta=c["zeta"],
                          update_rule=c["rule"], attribute_distance="cosine", network_distance="cosine", device="cpu").fit(x, P(a)).y_pred


def noisy(seed, s=CFG["compare"]["noise_sigma"]):
    r = np.random.default_rng(seed); xs = [x + r.normal(0, s, x.shape) for x in Xt]; return xs, [graph(x) for x in xs]


noise = [noisy(s) for s in (1, 2, 3)]
for name, c in VARIANTS:
    t0 = time.time(); L = [run(Xt[0], A[0], c)]
    for t in range(1, len(Xt)):
        L.append(align(L[-1], run(Xt[t], A[t], c))); print(f"{name}: месяц {t + 1}/{len(Xt)}  {time.time() - t0:.0f}с", flush=True)
    stab = np.mean([ARI(L[-1], run(xs[-1], As[-1], c)) for xs, As in noise])
    r = row(name, L, stab, Xt, A, reg); print(f"{name}: churn {r['churn']:.3f}  ARI при шуме {stab:.3f}  SW {r['SW']:.3f}  AVI {r['AVI']:.3f}  ({time.time() - t0:.0f}с)")
    base = pd.read_csv("data/processed/compare_all.csv"); base = base[base.метод != name]
    pd.concat([base, pd.DataFrame([r])]).to_csv("data/processed/compare_all.csv", index=False)
