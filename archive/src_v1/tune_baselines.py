"""Справедливая настройка методов жюри (CANUS, KEFRiN): в таблице сравнения они стояли с параметрами по умолчанию, а наш метод — с подобранными.
Протокол. Один валидационный месяц (середина периода). Для каждого метода — сетка параметров из документации авторов (CANUS: шаг и число эпох градиента,
порог фильтра τ, веса ρ/ζ, правило обновления; KEFRiN: веса ρ/ξ и расстояние). По каждой конфигурации: SW, AVI, Q на этом месяце и ARI с запуском при шуме 0,15σ
(тот же протокол шума, что в таблице сравнения). Выбор: лучший средний ранг по этим четырём критериям.
Выбор делается В ПОЛЬЗУ базовых методов: критерии совпадают с теми, по которым их потом сравнивают, а валидационный месяц входит в оценку.
Это сознательная щедрость к конкурентам, а не подгонка под нас. Результаты — tune_baselines.csv, выбранные параметры — configs/baselines_tuned.yaml."""
import sys, time; sys.path.insert(0, "src"); sys.path.insert(0, "external/KEFRiN"); sys.path.insert(0, "external/CANUS")
import numpy as np, pandas as pd, torch, yaml, logging
from sklearn.metrics import adjusted_rand_score as ARI
import kefrin; logging.disable(logging.CRITICAL)
from canus import CANUSClusterer
from features import build
from pipeline import month_features, graph
from icvi import all_indices
from config import CFG, SEED

K = CFG["compare"]["k"]; d = build(); Xt, months = month_features(d, trim=2); tv = len(Xt) // 2
x0 = Xt[tv]; a0 = graph(x0); P0 = (a0.toarray() > 0).astype(float)
noise = []
for s in (1, 2):
    r = np.random.default_rng(s); xn = x0 + r.normal(0, CFG["compare"]["noise_sigma"], x0.shape); noise.append((xn, (graph(xn).toarray() > 0).astype(float)))

CANUS_GRID = [dict(name="по умолчанию", lr=1e-3, epochs=300, tau=1.0, rho=1.0, zeta=1.0, rule="vanilla_filtered"),
              dict(name="lr 1e-2", lr=1e-2, epochs=300, tau=1.0, rho=1.0, zeta=1.0, rule="vanilla_filtered"),
              dict(name="lr 1e-1", lr=1e-1, epochs=300, tau=1.0, rho=1.0, zeta=1.0, rule="vanilla_filtered"),
              dict(name="lr 1e-2, 1000 эпох", lr=1e-2, epochs=1000, tau=1.0, rho=1.0, zeta=1.0, rule="vanilla_filtered"),
              dict(name="lr 1e-2, τ=2", lr=1e-2, epochs=300, tau=2.0, rho=1.0, zeta=1.0, rule="vanilla_filtered"),
              dict(name="lr 1e-2, ρ:ζ=1:0.5", lr=1e-2, epochs=300, tau=1.0, rho=1.0, zeta=0.5, rule="vanilla_filtered"),
              dict(name="lr 1e-2, ρ:ζ=0.5:1", lr=1e-2, epochs=300, tau=1.0, rho=0.5, zeta=1.0, rule="vanilla_filtered"),
              dict(name="Adam lr 1e-2", lr=1e-2, epochs=300, tau=1.0, rho=1.0, zeta=1.0, rule="adam")]
KEF_GRID = [dict(name=f"{m} ρ={r} ξ={x}", metric=m, rho=r, xi=x) for m in ("euclidean", "cosine") for r, x in ((1, 1), (1, .5), (.5, 1), (1, 2), (2, 1))]


def run_canus(x, P, c, seed=SEED):
    torch.manual_seed(seed)       # бутстрэп полосы фильтрации у авторов идёт на глобальном состоянии torch (см. run_canus.py)
    return CANUSClusterer(n_clusters=K, seed=seed, epochs=c["epochs"], learning_rate=c["lr"], tau=c["tau"], rho=c["rho"], zeta=c["zeta"], update_rule=c["rule"],
                          attribute_distance="cosine", network_distance="cosine", device="cpu").fit(x, P).y_pred


def run_kef(x, P, c):
    DM = kefrin.DistanceMetric; cfg = kefrin.KEFRiNConfig(n_clusters=K, rho=c["rho"], xi=c["xi"], distance_metric=DM(c["metric"]) if not isinstance(c["metric"], DM) else c["metric"],
                                                         kmeans_plus_plus=True, random_state=SEED, preprocessing_y=kefrin.PreprocessingMethod("z_score"),
                                                         preprocessing_p=kefrin.PreprocessingMethod("none"))
    return kefrin.KEFRiN(cfg).fit_predict(x, P)


rows = []
for method, grid, fn in (("CANUS", CANUS_GRID, run_canus), ("KEFRiN", KEF_GRID, run_kef)):
    for c in grid:
        t0 = time.time(); lab = fn(x0, P0, c); ev = all_indices(x0, a0, lab)
        st = float(np.mean([ARI(lab, fn(xn, Pn, c)) for xn, Pn in noise]))
        rows.append(dict(метод=method, конфигурация=c["name"], SW=ev["SW"], AVI=ev["AVI"], Q=ev["Q"], ARI_при_шуме=st, размеры=np.bincount(lab, minlength=K).tolist(), секунд=round(time.time() - t0), **{k: v for k, v in c.items() if k != "name"}))
        print(method, c["name"], {k: round(v, 3) for k, v in rows[-1].items() if k in ("SW", "AVI", "Q", "ARI_при_шуме")}, rows[-1]["размеры"], f"{rows[-1]['секунд']}с", flush=True)
r = pd.DataFrame(rows)
r["средний_ранг"] = r.groupby("метод")[["SW", "AVI", "Q", "ARI_при_шуме"]].rank(ascending=False).mean(axis=1)
r["выбран"] = r.groupby("метод")["средний_ранг"].transform("min") == r["средний_ранг"]
r["по_умолчанию"] = (r.конфигурация == "по умолчанию") | r.конфигурация.str.contains("ρ=1 ξ=1") & r.конфигурация.str.startswith("cosine")
r.round(4).to_csv("data/processed/tune_baselines.csv", index=False)
sel = {}
for m, g in r.groupby("метод"):
    best = g[g["выбран"]].iloc[0]; sel[m] = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items() if k in ("lr", "epochs", "tau", "rho", "zeta", "rule", "metric", "xi", "конфигурация")}
    sel[m] = {k: v for k, v in sel[m].items() if not (isinstance(v, float) and np.isnan(v))}
yaml.safe_dump(sel, open("configs/baselines_tuned.yaml", "w"), allow_unicode=True, sort_keys=False)
print("\nВыбрано:", sel)
