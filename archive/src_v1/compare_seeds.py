"""Таблица сравнения методов с разбросом по сидам. В основной таблице (compare_all.csv) у каждого метода один запуск, и соседние места в рейтинге
могут быть шумом инициализации. Здесь каждый метод со случайностью запускается на SEEDS сидах; по каждому сиду считаются индексы конкурса
(среднее по месяцам CHECK) и, где метод дешёвый, churn по всем месяцам; затем — среднее, стандартное отклонение и согласие между сидами (ARI).
Методы без случайности (Ward, паттерны) идут как есть. CANUS и KEFRiN — в двух вариантах: по умолчанию и с подобранными параметрами (configs/baselines_tuned.yaml).
CANUS дорогой, поэтому для него считаются только месяцы CHECK (churn не считается)."""
import sys, time; sys.path.insert(0, "src"); sys.path.insert(0, "external/KEFRiN"); sys.path.insert(0, "external/CANUS")
import numpy as np, pandas as pd, torch, yaml, logging, itertools
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
import kefrin; logging.disable(logging.CRITICAL)
from canus import CANUSClusterer
from features import build
from pipeline import month_features, graph, to_ig, snap, calibrate
from icvi import all_indices
from evalkit import align, churn
from methods2 import dmon, pooled_kmeans_viterbi
from headline import fit
from config import CFG, SEED

K = CFG["compare"]["k"]; SEEDS = [SEED + i for i in range(5)]
d = build(); Xt, months = month_features(d, trim=2); T = len(Xt); CHECK = (0, T // 2, T - 1)
A = [graph(x) for x in Xt]; G = [to_ig(a) for a in A]; gam = calibrate(G[T // 2], K); P = lambda a: (a.toarray() > 0).astype(float)
TUNED = yaml.safe_load(open("configs/baselines_tuned.yaml")) if Path("configs/baselines_tuned.yaml").exists() else {}
DM = kefrin.DistanceMetric


def kef(metric, rho=1.0, xi=1.0):
    def run(t, seed):
        cfg = kefrin.KEFRiNConfig(n_clusters=K, rho=rho, xi=xi, distance_metric=DM(metric), kmeans_plus_plus=True, random_state=seed,
                                  preprocessing_y=kefrin.PreprocessingMethod("z_score"), preprocessing_p=kefrin.PreprocessingMethod("none"))
        return kefrin.KEFRiN(cfg).fit_predict(Xt[t], P(A[t]))
    return run


def canus(lr=1e-3, epochs=300, tau=1.0, rho=1.0, zeta=1.0, rule="vanilla_filtered"):
    def run(t, seed):
        torch.manual_seed(seed)      # см. run_canus.py: seed авторов не засевает бутстрэп фильтра
        return CANUSClusterer(n_clusters=K, seed=seed, epochs=epochs, learning_rate=lr, tau=tau, rho=rho, zeta=zeta, update_rule=rule,
                              attribute_distance="cosine", network_distance="cosine", device="cpu").fit(Xt[t], P(A[t])).y_pred
    return run


# (имя, функция запуска (месяц, сид) → метки, случайность есть?, считать ли все месяцы)
METHODS = [("k-means", lambda t, s: KMeans(K, n_init=5, random_state=s).fit_predict(Xt[t]), True, True),
           ("Ward", lambda t, s: AgglomerativeClustering(K).fit_predict(Xt[t]), False, True),
           ("GMM", lambda t, s: GaussianMixture(K, random_state=s, covariance_type="diag").fit_predict(Xt[t]), True, True),
           ("спектральный", lambda t, s: SpectralClustering(K, affinity="precomputed", random_state=s, assign_labels="cluster_qr").fit_predict(A[t]), True, True),
           ("Leiden (срезы)", lambda t, s: snap(G[t], gam, seed=s), True, True),
           ("DMoN (срезы)", lambda t, s: dmon(Xt[t], A[t], K, seed=s), True, True),
           ("KEFRiNe", kef("euclidean"), True, True), ("KEFRiNc", kef("cosine"), True, True), ("KEFRiNm", kef("manhattan"), True, True),
           ("CANUS", canus(), True, False)]
if "KEFRiN" in TUNED:
    t_ = TUNED["KEFRiN"]; METHODS.append((f"KEFRiN (подобран: {t_.get('конфигурация', '')})", kef(t_["metric"], t_.get("rho", 1.0), t_.get("xi", 1.0)), True, True))
if "CANUS" in TUNED:
    t_ = TUNED["CANUS"]; METHODS.append((f"CANUS (подобран: {t_.get('конфигурация', '')})", canus(t_.get("lr", 1e-3), int(t_.get("epochs", 300)), t_.get("tau", 1.0), t_.get("rho", 1.0), t_.get("zeta", 1.0), t_.get("rule", "vanilla_filtered")), True, False))

rows = []; seeds_lab = {}
for name, fn, stoch, allm in METHODS:
    ss = SEEDS if stoch else SEEDS[:1]; t0 = time.time()
    for s in ss:
        if allm:
            L = [fn(0, s)]
            for t in range(1, T): L.append(align(L[-1], fn(t, s)))
            lab = {t: L[t] for t in CHECK}; ch = churn(L)
        else:
            lab = {t: fn(t, s) for t in CHECK}; ch = np.nan
        ev = [all_indices(Xt[t], A[t], lab[t]) for t in CHECK]; seeds_lab.setdefault(name, []).append(lab[CHECK[-1]])
        rows.append(dict(метод=name, сид=s, churn=ch, **{k: float(np.mean([e[k] for e in ev])) for k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q")}))
        print(f"{name} сид {s}: SW {rows[-1]['SW']:.3f} AVI {rows[-1]['AVI']:.3f}  ({time.time() - t0:.0f}с)", flush=True)
    pd.DataFrame(rows).to_csv("data/processed/compare_seeds_raw.csv", index=False)
# заголовочный метод и «общий k-means + Витерби» (случайность — только в инициализации k-means); время согласовано по построению
for name, fn in (("GS-TKM (наш, K=5)", lambda s: fit(Xt, K, seed=s)[0]), ("k-means общий + Витерби λ=0.5", lambda s: pooled_kmeans_viterbi(Xt, K, CFG["headline"]["lam"], seed=s)[0])):
    for s in SEEDS:
        L = list(fn(s)); ev = [all_indices(Xt[t], A[t], L[t]) for t in CHECK]; seeds_lab.setdefault(name, []).append(L[CHECK[-1]])
        rows.append(dict(метод=name, сид=s, churn=churn(L), **{k: float(np.mean([e[k] for e in ev])) for k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q")}))
raw = pd.DataFrame(rows); raw.to_csv("data/processed/compare_seeds_raw.csv", index=False)
agg = raw.groupby("метод", sort=False).agg(**{f"{k}_ср": (k, "mean") for k in ("churn", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q")},
                                          **{f"{k}_сд": (k, "std") for k in ("churn", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q")}, сидов=("сид", "size")).reset_index()
agg["ARI_между_сидами"] = [np.mean([ARI(a, b) for a, b in itertools.combinations(seeds_lab[m], 2)]) if len(seeds_lab[m]) > 1 else 1.0 for m in agg.метод]
agg.round(4).to_csv("data/processed/compare_seeds.csv", index=False)
pd.set_option("display.width", 250); print(agg[["метод", "сидов", "SW_ср", "SW_сд", "AVI_ср", "AVI_сд", "Q_ср", "churn_ср", "ARI_между_сидами"]].round(3).to_string(index=False))
