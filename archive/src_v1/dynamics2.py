"""Динамика типов (заголовочный метод, оба уровня K): матрица переходов, время пребывания, поток «начало→конец» (Sankey),
тест пространственной диффузии по дорожным соседям (шоссе, k=6)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, plotly.graph_objects as go
from config import CFG, SEED
from features import build
from headline import labels
from names import names
rng = np.random.default_rng(SEED)
d = build(); ids = np.array(d["ids"]); months = d["months"][CFG["features"]["trim_months"]:]; N = len(ids)
c = pd.read_parquet("data/raw/hack/hackathonlicence/connection.parquet"); c = c[c.type == "highway"]
idx = {int(t): i for i, t in enumerate(ids)}; c = c[c.territory_id_x.isin(idx) & c.territory_id_y.isin(idx) & (c.territory_id_x != c.territory_id_y)]
D = np.full((N, N), np.inf); D[c.territory_id_x.map(idx).values, c.territory_id_y.map(idx).values] = c.distance.values; D = np.minimum(D, D.T)
nb = np.argsort(D, 1)[:, :6]; has = np.isfinite(D[np.arange(N)[:, None], nb]).all(1)
res = []; smk = []
for K in CFG["headline"]["Ks"]:
    L = labels(K); T = L.shape[0]; nm = names(K); types = list(range(K))
    M = sum(pd.crosstab(L[t], L[t + 1]).reindex(index=types, columns=types, fill_value=0).values for t in range(T - 1)); P = M / np.maximum(M.sum(1, keepdims=True), 1)
    Pdf = pd.DataFrame(P, index=[f"{t} {nm[t]}" for t in types], columns=types).round(4); Pdf.to_csv(f"data/processed/transition_month_K{K}.csv")
    print(f"\n=== K={K}: вероятности перехода за месяц"); print(Pdf.to_string())
    # время пребывания 1/(1−P_tt) цензурируется периодом наблюдения: при P_tt → 1 оценка не отличима от «не менялся»
    dwell_num = [min(1 / max(1 - P[t, t], 1e-12), T) for t in types]
    dwell = {nm[t]: (f">{T}" if 1 - P[t, t] <= 1 / T else round(1 / (1 - P[t, t]), 1)) for t in types}
    print(f"время пребывания, мес. (цензура на {T} мес. наблюдения):", dwell)
    print(f"МО, ни разу не менявшие тип: {np.mean([len(set(L[:, i])) == 1 for i in range(N)]):.1%} | сменили тип к концу периода: {(L[0] != L[-1]).mean():.1%}")
    ct = pd.crosstab(L[0], L[-1]).reindex(index=types, columns=types, fill_value=0); ct.to_csv(f"data/processed/flow_start_end_K{K}.csv")
    PALS = ["#5b8c6a", "#d98e32", "#4f7cb8", "#6a4fa3", "#b0413e", "#c27ba0"]   # те же цвета, что в dashboard.html
    rgba = lambda h, a: f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"
    labs = [f"{nm[t]} · {ct.loc[t].sum()}" for t in types] + [f"{nm[t]} · {ct[t].sum()}" for t in types]
    src, tgt, val, col = [], [], [], []
    for a in types:
        for b in types:
            if ct.loc[a, b] >= 3:                 # потоки меньше 3 МО не рисуем; смены цветные, «остались» бледно-серые
                src.append(a); tgt.append(K + b); val.append(int(ct.loc[a, b])); col.append("rgba(150,150,150,0.25)" if a == b else rgba(PALS[a], 0.65))
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=labs, pad=22, thickness=18, color=[PALS[t] for t in types] * 2, line=dict(width=0)),
                              link=dict(source=src, target=tgt, value=val, color=col)))
    fig.update_layout(title=f"Потоки между типами, K={K}: слева {months[0]}, справа {months[-1]} (цветом — смены типа, серым — остались)", font=dict(size=13), height=620)
    fig.write_html(f"outputs/sankey_K{K}.html")
    obs, nul, nev = [], [], 0
    for t in range(T - 1):
        for i in np.where((L[t] != L[t + 1]) & has)[0]:
            a, b = L[t, i], L[t + 1, i]
            if t + 4 <= T and not (L[t + 1:t + 4, i] == b).all(): continue          # только устойчивые переходы (≥3 мес.)
            pool = np.where((L[t] == a) & has)[0]; j = rng.choice(pool, 200)
            obs.append(np.mean(L[t, nb[i]] == b)); nul.append(np.mean([np.mean(L[t, nb[jj]] == b) for jj in j])); nev += 1
    obs, nul = np.array(obs), np.array(nul); dif = obs.mean() - nul.mean(); z = dif / ((obs - nul).std() / np.sqrt(len(obs))) if nev > 2 else np.nan
    print(f"диффузия: устойчивых переходов {nev}; доля соседей уже в новом типе у переходящих {obs.mean():.3f} против {nul.mean():.3f}; разница {dif:+.3f} (z={z:.1f})")
    # Spatial Markov: вероятность смены типа в зависимости от того, совпадает ли тип МО с модальным типом дорожных соседей
    ag_n = ag_c = dg_n = dg_c = to_maj = 0
    for t in range(T - 1):
        cnt = np.stack([(L[t, nb] == k).sum(1) for k in types], 1); maj = cnt.argmax(1)
        for i in np.where(has)[0]:
            ch = L[t + 1, i] != L[t, i]
            if maj[i] == L[t, i]: ag_n += 1; ag_c += ch
            else:
                dg_n += 1; dg_c += ch
                if ch and L[t + 1, i] == maj[i]: to_maj += 1
    p_ag, p_dg = ag_c / ag_n, dg_c / dg_n; pp = (ag_c + dg_c) / (ag_n + dg_n); zsm = (p_dg - p_ag) / np.sqrt(pp * (1 - pp) * (1 / ag_n + 1 / dg_n))
    print(f"Spatial Markov: P(смена | тип совпадает с типом соседей) = {p_ag:.4f}, P(смена | не совпадает) = {p_dg:.4f} (в {p_dg / max(p_ag, 1e-12):.1f} раза чаще, z={zsm:.1f}); из смен у несовпадающих {to_maj / max(dg_c, 1):.0%} идут в тип соседей")
    smk.append(dict(K=K, P_смена_если_совпадает=p_ag, P_смена_если_не_совпадает=p_dg, отношение=p_dg / max(p_ag, 1e-12), z=zsm, доля_смен_в_тип_соседей=to_maj / max(dg_c, 1), n_совпадает=ag_n, n_не_совпадает=dg_n))
    res.append(dict(K=K, событий=nev, доля_соседей_переходящих=obs.mean(), доля_случайных=nul.mean(), z=z,
                    без_смены=np.mean([len(set(L[:, i])) == 1 for i in range(N)]),
                    churn_мес=float(np.mean([(L[t] != L[t + 1]).mean() for t in range(T - 1)])),
                    пребывание_мин_мес=min(dwell_num), пребывание_цензура=max(dwell_num) >= T))
pd.DataFrame(res).round(3).to_csv("data/processed/diffusion_test.csv", index=False)
pd.DataFrame(smk).round(4).to_csv("data/processed/spatial_markov.csv", index=False)
