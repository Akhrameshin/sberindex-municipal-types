import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from features import build
from pipeline import *
from config import CFG
TG = int(sys.argv[1]) if len(sys.argv) > 1 else CFG['temporal']['headline_target']
d = build(); Xt, months = month_features(d, trim=2); G = [to_ig(graph(x)) for x in Xt]
from canon import canonicalize
L0 = np.load(f"data/processed/L_k{TG}_om0.2.npy"); L = canonicalize(L0); np.save(f"data/processed/L_k{TG}_om0.2_canon.npy", L); T, N = L.shape
print("меток до/после выравнивания:", len(set(L0.ravel())), len(set(L.ravel())), "| среднемесячная смена до:", round(np.mean([(L0[t]!=L0[t+1]).mean() for t in range(T-1)]),3))
gam = float(pd.read_csv("data/processed/grid.csv").query(f"target=={TG} and omega==0.2").gamma.iloc[0])
ch = np.array([(L[t] != L[t+1]).mean() for t in range(T-1)])
# устойчивые переходы: новая метка держится ≥3 мес. подряд (или до конца ряда)
dur = blip = 0
for t in range(T-1):
    mv = np.where(L[t] != L[t+1])[0]
    for i in mv:
        end = min(t+4, T); hold = (L[t+1:end, i] == L[t+1, i]).all()
        if hold and (end - (t+1) >= 3 or end == T): dur += 1
        else: blip += 1
print(f"всего смен {dur+blip}; устойчивых (≥3 мес) {dur} ({dur/(dur+blip):.0%}); мигающих {blip}")
# сравнение с шумом: две реплики на зашумлённых признаках, расхождение по месяцам после сопоставления меток
def align_to(ref, cur):
    P, C = sorted(set(ref)), sorted(set(cur)); Mx = np.array([[((cur == c) & (ref == p)).sum() for p in P] for c in C])
    r, c = linear_sum_assignment(-Mx); mp = {C[i]: P[j] for i, j in zip(r, c)}; return np.array([mp.get(v, -1) for v in cur])
rng = np.random.default_rng(5); xs = [x + rng.normal(0, 0.15, x.shape) for x in Xt]
Ln = temporal([to_ig(graph(x)) for x in xs], 0.2, gam, seed=3)
dis = np.mean([(align_to(L[t], Ln[t]) != L[t]).mean() for t in range(T)])
print(f"расхождение с зашумлённой репликой (шум 0.15σ): {dis:.3f} vs среднемесячная смена {ch.mean():.3f}")
# общий итог начало → конец и события между тремя срезами
idx = [0, 9, T-1]
def jac(a, b, la, lb): A, B = set(np.where(a == la)[0]), set(np.where(b == lb)[0]); return len(A & B) / len(A | B)
ev = []
for s, (i, j) in enumerate(zip(idx[:-1], idx[1:])):
    a, b = L[i], L[j]; la_, lb_ = sorted(set(a)), sorted(set(b))
    for x in lb_:
        src = [(y, jac(a, b, y, x)) for y in la_ if jac(a, b, y, x) >= 0.2]
        if not src: ev.append((months[j], "рождение", x, "", ""))
        elif len(src) > 1: ev.append((months[j], "слияние", x, ",".join(str(y) for y, _ in src), ""))
    for y in la_:
        dst = [(x, jac(a, b, y, x)) for x in lb_ if jac(a, b, y, x) >= 0.2]
        if not dst: ev.append((months[j], "исчезновение", y, "", ""))
        elif len(dst) > 1: ev.append((months[j], "раскол", y, ",".join(str(x) for x, _ in dst), ""))
print(pd.DataFrame(ev, columns=["к месяцу", "событие", "кластер", "связанные", ""]).drop(columns="").to_string(index=False))
import plotly.graph_objects as go
labels, src, tgt, val, nid = [], [], [], [], {}
for s, t in enumerate(idx):
    for c in sorted(set(L[t])): nid[(s, c)] = len(labels); labels.append(f"{months[t]} · кл.{c} ({(L[t]==c).sum()})")
for s in range(len(idx)-1):
    ct = pd.crosstab(L[idx[s]], L[idx[s+1]])
    for a in ct.index:
        for b in ct.columns:
            if ct.loc[a, b] >= 5: src.append(nid[(s, a)]); tgt.append(nid[(s+1, b)]); val.append(int(ct.loc[a, b]))
go.Figure(go.Sankey(node=dict(label=labels, pad=15), link=dict(source=src, target=tgt, value=val))).write_html(f"outputs/sankey_k{TG}.html")
print("sankey.html записан;", T, "месяцев; срезы:", [months[i] for i in idx])
