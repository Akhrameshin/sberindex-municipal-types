"""Описание типов формальными понятиями (FCA, в духе Кузнецова) с индексом устойчивости понятия.

Контекст: объекты — МО, признаки — «верхний квартиль» и «нижний квартиль» каждого из профильных показателей (доли трат, уровень, зарплата, занятость).
Для каждого типа ищется понятие (замкнутое множество признаков и все МО, которые им обладают), лучше всего описывающее тип: поиск лучом по F1
(точность × полнота относительно типа), размер описания не больше MAX_LEN. Описание «тип X — это МО с высокой долей… и низкой…» читается без знания метода.
Устойчивость по Кузнецову (доля подмножеств экстента, дающих тот же интент) считается, но на экстентах в сотни МО она насыщается до 1: любое случайное
подмножество такого размера даёт тот же интент, и индекс ничего не различает. Поэтому главная мера устойчивости здесь другая: доля 80%-подвыборок МО,
на которых поиск заново находит ТО ЖЕ описание (общие подвыборки evalkit.subsamples)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from config import CFG, SEED
from features import build, load, PARTS_ALL, PARTS_RU
from headline import labels
from names import names
from evalkit import subsamples

MAX_LEN, BEAM, MC = 4, 12, 4000
KK = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["headline"]["Ks"][0]
d = build(); ids = d["ids"]; T = d["tensor"]; sh = d["shares"]; p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
F = pd.DataFrame(index=ids)
for i, c in enumerate(PARTS_ALL): F[f"доля {PARTS_RU[c]}"] = sh[:, :, i].mean(0)
F["расходы на душу"] = np.expm1(T[:, :, 0]).mean(0); F["население"] = p["pop"]; F["зарплата"] = p["wage"]
for g, nm in (("agri", "с/х"), ("mining", "добыча"), ("manuf", "обработка"), ("trade", "торговля"), ("gov", "госуправл."), ("edu", "образование"), ("ict", "ИКТ")): F[f"занятость {nm}"] = p["emp_" + g]
F = F.fillna(F.median())
q1, q3 = F.quantile(.25), F.quantile(.75)
items, cols = [], []
for c in F:
    items.append(f"{c} высокая"); cols.append((F[c] >= q3[c]).values)
    items.append(f"{c} низкая"); cols.append((F[c] <= q1[c]).values)
M = np.stack(cols, 1)                                            # объекты × признаки (булева матрица контекста)
lab = labels(KK)[-1]; NM = names(KK); rng = np.random.default_rng(SEED)


def extent(S): return M[:, list(S)].all(1) if S else np.ones(len(M), bool)
def intent(ext): return set(np.where(M[ext].all(0))[0]) if ext.any() else set()


def f1(ext, tgt):
    tp = (ext & tgt).sum(); return 0.0 if tp == 0 else 2 * tp / (ext.sum() + tgt.sum())


def stability(ext, intn):
    """Оценка Монте-Карло устойчивости по Кузнецову: P(пересечение интентов случайного подмножества экстента == интент)."""
    idx = np.where(ext)[0]; hit = 0
    for _ in range(MC):
        sub = idx[rng.random(len(idx)) < .5]
        if len(sub) and set(np.where(M[sub].all(0))[0]) == intn: hit += 1
    return hit / MC


def search(tgt, keep):
    """Луч по F1 на объектах keep; возвращает (лучший набор признаков, F1)."""
    Mk = M[keep]; tk = tgt[keep]; beam = [((), 0.0)]; best = ((), 0.0)
    for _ in range(MAX_LEN):
        cand = {}
        for S, _ in beam:
            for j in range(Mk.shape[1]):
                if j in S: continue
                T2 = tuple(sorted(S + (j,))); e = Mk[:, list(T2)].all(1)
                if e.sum() < 5: continue
                tp = (e & tk).sum(); cand[T2] = 0.0 if tp == 0 else 2 * tp / (e.sum() + tk.sum())
        beam = sorted(cand.items(), key=lambda kv: -kv[1])[:BEAM]
        if beam and beam[0][1] > best[1]: best = beam[0]
    return best


SUB = subsamples(len(M), reps=20, frac=0.8); allobj = np.arange(len(M)); rows = []
for t in sorted(set(lab)):
    tgt = lab == t; S, bf1 = search(tgt, allobj); e = extent(S); intn = intent(e)
    again = np.mean([search(tgt, keep)[0] == S for keep in SUB])
    rows.append(dict(тип=f"{t}: {NM.get(t, '?')}", n_типа=int(tgt.sum()), описание=" И ".join(items[j] for j in sorted(S)),
                     МО_с_описанием=int(e.sum()), точность=float((e & tgt).sum() / max(e.sum(), 1)), полнота=float((e & tgt).sum() / tgt.sum()), F1=bf1,
                     найдено_заново_на_подвыборках=float(again), замкнутое_описание=" И ".join(items[j] for j in sorted(intn)),
                     устойчивость_по_Кузнецову_МК=stability(e, intn)))
    r = rows[-1]; print(f"{r['тип']:55} F1 {bf1:.2f}  точн {r['точность']:.2f}  полн {r['полнота']:.2f}  найдено заново {again:.0%}  | {r['описание']}", flush=True)
pd.DataFrame(rows).round(3).to_csv(f"data/processed/fca_K{KK}.csv", index=False)
