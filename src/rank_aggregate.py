"""Агрегирование ранжировок методов по индексам качества: пороговое агрегирование (в духе Алескерова) и Борда.

Пороговое агрегирование: по каждому критерию метод получает оценку 1/2/3 по месту в таблице (первая, средняя и последняя треть мест).
Методы упорядочиваются по числу худших оценок (меньше — лучше), при равенстве по числу лучших (больше — лучше), затем по Борда.
Это наше прочтение опубликованного описания процедуры; пороги — трети мест, как в описании.
Борда: сумма мест по критериям (меньше — лучше).

Варианты: набор критериев (шесть индексов конкурса SW, CH, S_Dbw, AVI, AVU, MQ; те же без AVU; те же плюс динамические churn и ARI при шуме)
× способ агрегирования. По каждому методу считается худшее место по всем вариантам и доля вариантов, где он в первой двойке:
устойчивость вывода к выбору процедуры важнее победы в одной из них.
Оговорка: MQ — вариант Манкоридиса и по построению близок к K·AVI (семейство AVI), поэтому в наборе без MQ-дубля см. колонки *_без_MQ."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd

DIR = {"SW": 1, "CH": 1, "S_Dbw": -1, "AVI": 1, "AVU": -1, "MQ": 1, "churn": -1, "stab": 1}
SETS = {"6 индексов": ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ"], "без AVU": ["SW", "CH", "S_Dbw", "AVI", "MQ"],
        "без MQ": ["SW", "CH", "S_Dbw", "AVI", "AVU"], "6 индексов + динамика": ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "churn", "stab"]}


def places(c, cols):
    return pd.DataFrame({k: (c[k] * DIR[k]).rank(ascending=False, method="min") for k in cols})


def grades(P):
    n = len(P); return pd.DataFrame({k: np.where(P[k] <= np.ceil(n / 3), 1, np.where(P[k] <= np.ceil(2 * n / 3), 2, 3)) for k in P}, index=P.index)


def threshold_rank(c, cols):
    P = places(c, cols); G = grades(P); n3 = (G == 3).sum(axis=1); n1 = (G == 1).sum(axis=1); b = P.sum(axis=1)
    order = pd.DataFrame({"n3": n3, "n1": -n1, "b": b}).sort_values(["n3", "n1", "b"]).index
    return pd.Series(range(1, len(order) + 1), index=order).reindex(c.index), G


def main():
    c = pd.read_csv("data/processed/compare_ranked.csv").set_index("метод")
    out = pd.DataFrame(index=c.index); out["K"] = c["K"].round(1)
    for nm, cols in SETS.items():
        t, G = threshold_rank(c, cols); out[f"порог: {nm}"] = t
        out[f"Борда: {nm}"] = places(c, cols).sum(axis=1).rank(method="min").astype(int)
        out[f"оценки: {nm}"] = G.astype(str).agg("".join, axis=1)
    pl = [x for x in out.columns if x.startswith(("порог:", "Борда:"))]
    out["худшее_место"] = out[pl].max(axis=1); out["лучшее_место"] = out[pl].min(axis=1); out["доля_в_первой_двойке"] = (out[pl] <= 2).mean(axis=1)
    out = out.sort_values(["худшее_место", "доля_в_первой_двойке"], ascending=[True, False])
    out.to_csv("data/processed/method_ranking.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out[["K", *pl, "худшее_место", "доля_в_первой_двойке"]].to_string())


def main_ci(n=2000, seed=0):
    """Доверительные интервалы на места: на каждом из n повторов каждый метод получает случайный сид из compare_seeds_raw.csv, места считаются обоими
    способами агрегирования по наборам без динамики (churn есть не у всех методов). Интервал 5–95% — разброс места из-за инициализации."""
    raw = pd.read_csv("data/processed/compare_seeds_raw.csv"); rng = np.random.default_rng(seed); groups = {m: g.reset_index(drop=True) for m, g in raw.groupby("метод", sort=False)}
    sets = {k: v for k, v in SETS.items() if "динамика" not in k}; rec = {(m, k, a): [] for m in groups for k in sets for a in ("порог", "Борда")}
    for _ in range(n):
        c = pd.DataFrame([g.iloc[rng.integers(len(g))] for g in groups.values()]).set_index("метод")
        for k, cols in sets.items():
            t, _ = threshold_rank(c, cols); b = places(c, cols).sum(axis=1).rank(method="min")
            for m in groups: rec[(m, k, "порог")].append(t[m]); rec[(m, k, "Борда")].append(b[m])
    out = pd.DataFrame([dict(метод=m, набор=k, агрегирование=a, место_медиана=float(np.median(v)), место_p05=float(np.percentile(v, 5)), место_p95=float(np.percentile(v, 95)))
                        for (m, k, a), v in rec.items()])
    out.to_csv("data/processed/method_ranking_ci.csv", index=False)
    pd.set_option("display.width", 250); print(out.pivot_table(index="метод", columns=["набор", "агрегирование"], values="место_медиана").round(1).to_string())


if __name__ == "__main__":
    main_ci() if "ci" in sys.argv[1:] else main()
