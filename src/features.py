"""Признаки МО.
Расходы СберИндекса — композиция из 6 частей (5 категорий + «прочее»), переводится в CLR (Айчисон) → евклидова геометрия корректна.
Динамические дескрипторы (волатильность, сезонная амплитуда, тренд маркетплейсов) — «ext».
Структура экономики Росстата — «ros». Режимы (lens): spend | behavioral (spend + ext) | combined (spend + ext + ros)."""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

PARTS = ["health", "catering", "food", "market", "transport"]
PARTS_ALL = PARTS + ["other"]
PARTS_RU = {"health": "здоровье", "catering": "общепит", "food": "продовольствие", "market": "маркетплейсы",
            "transport": "транспорт", "other": "прочее"}   # подписи для таблиц и графиков
GROUPS = {"emp_primary": ["agri", "mining"], "emp_industry": ["manuf", "energy", "water", "constr"],
          "emp_trade": ["trade", "hotel"], "emp_transport": ["transp"],
          "emp_services": ["ict", "fin", "realty", "science", "admin", "other"],
          "emp_public": ["gov", "edu", "health", "culture"]}


def load(path="data/processed/panel.parquet"):
    return pd.read_parquet(path)


def clr(shares):
    lg = np.log(shares)
    return lg - lg.mean(-1, keepdims=True)


def spend_tensor(p):
    """(ids, months, cols, T×N×F, shares T×N×6). cols = log_total + 6 CLR-координат. Пропуски месяцев — интерполяция."""
    p = p.copy()
    p["other"] = (p.total - p[PARTS].sum(axis=1)).clip(lower=1.0)
    months, ids = sorted(p.month.unique()), sorted(p.tid.unique())
    mi, ii = {m: i for i, m in enumerate(months)}, {o: i for i, o in enumerate(ids)}
    base = ["total"] + PARTS_ALL
    arr = np.full((len(months), len(ids), len(base)), np.nan)
    arr[p.month.map(mi), p.tid.map(ii)] = p[base].values
    for j in range(len(ids)):                       # интерполяция по времени внутри МО
        for f in range(len(base)):
            arr[:, j, f] = pd.Series(arr[:, j, f]).interpolate(limit_direction="both").values
    parts = arr[:, :, 1:]
    shares = parts / parts.sum(-1, keepdims=True)
    cols = ["log_total"] + ["clr_" + c for c in PARTS_ALL]
    return ids, months, cols, np.concatenate([np.log(arr[:, :, :1]), clr(shares)], axis=-1), shares


def ext_features(T, shares):
    """Динамические дескрипторы по ряду каждого МО: волатильность, сезонная амплитуда, тренд маркетплейсов (CLR)."""
    t = np.arange(T.shape[0])
    lt = T[:, :, 0]
    slope = lambda y: np.polyfit(t, y, 1)[0]
    trend = np.array([slope(lt[:, j]) for j in range(lt.shape[1])])
    resid = lt - (trend[None] * (t[:, None] - t.mean()) + lt.mean(0)[None])
    vol = resid.std(0)
    n12 = T.shape[0] // 2
    season = np.stack([lt[m::12].mean(0) - lt.mean(0) for m in range(12) if m < T.shape[0]])
    amp = season.max(0) - season.min(0)
    mk = T[:, :, 1 + PARTS_ALL.index("market")]
    mtrend = np.array([slope(mk[:, j]) for j in range(mk.shape[1])])
    return pd.DataFrame({"volatility": vol, "season_amp": amp, "trend_level": trend, "trend_market": mtrend})


def ros_features(p, ids):
    s = p.drop_duplicates("tid").set_index("tid").reindex(ids)
    out = pd.DataFrame(index=ids)
    out["log_pop"] = np.log(s["pop"])
    out["log_wage"] = np.log(s["wage"])
    for g, secs in GROUPS.items():
        out[g] = s[["emp_" + x for x in secs]].sum(axis=1, min_count=1)
    return out, out.isna().any(axis=1)


def _z(df):
    from config import CFG
    c = CFG['features']['clip_z']
    return ((df - df.mean()) / df.std()).clip(-c, c)


def complete_only(p, min_months=None):
    """Оставляет МО, у которых есть данные минимум за min_months месяцев (по config: все 24).
    Раньше МО с пропусками дорисовывались интерполяцией с constant-экстраполяцией по краям; у 59 МО было меньше 12 месяцев,
    у 6 — один, то есть значительная часть ряда была выдумкой. Исключённые МО перечислены в data/processed/excluded_incomplete.csv."""
    from config import CFG
    k = CFG["data"]["min_months"] if min_months is None else min_months
    n = p.groupby("tid").month.nunique()
    return p[p.tid.isin(n[n >= k].index)]


def build(path="data/processed/panel.parquet"):
    p = complete_only(load(path))
    ids, months, cols, T, shares = spend_tensor(p)
    mean_sp = pd.DataFrame(T.mean(0), index=ids, columns=cols)
    ext = pd.DataFrame(ext_features(T, shares).values, index=ids, columns=["volatility", "season_amp", "trend_level", "trend_market"])
    ros, miss = ros_features(p, ids)
    full = pd.concat([mean_sp, ext, ros], axis=1)
    imp = pd.DataFrame(KNNImputer(n_neighbors=10).fit_transform(_z(full)), index=ids, columns=full.columns)
    ns, ne = mean_sp.shape[1], ext.shape[1]
    meta = p.drop_duplicates("tid").set_index("tid").reindex(ids)[
        ["municipal_district_name", "region_name", "municipal_district_center_lat", "municipal_district_center_lon", "market_access"]]
    return dict(ids=ids, months=months, cols=cols, tensor=T, shares=shares, X=imp, rosstat_missing=miss, meta=meta,
                n_spend=ns, n_ext=ne, ext_cols=list(ext.columns), ros_cols=list(ros.columns))


def static_block(d, lens):
    """Статические столбцы, добавляемые к месячным расходам: spend → ничего, behavioral → ext, combined → ext + ros."""
    X, a, b = d["X"], d["n_spend"], d["n_spend"] + d["n_ext"]
    return {"spend": X.iloc[:, :0], "behavioral": X.iloc[:, a:b], "combined": X.iloc[:, a:]}[lens].values


if __name__ == "__main__":
    d = build()
    print("МО:", len(d["ids"]), "| месяцев:", len(d["months"]), "| матрица:", d["X"].shape, "| с импутацией Росстата:", int(d["rosstat_missing"].sum()))
    print(d["X"].describe().loc[["mean", "std", "min", "max"]].round(2).T.to_string())
