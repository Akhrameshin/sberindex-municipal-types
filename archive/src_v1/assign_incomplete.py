"""Назначение типов МО, исключённым из типологии из-за неполного ряда (features.complete_only), с калиброванной надёжностью.

Метод. Для каждого исключённого МО в каждом месяце, где есть данные (после 2 месяцев окна), строится вектор признаков так же, как у полных МО:
скользящее среднее по доступным месяцам окна, стандартизация ПО СТАТИСТИКАМ полной выборки того же месяца, Росстат — по статистикам полной выборки.
Графовое сглаживание, которое видят центры типов, воспроизводится по ближайшим полным МО (10 соседей по косинусу в доступных измерениях: самих признаков
и Росстата; динамические дескрипторы по неполному ряду не считаются и берутся у соседей). МО относится к ближайшему центру заголовочной типологии;
итоговый тип — модальный по месяцам, уверенность — доля месяцев в модальном типе.
Калибровка надёжности (главное): тот же алгоритм применяется к ПОЛНЫМ МО, у которых оставлен случайный непрерывный отрезок из m месяцев; точность —
совпадение с типом, который у них получился по полному ряду в те же месяцы. Так видно, сколько месяцев достаточно, и каждому исключённому МО
приписывается ожидаемая точность по числу его месяцев. Назначенные МО в статистику типов (устойчивость, динамика, валидация) не входят."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from config import CFG, SEED
from features import build, load, PARTS, PARTS_ALL, clr, ros_features
from pipeline import month_features
from headline import centroids
from names import names

TR = CFG["features"]["trim_months"]; WIN = CFG["features"]["smooth_window"]; CLIP = CFG["features"]["clip_z"]; KNB = 10
d = build(); ids = list(d["ids"]); p_all = load(); months = d["months"]; Tn = len(months)
inc_ids = sorted(set(p_all.tid.unique()) - set(ids)); p = p_all.copy(); p["other"] = (p.total - p[PARTS].sum(axis=1)).clip(lower=1.0)
mi = {m: i for i, m in enumerate(months)}; allids = ids + inc_ids; ii = {o: i for i, o in enumerate(allids)}


def raw_tensor():
    """T×N×7 (log_total + CLR) БЕЗ интерполяции: пропущенные месяцы остаются NaN."""
    base = ["total"] + PARTS_ALL; arr = np.full((Tn, len(allids), len(base)), np.nan)
    q = p[p.tid.isin(ii)]; arr[q.month.map(mi), q.tid.map(ii)] = q[base].values
    parts = arr[:, :, 1:]; sh = parts / parts.sum(-1, keepdims=True)
    return np.concatenate([np.log(arr[:, :, :1]), clr(sh)], axis=-1)


R = raw_tensor(); nM = len(ids)
assert np.allclose(R[:, :nM], d["tensor"], equal_nan=False), "сырой тензор полных МО не совпал с тензором конвейера"
# статистики полной выборки: те же, что в month_features (скользящее по 3 месяца с полным окном, стандартизация внутри месяца)
sm_main = pd.DataFrame(R[:, :nM].reshape(Tn, -1)).rolling(WIN, min_periods=WIN).mean().values.reshape(Tn, nM, -1)
mu = sm_main.mean(1); sd = sm_main.std(1, ddof=1)       # np.std в month_features без ddof; приводим к тому же
sd0 = sm_main.std(1)
ros_main, _ = ros_features(p_all, ids); rmu = ros_main.mean(); rsd = ros_main.std()
Xt, _ = month_features(d); K_LIST = CFG["headline"]["Ks"]


def feats_partial(sub, ros_z):
    """sub: T×7 сырые ряды МО (NaN где нет данных); возвращает T×19 (NaN в недоступных измерениях) в тех же осях, что месячные признаки полной выборки."""
    sm = pd.DataFrame(sub).rolling(WIN, min_periods=1).mean().values
    z = np.clip((sm - mu) / sd0, -CLIP, CLIP); out = np.full((Tn, 19), np.nan); out[:, :7] = z; out[:, 11:] = ros_z[None]
    return out


def assign(sub, ros_z, C, nn_cache, exclude=None):
    """Тип по каждому доступному месяцу (t ≥ TR): ближайший центр для вектора, усреднённого с 10 ближайшими полными МО (воспроизводит графовое сглаживание)."""
    X = feats_partial(sub, ros_z); labs = {}
    for t in range(TR, Tn):
        if np.isnan(sub[t]).any(): continue
        x = X[t]; avail = ~np.isnan(x); Xm = Xt[t - TR]; xs = np.nan_to_num(x, nan=0.0)
        a, b = xs[avail], Xm[:, avail]; sim = (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-12)
        if exclude is not None: sim[exclude] = -np.inf
        nb = np.argsort(-sim)[:KNB]; stack = np.vstack([x[None], Xm[nb]]); xt = np.nanmean(stack, axis=0)
        labs[t] = int(((C - xt[None]) ** 2).sum(1).argmin())
    return labs


def mode_of(labs):
    if not labs: return None, 0.0, 0
    v = np.array(list(labs.values())); m = np.bincount(v).argmax(); return int(m), float((v == m).mean()), len(v)


rng = np.random.default_rng(SEED); res = {}
for K in K_LIST:
    C, L = centroids(Xt, K)
    # --- калибровка на полных МО: непрерывные отрезки из m месяцев
    cal = []; pick = rng.choice(nM, 300, replace=False)
    for m in (3, 4, 6, 9, 12, 18):
        ok = tot = 0
        for i in pick:
            st = int(rng.integers(0, Tn - m + 1)); sub = np.full((Tn, 7), np.nan); sub[st:st + m] = R[st:st + m, i]
            ros_z = np.clip(((ros_main.iloc[i] - rmu) / rsd).values, -CLIP, CLIP); ros_z = np.nan_to_num(ros_z, nan=0.0)
            labs = assign(sub, ros_z, C, None, exclude=i)
            if not labs: continue
            truth = np.array([L[t - TR, i] for t in labs]); mm, conf, n = mode_of(labs)
            tm = np.bincount(truth).argmax(); ok += int(mm == tm); tot += 1
        cal.append(dict(K=K, месяцев=m, точность=ok / max(tot, 1), n_проверок=tot))
    cal = pd.DataFrame(cal); cal.to_csv(f"data/processed/assign_calibration_K{K}.csv", index=False); print(f"K={K} калибровка:"); print(cal.round(3).to_string(index=False))
    # --- назначение исключённых МО
    inc = p_all[p_all.tid.isin(inc_ids)].drop_duplicates("tid").set_index("tid"); rz, _ = ros_features(p_all, inc_ids); rows = []; nm = names(K)
    for o in inc_ids:
        j = ii[o]; sub = R[:, j]; nmon = int((~np.isnan(sub).any(axis=1)).sum())
        rz_i = np.nan_to_num(np.clip(((rz.loc[o] - rmu) / rsd).values, -CLIP, CLIP), nan=0.0)
        mm, conf, n = mode_of(assign(sub, rz_i, C, None)) if nmon >= 3 else (None, 0.0, 0)
        acc = np.nan
        if mm is not None:
            c2 = cal.copy(); acc = float(np.interp(min(max(nmon, 3), 18), c2.месяцев, c2.точность))
        rows.append(dict(territory_id=o, МО=inc.loc[o, "municipal_district_name"], регион=inc.loc[o, "region_name"], месяцев_с_данными=nmon, назначено_по_месяцам=n,
                         тип=mm if mm is not None else -1, название_типа=nm[mm] if mm is not None else "не назначен (меньше 3 месяцев)", уверенность=conf, ожидаемая_точность=acc))
    out = pd.DataFrame(rows); out.round(3).to_csv(f"data/processed/assigned_incomplete_K{K}.csv", index=False)
    g = out[out["тип"] >= 0]; print(f"назначено {len(g)} из {len(out)}; не назначено {len(out) - len(g)}; ожидаемая точность в среднем {g['ожидаемая_точность'].mean():.2f}; типы:", g["тип"].value_counts().sort_index().to_dict())
