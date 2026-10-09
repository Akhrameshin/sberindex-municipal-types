"""Автономная интерактивная панель «паспорт МО»: карта типов по месяцам + сеть сходства на карте + профиль и траектория выбранного МО.
Один HTML без сервера и без внешних запросов (без внешних библиотек: карта рисуется SVG). Данные берутся из тех же меток и графа, что и в отчёте.
Новое по сравнению с map_K*.html: клик по МО показывает его соседей по сети сходства (в том числе из других регионов) линиями на карте,
ленту его типов по 22 месяцам и профиль относительно типа; режим «только смены типа» гасит 96% МО, которые не меняли тип."""
import sys, json; sys.path.insert(0, "src")
import numpy as np, pandas as pd, geopandas as gpd
from config import CFG
from features import build, load
from headline import labels
from names import names
from pipeline import month_features, graph

d = build(); ids = d["ids"]; N = len(ids); months = list(d["months"][CFG["features"]["trim_months"]:]); T = len(months); meta = d["meta"]
poly = gpd.read_file("data/raw/dict/t_dict_municipal_districts_poly.gpkg").sort_values("year_to").drop_duplicates("territory_id", keep="last")
poly["territory_id"] = poly.territory_id.astype(str); poly = poly.set_index("territory_id").loc[[str(i) for i in ids]]
assert len(poly) == N
poly["geometry"] = poly.geometry.simplify(0.06, preserve_topology=True)
# Собственная проекция: Альберс с центром на 100° в. д.; долготы < 0 (Чукотка за 180-м меридианом) сдвигаем на +360, иначе она оказывается на другом краю карты.
import shapely
from pyproj import Transformer
TR = Transformer.from_crs("EPSG:4326", "+proj=aea +lat_1=52 +lat_2=68 +lat_0=0 +lon_0=100 +datum=WGS84", always_xy=True)
def proj(xy):
    x = np.where(xy[:, 0] < 0, xy[:, 0] + 360, xy[:, 0]); X, Y = TR.transform(x, xy[:, 1]); return np.c_[X, -Y]
geoms = [shapely.transform(g, proj) for g in poly.geometry]
minx = min(g.bounds[0] for g in geoms); maxx = max(g.bounds[2] for g in geoms); miny = min(g.bounds[1] for g in geoms); maxy = max(g.bounds[3] for g in geoms)
SC = 1000 / (maxx - minx); W, H = 1000, int((maxy - miny) * SC) + 1
def path(g):
    out = []
    for pg in getattr(g, "geoms", [g]):
        for ring in [pg.exterior, *pg.interiors]:
            c = np.asarray(ring.coords); c = np.c_[(c[:, 0] - minx) * SC, (c[:, 1] - miny) * SC].round(1)
            out.append("M" + "L".join(f"{a:g} {b:g}" for a, b in c) + "Z")
    return "".join(out)
paths = [path(g) for g in geoms]
cen = [g.representative_point() for g in geoms]; px = [round((c.x - minx) * SC, 1) for c in cen]; py = [round((c.y - miny) * SC, 1) for c in cen]

p = load().drop_duplicates("tid").set_index("tid").reindex(ids); Tn = d["tensor"]; SH = d["shares"][2:].mean(0)
F = pd.DataFrame({"Расходы на душу (лог)": Tn[2:, :, 0].mean(0), "Продовольствие": SH[:, 2], "Общепит": SH[:, 1], "Маркетплейсы": SH[:, 3],
                  "Здоровье": SH[:, 0], "Транспорт": SH[:, 4], "Население (лог)": np.log(p["pop"].values), "Зарплата (лог)": np.log(p["wage"].values),
                  "Занятость: с/х": p.emp_agri.values, "Занятость: добыча": p.emp_mining.values, "Занятость: обработка": p.emp_manuf.values,
                  "Занятость: торговля": p.emp_trade.values, "Занятость: госуправл.": p.emp_gov.values, "Занятость: образование": p.emp_edu.values}, index=ids)
Z = ((F - F.median()) / F.std()).clip(-3, 3).fillna(0)

Xt, _ = month_features(d); A = graph(Xt[-1]).tocsr(); X = Xt[-1]; Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
NB, NS = [], []
for i in range(N):
    r = A.getrow(i); o = np.argsort(-r.data)[:8]; j = r.indices[o]; NB.append(j.tolist()); NS.append([round(float(Xn[i] @ Xn[k]), 2) for k in j])

PAL = ["#5b8c6a", "#d98e32", "#4f7cb8", "#6a4fa3", "#b0413e", "#c27ba0"]; GREY = "#d9d9d9"
D = dict(months=months, ids=[str(i) for i in ids], name=meta.municipal_district_name.fillna("?").tolist(), region=meta.region_name.fillna("?").tolist(),
         nb=NB, ns=NS, feat=list(F.columns), z=Z.round(2).values.tolist(), pal=PAL, grey=GREY, K={})
for K in CFG["headline"]["Ks"]:
    L = labels(K); nm = names(K); typ = L[-1]
    D["K"][str(K)] = dict(L=L.tolist(), names=[nm[k] for k in range(K)], n=np.bincount(typ, minlength=K).tolist(),
                          tmed=[Z[typ == k].median().round(2).tolist() for k in range(K)],
                          changed=[int(len(set(L[:, i])) > 1) for i in range(N)])
D.update(paths=paths, px=px, py=py, W=W, H=H)
tpl = open("src/dashboard_template.html", encoding="utf-8").read()
open("outputs/dashboard.html", "w", encoding="utf-8").write(tpl.replace("%%DATA%%", json.dumps(D, ensure_ascii=False, separators=(",", ":"))))
import os; print("outputs/dashboard.html", round(os.path.getsize("outputs/dashboard.html") / 2**20, 1), "МБ")
