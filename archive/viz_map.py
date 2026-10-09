import sys, json; sys.path.insert(0, "src")
import numpy as np, pandas as pd, geopandas as gpd, plotly.graph_objects as go
from config import CFG
from features import build
from headline import labels
from names import names
KK = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["headline"]["Ks"][-1]
NAMES = names(KK)
PAL = ["#8fb996", "#e0a458", "#7a9cc6", "#4a3f8f", "#a4454f", "#c27ba0"]; COL = {k: PAL[k] for k in range(KK)}
d = build(); ids = d["ids"]; months = d["months"][2:]; meta = d["meta"]
L = labels(KK)
tid = pd.Series({i: i for i in ids})                    # ключ МО = territory_id (строка)
poly = gpd.read_file("data/raw/dict/t_dict_municipal_districts_poly.gpkg"); poly = poly.sort_values("year_to").drop_duplicates("territory_id", keep="last")
poly["territory_id"] = poly.territory_id.astype(str)
poly["geometry"] = poly.geometry.simplify(0.05, preserve_topology=True)
ok = [i for i, o in enumerate(ids) if o in tid.index and tid[o] in set(poly.territory_id)]
print("МО с полигоном:", len(ok), "из", len(ids))
sub = poly.set_index("territory_id").loc[[tid[ids[i]] for i in ok]].reset_index()
sub["key"] = [ids[i] for i in ok]; gj = json.loads(sub[["key", "geometry"]].set_index("key").to_json())
names = meta.municipal_district_name.values[ok]; regs = meta.region_name.values[ok]
txt = [f"{n}<br>{r}" for n, r in zip(names, regs)]
scale = [pt for k in range(KK) for pt in ([k / KK, COL[k]], [(k + 1) / KK, COL[k]])]
def tr(t): return go.Choroplethmap(geojson=gj, locations=[ids[i] for i in ok], z=L[t][ok], zmin=0, zmax=KK-0.0001, colorscale=scale, showscale=False,
                                 text=txt, hovertemplate="%{text}<br>тип: %{customdata}<extra></extra>", customdata=[NAMES[v] for v in L[t][ok]], marker_line_width=0.2, marker_line_color="#ffffff")
fig = go.Figure(data=[tr(0)], frames=[go.Frame(data=[go.Choroplethmap(z=L[t][ok], customdata=[NAMES[v] for v in L[t][ok]])], name=months[t]) for t in range(len(months))])
fig.update_layout(map=dict(style="white-bg", center=dict(lat=62, lon=96), zoom=2.1))
fig.update_layout(height=620, margin=dict(l=0, r=0, t=40, b=0), title="Типы экономики муниципалитетов по месяцам",
    sliders=[dict(steps=[dict(method="animate", label=m, args=[[m], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for m in months], currentvalue=dict(prefix="Месяц: "))],
    updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05, buttons=[dict(label="▶", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])],
    annotations=[dict(x=0.99, y=0.97-0.06*k, xref="paper", yref="paper", showarrow=False, align="right", text=f"<span style='color:{COL[k]}'>■</span> {NAMES[k]}") for k in range(KK)])
fig.write_html(f"outputs/map_K{KK}.html", include_plotlyjs="cdn")
import os; print(f"outputs/map_K{KK}.html", round(os.path.getsize(f"outputs/map_K{KK}.html") / 2**20, 1), "МБ")
