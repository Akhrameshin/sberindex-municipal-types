import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import build, load
d = build(); ids = d["ids"]; meta = d["meta"]
ts = pd.Series([np.bincount(c).argmax() for c in np.load("data/processed/L_spend_only.npy").T], index=ids)
p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
reg = pd.Series(meta.region_name.fillna("?").values, index=ids)
def om2(x, g):
    m = x.notna(); x, g = x[m], g[m]; k = g.nunique(); n = len(x); gm = x.mean()
    ssb = sum(len(x[g == a]) * (x[g == a].mean() - gm) ** 2 for a in g.unique()); sst = ((x - gm) ** 2).sum(); msw = (sst - ssb) / (n - k)
    return max(0, (ssb - (k - 1) * msw) / (sst + msw))
v = pd.DataFrame({"log зарплата": np.log(p["wage"]), "log население": np.log(p["pop"]), "торговля": p["emp_trade"], "ИКТ": p["emp_ict"], "госуправление": p["emp_gov"], "с/х": p["emp_agri"], "добыча": p["emp_mining"]})
r = pd.DataFrame({"ω² типы по расходам (5)": {c: om2(v[c], ts) for c in v}, f"ω² регионы ({reg.nunique()})": {c: om2(v[c], reg) for c in v}}).round(3); print(r.to_string())
