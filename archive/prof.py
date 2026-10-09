import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import build, load
d = build(); L = np.load("data/processed/labels_om0.2.npy"); ids = d["ids"]
p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
T = d["tensor"]; cols = d["cols"]
last = L[-1]; first = L[0]
df = pd.DataFrame(T[12:].mean(0), index=ids, columns=cols)             # 2024 среднее
df["cl"] = last; df["pop"] = p["pop"].values; df["wage"] = p["wage"].values
for c in ["agri","mining","manuf","trade","gov","edu"]: df["e_"+c] = p["emp_"+c].values
df["spend"] = np.expm1(df.log_total)
g = df.groupby("cl").agg(n=("cl","size"), spend=("spend","median"), health=("sh_health","mean"), cater=("sh_catering","mean"),
    food=("sh_food","mean"), market=("sh_market","mean"), trans=("sh_transport","mean"), pop=("pop","median"), wage=("wage","median"),
    agri=("e_agri","mean"), mining=("e_mining","mean"), manuf=("e_manuf","mean"), gov=("e_gov","mean")).round(3)
print(g.to_string())
ct = pd.crosstab(first, last, rownames=["янв-23"], colnames=["дек-24"]); print(ct.to_string())
print("сменили кластер за 2 года:", round((first != last).mean(), 3))
