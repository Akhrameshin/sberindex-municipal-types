import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import build, load
d = build(); L = np.load("data/processed/L_k5_om0.2_canon.npy"); ids = d["ids"]; T = d["tensor"]; cols = d["cols"]
p = load().drop_duplicates("tid").set_index("tid").reindex(ids)
Lt = L; avg = pd.DataFrame(T[2:].mean(0), index=ids, columns=cols)             # среднее по периоду
mode = pd.Series([np.bincount(Lt[:, i]).argmax() for i in range(len(ids))], index=ids)   # модальный тип МО
st = pd.DataFrame({"pop": p["pop"], "wage": p["wage"], **{"e_"+c: p["emp_"+c] for c in ["agri","mining","manuf","trade","transp","gov","edu","health","ict","fin"]}}, index=ids)
df = pd.concat([avg, st], axis=1); df["type"] = mode
g = df.groupby("type").agg(n=("pop","size"), spend=("log_total", lambda s: np.expm1(s).median()), food=("sh_food","mean"), cater=("sh_catering","mean"),
    market=("sh_market","mean"), health=("sh_health","mean"), trans=("sh_transport","mean"), pop=("pop","median"), wage=("wage","median"),
    agri=("e_agri","mean"), mining=("e_mining","mean"), manuf=("e_manuf","mean"), trade=("e_trade","mean"), gov=("e_gov","mean"), edu=("e_edu","mean")).round(3)
print(g.to_string())
osc = ((Lt == 3).any(0)) & ((Lt == 0).any(0)); print("МО, бывавших и в 0, и в 3:", int(osc.sum()), "| всегда в одном типе:", int((np.array([len(set(Lt[:, i])) for i in range(len(ids))]) == 1).sum()))
cross = np.array([len(set(Lt[:, i])) for i in range(len(ids))]); print("число разных типов у МО:", dict(zip(*np.unique(cross, return_counts=True))))
