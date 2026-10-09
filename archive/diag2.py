import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import build
d = build(); L = np.load("data/processed/labels_trim.npy"); T = d["tensor"]; cols = d["cols"]
m0, m1 = 2, 23                                    # индексы месяцев в исходном тензоре
def prof(mask, t): return pd.Series(T[t][mask].mean(0), index=cols).round(3)
rows = {}
for name, mask in [("кл.1→1", (L[0]==1)&(L[-1]==1)), ("кл.1→8", (L[0]==1)&(L[-1]==8)), ("кл.2→2", (L[0]==2)&(L[-1]==2)), ("кл.2→4", (L[0]==2)&(L[-1]==4))]:
    a, b = prof(mask, m0), prof(mask, m1); rows[name] = pd.concat([a.add_suffix(" янв"), b.add_suffix(" дек")])
r = pd.DataFrame(rows).T
r["n"] = [int(((L[0]==a)&(L[-1]==b)).sum()) for a, b in [(1,1),(1,8),(2,2),(2,4)]]
print(r[[c for c in r.columns if any(k in c for k in ("sh_market","sh_food","log_total","sh_catering"))]+["n"]].to_string())
meta = d["meta"]; ids = np.array(d["ids"])
for a, b in [(1,8),(2,4)]:
    m = (L[0]==a)&(L[-1]==b); print(f"{a}→{b} примеры:", meta.municipal_district_name[m].head(5).tolist(), "| регионы:", meta.region_name[m].value_counts().head(3).to_dict())
