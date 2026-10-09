import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import build
d = build(); L = np.load("data/processed/L_k7_om0.2.npy"); T = d["tensor"]; cols = d["cols"]; meta = d["meta"]
labs = sorted(set(L.ravel()))
print(pd.DataFrame({c: [(L[t]==c).sum() for t in range(L.shape[0])] for c in labs}, index=d["months"][2:]).iloc[::3].to_string())
m = (L == 5).any(0); print("МО, побывавших в кл.5:", m.sum())
peak = int(np.argmax([(L[t]==5).sum() for t in range(L.shape[0])])); mask = L[peak]==5
print("пик в", d["months"][2+peak], "размер", mask.sum())
print("регионы:", meta.region_name[mask].value_counts().head(6).to_dict())
print("примеры:", meta.municipal_district_name[mask].head(6).tolist())
for nm, lab in [("кл.5", 5)]:
    print(pd.Series(T[2+peak][mask].mean(0), index=cols).round(3).to_dict(), "| все МО:", pd.Series(T[2+peak].mean(0), index=cols).round(3).to_dict())
