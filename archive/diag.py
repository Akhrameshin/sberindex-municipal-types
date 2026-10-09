import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from features import build, load
from pipeline import *
d = build(); Xt, months = month_features(d, trim=2); G = [to_ig(graph(x)) for x in Xt]
gam = calibrate(G[len(G)//2]); L = temporal(G, 0.2, gam)
print("месяцев", len(months), "γ", round(gam, 3), "| меток в первый/последний месяц:", len(set(L[0])), len(set(L[-1])))
print("меток по месяцам:", [len(set(l)) for l in L])
np.save("data/processed/labels_trim.npy", L)
ct = pd.crosstab(L[0], L[-1]); print(ct.to_string())
