"""Список МО, не вошедших в типологию из-за неполного ряда расходов (см. features.complete_only)."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from config import CFG
from features import load
p = load(); n = p.groupby("tid").month.nunique().rename("месяцев_с_данными")
meta = p.drop_duplicates("tid").set_index("tid")[["municipal_district_name", "region_name"]]
j = meta.join(n); ex = j[j["месяцев_с_данными"] < CFG["data"]["min_months"]].sort_values("месяцев_с_данными")
ex.to_csv("data/processed/excluded_incomplete.csv")
print(f"всего МО в панели: {len(n)} | в типологии (полный ряд {CFG['data']['min_months']} мес.): {int((n >= CFG['data']['min_months']).sum())} | исключено: {len(ex)}")
print("распределение числа месяцев у исключённых:", ex["месяцев_с_данными"].describe()[["min", "50%", "max"]].to_dict())
