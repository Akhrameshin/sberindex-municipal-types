"""Сборка панели МО: расходы СберИндекса (МО × месяц) + статические признаки Росстата (2023)."""
import re, sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
from config import CFG
YEAR = str(CFG["data"]["rosstat_year"])
SECT = {"А": "agri", "В": "mining", "C": "manuf", "D": "energy", "Е": "water", "F": "constr",
        "G": "trade", "Н": "transp", "I": "hotel", "J": "ict", "K": "fin", "L": "realty",
        "M": "science", "N": "admin", "O": "gov", "P": "edu", "Q": "health", "R": "culture", "S": "other"}


def oktmo8(s):
    return re.sub(r"\D", "", str(s))[:8]


# --- справочник: для каждого territory_id версия, действовавшая в YEAR (иначе последняя)
dic = pd.read_excel(RAW / "dict/t_dict_municipal_districts.xlsx")
dic["oktmo8"] = dic.oktmo.map(oktmo8)
act = dic[(dic.year_from <= int(YEAR)) & (dic.year_to >= int(YEAR))]
rest = dic[~dic.territory_id.isin(act.territory_id)].sort_values("year_to").drop_duplicates("territory_id", keep="last")
key = pd.concat([act.sort_values("year_from").drop_duplicates("territory_id", keep="last"), rest]).set_index("territory_id")

# --- расходы (архив хакатона СберИндекса, ключ territory_id)
sp = pd.read_parquet(RAW / "hack/hackathonlicence/consumption.parquet")
cat = {"Все категории": "total", "Здоровье": "health", "Общественное питание": "catering",
       "Продовольствие": "food", "Маркетплейсы": "market", "Транспорт": "transport"}
sp["cat"] = sp.category.map(cat)
spend = sp.pivot_table(index=["territory_id", "date"], columns="cat", values="value", aggfunc="first").reset_index().rename(columns={"date": "month"})
spend["oktmo8"] = spend.territory_id.map(key.oktmo8)
spend["tid"] = spend.territory_id.astype(str)
print(f"расходы: МО {spend.tid.nunique()}, месяцев {spend.month.nunique()}")

# --- Росстат (верхний уровень МО, годовые значения)
def rosstat(code):
    f = RAW / f"rosstat/{code}_{YEAR}.csv"
    cols = ["okved2", "mun_level", "oktmo", "indicator_value", "indicator_period"]
    d = pd.read_csv(f, sep=";", dtype=str, usecols=lambda c: c in cols)
    if "okved2" not in d:
        d["okved2"] = ""
    return d[d.mun_level.str.contains("верхнего")].assign(oktmo8=lambda t: t.oktmo.map(oktmo8),
                                                          v=lambda t: pd.to_numeric(t.indicator_value, errors="coerce"))


pop = rosstat("Y48112027")
pop = pop[pop.indicator_period == "На 1 января"].groupby("oktmo8").v.max().rename("pop")

emp = rosstat("Y48423005")
emp = emp[emp.indicator_period == "Январь-декабрь"]
emp["sec"] = emp.okved2.map(lambda s: "all" if s.startswith("Всего") else SECT.get(s.split()[1]))
emp = emp.dropna(subset=["sec"]).pivot_table(index="oktmo8", columns="sec", values="v", aggfunc="first")
share = emp.drop(columns="all").div(emp["all"], axis=0).fillna(0).add_prefix("emp_")
share["emp_total"] = emp["all"]

wage = rosstat("Y48423007")
wage = wage[(wage.indicator_period == "Январь-декабрь") & wage.okved2.str.startswith("Всего")]
wage = wage.groupby("oktmo8").v.first().rename("wage")

static = pd.concat([pop, share, wage], axis=1).reset_index()
panel = spend.merge(static, on="oktmo8", how="left")
meta = key.reset_index()[["territory_id", "municipal_district_name", "region_name", "municipal_district_center_lat", "municipal_district_center_lon"]]
panel = panel.merge(meta, on="territory_id", how="left")
ma = pd.read_parquet(RAW / "hack/hackathonlicence/market_access.parquet")        # только для валидации, в признаки не входит
panel = panel.merge(ma, on="territory_id", how="left")
panel.to_parquet(OUT / "panel.parquet", index=False)

per = panel.drop_duplicates("tid")
print("панель:", panel.shape, "| МО:", len(per), "| месяцев:", panel.month.nunique())
print("покрытие Росстата, % МО:", per[["pop", "emp_total", "wage", "market_access", "municipal_district_center_lat"]].notna().mean().round(3).to_dict())
