"""Download original BDMO CSV members, never another entrant's processed data.

Each raw member and every derived table is hashed. Offline builds use the supplied
prepared tables. Refresh is explicit; it never silently updates the source vintage.
"""
from pathlib import Path
import sys, hashlib, json, re, gzip, shutil
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rangezip import open_zip
ROOT=Path(__file__).resolve().parents[1]
URL='https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/by_indicator/data_section{}_112_v20250918.zip'
SOURCES={2:[('Y48401003',2024),('Y48401006',2024)],
         37:[('Y48010001',2024)],9:[('Y48109001',2023)],
         60:[('Y48060002',2024)],32:[('Y48423005',2023)]}

def download_section(section, indicators):
    folder=ROOT/'data/external/raw';folder.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data/external/source_manifest.json'
    known={row['file']:row['sha256'] for row in json.loads(manifest.read_text())} if manifest.exists() else {}
    z=None;out=[]
    for code,year in indicators:
        target=folder/f'{code}_{year}.csv'
        compressed=target.with_suffix('.csv.gz')
        if not target.exists() and compressed.exists():
            with gzip.open(compressed,'rb') as source,target.open('wb') as dest:
                shutil.copyfileobj(source,dest)
        if not target.exists():
            z=z or open_zip(URL.format(section))
            member=f'data_{code}_parts/data_{code}_year{year}_112_v20250918.csv'
            if member in z.namelist():
                target.write_bytes(z.read(member))
            else:
                member=f'data_{code}_112_v20250918.csv'
                full=pd.read_csv(z.open(member),sep=';',dtype=str)
                year_col=next(c for c in ['year','indicator_year'] if c in full)
                full=full[full[year_col].eq(str(year))]
                full.to_csv(target,sep=';',index=False)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        expected=known.get(str(target.relative_to(ROOT)))
        if expected and expected!=digest:raise ValueError('Source SHA-256 mismatch: '+str(target))
        out.append({'code':code,'year':year,'section':section,'url':URL.format(section),
                    'file':str(target.relative_to(ROOT)),
                    'sha256':digest})
        print('source',code,year,target.stat().st_size,flush=True)
    return out

def clean_sources():
    import numpy as np
    folder=ROOT/'data/external/raw'; tables=[]
    names={'Y48401003':('retail',2024,1000), 'Y48401006':('catering',2024,1000),
           'Y48010001':('housing',2024,1),'Y48109001':('investment',2023,1000),
           'Y48060002':('hotel_beds',2024,1)}
    for code,(name,year,scale) in names.items():
        f=folder/f'{code}_{year}.csv'
        if not f.exists():continue
        d=pd.read_csv(f,sep=';',dtype=str)
        d=d[d.mun_level.str.contains('верхнего',na=False)]
        for col in ['okved2','istinv']:
            if col in d:d=d[d[col].fillna('').str.startswith('Всего')]
        if name in ('retail','catering'):
            d=d[d.indicator_period.eq('Январь-декабрь')]
        d['oktmo8']=d.oktmo.map(lambda s:re.sub(r'\D','',str(s))[:8])
        d['value']=pd.to_numeric(d.indicator_value.str.replace(',','.',regex=False),errors='coerce')*scale
        # Repeated indicator-period records are not averaged across definitions.
        if d.duplicated('oktmo8').any():
            print(name,'duplicate definitions: retained in raw source; omitting ambiguous keys',flush=True)
            d=d[~d.oktmo8.duplicated(keep=False)]
        tables.append(d[['oktmo8','value']].assign(indicator=name,year=year,source_unit='; '.join(d.indicator_unit.unique()),unit_multiplier=scale))
        print('clean',name,len(d),list(d.indicator_period.unique()),flush=True)
    if tables:pd.concat(tables).to_csv(ROOT/'data/external/validation.csv',index=False)
    f=folder/'Y48423005_2023.csv'
    if f.exists():
        d=pd.read_csv(f,sep=';',dtype=str)
        d=d[d.mun_level.str.contains('верхнего',na=False)&d.indicator_period.eq('Январь-декабрь')].copy()
        aliases={'А':'A','В':'B','С':'C','Е':'E','Н':'H','К':'K','М':'M','О':'O','Р':'P'}
        sectors=dict(zip('ABCDEFGHIJKLMNOPQRS',['agri','mining','manuf','energy','water','constr','trade','transp','hotel','ict','fin','realty','science','admin','gov','edu','health','culture','other']))
        def sector(s):
            if str(s).startswith('Всего'):return 'total'
            m=re.search(r'Раздел\s+([A-ZА-Я])',str(s))
            return sectors.get(aliases.get(m[1],m[1])) if m else None
        d['sector']=d.okved2.map(sector)
        d['oktmo8']=d.oktmo.map(lambda s:re.sub(r'\D','',str(s))[:8])
        d['value']=pd.to_numeric(d.indicator_value.str.replace(',','.',regex=False),errors='coerce')
        d=d.dropna(subset=['sector'])
        if d.duplicated(['oktmo8','sector']).any():raise ValueError('Ambiguous employment definitions')
        e=d.pivot(index='oktmo8',columns='sector',values='value')
        total=e.pop('total');shares=e.div(total.where(total>0),axis=0).add_prefix('emp_')
        # A missing published cell stays missing. Observed zeros remain zeros.
        shares['emp_total']=total;shares.to_csv(ROOT/'data/external/employment_2023.csv')
        print('employment coverage',shares.notna().mean().round(2).to_dict(),flush=True)

if __name__=='__main__':
    from concurrent.futures import ThreadPoolExecutor,as_completed
    manifest=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(download_section,s,ind):s for s,ind in SOURCES.items()}
        for future in as_completed(futures):
            try:manifest.extend(future.result())
            except Exception as e:errors.append((futures[future],repr(e)));print('download failed',futures[future],repr(e),flush=True)
    if errors:raise RuntimeError('Some primary sources failed; the previous manifest was preserved: '+repr(errors))
    (ROOT/'data/external/source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    clean_sources()
