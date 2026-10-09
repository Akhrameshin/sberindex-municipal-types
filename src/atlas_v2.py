"""Auditable retrospective typology. No inference of causal changes or future forecasts.

Pipeline: aggregate ORIGINAL levels -> trailing-year spending/CLR -> train-fitted
standardisation and imputation -> explicit block weights -> spending network ->
graph smoothing -> pooled k-means -> exact conditional Viterbi.
"""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from graph_support import weighted_knn

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'configs/atlas.yaml').read_text())
PARTS = ['health', 'catering', 'food', 'market', 'transport', 'other']
GROUPS = {
 'primary':['agri','mining'], 'industry':['manuf','energy','water','constr'],
 'trade':['trade','hotel'], 'transport':['transp'],
 'services':['ict','fin','realty','science','admin','other'],
 'public':['gov','edu','health','culture'],
}
SPEND_COLUMNS = ['log_total'] + ['clr_'+x for x in PARTS]
CONTEXT_COLUMNS = ['log_pop','log_wage'] + ['emp_'+x for x in GROUPS] + ['emp_unreported']
SPEND_LABELS = ['Уровень расходов (лог)'] + ['Здоровье','Общепит','Продовольствие','Маркетплейсы','Транспорт','Прочее']
CONTEXT_LABELS = ['Население (лог)','Зарплата (лог)','Первичный сектор: раскрытая доля','Промышленность и стройка: раскрытая доля','Торговля и гостиницы: раскрытая доля','Транспорт: раскрытая доля','Рыночные услуги: раскрытая доля','Государственный сектор: раскрытая доля','Нераспределённая занятость']


def aggregate_panel(panel, policy='aggregate', coverage=0.99):
    """Population weights for per-resident levels; employment weights for labour shares.

    Ratios and logarithms are formed AFTER aggregation. All city members determine
    population; missing observations are allowed only above the coverage threshold.
    Synthetic city IDs are explicitly separate from Sber territory_id.
    """
    p = panel.copy()
    p['tid'] = p['tid'].astype(str)
    districts = p.municipal_district_name.str.contains('внутригород', case=False, na=False)
    if policy == 'original':
        return p, pd.DataFrame(columns=['source_tid','analysis_tid','region'])
    mapping = []
    if policy == 'drop':
        return p[~districts].copy(), pd.DataFrame(columns=['source_tid','analysis_tid','region'])
    pieces = [p[~districts].copy()]
    for region, synthetic in [('Москва','city-moscow'),('Санкт-Петербург','city-spb')]:
        city = p[districts & p.region_name.eq(region)]
        if city.empty: continue
        static = city.drop_duplicates('tid').set_index('tid')
        population = static['pop']
        if population.isna().any() or (population <= 0).any():
            raise ValueError('Cannot aggregate a city without valid population weights')
        total_population = float(population.sum())
        emp = static['emp_total']; valid_emp = emp.notna() & (emp > 0)
        total_employment = float(emp[valid_emp].sum())
        base = {c: np.nan for c in p.columns}
        base.update(tid=synthetic, territory_id=-9001 if region=='Москва' else -9002,
                    municipal_district_name=region+' (агрегат)', region_name=region,
                    pop=total_population, emp_total=total_employment, oktmo8='')
        for col in ['municipal_district_center_lat','municipal_district_center_lon','market_access']:
            good = static[col].notna()
            base[col] = float(np.average(static.loc[good,col], weights=population[good])) if good.any() else np.nan
        good = static.wage.notna() & valid_emp
        base['wage'] = float(np.average(static.loc[good,'wage'],weights=emp[good])) if good.any() else np.nan
        for col in [c for c in p.columns if c.startswith('emp_') and c != 'emp_total']:
            good = static[col].notna() & valid_emp
            base[col] = float((static.loc[good,col]*emp[good]).sum()/total_employment) if good.any() and total_employment>0 else np.nan
        for month, part in city.groupby('month',sort=True):
            row = dict(base); row['month'] = month
            complete = part[['total','health','catering','food','market','transport']].notna().all(axis=1)
            part = part[complete].set_index('tid')
            weights = population.reindex(part.index)
            observed = float(weights.sum()/total_population)
            if observed < coverage: continue
            for col in ['total','health','catering','food','market','transport']:
                row[col] = float(np.average(part[col],weights=weights))
            pieces.append(pd.DataFrame([row]))
        mapping.extend({'source_tid':t,'analysis_tid':synthetic,'region':region} for t in static.index)
    return pd.concat(pieces,ignore_index=True), pd.DataFrame(mapping)


def load_panel(policy=None):
    p = pd.read_parquet(ROOT/'data/processed/panel.parquet')
    corrected = ROOT/'data/external/employment_2023.csv'
    if not corrected.exists():raise FileNotFoundError('Version 2 requires employment_2023.csv; run make sources')
    if corrected.exists():
        emp = pd.read_csv(corrected, dtype={'oktmo8':str}).set_index('oktmo8')
        for col in [c for c in emp if c.startswith('emp_')]:
            p[col] = p.oktmo8.map(emp[col])
    policy = policy or CFG['territories']['city_policy']
    p, mapping = aggregate_panel(p,policy,CFG['territories']['minimum_population_coverage'])
    return p,mapping


def build_arrays(panel, window=None, minimum_months=None):
    window=CFG['features']['window'] if window is None else window
    minimum_months=CFG['features']['minimum_months'] if minimum_months is None else minimum_months
    p = panel.copy(); p['tid'] = p.tid.astype(str)
    levels = ['total','health','catering','food','market','transport']
    good = p[levels].notna().all(axis=1) & (p.total > 0)
    counts = p[good].groupby('tid').month.nunique()
    ids = sorted(counts[counts >= minimum_months].index)
    months = sorted(p.month.unique())
    rows = p[p.tid.isin(ids)].copy()
    if rows.duplicated(['tid','month']).any(): raise ValueError('Duplicate territory-month')
    meta = rows.drop_duplicates('tid').set_index('tid').reindex(ids)
    mi, ii = {m:i for i,m in enumerate(months)}, {t:i for i,t in enumerate(ids)}
    raw = np.full((len(months),len(ids),6),np.nan)
    raw[rows.month.map(mi),rows.tid.map(ii)] = rows[levels].to_numpy()
    if not np.isfinite(raw).all(): raise ValueError('Main model requires complete observed series')
    remainder = raw[:,:,0]-raw[:,:,1:].sum(2)
    if (remainder < -1e-6).any(): raise ValueError('Negative other-spending category')
    raw_parts = np.concatenate([raw[:,:,1:],remainder[:,:,None]],2)
    shares = raw_parts / raw[:,:,0,None]
    spending = []
    for end in range(window-1,len(months)):
        total = raw[end-window+1:end+1,:,0].mean(0)
        part = raw_parts[end-window+1:end+1].sum(0)
        annual_shares = part/part.sum(1,keepdims=True)
        # Multiplicative replacement avoids changing composition sum after a zero.
        annual_shares = np.maximum(annual_shares,1e-6)
        annual_shares /= annual_shares.sum(1,keepdims=True)
        logs = np.log(annual_shares)
        spending.append(np.c_[np.log(total),logs-logs.mean(1,keepdims=True)])
    sector_cols = [c for c in meta.columns if c.startswith('emp_') and c != 'emp_total']
    meta['sector_coverage'] = meta[sector_cols].sum(1,min_count=1)
    # Published sectors are lower bounds when cells are withheld. Their missing
    # mass receives an explicit coordinate; it is never called zero employment.
    valid_emp = meta.emp_total.notna() & (meta.emp_total>0)
    groups=np.column_stack([meta[['emp_'+x for x in sectors]].fillna(0).sum(1).to_numpy() for sectors in GROUPS.values()])
    groups/=np.maximum(groups.sum(1),1)[:,None]  # reconcile sub-per-mille rounding excess
    unreported=np.maximum(1-groups.sum(1),0)
    groups[~valid_emp]=np.nan;unreported[~valid_emp]=np.nan
    context=np.c_[np.log(meta['pop'].where(meta['pop']>0)),np.log(meta['wage'].where(meta['wage']>0)),groups,unreported]
    meta['context_missing'] = ~np.isfinite(context).all(1)
    meta['sector_withheld'] = meta.sector_coverage.lt(.98) | ~valid_emp
    return {'ids':ids,'months':months[window-1:],'all_months':months,'spending':np.asarray(spending),
            'context':context,'raw':raw,'shares':shares,'meta':meta,'window':window}


def scale_arrays(arrays, lens='combined', spending_weight=None, training_indices=None):
    """Fit all transformations on the specified training MO, including bootstrap/CV."""
    s = arrays['spending']; c = arrays['context']; n = s.shape[1]
    train = np.arange(n) if training_indices is None else np.asarray(training_indices)
    weight = CFG['features']['spending_weight'] if spending_weight is None else spending_weight
    clip = CFG['features']['clip_z']
    scaler = StandardScaler().fit(s[:,train].reshape(-1,7))
    zs = np.clip(scaler.transform(s.reshape(-1,7)).reshape(s.shape),-clip,clip)
    zc = np.empty((n,0)); imputer = None; cscale = None
    if lens == 'combined':
        imputer = SimpleImputer(strategy='median',keep_empty_features=True).fit(c[train])
        filled = imputer.transform(c)
        cscale = StandardScaler().fit(filled[train])
        zc = np.clip(cscale.transform(filled),-clip,clip)
        zs *= np.sqrt(weight/7); zc *= np.sqrt((1-weight)/c.shape[1])
    else:
        zs /= np.sqrt(7)
    xt = np.concatenate([zs,np.broadcast_to(zc,(len(s),n,zc.shape[1]))],2)
    return xt,{'spend_scaler':scaler,'context_imputer':imputer,'context_scaler':cscale,'lens':lens,'weight':weight}


def centre_windows(arrays, columns=None):
    """Копия массивов, где выбранные координаты расходов центрированы по медиане территорий внутри каждого окна.
    Так убирается общий для всей страны дрейф (номинальный рост, смещение к маркетплейсам): остаётся только положение территории относительно остальных."""
    s = arrays['spending'].copy(); cols = slice(None) if columns is None else columns
    s[:,:,cols] -= np.median(s[:,:,cols],axis=1,keepdims=True)
    out = dict(arrays); out['spending'] = s
    return out


def make_graphs(xt, source=None, neighbors=None, metric=None):
    source=CFG['graph']['source'] if source is None else source
    neighbors=CFG['graph']['neighbors'] if neighbors is None else neighbors
    metric=CFG['graph']['metric'] if metric is None else metric
    return [weighted_knn(x[:,:7] if source=='spending' else x,metric,k=neighbors,
                         minimum=CFG['graph']['minimum_neighbors']) for x in xt]


def smooth_features(xt, graphs, steps=1):
    if steps == 0: return np.asarray(xt).copy()
    out=[]
    for x,g in zip(xt,graphs):
        a=g+sparse.eye(len(x)); inv=np.power(np.asarray(a.sum(1)).ravel(),-.5)
        a=sparse.diags(inv)@a@sparse.diags(inv)
        for _ in range(steps): x=a@x
        out.append(x)
    return np.asarray(out)


def viterbi_costs(distances, penalty):
    d=np.asarray(distances); t,n,k=d.shape; cost=d[0].copy(); back=np.zeros((t,n,k),np.int16)
    for h in range(1,t):
        arg=cost.argmin(1); best=cost.min(1,keepdims=True)
        stay=cost<=best+penalty
        back[h]=np.where(stay,np.arange(k)[None],arg[:,None])
        cost=d[h]+np.minimum(cost,best+penalty)
    labels=np.zeros((t,n),np.int16);labels[-1]=cost.argmin(1)
    for h in range(t-1,0,-1):labels[h-1]=back[h,np.arange(n),labels[h]]
    return labels,float(cost.min(1).sum())


def fit_model(xt, k, graphs=None, seed=None, steps=None, lam=None):
    seed=CFG['seed'] if seed is None else seed
    steps=CFG['model']['smoothing_steps'] if steps is None else steps
    lam=CFG['model']['transition_penalty'] if lam is None else lam
    graphs=graphs or make_graphs(xt,neighbors=CFG['graph']['neighbors'])
    sm=smooth_features(xt,graphs,steps)
    km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(sm.reshape(-1,sm.shape[-1]))
    distances=np.sum((sm[:,:,None]-km.cluster_centers_[None,None])**2,3)
    penalty=float(lam*np.median(distances.min(2)))
    labels,objective=viterbi_costs(distances,penalty)
    order=np.argsort(-np.bincount(labels[-1],minlength=k),kind='stable')
    mapping=np.empty(k,int);mapping[order]=np.arange(k)
    centers=km.cluster_centers_[order];labels=mapping[labels]
    distances=distances[:,:,order]
    return {'labels':labels,'centers':centers,'distances':distances,'penalty':penalty,
            'objective':objective,'smoothed':sm,'graphs':graphs,'seed':seed,'steps':steps,'lam':lam}


def match_labels(reference, candidate, k):
    overlap=np.array([[np.sum((reference==a)&(candidate==b)) for b in range(k)] for a in range(k)])
    a,b=linear_sum_assignment(-overlap);mp=np.empty(k,int);mp[b]=a
    return mp[candidate]


def assigned_margin(distances, labels):
    own=np.take_along_axis(distances,labels[:,:,None],axis=2)[:,:,0]
    other=distances.copy();np.put_along_axis(other,labels[:,:,None],np.inf,axis=2)
    alt=other.min(2)
    return (alt-own)/(alt+own+1e-12),other.argmin(2)


def metrics(xt, graph, labels):
    from icvi import all_indices
    from icvi_reference import graph_indices_pattern
    values=all_indices(xt,graph,labels)
    values.update(graph_indices_pattern(graph,labels))
    return {key:float(value) for key,value in values.items()}


def bootstrap_model(arrays,k,reference,reps=50,fraction=.8,seed=142,lens='combined'):
    """Rebuild scaling, imputation, kNN and centroids for every subset."""
    rng=np.random.default_rng(seed);n=len(arrays['ids']);hits=np.zeros((len(reference),n));counts=np.zeros(n)
    aris=[];jaccards=[];changed=np.zeros(n);present=np.zeros(n)
    for r in range(reps):
        keep=np.sort(rng.choice(n,int(n*fraction),replace=False))
        local=dict(arrays);local['spending']=arrays['spending'][:,keep];local['context']=arrays['context'][keep]
        xt,_=scale_arrays(local,lens=lens)
        fit=fit_model(xt,k,seed=seed+r,steps=CFG['model']['smoothing_steps'],lam=CFG['model']['transition_penalty'])
        lab=match_labels(reference[:,keep],fit['labels'],k)
        hits[:,keep]+=(lab==reference[:,keep]);counts[keep]+=1
        changed[keep]+=(lab[0]!=lab[-1]);present[keep]+=1
        aris.append(adjusted_rand_score(reference[-1,keep],lab[-1]))
        per=[]
        for a in range(k):
            base=reference[-1,keep]==a;sub=lab[-1]==a
            per.append(float((base&sub).sum()/max((base|sub).sum(),1)))
        jaccards.append(per)
        if (r+1)%10==0:print(f'bootstrap K={k} {r+1}/{reps}',flush=True)
    return {'ari':np.asarray(aris),'jaccard':np.asarray(jaccards),
            'confidence':hits/np.maximum(counts,1)[None],
            'changed_share':changed/np.maximum(present,1),'runs':counts}


def semantic_names(arrays, labels, lens='combined'):
    """Profile descriptions derived from evidence, not geographic stereotypes."""
    k=int(labels.max()+1);m=arrays['meta'];sh=arrays['shares'];raw=arrays['raw'];rows=[]
    for a in range(k):
        use=labels[-1]==a
        rows.append({'type':a,'n':int(use.sum()),'spend':float(np.median(raw[-12:,use,0].mean(0))),
                     'wage':float(np.nanmedian(m.loc[use,'wage'])),
                     'food':float(np.median(sh[-12:,use,2].mean(0))),
                     'catering':float(np.median(sh[-12:,use,1].mean(0))),
                     'market':float(np.median(sh[-12:,use,3].mean(0))),
                     'mining':float(np.nanmedian(m.loc[use,'emp_mining'].fillna(0))),
                     'manufacturing':float(np.nanmedian(m.loc[use,'emp_manuf'].fillna(0))),
                     'public':float(np.nanmedian(arrays['context'][use,CONTEXT_COLUMNS.index('emp_public')]))})
    profiles=pd.DataFrame(rows).set_index('type')
    descriptions={a:f'Профиль {a+1}' for a in range(k)}
    high=int(profiles.spend.idxmax());low=int(profiles.spend.idxmin())
    descriptions[high]='Высокий уровень расходов и зарплат' if profiles.loc[high,'wage']>profiles.wage.median()*1.25 else 'Высокий уровень расходов'
    descriptions[low]='Низкие расходы, продовольственный профиль'
    remaining=[a for a in range(k) if a not in (high,low)]
    if remaining:
        mining=int(profiles.loc[remaining,'mining'].idxmax())
        if profiles.loc[mining,'mining']>.05:
            descriptions[mining]='Ресурсная специализация и высокие зарплаты';remaining.remove(mining)
    if remaining:
        industrial=int(profiles.loc[remaining,'manufacturing'].idxmax())
        if profiles.loc[industrial,'manufacturing']>.12:
            descriptions[industrial]='Промышленность и сервисное потребление';remaining.remove(industrial)
    for a in remaining:
        descriptions[a]='Государственный сектор и умеренные расходы' if profiles.loc[a,'public']>.55 else 'Смешанная занятость и средние расходы'
    if lens=='spend':
        descriptions={a:f'Потребление: профиль {a+1}' for a in range(k)}
        descriptions[high]='Потребление: высокие расходы';descriptions[low]='Потребление: низкие расходы'
    return descriptions,profiles.reset_index()


def lineage():
    dependencies={}
    for name in ['numpy','pandas','scipy','scikit-learn','pyarrow','igraph','leidenalg','matplotlib','PyYAML']:
        try:dependencies[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    h=hashlib.sha256()
    inputs={}
    for path in [ROOT/'data/processed/panel.parquet',ROOT/'configs/atlas.yaml',
                 *sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),
                 *sorted((ROOT/'data/external').glob('*.csv'))]:
        digest=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.relative_to(ROOT))]=digest
        h.update(str(path.relative_to(ROOT)).encode());h.update(digest.encode())
    return {'run_id':h.hexdigest()[:16],'python':platform.python_version(),'dependencies':dependencies,'inputs':inputs}


def write_json(path,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(np.bool_,)):return bool(x)
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        return x
    Path(path).write_text(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':')))
