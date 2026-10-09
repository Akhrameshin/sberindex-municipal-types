"""Comparisons and validation with explicit experimental units and data boundaries."""
import time, warnings, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from atlas_v2 import *


def graph_summary(graph, regions):
    e=sparse.triu(graph,k=1).tocoo();deg=np.asarray((graph>0).sum(1)).ravel()
    return {'edges':len(e.data),'degree_min':int(deg.min()),'degree_median':float(np.median(deg)),
            'degree_max':int(deg.max()),'isolates':int((deg==0).sum()),
            'within_region':float(np.mean(regions[e.row]==regions[e.col]))}


def graph_overlap(a,b):
    aa=(a>0).astype(int);bb=(b>0).astype(int)
    inter=aa.multiply(bb).nnz;union=(aa+bb).nnz
    return inter/union if union else 0.


def network_comparison(arrays,xt,reference,k):
    """Spend profile, time co-movement, overlap-normalised lags, DTW and geography."""
    from edge_rules import lag_corr_dist,dtw_dist
    n=len(arrays['ids']);s=xt[:,:,:7];g=reference['graphs'][-1]
    coords=arrays['meta'][['municipal_district_center_lat','municipal_district_center_lon']].to_numpy(float)
    lat=np.radians(coords[:,0]);lon=np.radians(coords[:,1])
    hav=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    geo=6371*2*np.arcsin(np.sqrt(np.clip(hav,0,1)));geo[~np.isfinite(geo)]=np.inf;np.fill_diagonal(geo,0)
    growth=np.diff(np.log(arrays['raw'][:,:,0]),axis=0).T
    lag,_=lag_corr_dist(list(s),max_lag=3)
    dtw=dtw_dist(list(s),band=3)
    rules=[('Косинус профиля',lambda:reference['graphs']),
           ('Евклид профиля',lambda:make_graphs(xt,metric='euclidean')),
           ('Корреляция приростов',lambda:[weighted_knn(growth,'correlation',15,3)]*len(xt)),
           ('Лаговая корреляция',lambda:[weighted_knn(lag,'precomputed',15,3)]*len(xt)),
           ('DTW годовых профилей',lambda:[weighted_knn(dtw,'precomputed',15,3)]*len(xt)),
           ('География: haversine',lambda:[weighted_knn(geo,'precomputed',15,3)]*len(xt))]
    rows=[];regions=arrays['meta'].region_name.to_numpy()
    for name,make in rules:
        graphs=make();f=fit_model(xt,k,graphs=graphs)
        rows.append({'rule':name,'ARI_to_main':adjusted_rand_score(reference['labels'][-1],f['labels'][-1]),
                     'edge_jaccard':graph_overlap(g,graphs[-1]),
                     'changed':float(np.mean(f['labels'][0]!=f['labels'][-1])),
                     **graph_summary(graphs[-1],regions),**metrics(xt[-1],graphs[-1],f['labels'][-1])})
        print('network',name,flush=True)
    return pd.DataFrame(rows)


def sensitivity(arrays,xt,reference,k):
    rows=[];base=reference['labels'][-1]
    def row(kind,value,fit,common_base=base):
        lab=fit['labels'][-1]
        rows.append({'factor':kind,'value':str(value),'n':len(lab),
                     'ARI':adjusted_rand_score(common_base,lab),
                     'changed_share':float(np.mean(fit['labels'][0]!=lab))})
    for w in CFG['robustness']['weight_grid']:
        z,_=scale_arrays(arrays,spending_weight=w);row('spending_weight',w,fit_model(z,k))
    for neighbors in CFG['robustness']['graph_neighbors']:
        row('neighbors',neighbors,fit_model(xt,k,graphs=make_graphs(xt,neighbors=neighbors)))
    for s in CFG['robustness']['smoothing_grid']:row('smoothing_steps',s,fit_model(xt,k,steps=s))
    for lam in CFG['robustness']['penalty_grid']:row('transition_penalty',lam,fit_model(xt,k,lam=lam))
    for seed in [0,1,2,3,4]:row('seed',seed,fit_model(xt,k,seed=seed))
    z=xt[:,:,:-1].copy()
    z[:,:,7:]*=np.sqrt(9/8)
    row('unreported_coordinate','excluded',fit_model(z,k))
    for policy in ['original','drop']:
        p,_=load_panel(policy);a=build_arrays(p);z,_=scale_arrays(a);f=fit_model(z,k)
        ix={tid:i for i,tid in enumerate(a['ids'])};common=[i for i,tid in enumerate(arrays['ids']) if tid in ix]
        loc=[ix[arrays['ids'][i]] for i in common]
        ff={'labels':f['labels'][:,loc]};row('city_policy',policy,ff,base[common])
    p,_=load_panel();a=build_arrays(p,window=3);z,_=scale_arrays(a);f=fit_model(z,k)
    row('window_months',3,f)
    for name,columns in [('level',[0]),('all_spending',None)]:     # убираем общий дрейф страны: центрирование координат расходов в каждом окне
        z,_=scale_arrays(centre_windows(arrays,columns));row('window_centering',name,fit_model(z,k))
    return pd.DataFrame(rows)


def baseline_comparison(xt,graphs,k,main,extended=False):
    """Common original-space ICVI on 3 dates; seed variability reported separately.

    Snapshot algorithms do not receive the temporal regulariser. This difference
    is explicit; no universal dominance or equal tuning budget is claimed.
    """
    from pipeline import to_ig,calibrate,snap,temporal
    ts=[0,len(xt)//2,len(xt)-1];rows=[];seeds=[42,43,44]
    methods={
      'k-means (срезы)':lambda x,a,seed:KMeans(k,n_init=10,random_state=seed).fit_predict(x),
      'Ward (срезы)':lambda x,a,seed:AgglomerativeClustering(k,linkage='ward').fit_predict(x),
      'GMM diagonal (срезы)':lambda x,a,seed:GaussianMixture(k,covariance_type='diag',n_init=3,random_state=seed).fit_predict(x),
      'Spectral (срезы)':lambda x,a,seed:SpectralClustering(k,affinity='precomputed',random_state=seed,n_init=10).fit_predict(a),
    }
    gam=calibrate(to_ig(graphs[len(xt)//2]),target=k)
    methods['Leiden (срезы)']=lambda x,a,seed:snap(to_ig(a),gam,seed)
    for name,func in methods.items():
        start=time.perf_counter();ev=[];last=[]
        for t in ts:
            for seed in seeds:
                lab=func(xt[t],graphs[t],seed)
                ev.append({'method':name,'t':t,'seed':seed,'K_actual':len(np.unique(lab)),**metrics(xt[t],graphs[t],lab)})
                if t==ts[-1]:last.append(lab)
        stability=np.mean([adjusted_rand_score(last[0],l) for l in last[1:]])
        for r in ev:r.update(seed_ARI=float(stability),seconds=(time.perf_counter()-start)/len(ev))
        rows.extend(ev);print('baseline',name,round(time.perf_counter()-start,1),flush=True)
    for name,steps,lam in [('Общие центры, без сети',0,.5),('GS-TKM без Витерби',1,0),('GS-TKM',CFG['model']['smoothing_steps'],CFG['model']['transition_penalty'])]:
        start=time.perf_counter();ev=[];last=[]
        for seed in seeds:
            f=main if name=='GS-TKM' and seed==42 else fit_model(xt,k,graphs=graphs,seed=seed,steps=steps,lam=lam)
            last.append(f['labels'][-1])
            for t in ts:ev.append({'method':name,'t':t,'seed':seed,'K_actual':k,**metrics(xt[t],graphs[t],f['labels'][t])})
        st=np.mean([adjusted_rand_score(last[0],l) for l in last[1:]])
        for r in ev:r.update(seed_ARI=float(st),seconds=(time.perf_counter()-start)/len(ev))
        rows.extend(ev)
    G=[to_ig(g) for g in graphs];last=[];ev=[];start=time.perf_counter()
    for seed in seeds:
        lab=temporal(G,omega=.2,gam=gam,seed=seed);last.append(lab[-1])
        for t in ts:ev.append({'method':'Multilayer Leiden','t':t,'seed':seed,'K_actual':len(np.unique(lab[t])),**metrics(xt[t],graphs[t],lab[t])})
    st=np.mean([adjusted_rand_score(last[0],l) for l in last[1:]])
    for r in ev:r.update(seed_ARI=float(st),seconds=(time.perf_counter()-start)/len(ev))
    rows.extend(ev)
    if extended:
        import sys,logging,torch
        torch.set_num_threads(2);logging.disable(logging.CRITICAL)
        sys.path.insert(0,str(ROOT/'external/KEFRiN'));sys.path.insert(0,str(ROOT/'external/CANUS'))
        import kefrin
        from canus import CANUSClusterer
        from methods2 import dmon
        # Preserve block weights: authors' optional second z-score is disabled.
        def kef(x,a,seed,metric='euclidean',xi=1.):
            c=kefrin.KEFRiNConfig(n_clusters=k,rho=1.,xi=xi,distance_metric=kefrin.DistanceMetric(metric),
                n_init=10,max_iterations=1000,random_state=seed,
                preprocessing_y=kefrin.PreprocessingMethod.NONE,preprocessing_p=kefrin.PreprocessingMethod.NONE)
            return kefrin.KEFRiN(c).fit_predict(x,(a.toarray()>0).astype(float))
        def can(x,a,seed):
            torch.manual_seed(seed)
            return CANUSClusterer(n_clusters=k,seed=seed,epochs=300,learning_rate=.001,tau=1.,rho=1.,zeta=1.,
                update_rule='vanilla_filtered',attribute_distance='cosine',network_distance='cosine',device='cpu').fit(x,(a.toarray()>0).astype(float)).y_pred
        funcs={'KEFRiN Euclidean (авторы)':lambda x,a,s:kef(x,a,s),
               'KEFRiN Cosine (авторы)':lambda x,a,s:kef(x,a,s,metric='cosine'),
               'CANUS (авторы)':can,'DMoN':lambda x,a,s:dmon(x,a,k,epochs=300,seed=s)}
        for name,func in funcs.items():
            start=time.perf_counter();ev=[];last=[]
            for t in ts:
                for seed in seeds:
                    lab=func(xt[t],graphs[t],seed)
                    ev.append({'method':name,'t':t,'seed':seed,'K_actual':len(np.unique(lab)),**metrics(xt[t],graphs[t],lab)})
                    if t==ts[-1]:last.append(lab)
                    print('baseline',name,'date',t,'seed',seed,round(time.perf_counter()-start,1),flush=True)
            st=np.mean([adjusted_rand_score(last[0],l) for l in last[1:]])
            for r in ev:r.update(seed_ARI=float(st),seconds=(time.perf_counter()-start)/len(ev))
            rows.extend(ev)
    raw=pd.DataFrame(rows)
    summary=raw.groupby('method').mean(numeric_only=True).reset_index()
    # Separate ranks: feature geometry versus agreement with a feature-derived network.
    summary['feature_rank']=summary.SW.rank(ascending=False)+summary.CH.rank(ascending=False)+summary.S_Dbw.rank()
    summary['network_rank']=summary.AVI_ref.rank(ascending=False)
    return raw,summary


def _within_effect(y,lab,regions):
    codes,reg=np.unique(regions,return_inverse=True);g=len(codes);n=len(y)
    yr=y-pd.Series(y).groupby(reg).transform('mean').to_numpy()
    h=pd.get_dummies(lab).to_numpy(float)
    hr=h-pd.DataFrame(h).groupby(reg).transform('mean').to_numpy()
    beta,_,rank,_=np.linalg.lstsq(hr,yr,rcond=None)
    s0=yr@yr;s1=np.sum((yr-hr@beta)**2);sst=np.sum((y-y.mean())**2)
    if n<=g+rank or sst==0:return np.nan
    return float((s0/(n-g)-s1/(n-g-rank))/(sst/(n-1)))

def _controlled_effect(y,lab,regions,controls):
    """Increment over region effects and log population/wage, with actual ranks."""
    _,reg=np.unique(regions,return_inverse=True);g=reg.max()+1;n=len(y)
    yr=y-pd.Series(y).groupby(reg).transform('mean').to_numpy()
    c=controls-pd.DataFrame(controls).groupby(reg).transform('mean').to_numpy()
    h=pd.get_dummies(lab).to_numpy(float);hr=h-pd.DataFrame(h).groupby(reg).transform('mean').to_numpy()
    by,_,rank_c,_=np.linalg.lstsq(c,yr,rcond=None);bh=np.linalg.lstsq(c,hr,rcond=None)[0]
    yr=yr-c@by;hr=hr-c@bh
    beta,_,rank_h,_=np.linalg.lstsq(hr,yr,rcond=None)
    s0=yr@yr;s1=np.sum((yr-hr@beta)**2);sst=np.sum((y-y.mean())**2)
    if n<=g+rank_c+rank_h or sst==0:return np.nan
    return float((s0/(n-g-rank_c)-s1/(n-g-rank_c-rank_h))/(sst/(n-1)))


def external_validation(arrays,models,permutations=499):
    """Independent indicators excluded from model fitting; omissions stay visible."""
    m=arrays['meta'];regions=m.region_name.fillna('?').to_numpy();variables={}
    variables['Доступность рынков (лог)']=np.log(m.market_access.where(m.market_access>0)).to_numpy()
    f=ROOT/'data/external/validation.csv'
    if f.exists():
        d=pd.read_csv(f,dtype={'oktmo8':str})
        labels={'retail':'Оборот торговли / житель, 2024','catering':'Оборот общепита / житель, 2024',
                'housing':'Ввод жилья / житель, 2024','investment':'Инвестиции / житель, 2023',
                'hotel_beds':'Гостиничные места / житель, 2024'}
        for code in d.indicator.unique():
            dd=d[d.indicator.eq(code)].set_index('oktmo8').value
            y=m.oktmo8.map(dd)/m['pop']
            variables[labels[code]]=np.log1p(y.where(y>=0)).to_numpy()
    rows=[];rng=np.random.default_rng(542);crng=np.random.default_rng(1542)
    control=np.c_[np.log(m['pop'].where(m['pop']>0)),np.log(m.wage.where(m.wage>0))]
    for lens,lab in models.items():
        vars_=dict(variables)
        if lens=='spend':
            vars_.update({'Зарплата (лог), 2023':np.log(m.wage.where(m.wage>0)).to_numpy(),
                          'Население (лог), 2023':np.log(m['pop'].where(m['pop']>0)).to_numpy()})
        for name,y in vars_.items():
            use=np.isfinite(y);yy=y[use];ll=lab[use];rr=regions[use];obs=_within_effect(yy,ll,rr)
            null=[];groups=[np.flatnonzero(rr==r) for r in np.unique(rr)]
            for _ in range(permutations):
                pp=ll.copy()
                for ids in groups:pp[ids]=rng.permutation(pp[ids])
                null.append(_within_effect(yy,pp,rr))
            # Regions, not individual municipalities, are resampled for intervals.
            boot=[];u=np.unique(rr)
            for _ in range(200):
                chunks=[np.flatnonzero(rr==r) for r in rng.choice(u,len(u),replace=True)]
                ix=np.concatenate(chunks);br=np.concatenate([np.full(len(chunk),j) for j,chunk in enumerate(chunks)])
                boot.append(_within_effect(yy[ix],ll[ix],br))
            valid=use&np.isfinite(control).all(1);yc=y[valid];lc=lab[valid];rc=regions[valid];cc=control[valid];uc=np.unique(rc)
            controlled=_controlled_effect(yc,lc,rc,cc);cb=[]
            for _ in range(200):
                chunks=[np.flatnonzero(rc==r) for r in crng.choice(uc,len(uc),replace=True)]
                ix=np.concatenate(chunks);br=np.concatenate([np.full(len(chunk),j) for j,chunk in enumerate(chunks)])
                cb.append(_controlled_effect(yc[ix],lc[ix],br,cc[ix]))
            if name not in variables:controlled=np.nan;cb=[np.nan];valid[:]=False
            rows.append({'lens':lens,'indicator':name,'n':int(use.sum()),'regions':len(u),'coverage':float(use.mean()),
                         'delta_adj_r2':obs,'region_boot_low':float(np.nanquantile(boot,.025)),
                         'region_boot_high':float(np.nanquantile(boot,.975)),
                         'p_descriptive':(1+np.sum(np.asarray(null)>=obs))/(permutations+1),
                         'controlled_n':int(valid.sum()),'delta_adj_r2_region_population_wage':controlled,
                         'controlled_boot_low':float(np.nanquantile(cb,.025)),'controlled_boot_high':float(np.nanquantile(cb,.975))})
            print('validation',lens,name,int(use.sum()),round(obs,4),flush=True)
    out=pd.DataFrame(rows)
    p=out.p_descriptive.to_numpy();order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out['q_BH']=np.clip(q[np.argsort(order)],0,1)
    return out


def inductive_labels(train_x,test_x,fit,return_distances=False):
    """One-way GCN extension. Test nodes never alter the training graph/centroids."""
    nn=NearestNeighbors(n_neighbors=min(15,len(train_x)),metric='cosine').fit(train_x[:,:7])
    d,ix=nn.kneighbors(test_x[:,:7]);a=fit['graphs'][-1]
    # Undo mean-weight normalisation through the selected training-neighbour distances.
    nn0=NearestNeighbors(n_neighbors=min(16,len(train_x)),metric='cosine').fit(train_x[:,:7])
    dd,jj=nn0.kneighbors(train_x[:,:7]);lookup={(i,int(j)):float(z) for i,(js,ds) in enumerate(zip(jj,dd)) for j,z in zip(js,ds) if i!=j}
    co=a.tocoo();raw=[1/(lookup.get((int(i),int(j)),lookup.get((int(j),int(i)),0))+.001) for i,j in zip(co.row,co.col)]
    norm=float(np.mean(raw));w=1/(d+.001)/norm
    dq=1+w.sum(1);dt=1+np.asarray(a.sum(1)).ravel()
    sm=test_x/dq[:,None]+np.sum((w/np.sqrt(dq[:,None]*dt[ix]))[:,:,None]*train_x[ix],axis=1)
    distances=cdist(sm,fit['centers'],'sqeuclidean')
    return distances if return_distances else distances.argmin(1)


def practical_cv(arrays,k):
    """2023-only features, fold-local transformations/graphs/types, 2024 target.

    This tests spatial transfer of an association; folds' training targets already
    include observed 2024. It is explicitly NOT a historical future forecast.
    """
    n=len(arrays['ids']);raw=arrays['raw'];y=np.log(raw[12:,:,0].mean(0)/raw[:12,:,0].mean(0))
    regions=arrays['meta'].region_name.fillna('?').to_numpy();a=dict(arrays);a['spending']=a['spending'][:1]
    pred=[];rows=[]
    for protocol,split in [('МО случайно',KFold(5,shuffle=True,random_state=842)),('Регионы целиком',GroupKFold(5))]:
        predictions={name:np.full(n,np.nan) for name in ['Регион','Регион + признаки','Регион + признаки + тип','Нелинейный контроль по признакам']}
        foldid=np.zeros(n,int)
        for f,(train,test) in enumerate(split.split(np.arange(n),groups=regions)):
            xt,_=scale_arrays(a,training_indices=train)
            fit=fit_model(xt[:,train],k)
            lt=fit['labels'][0];lv=inductive_labels(xt[0,train],xt[0,test],fit)
            re=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(regions[train,None]);rr=re.transform(regions[:,None])
            te=OneHotEncoder(categories=[np.arange(k)],handle_unknown='ignore',sparse_output=False).fit(lt[:,None])
            tt=np.zeros((n,k));tt[train]=te.transform(lt[:,None]);tt[test]=te.transform(lv[:,None])
            designs={'Регион':rr,'Регион + признаки':np.c_[rr,xt[0]],'Регион + признаки + тип':np.c_[rr,xt[0],tt]}
            for name,x in designs.items():
                model=Ridge(alpha=CFG['validation']['ridge_alpha']).fit(x[train],y[train]);predictions[name][test]=model.predict(x[test])
            control=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,learning_rate=.05,l2_regularization=10,random_state=842)
            control.fit(designs['Регион + признаки'][train],y[train]);predictions['Нелинейный контроль по признакам'][test]=control.predict(designs['Регион + признаки'][test])
            foldid[test]=f
        for name,p in predictions.items():
            rows.append({'protocol':protocol,'model':name,'n':n,'MAE_log_points':mean_absolute_error(y,p),
                         'R2':r2_score(y,p)})
            pred.extend({'protocol':protocol,'model':name,'tid':tid,'region':r,'fold':int(f),'actual':float(yy),'prediction':float(pp)}
                        for tid,r,f,yy,pp in zip(arrays['ids'],regions,foldid,y,p))
        base=np.abs(y-predictions['Регион + признаки']);typ=np.abs(y-predictions['Регион + признаки + тип']);gain=base-typ
        rng=np.random.default_rng(942);u=np.unique(regions);boot=[]
        for _ in range(1000):
            ix=np.concatenate([np.flatnonzero(regions==r) for r in rng.choice(u,len(u),replace=True)])
            boot.append(float(gain[ix].mean()))
        for row in rows:
            if row['protocol']==protocol:row.update(type_MAE_gain=float(gain.mean()),gain_low=float(np.quantile(boot,.025)),gain_high=float(np.quantile(boot,.975)))
        print('CV',protocol,'type MAE gain',float(gain.mean()),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(pred)


def transitions(arrays,fit,boot):
    """At least 3 subsequent monthly endpoints required to confirm a transition."""
    lab=fit['labels'];margin,_=assigned_margin(fit['distances'],lab);rows=[]
    for t in range(1,len(lab)):
        for i in np.flatnonzero(lab[t]!=lab[t-1]):
            censored=t+2>=len(lab)
            persists=False if censored else bool(np.all(lab[t:t+3,i]==lab[t,i]))
            confidence=float(boot['confidence'][t,i])
            rows.append({'tid':arrays['ids'][i],'month':str(arrays['months'][t])[:7],
                         'name':arrays['meta'].iloc[i].municipal_district_name,'region':arrays['meta'].iloc[i].region_name,
                         'from':int(lab[t-1,i]),'to':int(lab[t,i]),'censored':censored,
                         'persists_3_endpoints':persists,'subsample_agreement':confidence,
                         'signed_margin':float(margin[t,i]),
                         'trajectory_support_lower_bound':None if censored else float(max(0,boot['confidence'][t-1:t+3,i].sum()-3)),
                         'robust_descriptive':bool(persists and max(0,boot['confidence'][t-1:t+3,i].sum()-3)>=.75 and margin[t,i]>=0)})
    return pd.DataFrame(rows)
