"""Conservative retrospective cold-start projection; no fabricated probabilities."""
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import linear_sum_assignment
from atlas_v2 import *
from atlas_experiments import inductive_labels

def spend_profile(raw):
    total=raw[:,0].mean();parts=np.r_[raw[:,1:].sum(0),np.sum(raw[:,0]-raw[:,1:].sum(1))]
    shares=np.maximum(parts/parts.sum(),1e-6);shares/=shares.sum();log=np.log(shares)
    return np.r_[np.log(total),log-log.mean()]

def context_profile(m):
    groups=np.array([m[['emp_'+x for x in sectors]].astype(float).fillna(0).sum() for sectors in GROUPS.values()])
    groups/=max(groups.sum(),1);unknown=max(1-groups.sum(),0)
    if not np.isfinite(m.emp_total) or m.emp_total<=0:groups[:]=np.nan;unknown=np.nan
    return np.r_[np.log(m['pop']) if m['pop']>0 else np.nan,np.log(m.wage) if m.wage>0 else np.nan,groups,unknown]

def transform_query(spending,context,transforms):
    weight=transforms['weight'];clip=CFG['features']['clip_z']
    z=np.clip(transforms['spend_scaler'].transform(spending),-clip,clip)*np.sqrt(weight/7)
    c=transforms['context_imputer'].transform(context);c=np.clip(transforms['context_scaler'].transform(c),-clip,clip)*np.sqrt((1-weight)/c.shape[1])
    return np.c_[z,c]

def wilson_lower(correct,n,z=1.96):
    if not n:return 0.
    p=correct/n
    return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def calibrate_and_assign(panel,arrays,k,reference):
    rng=np.random.default_rng(1142);n=len(arrays['ids']);size=min(200,n)
    selected=set(rng.choice(n,size,replace=False));regions=arrays['meta'].region_name.to_numpy();records=[]
    lengths=CFG['assignment']['calibration_months']
    for fold,(train,test) in enumerate(GroupKFold(5).split(np.arange(n),groups=regions)):
        query=np.array([i for i in test if i in selected]);xt,tr=scale_arrays(arrays,training_indices=train);fit=fit_model(xt[:,train],k)
        overlap=np.array([[np.sum((reference[:,train]==a)&(fit['labels']==b)) for b in range(k)] for a in range(k)])
        aa,bb=linear_sum_assignment(-overlap);mp=np.empty(k,int);mp[bb]=aa
        for months in lengths:
            s=np.array([spend_profile(arrays['raw'][-min(months,12):,i]) for i in query])
            xq=transform_query(s,arrays['context'][query],tr);dist=inductive_labels(xt[-1,train],xq,fit,True)
            lab=dist.argmin(1);sorted_=np.sort(dist,axis=1);margin=(sorted_[:,1]-sorted_[:,0])/(sorted_[:,1]+sorted_[:,0]+1e-12)
            for i,l,m in zip(query,lab,margin):records.append({'months':months,'tid':arrays['ids'][i],'fold':fold,'agreement':bool(mp[l]==reference[-1,i]),'margin':m,'selected':bool(m>=.15)})
    raw=pd.DataFrame(records);rows=[]
    for months,part in raw.groupby('months'):
        subset=part[part.selected];correct=int(subset.agreement.sum());nn=len(subset)
        rows.append({'months':months,'n':len(part),'coverage':nn/len(part),'agreement':part.agreement.mean(),
                     'n_selected':nn,'agreement_selected':correct/nn if nn else np.nan,'wilson_low':wilson_lower(correct,nn)})
    calibration=pd.DataFrame(rows)
    xt,tr=scale_arrays(arrays);fit=fit_model(xt,k);names,_=semantic_names(arrays,reference)
    need=set(arrays['all_months'][-12:]);eligible=float(calibration[calibration.months.eq(12)].wilson_low.iloc[0])>=.8
    output=[];p=panel.copy();p['tid']=p.tid.astype(str)
    for tid,part in p[~p.tid.isin(arrays['ids'])].groupby('tid'):
        m=part.iloc[0];row={'tid':tid,'name':m.municipal_district_name,'region':m.region_name,'months':part.month.nunique(),
                          'status':'unassigned','reason':'нет полного годового окна 2024','type':None,'type_name':None,'signed_margin':None}
        part=part[part.month.isin(need)].sort_values('month');levels=['total','health','catering','food','market','transport']
        if set(part.month)==need and part[levels].notna().all().all() and (part.total>0).all():
            s=spend_profile(part[levels].to_numpy())[None];c=context_profile(m)[None];xq=transform_query(s,c,tr)
            d=inductive_labels(xt[-1],xq,fit,True)[0];order=np.argsort(d);l=int(order[0]);mar=float((d[order[1]]-d[l])/(d[order[1]]+d[l]+1e-12))
            row['signed_margin']=mar
            if eligible and mar>=.15:
                row.update(status='provisional',reason='полный 2024; пройдена калибровка и порог запаса',type=l,type_name=names[l])
            else:row['reason']='не пройден порог запаса или калибровки'
        output.append(row)
    return pd.DataFrame(output),calibration,raw

if __name__=='__main__':
    import json
    out=ROOT/'results/v2';k=json.loads((out/'summary.json').read_text())['selected_k'];p,_=load_panel();a=build_arrays(p)
    lab=np.load(out/f'model_K{k}.npz')['labels'];assigned,calibration,raw=calibrate_and_assign(p,a,k,lab)
    assigned.to_csv(out/'incomplete_assignment.csv',index=False);calibration.to_csv(out/'assignment_calibration.csv',index=False);raw.to_csv(out/'assignment_calibration_raw.csv',index=False)
    print(calibration.to_string(index=False));print(assigned.status.value_counts().to_dict())
