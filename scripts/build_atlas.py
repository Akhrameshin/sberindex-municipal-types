"""Single entry point; all stages consume the same versioned source panel/config."""
from pathlib import Path
import sys, argparse, time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas_v2 import *
from atlas_experiments import *
OUT=ROOT/'results/v2';OUT.mkdir(parents=True,exist_ok=True)

def models(reps=50):
    panel,mapping=load_panel();a=build_arrays(panel);x,_=scale_arrays(a);xs,_=scale_arrays(a,lens='spend')
    mapping.to_csv(OUT/'territory_mapping.csv',index=False)
    gaps=panel.groupby('tid').month.nunique();gaps=gaps[~gaps.index.isin(a['ids'])]
    gaps.rename('observed_months').to_csv(OUT/'excluded.csv')
    fits={};selection=[]
    for k in CFG['model']['candidate_k']:
        f=fit_model(x,k);fits[k]=f
        b=bootstrap_model(a,k,f['labels'],reps=CFG['selection']['bootstrap_runs'])
        share=np.bincount(f['labels'][-1],minlength=k).min()/len(a['ids'])
        selection.append({'K':k,'min_share':share,'ARI_mean':b['ari'].mean(),'ARI_low':np.quantile(b['ari'],.025),
                          'eligible':bool(share>=CFG['selection']['minimum_cluster_share'] and b['ari'].mean()>=CFG['selection']['minimum_mean_ari']),
                          **metrics(x[-1],f['graphs'][-1],f['labels'][-1])})
        print('candidate',selection[-1],flush=True)
    table=pd.DataFrame(selection);table.to_csv(OUT/'k_selection.csv',index=False)
    eligible=table[table.eligible]
    chosen=int(eligible.K.max()) if len(eligible) else int(table.sort_values('ARI_mean',ascending=False).iloc[0].K)
    dashboard={};rob=[]
    for k in sorted(set(CFG['model']['displayed_k']+[chosen])):
        fit=fits[k];b=bootstrap_model(a,k,fit['labels'],reps=reps)
        fb=fit_model(xs,k);names,profiles=semantic_names(a,fit['labels']);profiles.to_csv(OUT/f'profiles_K{k}.csv',index=False)
        margin,alt=assigned_margin(fit['distances'],fit['labels'])
        np.savez_compressed(OUT/f'model_K{k}.npz',labels=fit['labels'],spend_labels=fb['labels'],distances=fit['distances'],
                            centers=fit['centers'],confidence=b['confidence'],margin=margin,alternative=alt,
                            ari=b['ari'],jaccard=b['jaccard'],changed_share=b['changed_share'],runs=b['runs'])
        for typ in range(k):
            robust={'K':k,'type':typ,'name':names[typ],'n':int((fit['labels'][-1]==typ).sum()),
                    'ARI_mean':b['ari'].mean(),'ARI_low':np.quantile(b['ari'],.025),'ARI_high':np.quantile(b['ari'],.975),
                    'Jaccard_mean':b['jaccard'][:,typ].mean(),'Jaccard_low':np.quantile(b['jaccard'][:,typ],.025),
                    'Jaccard_high':np.quantile(b['jaccard'][:,typ],.975),'resamples':reps}
            rob.append(robust)
        rows=[]
        for t,month in enumerate(a['months']):
            for i,tid in enumerate(a['ids']):
                rows.append({'tid':tid,'month':str(month)[:7],'type':int(fit['labels'][t,i]),'type_name':names[int(fit['labels'][t,i])],
                             'spend_type':int(fb['labels'][t,i]),'subsample_agreement':b['confidence'][t,i],
                             'signed_margin':margin[t,i],'nearest_alternative':int(alt[t,i])})
        pd.DataFrame(rows).to_csv(OUT/f'types_K{k}.csv',index=False)
        transitions(a,fit,b).to_csv(OUT/f'transitions_K{k}.csv',index=False)
        dashboard[str(k)]={'names':names,'profiles':profiles.to_dict('records'),'bootstrap_ARI':b['ari'],
                           'label_lens_ARI':adjusted_rand_score(fit['labels'][-1],fb['labels'][-1])}
    pd.DataFrame(rob).to_csv(OUT/'robustness.csv',index=False)
    info={'selected_k':chosen,'selection_fallback':bool(eligible.empty),'n':len(a['ids']),'t':len(x),'months':[str(m)[:7] for m in a['months']],
          'original_n':int(pd.read_parquet(ROOT/'data/processed/panel.parquet').tid.nunique()),
          'aggregated_panel_n':int(panel.tid.nunique()),'incomplete_n':len(gaps),
          'context_missing_n':int(a['meta'].context_missing.sum()),'graph':graph_summary(fits[chosen]['graphs'][-1],a['meta'].region_name.to_numpy()),
          'models':dashboard,'lineage':lineage()}
    write_json(OUT/'summary.json',info)
    print('SELECTED K',chosen,flush=True)

def validation_only():
    summary=json.loads((OUT/'summary.json').read_text());k=summary['selected_k']
    p,_=load_panel();a=build_arrays(p);saved=np.load(OUT/f'model_K{k}.npz')
    external_validation(a,{'combined':saved['labels'][-1],'spend':saved['spend_labels'][-1]},CFG['validation']['permutations']).to_csv(OUT/'external_validation.csv',index=False)
    cv,pred=practical_cv(a,k);cv.to_csv(OUT/'practical_cv.csv',index=False);pred.to_csv(OUT/'cv_predictions.csv',index=False)

def experiments(extended=False):
    summary=json.loads((OUT/'summary.json').read_text());k=summary['selected_k']
    p,_=load_panel();a=build_arrays(p);x,_=scale_arrays(a);f=fit_model(x,k)
    saved=np.load(OUT/f'model_K{k}.npz')
    if not np.array_equal(saved['labels'],f['labels']):raise RuntimeError('Models are stale; rebuild models first')
    sensitivity(a,x,f,k).to_csv(OUT/'sensitivity.csv',index=False)
    network_comparison(a,x,f,k).to_csv(OUT/'network_rules.csv',index=False)
    raw,table=baseline_comparison(x,f['graphs'],k,f,extended)
    raw.to_csv(OUT/'baselines_raw.csv',index=False);table.to_csv(OUT/'baselines.csv',index=False)
    validation_only()
    summary['lineage']=lineage();summary['extended_baselines']=extended;write_json(OUT/'summary.json',summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['models','experiments','validation','all'],default='all')
    parser.add_argument('--resamples',type=int,default=CFG['robustness']['bootstrap_runs']);parser.add_argument('--extended',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    if args.stage in ['models','all']:models(args.resamples)
    if args.stage in ['experiments','all']:experiments(args.extended)
    if args.stage=='validation':validation_only()
    print('Elapsed seconds',round(time.perf_counter()-start,1),flush=True)
