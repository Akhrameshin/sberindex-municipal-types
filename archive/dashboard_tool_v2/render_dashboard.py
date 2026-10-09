"""All UI numbers are derived from model outputs; no separate hand-entered results."""
from pathlib import Path
import sys,json,base64,io,zipfile
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas_v2 import *
from atlas_experiments import transitions
OUT=ROOT/'results/v2'

def code_archive():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ['src','scripts','configs','tests','assets','results/v2']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.name not in ['dashboard_payload.json','manifest.json'] and '__pycache__' not in p.parts and (p.suffix not in ['.html','.png','.log','.pdf'] or p.name=='tests.log'):
                    z.write(p,str(p.relative_to(ROOT)))
        for p in [ROOT/'src/atlas_dashboard.html',ROOT/'data/processed/panel.parquet',
                  *sorted((ROOT/'data/external').glob('*.csv')),*sorted((ROOT/'data/external').glob('*.json')),
                  *[ROOT/name for name in ['README.md','Makefile','.gitignore','requirements.txt','requirements-core.txt','requirements-research.txt','SHA256SUMS','LICENSE']]]:
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    return b.getvalue()

def render():
    if CFG['features']['window']!=12:raise ValueError('The primary dashboard describes annual windows; use network/sensitivity experiments for other windows')
    p,mapping=load_panel();a=build_arrays(p);x,_=scale_arrays(a);xs,_=scale_arrays(a,lens='spend');info=json.loads((OUT/'summary.json').read_text())
    geo=json.loads((ROOT/'assets/geography.json').read_text())
    for tid,group in mapping.groupby('analysis_tid'):
        parts=[geo['municipalities'][str(t)] for t in group.source_tid if str(t) in geo['municipalities']]
        if parts:geo['municipalities'][tid]={'path':''.join(g['path'] for g in parts),'x':float(np.mean([g['x'] for g in parts])),'y':float(np.mean([g['y'] for g in parts]))}
    nodes=[]
    raw=a['raw'];other=raw[:,:,0]-raw[:,:,1:].sum(2);allparts=np.concatenate([raw[:,:,1:],other[:,:,None]],2)
    for i,tid in enumerate(a['ids']):
        m=a['meta'].iloc[i];annual=[]
        for t in range(13):annual.append(np.c_[raw[t:t+12,:,0].mean(0),allparts[t:t+12].mean(0)][i].round(4).tolist())
        nodes.append({'tid':tid,'name':m.municipal_district_name,'region':m.region_name,'population':m['pop'],'wage':m.wage,
                      'annual':annual,'contextMissing':bool(m.context_missing),'sectorWithheld':bool(m.sector_withheld),
                      'sectorCoverage':float(m.sector_coverage),'aggregate':tid.startswith('city-')})
    def nearest(z):
        out=[]
        for zz in z:
            d=cdist(zz,zz);np.fill_diagonal(d,np.inf);ix=np.argsort(d,axis=1,kind='stable')[:,:10]
            out.append([[[int(j),round(float(d[i,j]),4)] for j in js] for i,js in enumerate(ix)])
        return out
    analogs={'combined':nearest(x),'spend':nearest(xs)};graphs=make_graphs(x);neighbors=[]
    for g in graphs:
        layer=[]
        for i in range(len(nodes)):
            row=g.getrow(i);order=np.argsort(-row.data,kind='stable')[:10]
            layer.append([[int(row.indices[j]),round(float(row.data[j]),4)] for j in order])
        neighbors.append(layer)
    models={};rob=pd.read_csv(OUT/'robustness.csv')
    for k in info['models']:
        kk=int(k);z=np.load(OUT/f'model_K{k}.npz');f=fit_model(x,kk,graphs=graphs);fb=fit_model(xs,kk,graphs=graphs)
        if not np.array_equal(f['labels'],z['labels']):raise RuntimeError('Dashboard/model mismatch')
        model={}
        for lens,fit in [('combined',f),('spend',fb)]:
            nm,pr=semantic_names(a,fit['labels'],lens);mar,alt=assigned_margin(fit['distances'],fit['labels'])
            model[lens]={'labels':fit['labels'],'names':[nm[i] for i in range(kk)],'profiles':pr.to_dict('records'),
                         'confidence':z['confidence'].round(4) if lens=='combined' else None,
                         'margin':mar.round(4),'alternative':alt}
            if lens=='combined':
                pr.to_csv(OUT/f'profiles_K{k}.csv',index=False)
                info['models'][k]['names']=nm;info['models'][k]['profiles']=pr.to_dict('records')
                for typ,name in nm.items():rob.loc[(rob.K==kk)&(rob.type==typ),'name']=name
                frame=pd.read_csv(OUT/f'types_K{k}.csv',dtype={'tid':str});frame['type_name']=frame.type.map(nm);frame.to_csv(OUT/f'types_K{k}.csv',index=False)
        model['lensARI']=info['models'][k]['label_lens_ARI'];models[k]=model
        transitions(a,f,{'confidence':z['confidence']}).to_csv(OUT/f'transitions_K{k}.csv',index=False)
    rob.to_csv(OUT/'robustness.csv',index=False)
    info['lineage']=lineage();write_json(OUT/'summary.json',info)
    def read(name):
        f=OUT/name;return pd.read_csv(f).to_dict('records') if f.exists() else []
    main=int(info['selected_k']);z=np.load(OUT/f'model_K{main}.npz');uncertain=(z['confidence'][-1]<.75)|(z['margin'][-1]<0)
    inds=[];prof=models[str(main)]['combined']['profiles'];ratio=max(r['spend'] for r in prof)/min(r['spend'] for r in prof)
    inds.append({'big':f'{ratio:.1f}×','title':'Выбирайте сопоставимые территории','text':'Крайние типы различаются по медианным расходам. Для анализа муниципалитета используйте и уровень, и структуру потребления; карта предлагает близкие профили в других регионах.'})
    inds.append({'big':str(int(uncertain.sum())),'title':'Пограничные территории видны','text':'Согласие менее 75% или отрицательный запас до альтернативного центра. Их тип — повод проверить соседние профили и исходные показатели, а не основание для автоматического решения.'})
    inds.append({'big':f'{100*(1-info["graph"]["within_region"]):.1f}%','title':'Сеть выходит за границы региона','text':'Доля связей между разными регионами в декабре 2024. Сеть соединяет близкие профили расходов и помогает находить территории для сравнения.'})
    # Cases: high confidence; borderline; persistent transition with trajectory support.
    high=int(np.flatnonzero(z['labels'][-1]==np.argmax([r['spend'] for r in prof]))[np.argmax(z['confidence'][-1,z['labels'][-1]==np.argmax([r['spend'] for r in prof])])])
    border=int(np.argmin(z['confidence'][-1]+np.maximum(z['margin'][-1],0)))
    tr=pd.read_csv(OUT/f'transitions_K{main}.csv',dtype={'tid':str});stable=tr[tr.robust_descriptive.eq(True)]
    cases=[{'kind':'Устойчивый профиль','index':high,'text':'Высокий уровень расходов: проверьте структуру, зарплату и близкие территории. Высокое согласие пересборок помогает выбрать пример для содержательной интерпретации.'},
           {'kind':'Граница типов','index':border,'text':'Тип зависит от состава выборки. Сравните текущий центр с ближайшей альтернативой и переключите линзу на потребление.'}]
    if len(stable):
        row=stable.sort_values('trajectory_support_lower_bound',ascending=False).iloc[0];ii=a['ids'].index(row.tid)
        cases.append({'kind':'Устойчивая описательная смена','index':ii,'text':f'Смена в {row.month}; новый тип сохраняется на трёх конечных точках. Нижняя граница совместного согласия траектории: {row.trajectory_support_lower_bound:.0%}. Причина изменения требует отдельного исследования.'})
    else:
        cases.append({'kind':'Смена требует проверки','index':int(np.argmax(z['changed_share'])),'text':'Сопоставьте историю и пограничные присвоения. На последних двух окнах устойчивость смены ещё нельзя подтвердить тремя конечными точками.'})
    report=ROOT/'report/methodology.pdf';md=ROOT/'report/methodology.md'
    files={name:(OUT/name).read_text() for name in ['external_validation.csv','baselines_raw.csv','sensitivity.csv','practical_cv.csv','incomplete_assignment.csv'] if (OUT/name).exists()}
    payload={'mainK':main,'months':info['months'],'nodes':nodes,'models':models,'geo':geo,'neighbors':neighbors,'analogs':analogs,'graph':info['graph'],
             'parts':['Здоровье','Общепит','Продовольствие','Маркетплейсы','Транспорт','Прочее'],
             'runid':info['lineage']['run_id'],'selection':read('k_selection.csv'),'robustness':rob.to_dict('records'),
             'baselines':read('baselines.csv'),'networks':read('network_rules.csv'),'sensitivity':read('sensitivity.csv'),
             'external':read('external_validation.csv'),'cv':read('practical_cv.csv'),'insights':inds,'cases':cases,
             'reportPDF':base64.b64encode(report.read_bytes()).decode() if report.exists() else '',
             'reportMD':md.read_text() if md.exists() else '', 'codeZIP':base64.b64encode(code_archive()).decode(),'files':files}
    tests=ROOT/'results/v2/verification.json'
    if tests.exists():payload['verification']=json.loads(tests.read_text()).get('display','')
    target=OUT/'dashboard_payload.json';write_json(target,payload)
    repo=CFG.get('presentation',{}).get('repository_url','');link=f' · <a href="{repo}" target="_blank" rel="noopener">код</a>' if repo else ''
    template=(ROOT/'src/atlas_dashboard.html').read_text().replace('__REPO_LINK__',link);html=template.replace('__PAYLOAD__',target.read_text().replace('</','<\\/'))
    (ROOT/'dashboard_story.html').write_text(html);print('Dashboard bytes',len(html.encode()),flush=True)

if __name__=='__main__':render()
