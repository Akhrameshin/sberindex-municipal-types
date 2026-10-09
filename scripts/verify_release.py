"""Deterministic checks of submitted artifacts, independent of UI formatting."""
from pathlib import Path
import sys,json,hashlib,re,zipfile,base64,gzip
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas_v2 import *
OUT=ROOT/'results/v2'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    from package_release import prepare_sources
    prepare_sources()
    info=json.loads((OUT/'summary.json').read_text());p,_=load_panel();a=build_arrays(p);xt,_=scale_arrays(a);graphs=make_graphs(xt)
    html=(ROOT/'dashboard_story.html').read_text();match=re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S)
    data=json.loads(match[1]);assert len(data['ids'])==len(a['ids'])==info['n'];assert data['months']==info['months']
    assert data['ids']==a['ids']
    checks=[]
    for k in info['models']:
        f=fit_model(xt,int(k),graphs=graphs);z=np.load(OUT/f'model_K{k}.npz');table=pd.read_csv(OUT/f'types_K{k}.csv',dtype={'tid':str})
        assert np.array_equal(f['labels'],z['labels']);assert np.array_equal(data['K'][k]['L'],z['labels'])
        assert len(z['ari'])==50;assert np.all(z['runs']>0)
        for t,month in enumerate(a['months']):
            rows=table[table.month.eq(str(month)[:7])].set_index('tid').reindex(a['ids'])
            assert np.array_equal(rows.type.to_numpy(),z['labels'][t]);assert np.allclose(rows.subsample_agreement,z['confidence'][t])
        for i,ns in enumerate(data['nb']):
            assert all(graphs[-1][i,int(j)]>0 for j in ns)
        checks.append('K'+k+': repeated labels, all months, table agreement, graph edges')
    main=int(info['selected_k']);external=pd.read_csv(OUT/'external_validation.csv');assert external.indicator.nunique()==8
    assert len(pd.read_csv(OUT/'baselines_raw.csv'))>=81
    assert 'delta_adj_r2_region_population_wage' in external
    assert len(pd.read_csv(OUT/'practical_cv.csv'))==8
    assigned=pd.read_csv(OUT/'incomplete_assignment.csv');assert len(assigned)==info['incomplete_n']
    assert len(base64.b64decode(data['reportPDF']))>10000
    assert base64.b64decode(data['reportPDF'])==(ROOT/'report/methodology.pdf').read_bytes()
    source=json.loads((ROOT/'data/external/source_manifest.json').read_text())
    for row in source:
        raw=ROOT/row['file']
        if raw.exists():assert sha(raw)==row['sha256'],row['file']
        compressed=raw.with_suffix('.csv.gz')
        if compressed.exists():
            digest=hashlib.sha256()
            with gzip.open(compressed,'rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==row['sha256'],str(compressed)
    try:
        from pypdf import PdfReader
        pdf=PdfReader(ROOT/'report/methodology.pdf');pages=len(pdf.pages)
        assert all(len(page.extract_text())>60 for page in pdf.pages)
    except ImportError:
        import shutil,subprocess
        assert shutil.which('pdfinfo'),'нужен pypdf (requirements-core.txt) или pdfinfo'
        pages=int(re.search(r'Pages:\s+(\d+)',subprocess.run(['pdfinfo',str(ROOT/'report/methodology.pdf')],capture_output=True,text=True,check=True).stdout)[1])
    assert 10<=pages<=60
    tests=(OUT/'tests.log').read_text();count=int(re.search(r'(\d+) passed',tests)[1]);assert 'failed' not in tests.lower()
    result={'tests_passed':count,'model_checks':checks,'pdf_pages':pages,'node_count':len(a['ids']),
            'display':f'{count} автоматических тестов; повторный расчёт меток K=3/4/5 и всех 13 окон совпал с таблицами и дашбордом.'}
    write_json(OUT/'verification.json',result)
    info['lineage']=lineage();write_json(OUT/'summary.json',info)
    paths=[ROOT/'configs/atlas.yaml',ROOT/'README.md',ROOT/'Makefile',ROOT/'requirements-core.txt',ROOT/'requirements-research.txt',
           ROOT/'dashboard_story.html',ROOT/'report/methodology.pdf',ROOT/'report/methodology.md',ROOT/'data/processed/panel.parquet',
           *sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'data/external').glob('*.csv')),
           *sorted((ROOT/'data/external/raw').glob('*.csv.gz')),*sorted(OUT.glob('*.csv')),*sorted(OUT.glob('*.npz'))]
    manifest={'lineage':lineage(),'outputs':{str(path.relative_to(ROOT)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in paths}}
    write_json(OUT/'manifest.json',manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':verify()
