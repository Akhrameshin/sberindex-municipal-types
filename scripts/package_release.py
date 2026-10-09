"""Package reviewed artifacts; exclude caches, unlicensed author code and secrets."""
from pathlib import Path
import zipfile,hashlib,shutil,gzip
ROOT=Path(__file__).resolve().parents[1]
def prepare_sources():
    """Preserve source bytes with deterministic gzip; avoid oversized Git files."""
    for source in sorted((ROOT/'data/external/raw').glob('*.csv')):
        target=source.with_suffix('.csv.gz')
        if target.exists():continue
        temporary=target.with_suffix('.gz.tmp')
        with source.open('rb') as inp,temporary.open('wb') as out:
            with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0,compresslevel=6) as compressed:
                shutil.copyfileobj(inp,compressed)
        temporary.replace(target)
def package():
    prepare_sources()
    dest=ROOT/'dist';dest.mkdir(exist_ok=True);target=dest/'sberindex-cluster-share.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ROOT.rglob('*')):
            rel=p.relative_to(ROOT)
            if not p.is_file():continue
            if rel.parts[0] in ['external','dist']:continue
            if any(part in ['.venv','__pycache__'] or (part.startswith('.') and part!='.gitignore') for part in rel.parts):continue
            if p.name=='dashboard_payload.json' or p.suffix=='.pyc' or (p.suffix=='.log' and p.name!='tests.log'):continue
            if p.name=='main.pdf' and 'overleaf' in rel.parts:continue
            if rel.parts[:3]==('data','external','raw') and p.suffix=='.csv':continue
            z.write(p,'sberindex-cluster/'+str(rel))
    print('Release',target,'bytes',target.stat().st_size,'sha256',hashlib.sha256(target.read_bytes()).hexdigest())
if __name__=='__main__':package()
