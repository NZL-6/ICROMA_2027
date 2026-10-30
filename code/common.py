"""Isolated experiment paths, atomic records and source preservation."""
from pathlib import Path
import hashlib, importlib.util, json, math, os, sys
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
ENGINE=ROOT/'vendor'
SOURCE=ROOT
OUT=ROOT
PYTHON=Path(sys.executable)
SEED=20261026
STAGES={'A':('P1','P2',['Td','Tr']),'B':('P2','P3',['Td','Tr','O'])}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def clean(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if hasattr(v,'item'):return clean(v.item())
    return v
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
write=save
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def bootstrap_engine():
    if str(ENGINE) not in sys.path:sys.path.append(str(ENGINE))
def value(r,s):
    i,n=r['train'],int(r['visit'])
    if r['kind']=='Td':return s[i][n]['departure']-s[i][n]['arrival']
    if r['kind']=='Tr':return s[i][n+1]['arrival']-s[i][n]['departure']
    return int(s[i][n]['departure']<s[r['other_train']][int(r['other_visit'])]['departure'])
def verify_sources():
    for item in read(ROOT/'source_lock.json')['files']:
        p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
        assert sha(p)==item['sha256'],('release source changed',str(p))
    return True
def verify_frozen():
    if (ROOT/'execution_lock.json').exists():
        for item in read(ROOT/'execution_lock.json')['files']:
            p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
            assert sha(p)==item['sha256'],('frozen implementation changed',str(p))
    return True
def assert_ready():
    assert read(ROOT/'protocol.json')['authorized']

