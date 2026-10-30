"""Independent fresh proof preprocessing inside each timed solver job."""
from pathlib import Path
import hashlib,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
from acceptance import accepted
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def ensure(jobdir,direction):
    jobdir=Path(jobdir);job=read(jobdir/'job.json');out=jobdir/'common_bounds/results'/f"{job['case']}__common_{direction}_inf_dominance.json"
    meta=jobdir/f'bound_{direction}_provenance.json';script=jobdir/'relaxation_worker.py'
    if not script.exists():script.write_text((ROOT/'snapshots/core_template.py').read_text(encoding='utf-8'),encoding='utf-8')
    expected=dict(raw_sha256=sha(jobdir/'raw_input.json'),reference_sha256=sha(jobdir/'reference.json'),source_sha256=sha(script))
    actual=0.
    if not out.exists():
        started=time.perf_counter()
        with (jobdir/f'bound_{direction}_console.log').open('w',encoding='utf-8') as stream:
            p=subprocess.run([sys.executable,str(script),job['case'],'--direction',direction,'--budget','inf','--hint',str(jobdir/'reference.json')],
                stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        actual=time.perf_counter()-started
        if p.returncode:raise RuntimeError(('fresh bound failed',job['case'],job['variant'],direction))
        save(meta,dict(**expected,first_computation_seconds=actual,private_job_cache_only=True,target_optimum_read=False))
    assert meta.exists() and all(read(meta)[k]==v for k,v in expected.items())
    r=read(out);raw=read(jobdir/'raw_input.json')['data'];ref=read(jobdir/'reference.json')['schedule']
    upper=sum(max(0,ref[t['id']][-1]['arrival']-t['visits'][-1]['plan_arrival']) for t in raw['trains'] if t['runs'][0]['resource'].endswith('::'+direction))
    assert r['status']=='OPTIMAL' and accepted(r['relaxed_primal'],r['lower_bound']) and math.isfinite(r['lower_bound']) and r['lower_bound']<=upper+1e-6
    assert r['hint_info']['audit']['schedule_valid']
    assert r['parameters']['workers']==12 and r['parameters']['time_limit_seconds'] is None
    from audit_relaxation import audit_relaxation
    check=audit_relaxation(raw,r,direction);assert check['passed'],check
    r['independent_audit']=check;save(out,r)
    return dict(direction=direction,lower_bound=max(0,math.ceil(r['lower_bound']-1e-6)),proof_path=str(out),proof_sha256=sha(out),actual_computation_seconds=actual,private_cache=True)
