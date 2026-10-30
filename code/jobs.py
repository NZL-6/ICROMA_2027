"""One durable cost ledger per job, including failed/restarted attempts."""
import copy, datetime, os, subprocess, time, re, shutil
from common import *
bootstrap_engine()
from audit import audit_result

ENV={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def now():return datetime.datetime.now().astimezone().isoformat()
def create_job(case,key,policy,mask,reference=None,extra=None):
    folder=ROOT/'cases'/case;run=ROOT/'runs'/f'{case}_{key}'
    if (run/'job.json').exists():return run
    start=time.perf_counter();run.mkdir(exist_ok=True);shutil.copy2(folder/'common_input.json',run/'raw_input.json')
    save(run/'mask.json',mask)
    if reference is not None:
        save(run/'reference.json',reference)
        check=audit_result(read(run/'raw_input.json')['data'],reference,mask)
        assert check['passed'] and check['schedule_valid'],check
        save(run/'reference_audit.json',check)
    job=dict(case=case,key=key,stage='REFERENCE' if policy=='P1' else 'A' if policy=='P2' else 'B',policy=policy,
        variant='FULL' if key.startswith('FULL') else 'REFERENCE' if policy=='P1' else key.split('_')[0],
        reference_generation=reference is None,mask_sha256=sha(run/'mask.json'),reference_sha256=sha(run/'reference.json') if reference else None,
        target_optimum_available=False,solver_source_sha256=sha(ROOT/'solver_worker.py'),created_at=now())
    if extra:job.update(extra)
    save(run/'job.json',job);save(run/'job_preparation.json',dict(seconds=time.perf_counter()-start))
    return run

def execute(run):
    run=Path(run)
    if (run/'result.json').exists() and not (run/'active_attempt.json').exists():
        r=read(run/'result.json');assert r['accepted'] and r['audit']['passed'];return r
    if (run/'proven_infeasible.json').exists():return None
    attempts=read(run/'attempts.json') if (run/'attempts.json').exists() else []
    if (run/'active_attempt.json').exists():
        # Never discard an unfinished native search. Its time is an interval.
        active=read(run/'active_attempt.json');n=active['attempt']
        if active.get('worker_pid'):
            import psutil
            try:
                worker=psutil.Process(active['worker_pid'])
                if str(run) in ' '.join(worker.cmdline()) and worker.create_time()>=datetime.datetime.fromisoformat(active['started_at']).timestamp()-5:
                    worker.wait()  # Wait on the existing OS process; never start a duplicate search.
            except psutil.NoSuchProcess:pass
        arch=run/'interruptions'/f'attempt_{n:03d}';arch.mkdir(parents=True,exist_ok=True)
        for name in ['logs','common_bounds','worker_measurements.json','checkpoints','bound_F_console.log','bound_R_console.log']:
            p=run/name
            if p.is_dir():shutil.copytree(p,arch/name,dirs_exist_ok=True)
            elif p.exists():shutil.copy2(p,arch/name)
        upper=(datetime.datetime.now().astimezone()-datetime.datetime.fromisoformat(active['started_at'])).total_seconds()
        lower=0.;completed=set();started_stamp=datetime.datetime.fromisoformat(active['started_at']).timestamp()
        if (run/'worker_measurements.json').exists() and (run/'worker_measurements.json').stat().st_mtime>=started_stamp:
            calls=read(run/'worker_measurements.json').get('solve_calls',[]);lower=sum(x['native_wall_seconds'] for x in calls);completed={x['direction'] for x in calls}
        for log in (run/'logs').glob('*.log'):
            direction=log.stem.rsplit('_',1)[-1]
            if log.stat().st_mtime<started_stamp or direction in completed:continue
            text=log.read_text(encoding='utf-8',errors='replace');times=[float(x) for x in re.findall(r'^\S+\s+([0-9.]+)s\b',text,re.M)]
            if times:lower+=max(times)
        completed_helper=0.
        for direction in ['F','R']:
            outputs=list((run/'common_bounds/results').glob('*_'+direction+'_inf_dominance.json'))
            if outputs:
                if outputs[0].stat().st_mtime>=started_stamp:completed_helper+=read(outputs[0]).get('solver_wall_seconds',0.)
                continue
            for log in (run/'common_bounds/logs').glob('*_'+direction+'_inf_dominance.log'):
                if log.stat().st_mtime<started_stamp:continue
                times=[float(x) for x in re.findall(r'^\S+\s+([0-9.]+)s\b',log.read_text(encoding='utf-8',errors='replace'),re.M)]
                if times:lower+=max(times)
        attempts.append(dict(attempt=n,interrupted=True,process_wall_upper_seconds=upper,native_wall_lower_seconds=lower,
            separately_counted_completed_helper_seconds=completed_helper,archive=str(arch),timing_uncertain=True))
        save(run/'attempts.json',attempts);(run/'active_attempt.json').unlink()
    if (run/'result.json').exists():return read(run/'result.json')
    n=len(attempts)+1;started=time.perf_counter()
    save(run/'active_attempt.json',dict(attempt=n,started_at=now(),driver_pid=os.getpid()))
    with (run/f'attempt_{n:03d}.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen([str(PYTHON),'-B','-u',str(ROOT/'run_worker.py'),str(run)],stdout=log,stderr=subprocess.STDOUT,env=ENV)
        active=read(run/'active_attempt.json');active['worker_pid']=p.pid;save(run/'active_attempt.json',active)
        code=p.wait()
    elapsed=time.perf_counter()-started
    measure=read(run/'worker_measurements.json') if (run/'worker_measurements.json').exists() else {}
    save(run/f'measurements_{n:03d}.json',measure)
    attempts.append(dict(attempt=n,returncode=code,process_wall_seconds=elapsed,measurement_file=f'measurements_{n:03d}.json',ended_at=now(),interrupted=False))
    save(run/'attempts.json',attempts);(run/'active_attempt.json').unlink()
    if code:
        if (run/'proven_infeasible.json').exists() and read(run/'job.json')['reference_generation']:return None
        raise RuntimeError(('job_failed',str(run),code,(run/f'attempt_{n:03d}.log').read_text(encoding='utf-8')[-3000:]))
    result=read(run/'result.json');check=audit_result(read(run/'raw_input.json')['data'],result,read(run/'mask.json'))
    assert result['accepted'] and check['passed'] and check['schedule_valid'],check
    save(run/'independent_audit.json',check);return result

def publish(case,policy,result):
    result=copy.deepcopy(result);result.update(policy=policy,name=policy)
    dest=ROOT/'cases'/case/f'{policy}_optimal.json'
    if dest.exists():assert read(dest)['schedule']==result['schedule'];return
    save(dest,result)

def base_results(sc):
    case=sc['case'];folder=ROOT/'cases'/case
    if not sc['valid']:return False
    p1=execute(create_job(case,'REF_P1','P1',read(folder/'masks/P1.json')))
    if p1 is None:
        rec=read(folder/'input_validation.json');rec.update(valid=False,input_failure='Strategy-1 proven infeasible under approved local emergency and fixed-running rules')
        save(folder/'input_validation.json',rec);return False
    publish(case,'P1',p1)
    p2=execute(create_job(case,'FULL_P2','P2',read(folder/'masks/P2.json'),p1));publish(case,'P2',p2)
    p3=execute(create_job(case,'FULL_P3','P3',{},p1));publish(case,'P3',p3)
    return True

def costs(run):
    run=Path(run);native=0.;process=0.;cpu=0.;uncertain_extra=0.;completed_keys=set()
    for a in read(run/'attempts.json') if (run/'attempts.json').exists() else []:
        if a.get('interrupted'):
            native+=a['native_wall_lower_seconds'];uncertain_extra+=max(0,a['process_wall_upper_seconds']-a['native_wall_lower_seconds']-a.get('separately_counted_completed_helper_seconds',0.));process+=a['process_wall_upper_seconds']
        else:
            m=read(run/a['measurement_file']);native+=sum(x['native_wall_seconds'] for x in m.get('solve_calls',[]));cpu+=m.get('worker_cpu_seconds',0.);process+=a['process_wall_seconds']
    helper=0.
    for p in (run/'common_bounds/results').glob('*.json'):
        r=read(p);helper+=r.get('solver_wall_seconds',0.)
    # Each private helper result belongs only to this job, charged once.
    native+=helper
    return dict(solver_seconds=native,solver_seconds_upper=native+uncertain_extra,private_helper_seconds=helper,process_seconds=process,
        worker_cpu_seconds=cpu,interruption_uncertainty_seconds=uncertain_extra,attempts=len(read(run/'attempts.json')) if (run/'attempts.json').exists() else 0)

def refine(case,policy):
    folder=ROOT/'cases'/case;full=ROOT/'runs'/f'{case}_FULL_{policy}'
    contenders=[p for p in (ROOT/'runs').glob(case+'_*') if (p/'result.json').exists() and read(p/'job.json')['policy']==policy and not read(p/'job.json')['reference_generation'] and not p.name.startswith(case+'_FULL')]
    if not contenders:return
    best=min([read(p/'result.json') for p in contenders],key=lambda r:r['objective']);current=read(full/'result.json')
    if best['objective']>=current['objective']:return
    # Continuation is the same physical FULL, with only its own proof and an audited better incumbent.
    idx=1+len(list((full/'refinements').glob('*')));hist=full/'refinements'/f'{idx:03d}';hist.mkdir(parents=True)
    for name in ['result.json','job.json','solutions','worker_measurements.json']:
        p=full/name
        if p.is_dir():shutil.copytree(p,hist/name)
        elif p.exists():shutil.copy2(p,hist/name)
    for direction in ['F','R']:
        p=full/'solutions'/f'{policy}_{direction}.json'
        if p.exists():
            r=read(p);save(full/'checkpoints'/f'{policy}_{direction}_unfinished.json',dict(lower_bound=r['lower_bound']))
            p.unlink()
    (full/'result.json').unlink();save(full/'refinement_hint.json',best)
    job=read(full/'job.json');job.update(refinement=True,hint_objective=best['objective']);save(full/'job.json',job)
    final=execute(full);assert final['objective']<=best['objective']
    save(full/'reconciliation.json',dict(initial_objective=current['objective'],best_restricted=best['objective'],final_objective=final['objective'],frozen_reference_unchanged=True))
