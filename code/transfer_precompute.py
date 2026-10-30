"""Prediction-independent transfer references, serial solves overlapping training only."""
import datetime,os,time,traceback
from pathlib import Path
from common import *

CONTROL='transfer_precompute_control.json'
COMPLETE='transfer_full_complete.json'
FAILURE='transfer_precompute_failure.json'

def now():return datetime.datetime.now().astimezone().isoformat()

def wait_until_complete(root=None,interval=2.):
    """Gate both benchmark solver phases until the independent solver has finished."""
    root=Path(root) if root else ROOT
    if not (root/CONTROL).exists() or not read(root/CONTROL)['enabled']:return
    print('WAIT_FOR_TRANSFER_FULL_PRECOMPUTE',now(),flush=True)
    started=time.perf_counter()
    while True:
        if (root/COMPLETE).exists():
            assert read(root/COMPLETE)['complete'];break
        if (root/FAILURE).exists():raise RuntimeError(('transfer full precompute failed',read(root/FAILURE)))
        launch=root/'background_launch_transfer_full.json'
        if launch.exists():
            import psutil
            identity=read(launch);pid=identity.get('interpreter_pid',identity['root_pid'])
            try:
                proc=psutil.Process(pid)
                assert 'service_transfer_full.py' in ' '.join(proc.cmdline()),'parallel PID reused'
                created=identity.get('interpreter_creation_time',identity.get('root_creation_time',identity['started_at']))
                assert abs(proc.create_time()-datetime.datetime.fromisoformat(created).timestamp())<5,'parallel PID creation mismatch'
            except psutil.NoSuchProcess:
                # Completion/failure is written before a healthy driver exits.
                if (root/COMPLETE).exists():continue
                raise RuntimeError('Transfer-full driver exited without a completion marker; preserve costs and resume it.')
        elif time.perf_counter()-started>300:
            raise RuntimeError('Transfer-full overlap was enabled but no independent launch was recorded.')
        time.sleep(interval)
    if root==ROOT:
        record=dict(wait_seconds=time.perf_counter()-started,finished_at=now(),
            scope='Experiment scheduling wait; not native solver time and not added to hypothetical deployment pipeline latency.',
            no_two_target_solver_jobs_in_parallel=True)
        ledger=read(ROOT/'transfer_full_gate_waits.json') if (ROOT/'transfer_full_gate_waits.json').exists() else []
        ledger.append(record);save(ROOT/'transfer_full_gate_waits.json',ledger)
        save(ROOT/'transfer_full_gate_wait.json',record)
    print('TRANSFER_FULL_PRECOMPUTE_READY',now(),flush=True)

def progress(records,started_at):
    jobs=[];refs=[];failed=[]
    for sc in records:
        case=sc['case'];state=read(ROOT/'cases'/case/'input_validation.json')
        if not state['valid']:failed.append(dict(case=case,reason=state['input_failure']))
        for key in ['REF_P1','FULL_P2','FULL_P3']:
            run=ROOT/'runs'/f'{case}_{key}'
            if (run/'result.json').exists():
                result=read(run/'result.json');assert result['accepted'] and result['audit']['passed']
                (refs if key=='REF_P1' else jobs).append(dict(case=case,key=key))
    data=dict(stage='TRANSFER_FULL_PRECOMPUTE',complete=False,started_at=started_at,updated_at=now(),
              planned_targets=72,completed_targets=len(jobs),completed_references=len(refs),jobs=jobs,input_failures=failed)
    save(ROOT/'transfer_full_progress.json',data);return data

def main():
    from jobs import base_results,costs
    started=time.perf_counter();started_at=now();attempt=[]
    try:
        assert read(ROOT/CONTROL)['enabled'];verify_sources();verify_frozen()
        for item in read(ROOT/'input_lock.json')['files']:assert sha(item['path'])==item['sha256']
        assert read(ROOT/'base_full_complete.json')['complete']
        records=[sc for sc in read(ROOT/'case_manifest.json') if sc['cohort']=='transfer']
        assert len(records)==36
        for sc in records:
            actual=read(ROOT/'cases'/sc['case']/'input_validation.json')
            if actual['valid']:base_results(actual)
            progress(records,started_at)
            print('TRANSFER_FULL_CASE_DONE',sc['case'],flush=True)
        result=progress(records,started_at);valid=36-len(result['input_failures'])
        assert result['completed_targets']==2*valid and result['completed_references']==valid
        rows=[]
        for sc in records:
            for key in ['REF_P1','FULL_P2','FULL_P3']:
                run=ROOT/'runs'/f'{sc["case"]}_{key}'
                if (run/'job.json').exists():rows.append(dict(case=sc['case'],key=key,**costs(run)))
        save(ROOT/'transfer_full_costs.json',dict(jobs=rows,solver_costs_remain_in_each_original_run=True,
            full_native_solver_seconds=sum(r['solver_seconds'] for r in rows if r['key']!='REF_P1'),
            reference_native_solver_seconds=sum(r['solver_seconds'] for r in rows if r['key']=='REF_P1'),
            phase_wall_seconds=time.perf_counter()-started,started_at=started_at,ended_at=now(),
            may_overlap_training=True,phase_durations_not_additive_under_overlap=True))
        ledger=read(ROOT/'transfer_full_phase_ledger.json') if (ROOT/'transfer_full_phase_ledger.json').exists() else []
        ledger.append(dict(started_at=started_at,ended_at=now(),seconds=time.perf_counter()-started,returncode=0))
        save(ROOT/'transfer_full_phase_ledger.json',ledger)
        save(ROOT/COMPLETE,dict(**{k:v for k,v in result.items() if k!='complete'},complete=True,completed_at=now(),
            valid_transfer_cases=valid,no_prediction_or_training_read=True,solver_configuration_unchanged=True,
            original_job_results_reused_by_main_pipeline=True))
        print('TRANSFER_FULL_COMPLETE',result['completed_targets'],flush=True)
    except BaseException as exc:
        save(ROOT/FAILURE,dict(error=repr(exc),traceback=traceback.format_exc(),at=now(),driver_pid=os.getpid(),
            started_at=started_at,phase_wall_seconds=time.perf_counter()-started,all_job_attempts_retained=True))
        raise

if __name__=='__main__':main()
