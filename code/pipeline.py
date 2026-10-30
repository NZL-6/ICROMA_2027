"""Independent resumable sequential experiment; never supervise individual search."""
import datetime,os,subprocess,time,traceback
from common import *
from jobs import ENV
def phase(name,command,marker):
    if (ROOT/marker).exists():return
    save(ROOT/'TASK_STATE.json',dict(stage=name,complete=False,updated_at=datetime.datetime.now().astimezone().isoformat()))
    log=ROOT/'phase_logs';log.mkdir(exist_ok=True);at=time.perf_counter()
    with (log/(name+'.log')).open('a',encoding='utf-8') as out:
        p=subprocess.run([str(PYTHON),'-B','-u',*command],stdout=out,stderr=subprocess.STDOUT,env=ENV,cwd=str(ROOT))
    ledger=read(ROOT/'phase_ledger.json') if (ROOT/'phase_ledger.json').exists() else []
    ledger.append(dict(phase=name,seconds=time.perf_counter()-at,returncode=p.returncode,ended_at=datetime.datetime.now().astimezone().isoformat()));save(ROOT/'phase_ledger.json',ledger)
    if p.returncode or not (ROOT/marker).exists():raise RuntimeError((name,'failed',p.returncode,str(log/(name+'.log'))))
def main():
    try:
        verify_sources();verify_frozen()
        for item in read(ROOT/'input_lock.json')['files']:
            assert sha(item['path'])==item['sha256'],('frozen input changed',item['path'])
        phase('model_validation',['validate.py'],'preflight_complete.json')
        phase('base_full_references',['benchmark.py','full_base'],'base_full_complete.json')
        phase('reference_indicators',['features.py'],'training_data_complete.json')
        phase('indicator_associations',['associations.py'],'association_complete.json')
        phase('training',['train_all.py'],'training_validation.json')
        phase('base_restricted_evaluation',['benchmark.py','restricted_base'],'base_evaluation_complete.json')
        phase('frozen_transfer',['benchmark.py','transfer'],'completion.json')
        phase('final_audit_report',['analyze.py'],'final_validation.json')
        phase('evidence_package',['package.py'],'delivery_complete.json')
        save(ROOT/'TASK_STATE.json',dict(stage='AUDITED_PENDING_PAPER_UPDATE',complete=False,experiment_complete=True,paper_updated=False))
    except BaseException as exc:
        save(ROOT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=datetime.datetime.now().astimezone().isoformat()));raise
if __name__=='__main__':main()

