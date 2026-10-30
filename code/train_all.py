"""Run independent model/fold processes to bound memory and preserve recovery."""
import subprocess,time
from common import *
from jobs import ENV
def main():
    ledger=read(ROOT/'training_costs.json') if (ROOT/'training_costs.json').exists() else []
    for stage in STAGES:
        for fold in read(ROOT/'folds.json'):
            for method in ['STAT','ML']:
                f=fold['test_start'];folder=ROOT/'models'/method/stage/f'fold_{f:03d}'
                if (folder/'complete.json').exists():continue
                started=time.perf_counter();p=subprocess.run([str(PYTHON),'-B','-u',str(ROOT/'train.py'),'--method',method,'--stage',stage,'--test-start',str(f)],env=ENV)
                ledger.append(dict(stage=stage,fold=f,method=method,seconds=time.perf_counter()-started,returncode=p.returncode));save(ROOT/'training_costs.json',ledger)
                assert p.returncode==0
    subprocess.run([str(PYTHON),'-B',str(ROOT/'validate_training.py')],env=ENV,check=True)
if __name__=='__main__':main()

