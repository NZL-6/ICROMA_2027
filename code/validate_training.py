"""Verify every nested fit provenance and prediction probability/fixing count."""
import joblib
import pandas as pd
import numpy as np
from common import *
from models import *
def main():
    records=[];frozen=[]
    for stage in STAGES:
        for f in read(ROOT/'folds.json'):
            fold=f['test_start'];feat=joblib.load(ROOT/'shared_features'/stage/f'outer_{fold:03d}/final/features.pkl')
            assert set(feat.fit_cases)==set(f['train']) and not set(feat.fit_cases)&set(f['test'])
            for i in f['inner']:
                inner=joblib.load(ROOT/'shared_features'/stage/f'outer_{fold:03d}/inner_{i["validation_start"]:03d}/features.pkl')
                assert set(inner.fit_cases)==set(i['train']) and not set(inner.fit_cases)&set(i['validation'])
            for method in ['STAT','ML']:
                folder=ROOT/'models'/method/stage/f'fold_{fold:03d}';selection=read(folder/'selection.json')
                assert set(selection['training_cases'])==set(f['train']) and set(selection['test_cases'])==set(f['test'])
                assert (folder/'complete.json').exists()
                for case in f['test']:
                    df=pd.read_csv(ROOT/'predictions'/method/f'{case}_{stage}.csv.gz')
                    assert np.allclose(df[['p_decrease','p_keep','p_increase']].sum(axis=1),1.)
                    for kind,g in df.groupby('kind'):
                        assert int(g.fixed.sum())==len(g)//2
                    assert df.case.nunique()==1 and df.case.iloc[0]==case
                records.append(dict(stage=stage,fold=fold,method=method,train_cases=len(f['train']),test_cases=len(f['test']),passed=True))
    for directory in ['models','shared_features']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file():frozen.append(dict(path=str(p),sha256=sha(p)))
    save(ROOT/'model_lock.json',dict(files=frozen));save(ROOT/'training_validation.json',dict(passed=True,records=records,models=len(records),test_case_record_isolation=True))
    print('TRAINING_VALIDATED',len(records),flush=True)
if __name__=='__main__':main()

