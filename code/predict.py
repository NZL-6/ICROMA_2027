"""Fold-frozen inference and fixed-set construction, blind to target solutions."""
import copy,time
import numpy as np
import pandas as pd
import joblib
from common import *
from models import *
from features import build
bootstrap_engine()
from audit import audit_result

def predict(case,stage,method,tag,reference):
    target=ROOT/'frozen_predictions'/method/f'{case}_{stage}_{tag}.csv.gz';meta=target.with_suffix('.json')
    if target.exists() and meta.exists():
        info=read(meta);assert sha(target)==info['prediction_sha256'];return pd.read_csv(target,low_memory=False,float_precision='round_trip'),info
    df,ft=build(case,stage,tag,reference);sc=read(ROOT/'cases'/case/'scenario.json');fold=sc['outer_fold'] if sc['cohort']=='base' else 99
    directory=ROOT/'models'/method/stage/f'fold_{fold:03d}'
    at=time.perf_counter();feat=joblib.load(ROOT/'shared_features'/stage/f'outer_{fold:03d}/final/features.pkl');selection=read(directory/'selection.json')
    assert case not in selection['training_cases'];assert case not in feat.fit_cases
    if method=='STAT':model=joblib.load(directory/'model.joblib');modelpath=directory/'model.joblib'
    else:
        torch=torch_runtime();model=neural_model(sorted(feat.spec));modelpath=directory/'model.pt';model.load_state_dict(torch.load(modelpath,weights_only=True))
    loading=time.perf_counter()-at;at=time.perf_counter()
    if method=='STAT':prob=model.predict(df,feat);q=model.quantiles(df,feat)
    else:prob,q=neural_predict(model,df,feat)
    prob=calibrate(df,prob,selection['calibration'])
    for col in LABEL:df[col]=np.nan
    frame=prediction_frame(df,prob,q,method,stage,fold)
    inference=time.perf_counter()-at;target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(target,index=False,compression='gzip')
    info=dict(case=case,stage=stage,method=method,tag=tag,feature_costs=ft,model_loading_seconds=loading,inference_seconds=inference,
        model_sha256=sha(modelpath),prediction_sha256=sha(target),training_cases=selection['training_cases'],target_solution_read=False,fold=fold)
    save(meta,info);return frame,info

def mask(case,stage,pred,reference):
    started=time.perf_counter();folder=ROOT/'cases'/case
    raw=read(folder/'common_input.json')['data'];out=copy.deepcopy(read(folder/'masks'/f'{STAGES[stage][1]}.json'))
    optional=set(read(folder/'physical_variables.json')['S']);counts={}
    for kind,g in pred.groupby('kind'):
        expected=g.sort_values(['p_keep','unit_id'],ascending=[False,True],kind='stable').head(len(g)//2)
        actual=g[g.fixed.astype(str).str.lower()=='true'];assert set(expected.unit_id)==set(actual.unit_id)
        counts[kind]=dict(candidates=len(g),fixed=len(actual))
        for r in actual.to_dict('records'):
            i,n=r['train'],int(r['visit'])
            if kind=='O':
                j,k=r['other_train'],int(r['other_visit']);a,b=f'{i}:{n}',f'{j}:{k}'
                if not value(r,reference['schedule']):a,b=b,a
                out.setdefault('orders',[]).append(dict(kind='run',resource=r['resource'],first=a,second=b))
            else:
                assert kind!='Td' or f'{i}:{n}' not in optional
                v=int(value(r,reference['schedule']));assert v==r['reference_value']
                out.setdefault('dwell_bounds' if kind=='Td' else 'running_bounds',{})[f'{i}:{n}']=[v,v]
    if 'orders' in out:out['orders'].sort(key=lambda o:(o['resource'],o['first'],o['second']))
    check=audit_result(raw,reference,out);assert check['passed'] and check['schedule_valid'],check
    return out,dict(counts=counts,mask_and_audit_seconds=time.perf_counter()-started,reference_feasible=True,P2_mask_inherited=False)
