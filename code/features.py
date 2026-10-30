"""Reference-only indicators; labels are joined only in the base training step."""
import copy,gc,re,shutil,time
import numpy as np
import pandas as pd
from common import *
from models import IDENT,LABEL,CONTROL
bootstrap_engine()
AF=None
def source():
    global AF
    if AF is None:AF=module(ROOT/'snapshots/feature_source.py','local_reference_features')
    return AF

def view(case,stage,tag,reference):
    folder=ROOT/'reference_views'/f'{case}_{stage}_{tag}';folder.mkdir(parents=True,exist_ok=True)
    base=ROOT/'cases'/case
    for name in ['scenario.json','common_input.json','physical_variables.json','emergency_state.json']:
        if not (folder/name).exists():shutil.copy2(base/name,folder/name)
    policy=STAGES[stage][0];r=copy.deepcopy(reference);r.update(policy=policy,name=policy)
    p=folder/f'{policy}_optimal.json'
    if p.exists():assert read(p)['schedule']==r['schedule'],'reference mutation forbidden'
    else:save(p,r)
    shutil.copy2(ROOT/'snapshots/network_builder.py',folder/'network_builder.py')
    return folder

def build(case,stage,tag,reference):
    target=ROOT/'inference_inputs'/f'{case}_{stage}_{tag}.csv.gz';meta=target.with_suffix('.json')
    if target.exists() and meta.exists():
        info=read(meta);assert sha(target)==info['sha256']
        if info['indicator_source_sha256']==sha(ROOT/'snapshots/feature_source.py'):
            return pd.read_csv(target,low_memory=False,float_precision='round_trip'),info
        assert not (ROOT/'execution_lock.json').exists(),'Frozen feature source mismatch'
        archive=ROOT/'history/feature_validation'/info['sha256'][:12];archive.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target,archive/target.name);shutil.copy2(meta,archive/meta.name)
    folder=view(case,stage,tag,reference);policy=STAGES[stage][0];at=time.perf_counter()
    builder=module(folder/'network_builder.py',f'graph_{case}_{stage}_{tag}');network=builder.build(read(folder/f'{policy}_optimal.json'))
    save(folder/f'{policy}_network.json',network);graph_seconds=time.perf_counter()-at
    at=time.perf_counter();obj=source().Case(folder,policy);df=obj.rows();df=df[df.kind.isin(STAGES[stage][2])].copy()
    # Optional additional-stop decisions and their dwell intervals never fixed.
    optional=set(obj.units['S']);eligible=[not (r.kind=='Td' and f'{r.train}:{int(r.visit)}' in optional) for r in df.itertuples()]
    df=df.loc[eligible].reset_index(drop=True)
    df['reference_value']=[value(r,reference['schedule']) for r in df.to_dict('records')]
    df['physical_lower']=np.where(df.kind=='O',0,df.M04_domain_lower)
    df['physical_upper']=np.where(df.kind=='O',1,df.M04_domain_upper)
    df['can_decrease']=((df.kind!='O')&(df.reference_value>df.physical_lower)).astype(int)
    df['can_increase']=((df.kind=='O')|(df.reference_value<df.physical_upper)).astype(int)
    for col in df:
        if col not in IDENT:df[col]=pd.to_numeric(df[col],errors='coerce').astype('float32')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    feature_seconds=time.perf_counter()-at;at=time.perf_counter();target.parent.mkdir(exist_ok=True)
    df.to_csv(target,index=False,compression='gzip')
    info=dict(case=case,stage=stage,tag=tag,rows=len(df),columns=len(df.columns),graph_seconds=graph_seconds,feature_seconds=feature_seconds,
        feature_persistence_seconds=time.perf_counter()-at,reference_sha256=sha(folder/f'{policy}_optimal.json'),
        physical_input_sha256=sha(folder/'common_input.json'),indicator_source_sha256=sha(ROOT/'snapshots/feature_source.py'),
        graph_source_sha256=sha(folder/'network_builder.py'),sha256=sha(target),target_solution_read=False)
    save(meta,info);save(ROOT/'feature_definitions.json',source().REG)
    del obj;gc.collect();return df,info

def labels(frame,case,stage):
    target=read(ROOT/'cases'/case/f'{STAGES[stage][1]}_optimal.json')['schedule']
    df=frame.copy();df['target_value']=[value(r,target) for r in df.to_dict('records')]
    df['delta']=df.target_value-df.reference_value;df['changed']=(df.delta!=0).astype(int)
    df['action']=np.where(df.kind=='O',df.changed,np.sign(df.delta)).astype(int)
    df['ambiguous']=False
    # Detect disagreements only among actually retained equal-objective incumbents.
    run=ROOT/'runs'/f'{case}_FULL_{STAGES[stage][1]}'
    full=read(run/'result.json');directions={d:read(run/'solutions'/f'{STAGES[stage][1]}_{d}.json')['objective'] for d in ['F','R']}
    for p in (run/'incumbents').glob('*.json'):
        inc=read(p);direction=p.stem.split('_')[-2]
        if inc['objective']!=directions.get(direction):continue
        take=df.train.isin(inc['schedule'])
        if not take.any():continue
        vals=[value(r,inc['schedule']) for r in df.loc[take].to_dict('records')]
        df.loc[take,'ambiguous']|=np.array(vals)!=df.loc[take,'target_value'].to_numpy()
    df['label_weight']=np.where(df.ambiguous,.5,1.)
    return df

def training_data():
    records=[]
    for sc in read(ROOT/'case_manifest.json'):
        if sc['cohort']!='base' or not sc['valid']:continue
        case=sc['case']
        for stage in STAGES:
            out=ROOT/'features'/f'{case}_{stage}.csv.gz'
            if out.exists():continue
            ref=read(ROOT/'cases'/case/f'{STAGES[stage][0]}_optimal.json');df,info=build(case,stage,'FULL_REFERENCE',ref)
            df=labels(df,case,stage);df.to_csv(out,index=False,compression='gzip')
            save(out.with_suffix('.json'),dict(**info,labeled_sha256=sha(out),label_source=STAGES[stage][1],ambiguity_detected=int(df.ambiguous.sum())))
            records.append(dict(case=case,stage=stage,rows=len(df)));print('FEATURES_READY',case,stage,len(df),flush=True)
            del df;gc.collect()
    save(ROOT/'training_data_complete.json',dict(complete=True,files=len(list((ROOT/'features').glob('*.csv.gz'))),newly_built=records))
if __name__=='__main__':training_data()

