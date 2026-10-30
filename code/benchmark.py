"""Eight target jobs per valid case; references never silently change."""
import copy,datetime,time
from common import *
from jobs import create_job,execute,publish,base_results,refine

def update_progress(stage):
    done=[]
    for p in (ROOT/'runs').glob('*/result.json'):
        j=read(p.parent/'job.json')
        if not j['reference_generation']:done.append(dict(case=j['case'],key=j['key']))
    save(ROOT/'TASK_STATE.json',dict(stage=stage,complete=False,completed_initial_targets=len(done),registered_initial_targets=888,
        completed=done,updated_at=datetime.datetime.now().astimezone().isoformat()))

def refresh_manifest():
    records=[read(ROOT/'cases'/r['case']/'input_validation.json') for r in read(ROOT/'case_manifest.json')]
    save(ROOT/'case_manifest.json',records)
    return records

def full_base():
    assert read(ROOT/'preflight_complete.json')['passed']
    for sc in refresh_manifest():
        if sc['cohort']=='base' and sc['valid']:
            base_results(sc);update_progress('BASE_FULL_REFERENCES');print('BASE_FULL_DONE',sc['case'],flush=True)
    records=refresh_manifest()
    from prepare_cases import make_folds
    save(ROOT/'folds.json',make_folds(records))
    save(ROOT/'base_full_complete.json',dict(complete=True,valid_base=sum(s['valid'] and s['cohort']=='base' for s in records)))

def restricted(case,stage,method,tag,ref,upstream_key=None):
    from predict import predict,mask
    key=f'{method}_P2' if stage=='A' else f'{method}_P3_{tag}'
    run=ROOT/'runs'/f'{case}_{key}'
    if (run/'result.json').exists():return read(run/'result.json')
    # Different restricted-P2 timetables require independent cache namespaces.
    prediction_tag=f'{tag}_{method}' if tag=='CHAIN' else tag
    pp,pt=predict(case,stage,method,prediction_tag,ref)
    cached=ROOT/'reference_views'/f'{case}_{stage}_{prediction_tag}'/f'{STAGES[stage][0]}_optimal.json'
    assert read(cached)['schedule']==ref['schedule'],'Prediction indicators do not match this method reference'
    assert sha(cached)==pt['feature_costs']['reference_sha256']
    assert all(value(row,ref['schedule'])==row['reference_value'] for row in pp.to_dict('records'))
    m,mt=mask(case,stage,pp,ref)
    run=create_job(case,key,STAGES[stage][1],m,ref,dict(method=method,reference_mode=tag,upstream_key=upstream_key,
        prediction_tag=prediction_tag,prediction_sha256=pt['prediction_sha256'],model_sha256=pt['model_sha256'],fixed_counts=mt['counts']))
    save(run/'prediction_costs.json',dict(**pt,**mt))
    r=execute(run);update_progress('RESTRICTED_'+stage);return r

def evaluate_case(sc):
    case=sc['case'];folder=ROOT/'cases'/case
    p1=read(folder/'P1_optimal.json');p2=read(folder/'P2_optimal.json')
    stage2={}
    for method in ['STAT','ML']:stage2[method]=restricted(case,'A',method,'FULL_REFERENCE',p1,'REF_P1')
    for method in ['STAT','ML']:
        restricted(case,'B',method,'FULL_REFERENCE',p2,'FULL_P2')
        restricted(case,'B',method,'CHAIN',stage2[method],method+'_P2')
    refine(case,'P2');refine(case,'P3')
    save(folder/'evaluation_complete.json',dict(complete=True,initial_targets=8,reference_values_frozen=True))
    print('EVALUATED',case,flush=True)

def restricted_base():
    assert read(ROOT/'training_validation.json')['passed']
    from transfer_precompute import wait_until_complete
    wait_until_complete()
    for sc in read(ROOT/'case_manifest.json'):
        if sc['cohort']=='base' and sc['valid']:evaluate_case(sc)
    save(ROOT/'base_evaluation_complete.json',dict(complete=True))

def transfer():
    assert read(ROOT/'training_validation.json')['passed']
    from transfer_precompute import wait_until_complete
    wait_until_complete()
    for sc in refresh_manifest():
        if sc['cohort']!='transfer' or not sc['valid']:continue
        case=sc['case'];folder=ROOT/'cases'/case
        p1=execute(create_job(case,'REF_P1','P1',read(folder/'masks/P1.json')))
        if p1 is None:
            rec=read(folder/'input_validation.json');rec.update(valid=False,input_failure='Strategy-1 proven infeasible');save(folder/'input_validation.json',rec);continue
        publish(case,'P1',p1)
        full2=execute(create_job(case,'FULL_P2','P2',read(folder/'masks/P2.json'),p1));publish(case,'P2',full2)
        # Reference-only inference: an independently precomputed FULL-P3 is never read.
        from predict import predict
        for method in ['STAT','ML']:
            predict(case,'A',method,'FULL_REFERENCE',p1);predict(case,'B',method,'FULL_REFERENCE',full2)
        full3=execute(create_job(case,'FULL_P3','P3',{},p1));publish(case,'P3',full3)
        evaluate_case(sc)
    records=refresh_manifest();valid=sum(s['valid'] for s in records)
    actual=sum((ROOT/'runs'/f'{s["case"]}_{key}'/'result.json').exists() for s in records if s['valid'] for key in
        ['FULL_P2','FULL_P3','STAT_P2','ML_P2','STAT_P3_FULL_REFERENCE','ML_P3_FULL_REFERENCE','STAT_P3_CHAIN','ML_P3_CHAIN'])
    assert actual==8*valid
    save(ROOT/'completion.json',dict(complete=True,registered=111,valid_cases=valid,initial_target_results=actual,expected_for_valid_cases=8*valid,
        input_failures=111-valid,at=datetime.datetime.now().astimezone().isoformat()))

if __name__=='__main__':
    import sys
    {'full_base':full_base,'restricted_base':restricted_base,'transfer':transfer}[sys.argv[1]]()

