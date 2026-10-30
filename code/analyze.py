"""Independent final audit and case-paired quality/time reporting."""
import collections,html,math,time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,recall_score
from common import *
from jobs import costs
bootstrap_engine()
from audit import audit_result
from acceptance import accepted
from models import IDENT

KEYS=['FULL_P2','STAT_P2','ML_P2','FULL_P3','STAT_P3_FULL_REFERENCE','ML_P3_FULL_REFERENCE','STAT_P3_CHAIN','ML_P3_CHAIN']
def percent_loss(v,base):return 100*(v-base)/base if base else (0. if v==0 else np.nan)
def csv(name,rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    if not len(df.columns):
        columns={'input_failure_reference_costs.csv':['case','reason','solver_seconds','process_seconds'],
                 'input_failures.csv':['case','cohort','section','start_minutes','duration_minutes','input_failure'],
                 'unexecuted_input_failure_jobs.csv':['case','key','reason','objective','solver_seconds']}
        df=pd.DataFrame(columns=columns.get(name,['case']))
    df.to_csv(ROOT/'reports'/name,index=False);return df
def feature_cost(info):
    f=info.get('feature_costs',{});return sum(f.get(k,0.) for k in ['graph_seconds','feature_seconds','feature_persistence_seconds'])

def job_metrics(case,key):
    run=ROOT/'runs'/f'{case}_{key}';job=read(run/'job.json');result=read(run/'result.json');raw=read(run/'raw_input.json')['data'];mask=read(run/'mask.json')
    check=audit_result(raw,result,mask);assert check['passed'] and check['schedule_valid'] and accepted(result['objective'],result['lower_bound'])
    assert sha(run/'mask.json')==job['mask_sha256']
    assert sha(run/'raw_input.json')==sha(ROOT/'cases'/case/'common_input.json')
    if not job['reference_generation']:assert sha(run/'reference.json')==job['reference_sha256']
    for part in result.get('directional',[]):
        pars=part['parameters'];assert pars['workers']==12 and pars['seed']==SEED and pars['proof_slice_seconds'] is None and pars['relative_gap_limit']<=.001
    for proof in result.get('bound_proofs',[]):
        p=Path(proof['proof_path']);assert sha(p)==proof['proof_sha256'];assert read(p)['independent_audit']['passed']
    t=costs(run);info=read(run/'prediction_costs.json') if (run/'prediction_costs.json').exists() else {}
    failed_prep=read(run/'failed_preparation_costs.json') if (run/'failed_preparation_costs.json').exists() else {}
    inference=info.get('inference_seconds',0.)+failed_prep.get('inference_seconds',0.)
    load=info.get('model_loading_seconds',0.)+failed_prep.get('model_loading_seconds',0.);fc=feature_cost(info)
    prep=read(run/'job_preparation.json')['seconds']
    mask_time=info.get('mask_and_audit_seconds',0.)+failed_prep.get('validation_mask_and_audit_seconds',0.)
    row=dict(case=case,key=key,policy=job['policy'],method=job.get('method','FULL'),reference_mode=job.get('reference_mode','DIRECT'),
        objective=result['objective'],lower_bound=result['lower_bound'],relative_gap=result['relative_gap'],**t,
        feature_seconds=fc,inference_seconds=inference,model_loading_seconds=load,mask_generation_audit_seconds=mask_time,job_preparation_seconds=prep,
        failed_preparation_inference_seconds=failed_prep.get('inference_seconds',0.),failed_preparation_loading_seconds=failed_prep.get('model_loading_seconds',0.),
        prediction_solver_seconds=t['solver_seconds']+inference,prediction_solver_seconds_upper=t['solver_seconds_upper']+inference,
        online_increment_seconds=t['process_seconds']+fc+inference+load+prep+mask_time,audit_passed=True,
        reference_sha256=job.get('reference_sha256'),mask_sha256=job['mask_sha256'])
    save(run/'final_audit.json',dict(passed=True,physical=check,timing=row));return row

def predictions_for(case,key,stage,method,tag):
    job=read(ROOT/'runs'/f'{case}_{key}/job.json');prediction_tag=job.get('prediction_tag',tag)
    pp=ROOT/'frozen_predictions'/method/f'{case}_{stage}_{prediction_tag}.csv.gz';df=pd.read_csv(pp,low_memory=False)
    meta=read(pp.with_suffix('.json'));assert sha(pp)==meta['prediction_sha256'];assert case not in meta['training_cases']
    reference=read(ROOT/'runs'/f'{case}_{key}/reference.json')
    cached=ROOT/'reference_views'/f'{case}_{stage}_{prediction_tag}'/f'{STAGES[stage][0]}_optimal.json'
    assert read(cached)['schedule']==reference['schedule']
    assert sha(cached)==meta['feature_costs']['reference_sha256']
    assert np.array_equal(df.reference_value.to_numpy(),np.array([value(row,reference['schedule']) for row in df.to_dict('records')]))
    target=read(ROOT/'cases'/case/f'{STAGES[stage][1]}_optimal.json')['schedule']
    truth=np.array([value(r,target) for r in df.to_dict('records')]);delta=truth-df.reference_value.to_numpy();changed=delta!=0
    df['target_value']=truth;df['delta']=delta;df['changed']=changed;df['action']=np.where(df.kind=='O',changed,np.sign(delta))
    df['key']=key
    for kind,g in df.groupby('kind'):
        expected=g.sort_values(['p_keep','unit_id'],ascending=[False,True],kind='stable').head(len(g)//2)
        actual=g[g.fixed.astype(str).str.lower()=='true'];assert set(expected.unit_id)==set(actual.unit_id)
    target_path=ROOT/'evaluated_predictions'/method/f'{case}_{stage}_{tag}.csv.gz';target_path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(target_path,index=False,compression='gzip')
    return df

def summarize(df):
    rows=[]
    for (cohort,policy,key),g in df.groupby(['cohort','policy','key']):
        base=g.full_objective.to_numpy();loss=g.objective.to_numpy()-base
        rows.append(dict(cohort=cohort,policy=policy,key=key,cases=len(g),objective_sum=int(g.objective.sum()),
            mean_loss=float(loss.mean()),max_loss=float(loss.max()),mean_relative_loss_percent=float(g.relative_loss_percent.mean()),
            preserved=int((loss==0).sum()),preserved_percent=100*float((loss==0).mean()),
            pipeline_solver_seconds=float(g.pipeline_solver_seconds.sum()),pipeline_prediction_solver_seconds=float(g.pipeline_prediction_solver_seconds.sum()),
            pipeline_online_seconds=float(g.pipeline_online_seconds.sum()),
            solver_speedup=float(g.direct_solver_seconds.sum()/g.pipeline_solver_seconds.sum()),
            prediction_solver_speedup=float(g.direct_solver_seconds.sum()/g.pipeline_prediction_solver_seconds.sum()),
            online_speedup=float(g.direct_online_seconds.sum()/g.pipeline_online_seconds.sum()),
            median_paired_solver_speedup=float(np.median(g.direct_solver_seconds/g.pipeline_solver_seconds)),
            quality_preserved_and_solver_faster=int(((loss==0)&(g.pipeline_solver_seconds<g.direct_solver_seconds)).sum()),
            quality_preserved_and_online_faster=int(((loss==0)&(g.pipeline_online_seconds<g.direct_online_seconds)).sum())))
    return pd.DataFrame(rows)

def main():
    assert read(ROOT/'completion.json')['complete'];verify_sources();verify_frozen()
    from scheduling_audit import audit_schedule
    scheduling=audit_schedule()
    for item in read(ROOT/'input_lock.json')['files']:assert sha(item['path'])==item['sha256']
    for item in read(ROOT/'model_lock.json')['files']:assert sha(item['path'])==item['sha256']
    manifest=read(ROOT/'case_manifest.json');valid=[s for s in manifest if s['valid']];alljobs=[];pipeline=[];predictions=[];refcost=[]
    for sc in valid:
        case=sc['case'];ref=job_metrics(case,'REF_P1');refcost.append({**sc,**ref})
        jobs={k:job_metrics(case,k) for k in KEYS};alljobs += [dict(**r,cohort=sc['cohort'],group=sc['group'],section=sc['section'],start_minutes=sc['start_minutes'],duration_minutes=sc['duration_minutes']) for r in jobs.values()]
        assert all(r['objective']>=jobs['FULL_'+r['policy']]['objective'] for r in jobs.values())
        for key,r in jobs.items():
            parts=[r]
            if r['policy']=='P3' and key!='FULL_P3':parts.insert(0,jobs['FULL_P2' if 'FULL_REFERENCE' in key else r['method']+'_P2'])
            direct=jobs['FULL_'+r['policy']]
            row=dict({**sc,**r},full_objective=direct['objective'],loss=r['objective']-direct['objective'],relative_loss_percent=percent_loss(r['objective'],direct['objective']),
                pipeline_solver_seconds=sum(x['solver_seconds'] for x in parts),pipeline_solver_seconds_upper=sum(x['solver_seconds_upper'] for x in parts),
                pipeline_prediction_solver_seconds=sum(x['prediction_solver_seconds'] for x in parts),
                pipeline_online_seconds=ref['online_increment_seconds']+sum(x['online_increment_seconds'] for x in parts),
                direct_solver_seconds=direct['solver_seconds'],direct_solver_seconds_upper=direct['solver_seconds_upper'],direct_online_seconds=ref['online_increment_seconds']+direct['online_increment_seconds'],
                upstream_charged=[x['key'] for x in parts],common_P1_charged_once=True)
            pipeline.append(row)
            if r['method']!='FULL':
                stage='A' if r['policy']=='P2' else 'B';tag=r['reference_mode']
                pp=predictions_for(case,key,stage,r['method'],tag);pp['cohort']=sc['cohort'];predictions.append(pp)
    jobsdf=csv('job_metrics.csv',alljobs);paired=csv('paired_results.csv',pipeline);summary=csv('summary.csv',summarize(paired));csv('reference_generation_costs.csv',refcost)
    csv('input_failure_reference_costs.csv',[dict(case=s['case'],reason=s['input_failure'],**costs(ROOT/'runs'/f'{s["case"]}_REF_P1'))
        for s in manifest if not s['valid'] and (ROOT/'runs'/f'{s["case"]}_REF_P1').exists()])
    csv('quality_loss_cases.csv',paired[paired.loss>0]);csv('input_failures.csv',[s for s in manifest if not s['valid']])
    csv('unexecuted_input_failure_jobs.csv',[dict(case=s['case'],key=k,reason=s['input_failure'],objective=None,solver_seconds=None) for s in manifest if not s['valid'] for k in KEYS])
    preds=pd.concat(predictions,ignore_index=True);pm=[]
    for (cohort,key,kind),g in preds.groupby(['cohort','key','kind']):
        w=1/g.groupby('case').case.transform('size').to_numpy();truth=g.changed.to_numpy(bool);guess=g.predicted_action.to_numpy()!=0;fixed=g.fixed.astype(str).str.lower()=='true'
        pm.append(dict(cohort=cohort,key=key,kind=kind,cases=g.case.nunique(),records=len(g),changed_records=int(truth.sum()),
            recall=float(recall_score(truth,guess,sample_weight=w,zero_division=0)) if truth.any() else None,
            pr_auc=float(average_precision_score(truth,1-g.p_keep,sample_weight=w)) if truth.any() else None,
            fixed_disagreement_percent=100*float(np.average(truth[fixed],weights=w[fixed])) if fixed.any() else None,
            duration_mae=float(np.average(abs(g.predicted_delta-g.delta),weights=w)) if kind!='O' else None))
    csv('prediction_metrics.csv',pm)
    if (ROOT/'recovery_preparation_ledger.json').exists():
        save(ROOT/'reports/recovery_preparation_ledger.json',read(ROOT/'recovery_preparation_ledger.json'))
    strat=[]
    for field in ['group','section','start_minutes','duration_minutes']:
        for level,g in paired.groupby(field):
            out=summarize(g);out['stratification']=field;out['level']=str(level);strat.append(out)
    csv('stratified_summary.csv',pd.concat(strat,ignore_index=True))
    chain=[]
    for (case,method),g in paired[(paired.policy=='P3')&(paired.method!='FULL')].groupby(['case','method']):
        a=g[g.reference_mode=='FULL_REFERENCE'].iloc[0];b=g[g.reference_mode=='CHAIN'].iloc[0]
        chain.append(dict(case=case,cohort=a.cohort,method=method,full_reference_objective=int(a.objective),chain_objective=int(b.objective),
            chain_minus_full_reference_loss=int(b.objective-a.objective),full_reference_solver_seconds=a.pipeline_solver_seconds,
            chain_solver_seconds=b.pipeline_solver_seconds,solver_speedup=a.pipeline_solver_seconds/b.pipeline_solver_seconds,
            prediction_solver_speedup=a.pipeline_prediction_solver_seconds/b.pipeline_prediction_solver_seconds,
            online_speedup=a.pipeline_online_seconds/b.pipeline_online_seconds))
    csv('chained_reference_comparison.csv',chain)
    sensitivity=[]
    for (cohort,policy),g in paired.groupby(['cohort','policy']):
        full=g[g.key=='FULL_'+policy].sort_values('solver_seconds',ascending=False)
        for count in [1,3]:
            removed=set(full.head(count).case);out=summarize(g[~g.case.isin(removed)]);out['removed_cases']=';'.join(sorted(removed));out['excluded_largest_full']=count;sensitivity.append(out)
    csv('outlier_sensitivity.csv',pd.concat(sensitivity,ignore_index=True))
    # Paired scenario bootstrap, stratified by closure location (not by variable rows).
    rng=np.random.default_rng(SEED);boot=[]
    for (cohort,policy,key),g in paired[~paired.key.str.startswith('FULL')].groupby(['cohort','policy','key']):
        blocks=[b.reset_index(drop=True) for _,b in g.groupby('section')];values=[]
        for _ in range(2000):
            sample=pd.concat([b.iloc[rng.integers(0,len(b),len(b))] for b in blocks]);values.append((sample.loss.mean(),sample.direct_solver_seconds.sum()/sample.pipeline_solver_seconds.sum()))
        z=np.array(values);boot.append(dict(cohort=cohort,policy=policy,key=key,mean_loss_ci_low=np.quantile(z[:,0],.025),mean_loss_ci_high=np.quantile(z[:,0],.975),solver_speedup_ci_low=np.quantile(z[:,1],.025),solver_speedup_ci_high=np.quantile(z[:,1],.975)))
    csv('case_bootstrap_comparison.csv',boot)
    uncertain=paired[(paired.pipeline_solver_seconds_upper>paired.pipeline_solver_seconds)|(paired.direct_solver_seconds_upper>paired.direct_solver_seconds)].copy()
    uncertain['solver_speedup_lower']=uncertain.direct_solver_seconds/uncertain.pipeline_solver_seconds_upper
    uncertain['solver_speedup_upper']=uncertain.direct_solver_seconds_upper/uncertain.pipeline_solver_seconds
    csv('timing_uncertainty.csv',uncertain)
    save(ROOT/'reports/cost_scope.json',dict(solver='Native CP-SAT walltime: both directions, private helpers, every attempt and FULL refinement. P3 pipelines include upstream P2.',
        prediction_solver='Inference plus cumulative native solver; offline training excluded.',online='P1 generation once + feature construction/persistence + model loading/inference + mask audit/preparation + subprocess costs including model build/search/audit.',
        source_data_and_case_input_preparation='Offline design generation separately archived in phase ledger; P1 generation charged online.',
        nested_costs='Process time contains solver/helper time; these are alternative scopes, never added together.',
        incomplete_native_logs='Lower bound and conservative wall-clock upper bound retained; no unknown interval represented as exact solver time.',
        labels='Representative initial FULL solutions, frozen before model training; baseline reconciliation changes comparison objectives only.',
        training='training_costs.json and phase_ledger.json; no transfer labels used',
        concurrent_scheduling='User-authorized transfer FULL precomputation overlaps offline training only; actual native timings are retained. execution_schedule.csv and scheduling_overlap_audit.json record overlap; phase wall times are not additive.',
        solver_queue_wait='Experiment scheduling waits are retained in transfer_full_gate_waits.json; they are not added to hypothetical standalone deployment latency.'))
    lines=['# 多中断场景实验结果','',f'登记111例，有效{len(valid)}例；完成{len(alljobs)}个初次目标模型。旧结果未用于新规则的基准。',
        '', 'Strategy-3的完整参考和串联流程均包含取得对应Strategy-2运行图的成本。三个时间口径分别记录，不能混称。','',summary.to_markdown(index=False),
        '', '## 预测表现','',pd.DataFrame(pm).to_markdown(index=False),'', '## 串联与完整参考流程','',pd.DataFrame(chain).groupby(['cohort','method']).agg(cases=('case','size'),mean_quality_difference=('chain_minus_full_reference_loss','mean'),median_solver_speedup=('solver_speedup','median')).to_markdown(),
        '', '全部质量结论以受限求解的终到延误为依据。变量标签不一致可能来自等目标运行图的不同调整，不能直接解释为固定导致损失。',
        '结果是同一路网、同一计划运行图上的内部抽样验证和冻结模型时间参数迁移；不推断未见路网或未见封锁位置的适用性。',
        '完整关联结果、异常长案例敏感性、逐场景损失、计时分解及全部证据见随附表。论文需在这些真实结果基础上完成实质编辑。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    body='<h1>Disruption generalization experiment</h1><p>Audited results; manuscript update pending.</p>'+summary.to_html(index=False)+pd.DataFrame(pm).to_html(index=False)
    (ROOT/'reports/index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Experiment results</title><style>body{font:15px Arial;margin:28px}table{border-collapse:collapse;margin:24px 0}td,th{padding:7px;border:1px solid #ddd}th{background:#eef3f8}</style>'+body+'</html>',encoding='utf-8')
    save(ROOT/'final_validation.json',dict(passed=True,registered=111,valid_cases=len(valid),initial_target_results=len(alljobs),physical_and_mask_audits=True,
        all_gaps_strictly_below_point001=True,no_restricted_result_beats_final_FULL=True,training_and_transfer_isolation=True,source_hashes_preserved=True,
        source_and_model_locks_checked=True,all_attempt_costs_retained=True,paper_update_pending=True,
        scheduling_overlap_audited=scheduling['passed']))
    print('FINAL_AUDIT_PASSED',len(alljobs),flush=True)
if __name__=='__main__':main()
