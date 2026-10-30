"""Exercise fixed-set adapters and cost summaries using retained real pilots."""
import time
import numpy as np
import pandas as pd
from common import *
from features import build
from models import LABEL,legal_probs,prediction_frame
from predict import mask
from analyze import job_metrics,summarize
def main():
    records=[];tables=[]
    for case in ['D001','D026','D051']:
        sc=read(ROOT/'cases'/case/'scenario.json');p1=job_metrics(case,'REF_P1')
        for stage in STAGES:
            ref=read(ROOT/'cases'/case/f'{STAGES[stage][0]}_optimal.json');df,_=build(case,stage,'FULL_REFERENCE',ref)
            for c in LABEL:df[c]=np.nan
            probabilities=legal_probs(np.tile([.25,.5,.25],(len(df),1)),df)
            pred=prediction_frame(df,probabilities,np.zeros((len(df),2,3)),'VALIDATION_ONLY',stage,-1)
            m,check=mask(case,stage,pred,ref)
            assert all(v['fixed']==v['candidates']//2 for v in check['counts'].values())
            assert not m.get('stops') and not m.get('tracks')
            policy=STAGES[stage][1];job=job_metrics(case,'FULL_'+policy)
            row=dict(job,cohort='validation_only',full_objective=job['objective'],relative_loss_percent=0.,
                pipeline_solver_seconds=job['solver_seconds'],pipeline_prediction_solver_seconds=job['solver_seconds'],
                pipeline_online_seconds=p1['online_increment_seconds']+job['online_increment_seconds'],direct_solver_seconds=job['solver_seconds'],
                direct_online_seconds=p1['online_increment_seconds']+job['online_increment_seconds'])
            tables.append(row);records.append(dict(case=case,stage=stage,counts=check['counts'],solver_seconds=job['solver_seconds'],passed=True))
    result=summarize(pd.DataFrame(tables));assert np.allclose(result.solver_speedup,1.) and np.allclose(result.online_speedup,1.)
    save(ROOT/'validation/fixing_and_reporting.json',dict(passed=True,records=records,validation_predictions_not_used_in_experiment=True,
        nested_costs_not_added=True,full_self_comparison_speedup_exactly_one=True))
    print('FIXING_AND_REPORTING_VALIDATION_PASSED',flush=True)
if __name__=='__main__':main()
