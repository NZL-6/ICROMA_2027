"""Re-estimate associations on base cases only; never screen transfer outcomes."""
import time
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from common import *
from models import load_stage,IDENT,LABEL,CONTROL,PATTERN
def main():
    at=time.perf_counter();engine=module(ROOT/'snapshots/statistical_analysis.py','descriptive_associations')
    scene={s['case']:s for s in read(ROOT/'case_manifest.json')};allrows=[];allinventory=[]
    for stage in STAGES:
        df=load_stage(stage);assert all(scene[c]['cohort']=='base' for c in df.case.unique())
        for kind,g in df.groupby('kind',sort=True):
            g=g.reset_index(drop=True);features=[c for c in g if PATTERN.match(c)]
            frame=g[IDENT+features].copy();comparison=STAGES[stage][1]+'-'+STAGES[stage][0]
            allinventory+=engine.inventory(frame,comparison,kind)
            for response in ['changed','delta']:
                if kind=='O' and response=='delta':continue
                y=g[response].to_numpy(float);rows=engine.univariate(frame,y,comparison,kind,response)
                byfeature={r['feature']:r for r in rows};locations=np.array([scene[c]['section'] for c in g.case]);case=g.case.to_numpy()
                for begin in range(0,len(features),96):
                    names=features[begin:begin+96];x=g[names].to_numpy(float);estimates=[]
                    for location in sorted(set(locations)):
                        keep=locations!=location;xx=x[keep];yy=y[keep];cc=case[keep];finite=np.isfinite(xx);w=np.zeros_like(xx)
                        for c in np.unique(cc):
                            ix=cc==c;count=finite[ix].sum(axis=0);w[ix]=np.divide(finite[ix],count,out=np.zeros_like(xx[ix]),where=count>0)
                        w=np.divide(w,w.sum(axis=0),out=np.zeros_like(w),where=w.sum(axis=0)>0)
                        xx=np.nan_to_num(xx,nan=0,posinf=0,neginf=0);mx=np.sum(w*xx,axis=0);my=np.sum(w*yy[:,None],axis=0)
                        dx=np.where(finite,xx-mx,0);variance=np.sum(w*dx*dx,axis=0)
                        slope=np.divide(np.sum(w*dx*(yy[:,None]-my),axis=0),variance,out=np.full(len(names),np.nan),where=variance>1e-10)
                        estimates.append(slope)
                    for j,name in enumerate(names):
                        if name not in byfeature:continue
                        r=byfeature[name];ss=np.array([v[j] for v in estimates]);finite=np.isfinite(ss)
                        r['leave_location_sign_fraction']=float(np.mean(np.sign(ss[finite])==np.sign(r['slope_per_sd']))) if finite.any() else None
                        r['stage']=stage
                valid=[i for i,r in enumerate(rows) if r['p_two_way'] is not None and np.isfinite(r['p_two_way'])]
                adjusted=multipletests([rows[i]['p_two_way'] for i in valid],method='fdr_by')[1] if valid else []
                for r in rows:r['adjusted_p']=None
                for i,p in zip(valid,adjusted):rows[i]['adjusted_p']=float(p)
                allrows+=rows
        print('ASSOCIATION_STAGE_DONE',stage,flush=True)
        del df
    out=ROOT/'reports';pd.DataFrame(allrows).to_csv(out/'indicator_associations.csv',index=False)
    pd.DataFrame(allinventory).to_csv(out/'indicator_inventory.csv',index=False)
    definitions=read(ROOT/'feature_definitions.json');catalog=read(ROOT/'candidate_catalog.json')['records']
    coverage=[]
    for item in catalog:
        columns=[k for k in definitions if k.split('_')[0]==item['id']]
        coverage.append(dict(**item,columns=columns,status='computed_or_nonapplicable_by_row' if columns else 'not_computable_in_this_model'))
    save(out/'catalog_coverage.json',coverage)
    save(ROOT/'association_complete.json',dict(complete=True,base_only=True,rows=len(allrows),seconds=time.perf_counter()-at,
        inference='case/train dependence; dyadic train dependence for order pairs; BY correction; leave-onset and leave-location sensitivity',
        not_used_as_global_predictive_feature_filter=True))
if __name__=='__main__':main()
