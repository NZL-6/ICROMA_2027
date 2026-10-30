"""Case-balanced association study, dependence-aware inference and grouped CV.

Reports the chosen optimum's changes; coefficients are associations, not causal
effects of fixing a variable. All columns, including constants, are inventoried.
"""
from pathlib import Path
import json,time,os,gc,sys,subprocess
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'statistics';OUT.mkdir(exist_ok=True)
META=['case','start','duration','reference','unit_id','train','visit','other_train','other_visit','resource','station','direction','kind']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load_features(p):
    df=pd.read_csv(p,low_memory=False);extra=ROOT/'higher_order'/p.name
    if extra.exists():
        h=pd.read_csv(extra);assert (h.unit_id.to_numpy()==df.unit_id.to_numpy()).all()
        df=pd.concat([df,h.drop(columns='unit_id')],axis=1)
    return df
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def labels(frame,target):
    sch={c:read(ROOT/'cases'/c/f'{target}_optimal.json')['schedule'] for c in frame['case'].unique()}
    base={c:read(ROOT/'cases'/c/f'{frame.iloc[0]["reference"]}_optimal.json')['schedule'] for c in frame['case'].unique()}
    ys=[];values=[]
    for r in frame[META].itertuples(index=False):
        a=base[r.case];b=sch[r.case];i=r.train;n=r.visit
        if r.kind=='S':old,new=a[i][n]['stop'],b[i][n]['stop']
        elif r.kind=='Td':old,new=a[i][n]['departure']-a[i][n]['arrival'],b[i][n]['departure']-b[i][n]['arrival']
        elif r.kind=='Tr':old,new=a[i][n+1]['arrival']-a[i][n]['departure'],b[i][n+1]['arrival']-b[i][n]['departure']
        else:
            j,k=r.other_train,r.other_visit
            old=int(a[i][n]['departure']<a[j][k]['departure']);new=int(b[i][n]['departure']<b[j][k]['departure'])
        ys.append(new-old);values.append((old,new))
    return np.array(ys,dtype=float),np.array(values)
def caseweights(c):
    _,inv,cnt=np.unique(c,return_inverse=True,return_counts=True);return 1/(cnt[inv]*len(cnt))
def codes(x):return np.unique(x,return_inverse=True)[1]
def grouped_sum(x,g):
    z=np.zeros((max(g)+1,x.shape[1]));np.add.at(z,g,x);return z
def inventory(df,comparison,kind):
    rows=[]
    for k in df:
        if k in META:continue
        z=df[k].to_numpy(dtype=float);ok=np.isfinite(z);unique=len(np.unique(z[ok]));rows.append(dict(comparison=comparison,kind=kind,feature=k,catalog_id=k.split('_')[0],n_available=int(ok.sum()),missing_fraction=float(1-ok.mean()),n_unique=unique,status='all_missing' if not ok.any() else 'constant' if unique==1 else 'variable'))
    return rows

def univariate(df,y,comparison,kind,response):
    features=[k for k in df if k not in META];N=len(df);C=codes(df['case']);A=codes(df['start']);case=df['case'].to_numpy();G=len(np.unique(C));Gstart=len(np.unique(A))
    # Dependence spans all visits of a repeated train. O rows are dyadic:
    # any shared endpoint train can induce dependence, even for different pairs.
    train1=df['train'].astype(str).to_numpy(dtype=str)
    if kind=='O':
        train2=df['other_train'].astype(str).to_numpy(dtype=str);trainlevels=np.unique(np.r_[train1,train2]);U1=np.searchsorted(trainlevels,train1);U2=np.searchsorted(trainlevels,train2)
        pair=np.array(['|'.join(sorted([a,b])) for a,b in zip(train1,train2)]);U=codes(pair);CU=codes(np.char.add(np.char.add(case.astype(str),'@'),pair))
        CU1=codes(np.char.add(np.char.add(case.astype(str),'@'),train1));CU2=codes(np.char.add(np.char.add(case.astype(str),'@'),train2));Gunit=len(trainlevels)
    else:
        U=codes(train1);CU=codes(np.char.add(np.char.add(case.astype(str),'@'),train1));Gunit=len(np.unique(U))
    available=[];rows=[];is_binary=response=='changed';outcome_cases=sum(bool((y[C==c]>0).any()) for c in range(G)) if is_binary else G
    case_mean_varies=np.ptp([y[C==c].mean() for c in range(G)])>1e-12
    for begin in range(0,len(features),96):
        names=features[begin:begin+96];X=df[names].to_numpy(dtype=float);ok=np.isfinite(X)
        w=np.zeros_like(X)
        for c in range(G):
            sel=C==c;counts=ok[sel].sum(axis=0);w[sel]=np.divide(ok[sel],counts,out=np.zeros_like(X[sel]),where=counts>0)/G
        total=w.sum(axis=0);w=np.divide(w,total,out=w,where=total>0)
        X=np.nan_to_num(X,nan=0,posinf=0,neginf=0);mx=(w*X).sum(axis=0);dx=np.where(ok,X-mx,0);sx=np.sqrt((w*dx*dx).sum(axis=0));valid=sx>1e-10
        valid&=(np.max(np.where(ok,X,-np.inf),axis=0)>np.min(np.where(ok,X,np.inf),axis=0))
        z=np.divide(dx,sx,out=np.zeros_like(dx),where=valid);my=(w*y[:,None]).sum(axis=0);dy=y[:,None]-my
        beta=(w*z*dy).sum(axis=0);res=dy-z*beta;score=w*z*res
        def vc(g):
            sums=grouped_sum(score,g);ng=(grouped_sum(w,g)>0).sum(axis=0)
            correction=np.divide(ng,ng-1,out=np.ones_like(ng,dtype=float),where=ng>1)
            return np.sum(sums*sums,axis=0)*correction
        sc=vc(C)
        if kind=='O':
            sums=np.zeros((Gunit,score.shape[1]));np.add.at(sums,U1,score);np.add.at(sums,U2,score)
            nw=np.zeros_like(sums);np.add.at(nw,U1,w);np.add.at(nw,U2,w);observed_units=(nw>0).sum(axis=0)
            su=np.sum(sums*sums,axis=0)*Gunit/(Gunit-1)-vc(U)
            # Shared case-train intersections use one common index space.
            ctlevels={z:i for i,z in enumerate(sorted(set(zip(C,train1))|set(zip(C,train2))))}
            ss=np.zeros((len(ctlevels),score.shape[1]));np.add.at(ss,[ctlevels[z] for z in zip(C,train1)],score);np.add.at(ss,[ctlevels[z] for z in zip(C,train2)],score)
            si=np.sum(ss*ss,axis=0)*len(ss)/(len(ss)-1)-vc(CU)
        else:su=vc(U);si=vc(CU);observed_units=(grouped_sum(w,U)>0).sum(axis=0)
        v=sc+su-si
        fallback=v<=0;v=np.where(fallback,np.maximum(sc,su),v);se=np.sqrt(v)
        observed_cases=(grouped_sum(w,C)>0).sum(axis=0);observed_starts=(grouped_sum(w,A)>0).sum(axis=0)
        dfree=np.maximum(1,np.minimum(observed_cases-1,observed_units-1))
        p=2*stats.t.sf(np.divide(abs(beta),se,out=np.full_like(beta,np.inf),where=se>0),dfree);q=stats.t.ppf(.975,dfree)
        sa=np.sqrt(vc(A));pa=2*stats.t.sf(np.divide(abs(beta),sa,out=np.full_like(beta,np.inf),where=sa>0),np.maximum(1,observed_starts-1))
        for j,k in enumerate(names):
            if not valid[j]:continue
            keep=ok[:,j];rankx=stats.rankdata(X[keep,j]);ranky=stats.rankdata(y[keep]);ww=w[keep,j];ww=ww/ww.sum()
            aa=rankx-np.sum(ww*rankx);bb=ranky-np.sum(ww*ranky);spearman=float(np.sum(ww*aa*bb)/np.sqrt(np.sum(ww*aa*aa)*np.sum(ww*bb*bb))) if np.var(ranky)>0 else np.nan
            # Leave-one-start estimates: recompute case balance on remaining scenes.
            estimates=[]
            for a in range(Gstart):
                kk=keep&(A!=a);wt=caseweights(case[kk]);xx=X[kk,j];yy=y[kk];xx=xx-np.sum(wt*xx);den=np.sum(wt*xx*xx)
                estimates.append(float(np.sum(wt*xx*(yy-np.sum(wt*yy)))/den*sx[j]) if den>1e-10 else np.nan)
            ee=np.array(estimates);direction=float(np.nanmean(np.sign(ee)==np.sign(beta[j])))
            usable=observed_cases[j]>=10 and observed_starts[j]>=4 and outcome_cases>=5 and np.var(y)>0 and (case_mean_varies or not k.startswith(('D','I')))
            scale=100 if is_binary else 1
            rows.append(dict(comparison=comparison,kind=kind,response=response,feature=k,catalog_id=k.split('_')[0],n=int(keep.sum()),n_scenarios=int(len(np.unique(C[keep]))),positive_scenarios=outcome_cases,feature_mean=float(mx[j]),feature_sd=float(sx[j]),slope_per_sd=float(beta[j])*scale,ci_low=float(beta[j]-q[j]*se[j])*scale,ci_high=float(beta[j]+q[j]*se[j])*scale,effect_unit='percentage_points_per_SD' if is_binary else 'minutes_per_SD',spearman=spearman,p_two_way=float(p[j]) if usable else np.nan,p_start7=float(pa[j]) if usable else np.nan,se_fallback=bool(fallback[j]),leave_start_sign_fraction=direction,leave_start_min=float(np.nanmin(ee))*scale,leave_start_max=float(np.nanmax(ee))*scale,inference=('scenario_plus_dyadic_train_cluster' if kind=='O' else 'scenario_plus_train_cluster') if usable else 'descriptive_rare_or_invariant_outcome'))
    return rows

def predictive(df,y,comparison,kind,response):
    features=[k for k in df if k not in META];X=df[features].to_numpy(dtype=float);X[~np.isfinite(X)]=np.nan
    good=np.nanstd(X,axis=0)>1e-10;X=X[:,good];names=np.array(features)[good]
    # Exact duplicates are non-identifiable, retain their mapping, not a utility screen.
    signatures={};keep=[];dup=[]
    for j in range(X.shape[1]):
        signature=hash(np.nan_to_num(X[:,j],nan=-1.23456789e100).tobytes())
        if signature in signatures:dup.append(dict(feature=str(names[j]),same_as=str(names[signatures[signature]])))
        else:signatures[signature]=j;keep.append(j)
    X=X[:,keep];names=names[keep];pred=np.zeros(len(y));null=np.zeros(len(y));fold=[];coeff=[]
    for held in sorted(df['start'].unique()):
        tr=df['start'].to_numpy()!=held;te=~tr;xx=X[tr];xt=X[te];med=np.nanmedian(xx,axis=0);med=np.nan_to_num(med)
        missing_train=np.isnan(xx);missing_test=np.isnan(xt);xx=np.where(missing_train,med,xx);xt=np.where(missing_test,med,xt)
        w=caseweights(df.loc[tr,'case'].to_numpy())*tr.sum();mu=np.average(xx,axis=0,weights=w);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=w));sd[sd<1e-10]=1
        mm=missing_train.any(axis=0);xx=np.column_stack(((xx-mu)/sd,missing_train[:,mm].astype(float)));xt=np.column_stack(((xt-mu)/sd,missing_test[:,mm].astype(float)))
        model=Ridge(alpha=100,solver='lsqr',tol=1e-4,max_iter=1000);model.fit(xx,y[tr],sample_weight=w);yp=model.predict(xt)
        if response=='changed':yp=np.clip(yp,0,1)
        pred[te]=yp;null[te]=np.average(y[tr],weights=w);wt=caseweights(df.loc[te,'case'].to_numpy())
        fold.append(dict(comparison=comparison,kind=kind,response=response,held_start=int(held),n_test=int(te.sum()),mse=float(np.average((y[te]-yp)**2,weights=wt)),null_mse=float(np.average((y[te]-null[te])**2,weights=wt)),mae=float(np.average(abs(y[te]-yp),weights=wt))))
        for j,k in enumerate(names):coeff.append(dict(comparison=comparison,kind=kind,response=response,held_start=int(held),feature=str(k),coefficient_per_train_sd=float(model.coef_[j])))
    w=caseweights(df['case'].to_numpy());mse=np.average((y-pred)**2,weights=w);nmse=np.average((y-null)**2,weights=w)
    info=dict(comparison=comparison,kind=kind,response=response,n=len(y),features=len(names),exact_duplicates=len(dup),all_defined_nonconstant_features_enter=True,alpha=100,solver='ridge_lsqr',cv='leave_one_start_out',fold_count=int(df['start'].nunique()),weighted_mse=float(mse),null_mse=float(nmse),skill_vs_training_mean=float(1-mse/nmse) if nmse else None,mae=float(np.average(abs(y-pred),weights=w)))
    pd.DataFrame(dup).to_csv(OUT/f'duplicates_{comparison}_{kind}.csv',index=False)
    return info,fold,coeff,pred

def main():
    subprocess.run([sys.executable,str(ROOT/'higher_order_features.py')],check=True)
    started=time.perf_counter();allrows=[];allinv=[];des=[];cv=[];folds=[];coefs=[];casechange=[]
    files=sorted((ROOT/'features').glob('*_P1.csv.gz'))
    if len(files)!=35:raise RuntimeError(('Need all 35 scenes',len(files)))
    for ref,targets in [('P1',['P2','P3']),('P2',['P3'])]:
        parts=[load_features(f.with_name(f.name.replace('_P1','_'+ref))) for f in files]
        data=pd.concat(parts,ignore_index=True);del parts;gc.collect()
        for target in targets:
            comparison=target+'-'+ref
            for kind in ['S','Td','Tr','O']:
                df=data[data['kind']==kind].copy();df=df.loc[:,META+[k for k in df.columns if k not in META and df[k].notna().any()]]
                delta,values=labels(df,target);responses={'changed':(delta!=0).astype(float)}
                if kind in ['Td','Tr']:responses.update(signed=delta,absolute=abs(delta))
                allinv+=inventory(df,comparison,kind)
                lab=df[META].copy();lab['original_reference_value']=values[:,0];lab['target_value']=values[:,1];lab['signed_change']=delta;lab['changed']=(delta!=0).astype(int);lab.to_csv(OUT/f'labels_{comparison}_{kind}.csv.gz',index=False)
                for c in sorted(df['case'].unique()):
                    sel=df['case'].to_numpy()==c;casechange.append(dict(case=c,comparison=comparison,kind=kind,n=int(sel.sum()),changed=int(np.count_nonzero(delta[sel])),changed_fraction=float(np.mean(delta[sel]!=0)),mean_signed=float(np.mean(delta[sel])),mean_absolute=float(np.mean(abs(delta[sel])))))
                for response,y in responses.items():
                    checkpoint=OUT/f'associations_{comparison}_{kind}_{response}.csv.gz'
                    if np.var(y)==0:rr=[]
                    elif checkpoint.exists():rr=pd.read_csv(checkpoint).to_dict('records')
                    else:rr=univariate(df,y,comparison,kind,response);pd.DataFrame(rr).to_csv(checkpoint,index=False)
                    allrows+=rr
                    w=caseweights(df['case'].to_numpy());des.append(dict(comparison=comparison,kind=kind,response=response,n=len(y),scenarios=df['case'].nunique(),case_balanced_mean=float(np.sum(w*y)),min=float(y.min()),max=float(y.max())))
                    if response in ['changed','absolute'] and np.var(y)>0:
                        info,fo,cc,pred=predictive(df,y,comparison,kind,response);cv.append(info);folds+=fo;coefs+=cc
                    print(comparison,kind,response,'tested',len(rr),'seconds',round(time.perf_counter()-started,1),flush=True)
                del df;gc.collect()
        del data;gc.collect()
    result=pd.DataFrame(allrows)
    for col,suffix in [('p_two_way',''),('p_start7','_start7')]:
        ok=result[col].notna();result['q_BY'+suffix]=np.nan;result['q_BH'+suffix]=np.nan
        if ok.any():
            result.loc[ok,'q_BY'+suffix]=multipletests(result.loc[ok,col],method='fdr_by')[1];result.loc[ok,'q_BH'+suffix]=multipletests(result.loc[ok,col],method='fdr_bh')[1]
    result.to_csv(OUT/'all_associations.csv.gz',index=False);pd.DataFrame(allinv).to_csv(OUT/'all_feature_status.csv.gz',index=False)
    pd.DataFrame(casechange).to_csv(OUT/'case_decision_changes.csv',index=False);pd.DataFrame(des).to_csv(OUT/'response_summary.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'grouped_cv_folds.csv',index=False);pd.DataFrame(coefs).to_csv(OUT/'ridge_coefficients.csv.gz',index=False)
    save(OUT/'grouped_cv_summary.json',cv)
    selected=result[(result.q_BY<.05)&(result.leave_start_sign_fraction>=6/7)].sort_values('q_BY');selected.to_csv(OUT/'stable_associations.csv',index=False)
    save(OUT/'analysis_summary.json',dict(complete=True,seconds=time.perf_counter()-started,associations=len(result),hypotheses=int(result.p_two_way.notna().sum()),BY_significant=int((result.q_BY<.05).sum()),BY_start7_significant=int((result.q_BY_start7<.05).sum()),direction_stable_BY_significant=len(selected),policy='All comparisons, responses and covariates share one BY multiplicity family. Seven-start sensitivity separately adjusted. No causal or universal claim.'))
    from finalise_statistics import main as finalise
    finalise()
if __name__=='__main__':main()
