"""Nested grouped CV, out-of-fold calibration, and per-case predictions."""
import argparse, pickle, time
import numpy as np
import pandas as pd
import joblib
from common import *
from models import *

def feature_fit(train,folder):
    file=folder/'features.pkl'
    if file.exists():return joblib.load(file)
    started=time.perf_counter();feat=Features().fit(train)
    folder.mkdir(parents=True,exist_ok=True);joblib.dump(feat,file)
    save(folder/'features.json',dict(**feat.export(),fit_seconds=time.perf_counter()-started))
    return feat

def save_predictions(frame,method,stage,start,timing):
    target=ROOT/'predictions'/method;target.mkdir(parents=True,exist_ok=True)
    for sid,g in frame.groupby('case',sort=True):
        g.to_csv(target/f'{sid}_{stage}.csv.gz',index=False,compression='gzip')
    save(ROOT/'models'/method/stage/f'fold_{start:03d}'/'complete.json',dict(
        method=method,stage=stage,test_start=start,cases=sorted(frame.case.unique()),rows=len(frame),
        prediction_columns=list(frame.columns),**timing))

def train_stat(stage,df,fold):
    start=fold['test_start'];folder=ROOT/'models/STAT'/stage/f'fold_{start:03d}'
    if (folder/'complete.json').exists():return
    began=time.perf_counter();train=df[df.case.isin(fold['train'])].reset_index(drop=True)
    test=df[df.case.isin(fold['test'])].reset_index(drop=True);n=len(train)
    Cs=[.01,.1,1.,10.];alphas=[.001,.01,.1]
    oof={c:np.zeros((n,3)) for c in Cs};qscore={a:[] for a in alphas};inner_records=[]
    for inner in fold['inner']:
        validation=inner['validation_start'];idx=np.where(train.case.isin(inner['validation']).to_numpy())[0]
        tr=train[train.case.isin(inner['train'])].reset_index(drop=True);va=train.iloc[idx].reset_index(drop=True)
        ff=ROOT/'shared_features'/stage/f'outer_{start:03d}'/f'inner_{validation:03d}'
        feat=feature_fit(tr,ff)
        models={}
        for C in Cs:
            model=StatModel(C).fit(tr,feat);oof[C][idx]=model.predict(va,feat);models[C]=model
        # Quantile regularisation selected with median pinball loss only.
        for a in alphas:
            model=StatModel().fit_quantiles(tr,feat,a,quantiles=(.5,))
            score=pinball_score(va,model.quantiles(va,feat));qscore[a].append(score)
        inner_records.append(dict(validation_start=validation,train_cases=sorted(tr.case.unique()),
            validation_cases=sorted(va.case.unique()),feature_spec=str(ff.relative_to(ROOT))))
        print('STAT_INNER',stage,start,validation,round(time.perf_counter()-began,1),flush=True)
    scores={c:macro_loss(train,p) for c,p in oof.items()};C=min(Cs,key=lambda c:(scores[c],c))
    alpha=min(alphas,key=lambda a:(np.mean(qscore[a]),a));cal=fit_calibration(train,oof[C])
    feat=feature_fit(train,ROOT/'shared_features'/stage/f'outer_{start:03d}'/'final')
    model=StatModel(C).fit(train,feat).fit_quantiles(train,feat,alpha)
    folder.mkdir(parents=True,exist_ok=True);joblib.dump(model,folder/'model.joblib')
    save(folder/'selection.json',dict(logistic_C=C,logistic_C_scores=scores,quantile_alpha=alpha,
        quantile_alpha_scores={a:float(np.mean(s)) for a,s in qscore.items()},calibration=cal,inner=inner_records,
        training_cases=fold['train'],test_cases=fold['test'],training_seconds=time.perf_counter()-began))
    at=time.perf_counter()
    if len(test):
        prob=calibrate(test,model.predict(test,feat),cal);q=model.quantiles(test,feat)
        frame=prediction_frame(test,prob,q,'STAT',stage,start)
    else:frame=pd.DataFrame(columns=['case'])
    save_predictions(frame,'STAT',stage,start,dict(training_seconds=at-began,prediction_seconds=time.perf_counter()-at))
    print('STAT_FOLD_DONE',stage,start,round(time.perf_counter()-began,1),flush=True)

def train_ml(stage,df,fold):
    start=fold['test_start'];folder=ROOT/'models/ML'/stage/f'fold_{start:03d}'
    if (folder/'complete.json').exists():return
    began=time.perf_counter();train=df[df.case.isin(fold['train'])].reset_index(drop=True)
    test=df[df.case.isin(fold['test'])].reset_index(drop=True);oof=np.zeros((len(train),3));epochs=[];inner_records=[]
    torch=torch_runtime();folder.mkdir(parents=True,exist_ok=True)
    for inner in fold['inner']:
        validation=inner['validation_start'];idx=np.where(train.case.isin(inner['validation']).to_numpy())[0]
        tr=train[train.case.isin(inner['train'])].reset_index(drop=True);va=train.iloc[idx].reset_index(drop=True)
        ff=ROOT/'shared_features'/stage/f'outer_{start:03d}'/f'inner_{validation:03d}'
        feat=feature_fit(tr,ff);log=folder/f'inner_{validation:03d}.json';checkpoint=folder/f'inner_{validation:03d}.pt'
        if log.exists() and checkpoint.exists():
            record=read(log);net=neural_model(sorted(tr.kind.unique()));net.load_state_dict(torch.load(checkpoint,weights_only=True))
        else:
            net,record=fit_neural(tr,feat,va,seed=SEED+start*1000+validation)
            torch.save(net.state_dict(),checkpoint);record.update(train_cases=inner['train'],validation_cases=inner['validation']);save(log,record)
        p,q=neural_predict(net,va,feat);oof[idx]=p;epochs.append(record['best_epoch']);inner_records.append(record)
        print('ML_INNER',stage,start,validation,record['best_epoch'],round(time.perf_counter()-began,1),flush=True)
    cal=fit_calibration(train,oof);final_epochs=max(1,int(np.median(epochs)))
    feat=feature_fit(train,ROOT/'shared_features'/stage/f'outer_{start:03d}'/'final')
    net,record=fit_neural(train,feat,epochs=final_epochs,seed=SEED+start*1000+999)
    torch.save(net.state_dict(),folder/'model.pt')
    save(folder/'selection.json',dict(epochs=final_epochs,inner_best_epochs=epochs,calibration=cal,
        inner=inner_records,training_cases=fold['train'],test_cases=fold['test'],final_training=record,
        total_training_seconds=time.perf_counter()-began))
    at=time.perf_counter()
    if len(test):
        prob,q=neural_predict(net,test,feat);prob=calibrate(test,prob,cal)
        frame=prediction_frame(test,prob,q,'ML',stage,start)
    else:frame=pd.DataFrame(columns=['case'])
    save_predictions(frame,'ML',stage,start,dict(training_seconds=at-began,prediction_seconds=time.perf_counter()-at))
    print('ML_FOLD_DONE',stage,start,round(time.perf_counter()-began,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['STAT','ML','both'],default='both')
    ap.add_argument('--stage',choices=['A','B','both'],default='both');ap.add_argument('--test-start',type=int)
    args=ap.parse_args();folds=read(ROOT/'folds.json')
    for stage in STAGES:
        if args.stage not in ['both',stage]:continue
        df=load_stage(stage)
        if len(df)==0:raise RuntimeError('No valid training examples')
        for fold in folds:
            if args.test_start is not None and fold['test_start']!=args.test_start:continue
            if args.method in ['STAT','both']:train_stat(stage,df,fold)
            if args.method in ['ML','both']:train_ml(stage,df,fold)
    print('TRAINING_REQUEST_COMPLETE',flush=True)

if __name__=='__main__': main()
