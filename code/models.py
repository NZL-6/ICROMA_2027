"""Training-only shared feature selection, statistical and residual models."""
import copy, hashlib, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.exceptions import ConvergenceWarning
from common import ROOT, SEED, save
warnings.filterwarnings('ignore',category=ConvergenceWarning)

CONTROL=['reference_value','physical_lower','physical_upper','can_decrease','can_increase']
IDENT=['case','start','duration','reference','unit_id','train','visit','other_train','other_visit','resource','station','direction','kind']
LABEL=['target_value','delta','action','changed','ambiguous','label_weight']
PATTERN=re.compile(r'^(?:[DEIXTPFGHBR]\d\d|M0[3-6])_')

def load_stage(stage):
    parts=[]
    for file in sorted((ROOT/'features').glob(f'*_{stage}.csv.gz')):
        header=pd.read_csv(file,nrows=0).columns
        dtype={c:'float32' for c in header if c not in IDENT+['ambiguous']}
        parts.append(pd.read_csv(file,low_memory=False,dtype=dtype,usecols=lambda x:x in IDENT+LABEL+CONTROL or bool(PATTERN.match(x))))
    d=pd.concat(parts,ignore_index=True)
    for c in d.columns:
        if c not in IDENT: d[c]=pd.to_numeric(d[c],errors='coerce').astype('float32')
    d.replace([np.inf,-np.inf],np.nan,inplace=True)
    return d

def weights(df,ambiguity=True):
    n=df.groupby(['case','kind']).case.transform('size').to_numpy(float)
    w=1/n
    if ambiguity:w*=df.label_weight.to_numpy(float)
    return w/w.sum()*len(w)

def weighted_quantile(y,q,w):
    if not len(y):return 0.
    idx=np.argsort(y); y=np.asarray(y)[idx];w=np.asarray(w)[idx]
    return float(np.interp(q,np.cumsum(w)/w.sum(),y))

class Features:
    """Per-task selected identities and scalers are fitted without held-out labels."""
    def fit(self,df):
        self.spec={}; self.fit_cases=sorted(df.case.unique().tolist())
        candidates=sorted(c for c in df if PATTERN.match(c))
        for kind,g in df.groupby('kind',sort=True):
            w=weights(g); y=g.changed.to_numpy(float); y-=np.average(y,weights=w)
            # Same paper ranking, evaluated in bounded column blocks on 32 GB RAM.
            score=np.zeros(len(candidates));usable={};digests={};global_flags={}
            for begin in range(0,len(candidates),96):
                names=candidates[begin:begin+96];x=g[names].to_numpy(float);finite=np.isfinite(x)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(x,axis=0)
                med=np.where(np.isfinite(med),med,0.);filled=np.where(finite,x,med)
                ranks=pd.DataFrame(filled).rank(method='average').to_numpy(float);ranks-=np.average(ranks,axis=0,weights=w)
                den=np.sqrt(np.sum(w[:,None]*ranks*ranks,axis=0)*np.sum(w*y*y))
                score[begin:begin+len(names)]=np.divide(abs((w*y)@ranks),den,out=np.zeros(len(names)),where=den>0)
                global_flags.update((g[names].groupby(g.case).nunique(dropna=False).max(axis=0)<=1).to_dict())
                for k,c in enumerate(names):
                    usable[c]=finite[:,k].any() and np.max(filled[:,k])!=np.min(filled[:,k])
                    digests[c]=hashlib.sha256(np.c_[filled[:,k],finite[:,k]].tobytes()).digest()
                del x,finite,filled,ranks
            seen=set(); local=[]; glob=[]
            for j in sorted(range(len(candidates)),key=lambda j:(-score[j],candidates[j])):
                if not usable[candidates[j]]:continue
                digest=digests[candidates[j]]
                if digest in seen:continue
                seen.add(digest)
                if global_flags[candidates[j]]:
                    if len(glob)<8:glob.append(candidates[j])
                elif len(local)<24:local.append(candidates[j])
                if len(local)==24 and len(glob)==8:break
            spec=dict(local=local,global_=glob,parameters={})
            for c in local+glob+CONTROL:
                a=g[c].to_numpy(float); good=np.isfinite(a); median=float(np.median(a[good])) if good.any() else 0.
                a=np.where(good,a,median); mean=float(np.average(a,weights=w));sd=float(np.sqrt(np.average((a-mean)**2,weights=w)))
                spec['parameters'][c]=[median,mean,max(sd,1e-6)]
            self.spec[kind]=spec
        return self

    def transform(self,df):
        local=np.zeros((len(df),58),dtype=np.float32);glob=np.zeros((len(df),16),dtype=np.float32)
        for kind in self.spec:
            idx=np.where(df.kind.to_numpy()==kind)[0];g=df.iloc[idx];s=self.spec[kind]
            for output,columns in [(local,s['local']+CONTROL),(glob,s['global_'])]:
                for j,c in enumerate(columns):
                    a=g[c].to_numpy(float); miss=~np.isfinite(a);med,mean,sd=s['parameters'][c]
                    output[idx,2*j]=np.clip((np.where(miss,med,a)-mean)/sd,-20,20)
                    output[idx,2*j+1]=miss
        return local,glob

    def export(self):return dict(fit_cases=self.fit_cases,spec=self.spec,local_dimension=58,global_dimension=16)

def legal_probs(prob,df):
    prob=np.maximum(np.asarray(prob,float),0)
    prob[df.can_decrease.to_numpy()==0,0]=0
    prob[df.can_increase.to_numpy()==0,2]=0
    total=prob.sum(axis=1);prob[total<=0,1]=1
    return prob/prob.sum(axis=1,keepdims=True)

def macro_loss(df,prob):
    truth=df.action.to_numpy(int)+1;w=weights(df)
    return float(np.average(-np.log(np.maximum(prob[np.arange(len(df)),truth],1e-12)),weights=w))

class StatModel:
    def __init__(self,C=1.):self.C=C;self.models={};self.qmodels={}
    def fit(self,df,feat):
        l,g=feat.transform(df);x=np.c_[l,g]
        for kind in feat.spec:
            take=np.where(df.kind.to_numpy()==kind)[0];d=df.iloc[take];y=d.action.to_numpy(int)+1;w=weights(d)
            labels=np.unique(y)
            if len(labels)==1:self.models[kind]=int(labels[0])
            else:
                m=LogisticRegression(C=self.C,solver='lbfgs',max_iter=800,tol=1e-6)
                m.fit(x[take],y,sample_weight=w);self.models[kind]=m
        return self

    def predict(self,df,feat):
        l,g=feat.transform(df);x=np.c_[l,g];out=np.zeros((len(df),3))
        for kind,m in self.models.items():
            take=np.where(df.kind.to_numpy()==kind)[0]
            if isinstance(m,int):out[take,m]=1
            else:out[np.ix_(take,m.classes_)]=m.predict_proba(x[take])
        return legal_probs(out,df)

    def fit_quantiles(self,df,feat,alpha=.01,quantiles=(.1,.5,.9)):
        l,g=feat.transform(df);x=np.c_[l,g]
        for kind in feat.spec:
            if kind=='O':continue
            for sign in [-1,1]:
                take=np.where((df.kind.to_numpy()==kind)&(df.action.to_numpy()==sign))[0]
                d=df.iloc[take];y=abs(d.delta.to_numpy(float));w=weights(d) if len(d) else np.array([])
                for q in quantiles:
                    if len(d)<5 or len(np.unique(y))==1:self.qmodels[kind,sign,q]=weighted_quantile(y,q,w)
                    else:
                        m=QuantileRegressor(quantile=q,alpha=alpha,solver='highs')
                        m.fit(x[take],y,sample_weight=w);self.qmodels[kind,sign,q]=m
        return self

    def quantiles(self,df,feat):
        l,g=feat.transform(df);x=np.c_[l,g];out=np.full((len(df),2,3),np.nan)
        for (kind,sign,q),m in self.qmodels.items():
            take=np.where(df.kind.to_numpy()==kind)[0]
            out[take,0 if sign<0 else 1,[.1,.5,.9].index(q)]=m if isinstance(m,float) else m.predict(x[take])
        for kind in self.models.keys() | {k for k,s,q in self.qmodels}:
            take=np.where(df.kind.to_numpy()==kind)[0]
            for j in range(2):
                a=out[take,j]
                if not len(a):continue
                for col in range(3):
                    missing=~np.isfinite(a[:,col]);a[missing,col]=np.nan_to_num(a[missing,1],nan=0.)
                out[take,j]=a
        return np.sort(np.maximum(np.nan_to_num(out),0),axis=-1)

def fit_calibration(df,prob):
    params={}
    for kind,g in df.groupby('kind',sort=True):
        take=np.where(df.kind.to_numpy()==kind)[0];p=np.clip(prob[take,1],1e-6,1-1e-6)
        x=np.log(p/(1-p)).reshape(-1,1);y=(g.changed.to_numpy()==0).astype(int)
        if len(np.unique(y))<2:params[kind]=dict(constant=float(y[0]));continue
        m=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000).fit(x,y,sample_weight=weights(g,False))
        params[kind]=dict(slope=float(m.coef_[0,0]),intercept=float(m.intercept_[0]))
    return params

def calibrate(df,prob,params):
    out=prob.copy()
    for kind,s in params.items():
        take=np.where(df.kind.to_numpy()==kind)[0];p=np.clip(prob[take,1],1e-6,1-1e-6)
        new=np.full(len(take),s['constant']) if 'constant' in s else 1/(1+np.exp(-np.clip(s['slope']*np.log(p/(1-p))+s['intercept'],-40,40)))
        sides=prob[take][:,[0,2]];total=sides.sum(axis=1)
        # A legal, unobserved changing class retains a tiny mass for ranking.
        sides[total<=1e-12]=np.c_[df.iloc[take].can_decrease,df.iloc[take].can_increase][total<=1e-12]
        total=sides.sum(axis=1); sides/=np.maximum(total[:,None],1e-12)
        out[take,0]=sides[:,0]*(1-new);out[take,1]=new;out[take,2]=sides[:,1]*(1-new)
    return legal_probs(out,df)

def prediction_frame(df,prob,q,method,stage,test_start):
    out=df[IDENT+LABEL+CONTROL].copy();out['method']=method;out['stage']=stage;out['test_start']=test_start
    for j,n in enumerate(['p_decrease','p_keep','p_increase']):out[n]=prob[:,j]
    action=np.argmax(prob,axis=1)-1;out['predicted_action']=action
    signed=np.zeros((len(df),3))
    for sign,j in [(-1,0),(1,1)]:
        take=np.where((action==sign)&(df.kind.to_numpy()!='O'))[0]
        signed[take]=q[take,j] if sign>0 else -q[take,j,::-1]
    order=df.kind.to_numpy()=='O';signed[order]=action[order,None]
    lo=(df.physical_lower-df.reference_value).to_numpy();hi=(df.physical_upper-df.reference_value).to_numpy()
    for j,n in enumerate(['delta_q10','predicted_delta','delta_q90']):
        vals=np.clip(signed[:,j],lo,hi);vals[order]=action[order];out[n]=vals
    out['fixed']=False
    for (_,kind),g in out.groupby(['case','kind']):
        ids=g.sort_values(['p_keep','unit_id'],ascending=[False,True],kind='stable').index[:len(g)//2]
        out.loc[ids,'fixed']=True
    return out

def pinball_score(df,q):
    losses=[]
    for kind,g in df[df.kind!='O'].groupby('kind'):
        take=np.where((df.kind.to_numpy()==kind)&(df.action.to_numpy()!=0))[0]
        if not len(take):continue
        sign=(df.iloc[take].action.to_numpy()>0).astype(int)
        pred=q[take,sign,1];y=abs(df.iloc[take].delta.to_numpy())
        losses.append(np.average(abs(y-pred)*.5,weights=weights(df.iloc[take],False)))
    return float(np.mean(losses)) if losses else 0.

def torch_runtime():
    import torch
    torch.set_num_threads(1)
    return torch

def neural_model(kinds):
    torch=torch_runtime();nn=torch.nn
    class Residual(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,64))
        def forward(self,x):return torch.relu(x+self.net(x))
    class Network(nn.Module):
        def __init__(self):
            super().__init__();self.kinds=list(kinds)
            self.local=nn.ModuleDict({k:nn.Linear(58,48) for k in kinds})
            self.context=nn.ModuleDict({k:nn.Linear(16,16) for k in kinds})
            self.shared=nn.Sequential(Residual(),Residual())
            self.action=nn.ModuleDict({k:nn.Linear(64,2 if k=='O' else 3) for k in kinds})
            self.magnitude=nn.ModuleDict({k:nn.Linear(64,6) for k in kinds if k!='O'})
        def forward(self,l,g,k):
            z=self.shared(torch.cat([torch.relu(self.local[k](l)),torch.relu(self.context[k](g))],dim=1))
            logits=self.action[k](z)
            q=None if k=='O' else torch.cumsum(torch.nn.functional.softplus(self.magnitude[k](z).reshape(-1,2,3)),dim=-1)
            return logits,q
    return Network()

def neural_arrays(df,feat,inference=False):
    torch=torch_runtime();l,g=feat.transform(df)
    if inference:return dict(l=torch.from_numpy(l),g=torch.from_numpy(g),kind=df.kind.to_numpy())
    return dict(l=torch.from_numpy(l),g=torch.from_numpy(g),kind=df.kind.to_numpy(),
        y=torch.tensor(df.action.to_numpy(int)+1,dtype=torch.long),
        delta=torch.tensor(abs(df.delta.to_numpy(float)),dtype=torch.float32),
        w=torch.tensor(weights(df),dtype=torch.float32))

def neural_loss(net,arr,ids):
    torch=torch_runtime();loss=torch.tensor(0.);den=0.
    for kind in net.kinds:
        ix=ids[arr['kind'][ids]==kind]
        if not len(ix):continue
        logits,q=net(arr['l'][ix],arr['g'][ix],kind); y=arr['y'][ix]-(1 if kind=='O' else 0)
        ce=torch.nn.functional.cross_entropy(logits,y,reduction='none');w=arr['w'][ix]
        part=(ce*w).sum()
        if q is not None:
            changed=(arr['y'][ix]!=1);j=(arr['y'][ix][changed]==2).long()
            if changed.any():
                selected=q[changed,j];err=arr['delta'][ix][changed,None]-selected
                tau=torch.tensor([.1,.5,.9]);pin=torch.maximum(tau*err,(tau-1)*err).mean(dim=1)
                part=part+.5*(pin*w[changed]).sum()
        loss=loss+part;den+=float(w.sum())
    return loss/max(den,1e-9)

def fit_neural(train,feat,validation=None,epochs=None,seed=SEED):
    torch=torch_runtime();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=neural_model(sorted(train.kind.unique()));opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001)
    arr=neural_arrays(train,feat);val=neural_arrays(validation,feat) if validation is not None else None
    best=float('inf');best_state=None;best_epoch=0;stale=0;history=[];started=time.perf_counter()
    for epoch in range(1,(epochs or 200)+1):
        net.train();idx=rng.permutation(len(train))
        for a in range(0,len(idx),256):
            opt.zero_grad();loss=neural_loss(net,arr,idx[a:a+256]);loss.backward();opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad(): score=float(neural_loss(net,val,np.arange(len(validation))))
            history.append([epoch,score])
            if score<best-1e-6:best=score;best_epoch=epoch;best_state=copy.deepcopy(net.state_dict());stale=0
            else:stale+=1
            if stale>=20:break
    if best_state is not None:net.load_state_dict(best_state)
    return net,dict(epochs_run=epoch,best_epoch=best_epoch or epoch,validation_loss=best if np.isfinite(best) else None,
                    seconds=time.perf_counter()-started,history=history,seed=seed)

def neural_predict(net,df,feat):
    torch=torch_runtime();a=neural_arrays(df,feat,inference=True);out=np.zeros((len(df),3));qs=np.zeros((len(df),2,3));net.eval()
    with torch.no_grad():
        for kind in net.kinds:
            ids=np.where(a['kind']==kind)[0]
            logits,q=net(a['l'][ids],a['g'][ids],kind);p=torch.softmax(logits,dim=1).numpy()
            if kind=='O':out[ids,1:]=p
            else:out[ids]=p;qs[ids]=q.numpy()
    return legal_probs(out,df),qs
