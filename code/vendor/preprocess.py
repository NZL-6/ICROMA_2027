"""Target-blind interval consistency shared by both formulations."""
import copy,itertools
from collections import defaultdict

class EmptyDomain(ValueError):pass

def interval(data,t,n,phase):
    v=t['visits'][n]
    if 'fixed_'+phase in v:return [v['fixed_'+phase]]*2
    lo,hi=v.get(phase+'_bounds',[data['now'],data['horizon']])
    lo=max(lo,data['now'])
    if n==0 or phase=='departure' and v.get('commercial',False):lo=max(lo,v['plan_'+phase])
    return [lo,min(hi,data['horizon'])]

def preprocess(raw,mask=None):
    mask=mask or {};data=copy.deepcopy(raw);times={};dwells={};groups=defaultdict(list)
    for t in data['trains']:
        for n,v in enumerate(t['visits']):
            ident=f"{t['id']}:{n}"
            for phase in ('arrival','departure'):
                times[ident,phase]=interval(data,t,n,phase)
                if ident+':'+phase in mask.get('bounds',{}):
                    lo,hi=mask['bounds'][ident+':'+phase]
                    times[ident,phase]=[max(times[ident,phase][0],lo),min(times[ident,phase][1],hi)]
                if times[ident,phase][0]>times[ident,phase][1]:raise EmptyDomain('event:'+ident+':'+phase)
            modes=[0] if n in (0,len(t['visits'])-1) or v.get('stop')=='forbidden' else [1] if v.get('stop')=='required' else [0,1]
            for fixed in (v.get('fixed_stop'),mask.get('stops',{}).get(ident)):
                if fixed is not None:modes=[x for x in modes if x==fixed]
            if not modes:raise EmptyDomain('stop:'+ident)
            lo=v.get('minimum_dwell',1) if modes==[1] else 0
            hi=0 if modes==[0] else data['horizon']
            if ident in mask.get('dwell_bounds',{}):
                a,b=mask['dwell_bounds'][ident];lo=max(lo,a);hi=min(hi,b)
            if 'duration_bounds' in v:
                a,b=v['duration_bounds'];lo=max(lo,a);hi=min(hi,b)
            if hi<v.get('minimum_dwell',1):modes=[x for x in modes if x==0]
            if lo>0:modes=[x for x in modes if x==1]
            if not modes or lo>hi:raise EmptyDomain('dwell:'+ident)
            if len(modes)==1:v['fixed_stop']=modes[0]
            if modes==[0]:lo=hi=0
            elif modes==[1]:lo=max(lo,v.get('minimum_dwell',1))
            dwells[ident]=[lo,hi]
            if ident in mask.get('tracks',{}):
                fixed=mask['tracks'][ident]
                if 'fixed_track' in v and fixed!=v['fixed_track']:raise EmptyDomain('track history:'+ident)
                if fixed not in v.get('tracks',data['stations'].get(v['station'],{}).get('tracks',[])):raise EmptyDomain('track:'+ident)
                v['fixed_track']=fixed
        for n,r in enumerate(t['runs']):
            ident=f"{t['id']}:{n}"
            if ident in mask.get('running_bounds',{}):
                a,b=mask['running_bounds'][ident];r['minimum']=max(r['minimum'],a);r['maximum']=min(r['maximum'],b)
            if r['minimum']>r['maximum']:raise EmptyDomain('run:'+ident)
            groups[r['resource']].append((t,n,r))
    specified={}
    for o in mask.get('orders',[]):
        if o['kind']=='run':specified[o['resource'],o['first'],o['second']]=True
    def update(k,lo=None,hi=None):
        cur=times[k];old=cur[:]
        if lo is not None:cur[0]=max(cur[0],lo)
        if hi is not None:cur[1]=min(cur[1],hi)
        if cur[0]>cur[1]:raise EmptyDomain(f'event:{k}:{cur}')
        return old!=cur
    def difference(x,y,lo,hi):
        # x-y in [lo,hi].
        changed=update(x,lo=times[y][0]+lo,hi=times[y][1]+hi)
        changed|=update(y,lo=times[x][0]-hi,hi=times[x][1]-lo)
        return changed
    pairs=[(resource,x,y) for resource,rs in groups.items() for x,y in itertools.combinations(rs,2) if x[0]['id']!=y[0]['id']]
    def event(t,n,p):return (f"{t['id']}:{n}",p)
    def orientation(resource,x,y):
        tx,nx,rx=x;ty,ny,ry=y;rd=data['running_resources'][resource]
        if rx['direction']==ry['direction']:
            return times[event(tx,nx,'departure')][0]+rd['entry_headway']<=times[event(ty,ny,'departure')][1] and times[event(tx,nx+1,'arrival')][0]+rd['exit_headway']<=times[event(ty,ny+1,'arrival')][1]
        return times[event(tx,nx+1,'arrival')][0]+rd['opposite_clearance']<=times[event(ty,ny,'departure')][1]
    for iteration in range(2000):
        changed=False
        for t in data['trains']:
            for n,v in enumerate(t['visits']):
                ident=f"{t['id']}:{n}"
                changed|=difference((ident,'departure'),(ident,'arrival'),*dwells[ident])
            for n,r in enumerate(t['runs']):
                a=event(t,n+1,'arrival');d=event(t,n,'departure')
                # Supplements are zero in the approved network. For generic
                # validation examples retain a conservative range here.
                supplement=r.get('stop_depart_add',0)+r.get('stop_arrive_add',0)
                changed|=difference(a,d,r['minimum'],r['maximum']+supplement)
                for c in data.get('closures',[]):
                    if c['resource']!=r['resource']:continue
                    before=times[a][0]<=c['start'] and times[d][0]+r['minimum']<=c['start']
                    after=times[d][1]>=c['end'] and times[a][1]>=c['end']+r['minimum']
                    if not before and not after:raise EmptyDomain('closure:'+f"{t['id']}:{n}")
                    if not before:changed|=update(d,lo=c['end'])
                    if not after:changed|=update(a,hi=c['start'])
        for resource,x,y in pairs:
            xid=f"{x[0]['id']}:{x[1]}";yid=f"{y[0]['id']}:{y[1]}"
            ab=orientation(resource,x,y) and (resource,yid,xid) not in specified
            ba=orientation(resource,y,x) and (resource,xid,yid) not in specified
            if not ab and not ba:raise EmptyDomain('order:'+xid+':'+yid)
            if ab and ba:continue
            if ba:x,y=y,x
            tx,nx,rx=x;ty,ny,ry=y;rd=data['running_resources'][resource]
            edges=[(event(tx,nx,'departure'),event(ty,ny,'departure'),rd['entry_headway']),
                   (event(tx,nx+1,'arrival'),event(ty,ny+1,'arrival'),rd['exit_headway'])] if rx['direction']==ry['direction'] else [(event(tx,nx+1,'arrival'),event(ty,ny,'departure'),rd['opposite_clearance'])]
            for first,second,gap in edges:
                changed|=update(second,lo=times[first][0]+gap)
                changed|=update(first,hi=times[second][1]-gap)
        if not changed:break
    else:raise RuntimeError('interval propagation failed to converge')
    for t in data['trains']:
        for n,v in enumerate(t['visits']):
            ident=f"{t['id']}:{n}"
            for phase in ('arrival','departure'):v[phase+'_bounds']=times[ident,phase]
            lo,hi=dwells[ident]
            lo=max(lo,times[ident,'departure'][0]-times[ident,'arrival'][1]);hi=min(hi,times[ident,'departure'][1]-times[ident,'arrival'][0])
            if 0<n<len(t['visits'])-1:
                if lo>0:
                    lo=max(lo,v.get('minimum_dwell',1));v['fixed_stop']=1
                elif hi<v.get('minimum_dwell',1):lo=hi=0;v['fixed_stop']=0
            if lo>hi:raise EmptyDomain('final dwell:'+ident)
            v['duration_bounds']=[lo,hi]
    return data,{'iterations':iteration+1,'target_blind':True}

def physical_variables(data):
    out={'S':{},'K':{},'O':{},'T':{}};groups=defaultdict(list)
    for t in data['trains']:
        tid=t['id']
        for n,v in enumerate(t['visits'][1:-1],1):
            ident=f'{tid}:{n}'
            if v.get('stop')=='optional' and 'fixed_stop' not in v:out['S'][ident]={'train':tid,'n':n}
            choices=v.get('tracks',data['stations'][v['station']]['tracks'])
            if len(choices)>1 and 'fixed_track' not in v:out['K'][ident]={'train':tid,'n':n}
            lo,hi=v['duration_bounds']
            if lo<hi:out['T']['D:'+ident]={'kind':'dwell','train':tid,'n':n,'bounds':[lo,hi]}
        for n,r in enumerate(t['runs']):
            lo=max(r['minimum'],t['visits'][n+1]['arrival_bounds'][0]-t['visits'][n]['departure_bounds'][1])
            hi=min(r['maximum'],t['visits'][n+1]['arrival_bounds'][1]-t['visits'][n]['departure_bounds'][0])
            if lo<hi:out['T']['R:'+f'{tid}:{n}']={'kind':'run','train':tid,'n':n,'bounds':[lo,hi]}
            groups[r['resource']].append((t,n,r))
    for resource,items in groups.items():
        items=sorted(items,key=lambda x:(x[0]['id'],x[1]))
        rd=data['running_resources'][resource]
        for (ta,na,ra),(tb,nb,rb) in itertools.combinations(items,2):
            if ta['id']==tb['id']:continue
            va,wa=ta['visits'][na:na+2];vb,wb=tb['visits'][nb:nb+2]
            ab=va['departure_bounds'][0]+rd['entry_headway']<=vb['departure_bounds'][1] and wa['arrival_bounds'][0]+rd['exit_headway']<=wb['arrival_bounds'][1]
            ba=vb['departure_bounds'][0]+rd['entry_headway']<=va['departure_bounds'][1] and wb['arrival_bounds'][0]+rd['exit_headway']<=wa['arrival_bounds'][1]
            if ab and ba:
                aid=f"{ta['id']}:{na}";bid=f"{tb['id']}:{nb}"
                ident=resource+'|'+aid+'|'+bid
                out['O'][ident]={'resource':resource,'left':aid,'right':bid,'left_train':ta['id'],'right_train':tb['id'],'left_n':na,'right_n':nb}
    return out
