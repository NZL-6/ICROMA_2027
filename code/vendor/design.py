"""Frozen physical masks; Oracle dependence is explicit, never in the solver."""
import copy,math,random,time
from common import digest

def times_value(unit,schedule):
    es=schedule[unit['train']];n=unit['n']
    return es[n]['departure']-es[n]['arrival'] if unit['kind']=='dwell' else es[n+1]['arrival']-es[n]['departure']

def mask_for(units,schedule,groups,opened=None,width=None):
    opened=opened or {};mask={}
    for group in groups:
        for key,u in units[group].items():
            if key in opened.get(group,set()):continue
            if group=='S':mask.setdefault('stops',{})[key]=schedule[u['train']][u['n']]['stop']
            elif group=='K':mask.setdefault('tracks',{})[key]=schedule[u['train']][u['n']]['track']
            elif group=='O':
                left,right=u['left'],u['right']
                ld=schedule[u['left_train']][u['left_n']]['departure'];rd=schedule[u['right_train']][u['right_n']]['departure']
                first,second=(left,right) if ld<rd else (right,left)
                mask.setdefault('orders',[]).append(dict(kind='run',resource=u['resource'],first=first,second=second))
            else:
                value=times_value(u,schedule);lo,hi=u['bounds']
                interval=[value,value] if width is None else [max(lo,math.ceil(value-width*(hi-lo)/2)),min(hi,math.floor(value+width*(hi-lo)/2))]
                field='dwell_bounds' if u['kind']=='dwell' else 'running_bounds'
                mask.setdefault(field,{})[f"{u['train']}:{u['n']}"]=interval
    if 'orders' in mask:mask['orders'].sort(key=lambda x:(x['resource'],x['first'],x['second']))
    return mask

def suffixes(units,original,optimal):
    first={}
    for key,u in units['O'].items():
        i,j=u['left_train'],u['right_train'];n,m=u['left_n'],u['right_n']
        old=original[i][n]['departure']<original[j][m]['departure']
        new=optimal[i][n]['departure']<optimal[j][m]['departure']
        if old!=new:
            first[i]=min(first.get(i,n),n);first[j]=min(first.get(j,m),m)
    covered={i:set() for i in first}
    for key,u in units['O'].items():
        for tid,n in ((u['left_train'],u['left_n']),(u['right_train'],u['right_n'])):
            if tid in first and n>=first[tid]:covered[tid].add(key)
    return first,covered

def sec52_rows(case,raw,units,original,optimal):
    first,cover=suffixes(units,original,optimal)
    entry={}
    for t in raw['trains']:
        for n,r in enumerate(t['runs']):
            if r['resource'].startswith('H1--C::'):entry[t['id']]=original[t['id']][n]['departure']
    o_rank=sorted(cover,key=lambda x:(entry[x],x))
    t_rank=sorted(units['T'],key=lambda x:(-abs(times_value(units['T'][x],optimal)-times_value(units['T'][x],original)),units['T'][x]['train'],units['T'][x]['n'],units['T'][x]['kind']))
    diagnostic=dict(first_changed_station_index=first,
        first_changed_station={i:original[i][n]['station'] for i,n in first.items()},
        maximum_order_suffix_coverage=len(set().union(*cover.values())) if cover else 0,
        total_orders=len(units['O']),total_times=len(units['T']),order_ranking=o_rank,
        time_ranking=[dict(id=x,change=abs(times_value(units['T'][x],optimal)-times_value(units['T'][x],original))) for x in t_rank])
    rows=[]
    for groups,label in ((['O'],'O'),(['T'],'T'),(['O','T'],'OT')):
        for strategy in ('random','priority'):
            for draw in range(5 if strategy=='random' else 1):
                orank=o_rank[:] if strategy=='priority' else sorted(cover)
                trank=t_rank[:] if strategy=='priority' else sorted(units['T'])
                if strategy=='random':
                    seed=int(digest([20261024,case,label,draw])[:15],16);rng=random.Random(seed);rng.shuffle(orank);rng.shuffle(trank)
                for p in (.2,.4,.6,.8):
                    start=time.perf_counter();opened={};selected=[];reachable=True
                    if 'O' in groups:
                        target=math.ceil(p*len(units['O']));got=set()
                        for tid in orank:
                            if len(got)>=target:break
                            got.update(cover[tid]);selected.append(tid)
                        reachable=len(got)>=target;opened['O']=got
                    if 'T' in groups:opened['T']=set(trank[:math.ceil(p*len(units['T']))])
                    mask=mask_for(units,original,groups,opened)
                    generation=time.perf_counter()-start
                    actual={g:len(opened[g]) for g in groups}
                    denominator=sum(len(units[g]) for g in groups)
                    rows.append(dict(section='5.2',case=case,setting=label,strategy=strategy,draw=draw,proportion=p,
                        target_counts={g:math.ceil(p*len(units[g])) for g in groups},actual_open_counts=actual,
                        actual_proportion=sum(actual.values())/denominator if denominator else 0,
                        requested_proportion_reachable=reachable,selected_order_suffixes=selected,
                        oracle_information=('O' in groups or strategy=='priority'),mask=mask,generation_seconds=generation))
    return rows,diagnostic
