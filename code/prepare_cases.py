"""Approved 75+36 cells; a local, predetermined emergency-stopping rule."""
import copy, itertools, collections, datetime, random
from common import *
bootstrap_engine()
from preprocess import preprocess,physical_variables,EmptyDomain

LOCATIONS=[('H1--C',[13,31,53,71,111],[3,93],[31,71]),
           ('U13--H1',[9,23,49,63,103],[15,89],[23,63]),
           ('U11--U12',[7,35,47,73,107],[15,95],[35,73])]
DURATIONS=[10,30,60,90,120]
def clock(m):return f'{8+m//60:02d}:{m%60:02d}'
def design():
    rng=random.Random(SEED);duration_permutation=list(range(5));rng.shuffle(duration_permutation)
    records=[]
    for li,(section,starts,newstarts,knownstarts) in enumerate(LOCATIONS):
        for si,start in enumerate(starts):
            for di,duration in enumerate(DURATIONS):
                records.append(dict(case=f'D{len(records)+1:03d}',cohort='base',group='sampled_validation',section=section,
                    location_index=li,onset_index=si,start_minutes=start,end_minutes=start+duration,duration_minutes=duration,
                    outer_fold=(li+si+duration_permutation[di])%5,start_clock=clock(start),end_clock=clock(start+duration)))
    for group,which,ds in [('new_onset',2,[30,120]),('new_duration',3,[45,75]),('joint',2,[45,75])]:
        for li,location in enumerate(LOCATIONS):
            for start in location[which]:
                for duration in ds:
                    records.append(dict(case=f'T{len(records)-74:03d}',cohort='transfer',group=group,section=location[0],
                        location_index=li,start_minutes=start,end_minutes=start+duration,duration_minutes=duration,
                        outer_fold=None,start_clock=clock(start),end_clock=clock(start+duration)))
    assert len(records)==111 and len({(c['section'],c['start_minutes'],c['duration_minutes']) for c in records})==111
    return records
def order_mask(raw,sch):
    groups=collections.defaultdict(list);orders=[]
    for t in raw['trains']:
        for n,r in enumerate(t['runs']):groups[r['resource']].append((t['id'],n))
    for resource,items in groups.items():
        for (i,n),(j,k) in itertools.combinations(items,2):
            assert sch[i][n]['departure']!=sch[j][k]['departure']
            a,b=(f'{i}:{n}',f'{j}:{k}') if sch[i][n]['departure']<sch[j][k]['departure'] else (f'{j}:{k}',f'{i}:{n}')
            orders.append(dict(kind='run',resource=resource,first=a,second=b))
    return dict(orders=sorted(orders,key=lambda o:(o['resource'],o['first'],o['second'])))
def build_input(sc):
    raw=read(ROOT/'baseline/baseline_input.json');plan=read(ROOT/'baseline/baseline_schedule.json')['schedule']
    B,E=sc['start_minutes'],sc['end_minutes'];H=raw['horizon'];section=sc['section']
    raw.update(now=B,purpose='local_disruption_generalization',closures=[dict(resource=section+'::'+d,start=B,end=E) for d in ['F','R']])
    held=[];affected=[];committed=[]
    for t in raw['trains']:
        tid=t['id'];p=plan[tid];N=len(p);not_started=p[0]['departure']>=B;done=p[-1]['arrival']<=B
        crosses=[n for n,r in enumerate(t['runs']) if raw['running_resources'][r['resource']]['physical_section']==section]
        assert not any(p[n]['departure']<=B<p[n+1]['arrival'] for n in crosses),'closure occupied at onset'
        direct=any(p[n]['departure']<E and p[n+1]['arrival']>B for n in crosses)
        if direct:affected.append(tid)
        for n,(v,e) in enumerate(zip(t['visits'],p)):
            for k in list(v):
                if k.startswith('fixed_') or k=='duration_bounds':del v[k]
            for phase in ['arrival','departure']:
                observed=done or (not not_started and (e[phase]<B or (phase=='arrival' and e[phase]==B)))
                v[phase+'_bounds']=[e[phase],e[phase]] if observed else [max(B,e[phase]),H]
                if observed:v['fixed_'+phase]=e[phase]
            if e['departure']<B or done:v['fixed_stop']=e['stop']
            if 0<n<N-1 and e['arrival']<=B and not not_started:v['fixed_track']=e['track']
        for n,r in enumerate(t['runs']):
            planned=p[n+1]['arrival']-p[n]['departure'];r['original_duration']=planned
            if p[n]['departure']<B:
                r['minimum']=r['maximum']=planned;r['duration_state']='committed'
                v=t['visits'][n+1];v['fixed_arrival']=p[n+1]['arrival'];v['arrival_bounds']=[p[n+1]['arrival']]*2
                if p[n+1]['arrival']>B:committed.append(dict(train=tid,n=n,arrival=p[n+1]['arrival']))
            else:r['maximum']=H-max(B,p[n]['departure']);r['duration_state']='future_adjustable'
        if direct:
            assert not done
            n=0 if not_started else next(n for n,e in enumerate(p) if e['departure']>=B)
            assert n<N-1,('affected train already finishing',tid)
            v=t['visits'][n];e=p[n]
            kind='origin_hold' if n==0 else 'hold_current_station' if e['arrival']<=B else 'finish_committed_run_then_hold'
            if n==0:v['arrival_bounds'][0]=max(v['arrival_bounds'][0],E)
            else:
                v['fixed_arrival']=e['arrival'];v['arrival_bounds']=[e['arrival']]*2;v['fixed_stop']=1
            v['departure_bounds'][0]=max(v['departure_bounds'][0],E)
            held.append(dict(train=tid,n=n,station=v['station'],kind=kind,arrival=e['arrival'] if n else None,
                release=E,track=v.get('fixed_track'),track_assignment='observed' if 'fixed_track' in v else 'variable'))
    pools=collections.Counter((e['station'],next(t for t in raw['trains'] if t['id']==e['train'])['runs'][0]['resource'].split('::')[-1]) for e in held if e['n']>0)
    for (station,direction),count in pools.items():
        if count>len(raw['stations'][station]['direction_tracks'][direction]):
            raise ValueError(('simultaneous_prescribed_holds_exceed_capacity',station,direction,count))
    prepared,_=preprocess(raw);units=physical_variables(prepared);p2=order_mask(raw,plan);p1=copy.deepcopy(p2)
    for t in raw['trains']:
        for n,v in enumerate(t['visits'][1:-1],1):
            lo=v['plan_departure']-v['plan_arrival']
            p1.setdefault('dwell_bounds',{})[f'{t["id"]}:{n}']=[lo,H]
        for n,r in enumerate(t['runs']):p1.setdefault('running_bounds',{})[f'{t["id"]}:{n}']=[r['original_duration']]*2
    preprocess(raw,p1)
    return raw,units,dict(P1=p1,P2=p2,P3={}),dict(records=held,directly_affected=affected,committed_runs=committed,onset=B,release=E,
        selection='unfinished planned closure traversal overlaps [onset,reopening)',first_station_optimized=False,
        future_holding_track='selected by exact physical model; already occupied tracks immutable')

def make_folds(records):
    base=[r for r in records if r['cohort']=='base' and r.get('valid',True)]
    folds=[]
    for f in range(5):
        train=[r['case'] for r in base if r['outer_fold']!=f];test=[r['case'] for r in base if r['outer_fold']==f];inner=[]
        allocations={}
        for li in range(3):
            for si in range(5):
                block=sorted([r for r in base if r['location_index']==li and r['onset_index']==si and r['outer_fold']!=f],key=lambda r:r['duration_minutes'])
                for j,r in enumerate(block):allocations[r['case']]=(j+li+si)%4
        for k in range(4):
            val=[c for c in train if allocations[c]==k]
            inner.append(dict(validation_start=k,train=[c for c in train if c not in val],validation=val))
        folds.append(dict(test_start=f,train=train,test=test,inner=inner))
    # Final all-base fit: parameter selection/calibration through five base-only folds.
    folds.append(dict(test_start=99,train=[r['case'] for r in base],test=[],inner=[dict(validation_start=f['test_start'],train=f['train'],validation=f['test']) for f in folds]))
    return folds
def main():
    records=[]
    for sc in design():
        folder=ROOT/'cases'/sc['case'];folder.mkdir(exist_ok=True)
        if (folder/'input_validation.json').exists():records.append(read(folder/'input_validation.json'));continue
        save(folder/'scenario.json',sc)
        try:
            raw,units,masks,em=build_input(sc)
            save(folder/'common_input.json',dict(data=raw));save(folder/'physical_variables.json',units);save(folder/'emergency_state.json',em)
            save(folder/'original_schedule.json',read(ROOT/'baseline/baseline_schedule.json'))
            for p,m in masks.items():save(folder/'masks'/f'{p}.json',m)
            record=dict(**sc,valid=True,input_sha256=sha(folder/'common_input.json'),affected_trains=len(em['directly_affected']))
        except (ValueError,EmptyDomain) as exc:
            record=dict(**sc,valid=False,input_failure=repr(exc),quality=None,time=None)
        save(folder/'input_validation.json',record);records.append(record)
    save(ROOT/'case_manifest.json',records);save(ROOT/'folds.json',make_folds(records))
    save(ROOT/'input_validation.json',dict(passed=True,registered=len(records),valid=sum(r['valid'] for r in records),failures=[r for r in records if not r['valid']],no_substitution=True))
    print('INPUTS_REGISTERED',len(records),'VALID',sum(r['valid'] for r in records),flush=True)
if __name__=='__main__':main()
