"""Independent exported common-section timetable audit; no solver imports."""
import copy,itertools
from common import ENGINE
from preprocess import preprocess

def audit_relaxation(raw,proof,direction):
    sub=copy.deepcopy(raw);sub['trains']=[t for t in sub['trains'] if t['runs'][0]['resource'].endswith('::'+direction)]
    data,_=preprocess(sub,{});schedule=proof['common_schedule'];errors=[];groups={};pools={};cost=0
    index={t['id']:t for t in data['trains']}
    for info in proof['mapping']:
        tid=info['train'];t=index[tid]
        for n in info['indices']:
            v=t['visits'][n];e=schedule[tid][str(n)];a,d=e['arrival'],e['departure']
            if not(v['arrival_bounds'][0]<=a<=v['arrival_bounds'][1] and v['departure_bounds'][0]<=d<=v['departure_bounds'][1] and v['duration_bounds'][0]<=d-a<=v['duration_bounds'][1]):errors.append(('domain',tid,n))
            pools.setdefault(v['station'],[]).append((a,d+data['stations'][v['station']]['clearance']))
        for n in info['indices'][:-1]:
            r=t['runs'][n];d=schedule[tid][str(n)]['departure'];a=schedule[tid][str(n+1)]['arrival']
            if not r['minimum']<=a-d<=r['maximum']:errors.append(('runtime',tid,n))
            for c in data['closures']:
                if c['resource']==r['resource'] and d<c['end'] and a>c['start']:errors.append(('closure',tid,n))
            groups.setdefault(r['resource'],[]).append((d,a))
        cost+=max(info['base_delay'],schedule[tid][str(info['indices'][-1])]['departure']+info['min_tail_after_last_departure']-info['plan_terminal'])
    for resource,items in groups.items():
        rd=data['running_resources'][resource]
        for (da,aa),(db,ab) in itertools.combinations(items,2):
            if not(db>=da+rd['entry_headway'] and ab>=aa+rd['exit_headway'] or da>=db+rd['entry_headway'] and aa>=ab+rd['exit_headway']):errors.append(('headway',resource))
    for station,items in pools.items():
        cap=len(data['stations'][station]['direction_tracks'][direction]);count=0
        for tick,change in sorted([(a,1) for a,d in items]+[(d,-1) for a,d in items]):
            count+=change
            if count>cap:errors.append(('capacity',station,tick))
    for group in proof['dominance_groups']:
        if group['verified']:
            for a,b in zip(group['trains'],group['trains'][1:]):
                for va,vb in zip(schedule[a].values(),schedule[b].values()):
                    if va['arrival']>vb['arrival'] or va['departure']>vb['departure']:errors.append(('dominance',a,b))
    if abs(cost-proof['relaxed_primal'])>1e-6:errors.append(('objective',cost,proof['relaxed_primal']))
    return dict(passed=not errors,errors=errors[:10],recomputed_objective=cost)
