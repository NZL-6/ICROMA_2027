"""Offline lower-bound relaxation of H1--C--H2 with finite station pools.

Branch constraints and track labels are relaxed; all retained constraints are
necessary for a full physical schedule. Its certified lower bound is therefore
valid for the original problem, and is never used by formal algorithms.
"""
import argparse,itertools,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parents[1]))
from common import ENGINE
sys.path.append(str(ENGINE));sys.path.insert(0,str(ENGINE/"cp_deps"))
sys.modules['numexpr']=None
sys.modules['bottleneck']=None
from ortools.sat.python import cp_model
import ortools
from common import ROOT,OUT,read,write,sha
OUT=Path(__file__).resolve().parent
from preprocess import preprocess
from audit import audit_result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--direction',required=True);ap.add_argument('--budget',type=float,default=math.inf);ap.add_argument('--hint');args=ap.parse_args()
    ident=args.case+'__common_'+args.direction+'_'+str(args.budget)+'_dominance';output=OUT/'common_bounds/results'/(ident+'.json')
    if output.exists():print('cached',ident,flush=True);return
    raw=read(OUT/'raw_input.json')['data'];raw['trains']=[t for t in raw['trains'] if t['runs'][0]['resource'].endswith('::'+args.direction)]
    data,_=preprocess(raw);m=cp_model.CpModel();vars={};pool={};resources={};delays=[];mapping=[];start=time.perf_counter()
    for t in data['trains']:
        tid=t['id'];indices=[k for k,v in enumerate(t['visits']) if v['station'] in ('H1','C','H2')]
        for n in indices:
            v=t['visits'][n];a=m.NewIntVar(*v['arrival_bounds'],f'a[{tid},{n}]');d=m.NewIntVar(*v['departure_bounds'],f'd[{tid},{n}]');vars[tid,n]=(a,d)
            wl,wu=v['duration_bounds'];m.Add(d-a>=wl);m.Add(d-a<=wu)
            size=m.NewIntVar(1,data['horizon']+1,'size');end=m.NewIntVar(0,data['horizon']+1,'end');m.Add(size==d-a+data['stations'][v['station']]['clearance']);m.Add(end==d+data['stations'][v['station']]['clearance'])
            iv=m.NewIntervalVar(a,size,end,f'platform[{tid},{n}]');pool.setdefault(v['station'],[]).append(iv)
        for n in indices[:-1]:
            r=t['runs'][n];d=vars[tid,n][1];an=vars[tid,n+1][0]
            m.Add(an-d>=r['minimum']);m.Add(an-d<=r['maximum']);resources.setdefault(r['resource'],[]).append((tid,n,r))
            for c in data.get('closures',[]):
                if c['resource']==r['resource']:
                    b=m.NewBoolVar('before');m.Add(an<=c['start']).OnlyEnforceIf(b);m.Add(d>=c['end']).OnlyEnforceIf(b.Not())
        last=indices[-1];tail=sum(t['runs'][k]['minimum'] for k in range(last,len(t['runs'])))+sum(t['visits'][k]['duration_bounds'][0] for k in range(last+1,len(t['visits'])-1))
        base=max(0,t['visits'][-1]['arrival_bounds'][0]-t['visits'][-1]['plan_arrival'])
        delay=m.NewIntVar(base,data['horizon'],'delay');m.AddMaxEquality(delay,[base,vars[tid,last][1]+tail-t['visits'][-1]['plan_arrival']]);delays.append(delay)
        mapping.append(dict(train=tid,indices=indices,min_tail_after_last_departure=tail,base_delay=base,plan_terminal=t['visits'][-1]['plan_arrival']))
    for station,ivs in pool.items():
        cap=len(data['stations'][station]['direction_tracks'][args.direction]);m.AddCumulative(ivs,[1]*len(ivs),cap)
    for resource,items in resources.items():
        rd=data['running_resources'][resource]
        for phase in (0,1):
            events=[vars[i,n][1] if phase==0 else vars[i,n+1][0] for i,n,r in items];h=rd['entry_headway'] if phase==0 else rd['exit_headway']
            ivs=[m.NewIntervalVar(e,h,e+h,'headway') for e in events];m.AddNoOverlap(ivs)
        for (i,n,r),(j,k,s) in itertools.combinations(items,2):
            q=m.NewBoolVar('order')
            for (a,na),(b,nb),cond in (((i,n),(j,k),q),((j,k),(i,n),q.Not())):
                m.Add(vars[b,nb][1]>=vars[a,na][1]+rd['entry_headway']).OnlyEnforceIf(cond)
                m.Add(vars[b,nb+1][0]>=vars[a,na+1][0]+rd['exit_headway']).OnlyEnforceIf(cond)
    # In this RELAXATION ONLY, equal-type trains may be coordinatewise sorted.
    # Arrival/departure multisets (and hence pooled occupancy) are unchanged;
    # min-duration constraints are closed under min/max; maximum durations must
    # be redundant. Event domains must be componentwise ordered, and effective
    # due dates ordered, for the exchange not to increase total tardiness.
    train_index={t['id']:t for t in data['trains']};classes={};dominance=[]
    for info in mapping:
        t=train_index[info['train']];nn=info['indices']
        minimums=tuple(2 if t['visits'][n].get('commercial') else 0 for n in nn)
        runtimes=tuple(t['runs'][n]['minimum'] for n in nn[:-1])
        key=(minimums,runtimes,info['min_tail_after_last_departure'])
        classes.setdefault(key,[]).append(info)
    for (mins,rmins,tail),members in classes.items():
        members.sort(key=lambda z:(z['plan_terminal']-tail+z['base_delay'],z['plan_terminal'],z['train']))
        valid=True
        for info in members:
            t=train_index[info['train']]
            for pos,n in enumerate(info['indices']):
                v=t['visits'][n];wl,wu=v['duration_bounds'];al,au=v['arrival_bounds'];dl,du=v['departure_bounds']
                valid&=wl<=max(mins[pos],dl-au) and wu>=du-al
            for n in info['indices'][:-1]:valid&=t['runs'][n]['maximum']>=t['visits'][n+1]['arrival_bounds'][1]-t['visits'][n]['departure_bounds'][0]
        for a,b in zip(members,members[1:]):
            ta,tb=train_index[a['train']],train_index[b['train']]
            for na,nb in zip(a['indices'],b['indices']):
                for phase in ('arrival','departure'):
                    aa,bb=ta['visits'][na][phase+'_bounds'],tb['visits'][nb][phase+'_bounds']
                    valid&=aa[0]<=bb[0] and aa[1]<=bb[1]
        if valid:
            for a,b in zip(members,members[1:]):
                for na,nb in zip(a['indices'],b['indices']):
                    for phase in (0,1):m.Add(vars[a['train'],na][phase]<=vars[b['train'],nb][phase])
        dominance.append(dict(trains=[x['train'] for x in members],verified=bool(valid),min_dwells=mins,min_runs=rmins,tail=tail))
    m.Minimize(sum(delays))
    hint_info=None
    if args.hint:
        src=read(args.hint);hint=src.get('result',src).get('schedule',src.get('optimal_schedule'));hint={t['id']:hint[t['id']] for t in raw['trains']}
        obj=sum(max(0,hint[t['id']][-1]['arrival']-t['visits'][-1]['plan_arrival']) for t in raw['trains'])
        audit=audit_result(raw,dict(schedule=hint,objective=obj));assert audit['schedule_valid'],audit
        for (i,n),(a,d) in vars.items():m.AddHint(a,hint[i][n]['arrival']);m.AddHint(d,hint[i][n]['departure'])
        m.Add(sum(delays)<=obj);hint_info=dict(source=args.hint,objective=obj,audit=audit)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=max(.01,args.budget-(time.perf_counter()-start));solver.parameters.num_search_workers=12;solver.parameters.relative_gap_limit=math.nextafter(0.001,0.);solver.parameters.random_seed=20261026
    solver.parameters.num_full_subsolvers=4;solver.parameters.subsolvers.extend(['default_lp','max_lp','quick_restart','objective_lb_search']);solver.parameters.linearization_level=2;solver.parameters.log_search_progress=True;solver.parameters.log_to_stdout=False
    log=OUT/'common_bounds/logs'/(ident+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    def plan(get):return {i:{str(n):{'arrival':get(a),'departure':get(d)} for (j,n),(a,d) in vars.items() if j==i} for i in train_index}
    class Candidate(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            write(OUT/'common_bounds/incumbents'/(args.case+'_'+args.direction+'.json'),dict(case=args.case,direction=args.direction,
                relaxed_objective=int(round(self.ObjectiveValue())),lower_bound=self.BestObjectiveBound(),common_schedule=plan(self.Value),seconds=time.perf_counter()-start))
    with log.open('w',encoding='utf-8') as f:
        solver.log_callback=lambda s:(f.write(s+'\n'),f.flush());status=solver.Solve(m,Candidate())
    feasible=status in (cp_model.OPTIMAL,cp_model.FEASIBLE)
    result=dict(case=args.case,direction=args.direction,status=solver.StatusName(status),lower_bound=solver.BestObjectiveBound(),
        solver_wall_seconds=solver.WallTime(),response=solver.ResponseStats(),relaxed_primal=solver.ObjectiveValue() if feasible else None,relaxation_proven=status==cp_model.OPTIMAL,
        source_sha256=sha(__file__),version=ortools.__version__,parameters=dict(workers=solver.parameters.num_search_workers,seed=solver.parameters.random_seed,linearization_level=solver.parameters.linearization_level,num_full_subsolvers=solver.parameters.num_full_subsolvers,subsolvers=list(solver.parameters.subsolvers),time_limit_seconds=None if math.isinf(solver.parameters.max_time_in_seconds) else solver.parameters.max_time_in_seconds,relative_gap_limit=math.nextafter(0.001,0.)),hint_info=hint_info,mapping=mapping,seconds=time.perf_counter()-start,dominance_groups=dominance,
        common_schedule=plan(solver.Value) if feasible else None,
        scope='lower-bound relaxation only; omitted branch conflicts and station-track labels; retains common-section headways, no overtaking and finite station capacity')
    write(output,result);print({k:result[k] for k in ('case','direction','status','lower_bound','relaxed_primal','seconds')},flush=True)

if __name__=='__main__':main()
