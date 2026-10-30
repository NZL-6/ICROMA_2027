"""Exact three-policy pilot; shared emergency state, independent physical audit."""
from pathlib import Path
import copy, csv, hashlib, itertools, json, math, subprocess, sys, time

HERE = Path(sys.argv[1]).resolve()
LOADED_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ENGINE = Path(__file__).resolve().parent/'vendor'
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / 'cp_deps'))
sys.modules['numexpr'] = None
sys.modules['bottleneck'] = None
from ortools.sat.python import cp_model
import ortools
from preprocess import preprocess, physical_variables
from audit import audit_result
from design import mask_for
from acceptance import GAP_LIMIT, accepted, relative_gap

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def solve_direction(raw, mask, policy, direction, hint=None):
    output=HERE/'solutions'/f'{policy}_{direction}.json'
    if output.exists() and read(output).get('accepted'): return read(output)
    sub=copy.deepcopy(raw);sub['trains']=[t for t in sub['trains'] if t['runs'][0]['resource'].endswith('::'+direction)]
    tids={t['id'] for t in sub['trains']}
    dm=copy.deepcopy(mask)
    for field in ['dwell_bounds','running_bounds','stops','tracks','bounds']:
        if field in dm: dm[field]={k:v for k,v in dm[field].items() if k.split(':')[0] in tids}
    if 'orders' in dm: dm['orders']=[o for o in dm['orders'] if o['first'].split(':')[0] in tids and o['second'].split(':')[0] in tids]
    start=time.perf_counter();data,prop=preprocess(sub,dm)
    m=cp_model.CpModel();events={};assigned={};intervals={};pooled={};resources={}
    for t in data['trains']:
        tid=t['id'];N=len(t['visits'])
        for n,v in enumerate(t['visits']):
            a=m.NewIntVar(*v['arrival_bounds'],f'a[{tid},{n}]');d=m.NewIntVar(*v['departure_bounds'],f'd[{tid},{n}]')
            z=m.NewBoolVar(f'z[{tid},{n}]');events[tid,n]=(a,d,z)
            if n in (0,N-1):m.Add(a==d);m.Add(z==0);continue
            modes=[v['fixed_stop']] if 'fixed_stop' in v else [1] if v.get('stop')=='required' else [0] if v.get('stop')=='forbidden' else [0,1]
            if len(modes)==1:m.Add(z==modes[0])
            wl,wu=v['duration_bounds'];m.Add(d-a>=wl);m.Add(d-a<=wu)
            m.Add(d-a>=v.get('minimum_dwell',1)*z);m.Add(d-a<=max(0,v['departure_bounds'][1]-v['arrival_bounds'][0])*z)
            clearance=data['stations'][v['station']]['clearance']
            size=m.NewIntVar(clearance,data['horizon']+clearance,'size');end=m.NewIntVar(0,data['horizon']+clearance,'clear_end')
            m.Add(size==d-a+clearance);m.Add(end==d+clearance)
            choices=v.get('tracks',data['stations'][v['station']]['tracks'])
            assert sorted(choices)==sorted(data['stations'][v['station']]['direction_tracks'][direction])
            assigned[tid,n]={}
            for track in choices:
                selected=m.NewBoolVar(f'track[{tid},{n},{track}]')
                assigned[tid,n][track]=selected
                if 'fixed_track' in v:m.Add(selected==int(track==v['fixed_track']))
                interval=m.NewOptionalIntervalVar(a,size,end,selected,f'occupation[{tid},{n},{track}]')
                intervals.setdefault((v['station'],track),[]).append(interval)
            m.AddExactlyOne(list(assigned[tid,n].values()))
            pooled.setdefault((v['station'],tuple(choices)),[]).append(m.NewIntervalVar(a,size,end,'pooled'))
        for n,r in enumerate(t['runs']):
            a,d,z=events[tid,n];an,dn,zn=events[tid,n+1]
            addition=r.get('stop_depart_add',0)*z+r.get('stop_arrive_add',0)*zn
            m.Add(an-d>=r['minimum']+addition);m.Add(an-d<=r['maximum']+addition)
            for c in data['closures']:
                if c['resource']==r['resource']:
                    before=m.NewBoolVar('before');m.Add(an<=c['start']).OnlyEnforceIf(before);m.Add(d>=c['end']).OnlyEnforceIf(before.Not())
            resources.setdefault(r['resource'],[]).append((t,n,r))
    for ivs in intervals.values():m.AddNoOverlap(ivs)
    for (station,pool),ivs in pooled.items():m.AddCumulative(ivs,[1]*len(ivs),len(pool))
    fixed={(o['resource'],o['first'],o['second']) for o in dm.get('orders',[])}
    for resource,items in resources.items():
        rd=data['running_resources'][resource]
        # Redundant global headway propagation. Pair orders below still enforce
        # the same order at entry and exit, hence no within-section overtaking.
        for phase in (0,1):
            es=[events[t['id'],n][1] if phase==0 else events[t['id'],n+1][0] for t,n,r in items]
            h=rd['entry_headway'] if phase==0 else rd['exit_headway']
            m.AddNoOverlap([m.NewIntervalVar(e,h,e+h,'global_headway') for e in es])
        for (ti,ni,ri),(tj,nj,rj) in itertools.combinations(items,2):
            if ti['id']==tj['id']:continue
            i,j=ti['id'],tj['id'];q=m.NewBoolVar('order')
            if (resource,f'{i}:{ni}',f'{j}:{nj}') in fixed:m.Add(q==1)
            if (resource,f'{j}:{nj}',f'{i}:{ni}') in fixed:m.Add(q==0)
            for a,na,b,nb,cond in [(i,ni,j,nj,q),(j,nj,i,ni,q.Not())]:
                m.Add(events[b,nb][1]>=events[a,na][1]+rd['entry_headway']).OnlyEnforceIf(cond)
                m.Add(events[b,nb+1][0]>=events[a,na+1][0]+rd['exit_headway']).OnlyEnforceIf(cond)
    delays=[]
    for t in data['trains']:
        delay=m.NewIntVar(0,data['horizon'],'terminal_delay')
        m.AddMaxEquality(delay,[0,events[t['id'],len(t['visits'])-1][0]-t['visits'][-1]['plan_arrival']]);delays.append(delay)
    primary=sum(delays);m.Minimize(primary)
    certified_bound_source=None
    if direction in BLIND_BOUNDS:
        info=BLIND_BOUNDS[direction]
        m.Add(primary>=info['lower_bound']);certified_bound_source=info['proof_path']
    def extract(get):
        sch={t['id']:[dict(station=v['station'],arrival=get(events[t['id'],n][0]),departure=get(events[t['id'],n][1]),
                   stop=get(events[t['id'],n][2]),track=(next(k for k,x in assigned[t['id'],n].items() if get(x)) if (t['id'],n) in assigned else None))
                   for n,v in enumerate(t['visits'])] for t in data['trains']}
        return sch
    resume=HERE/'checkpoints'/f'{policy}_{direction}.json'
    if resume.exists() and not read(HERE/'job.json').get('refinement'): hint=read(resume).get('schedule',hint)
    oldbound=HERE/'checkpoints'/f'{policy}_{direction}_unfinished.json'
    if oldbound.exists(): m.Add(primary>=math.ceil(read(oldbound)['lower_bound']-1e-6))
    hint_upper=None
    if hint:
        hs={t['id']:hint[t['id']] for t in sub['trains']}
        hint_upper=sum(max(0,hs[t['id']][-1]['arrival']-t['visits'][-1]['plan_arrival']) for t in sub['trains'])
        assert audit_result(sub,dict(schedule=hs,objective=hint_upper),dm)['passed']
        # Valid optimization cutoff from an audited reference/own incumbent,
        # never from the known test target optimum. Add before presolve.
        m.Add(primary<=hint_upper)
    if hint:
        for (tid,n),(a,d,z) in events.items():
            e=hint[tid][n];m.AddHint(a,e['arrival']);m.AddHint(d,e['departure']);m.AddHint(z,e['stop'])
        for (tid,n),xx in assigned.items():
            for k,x in xx.items():m.AddHint(x,int(hint[tid][n]['track']==k))
    best={};history=[]
    class Callback(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            sch=extract(self.Value);obj=int(round(self.ObjectiveValue()));check=audit_result(sub,dict(schedule=sch,objective=obj),dm)
            if not check['passed']:raise RuntimeError(check)
            event=dict(objective=obj,lower_bound=self.BestObjectiveBound(),seconds=time.perf_counter()-start)
            history.append(event);best.update(schedule=sch,objective=obj)
            save(HERE/'incumbents'/f'{policy}_{direction}_{len(history):04d}.json',dict(**event,schedule=sch))
            save(HERE/'checkpoints'/f'{policy}_{direction}.json',dict(**event,schedule=sch,audit=check))
            print(json.dumps(dict(policy=policy,direction=direction,**event)),flush=True)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=12;solver.parameters.random_seed=20261026
    solver.parameters.num_full_subsolvers=4;solver.parameters.subsolvers.extend(['default_lp','max_lp','quick_restart','objective_lb_search']);solver.parameters.linearization_level=2;solver.parameters.max_time_in_seconds=float('inf');solver.parameters.relative_gap_limit=math.nextafter(GAP_LIMIT,0.);solver.parameters.log_search_progress=True;solver.parameters.log_to_stdout=False
    # No time limit: all six original-domain directional optima must be certified.
    log=HERE/'logs'/f'{policy}_{direction}.log';log.parent.mkdir(exist_ok=True)
    if log.exists():
        import shutil
        archived=log.parent/'previous_slices';archived.mkdir(exist_ok=True)
        shutil.copy2(log,archived/(log.stem+'_'+str(len(list(archived.glob(log.stem+'_*')))+1)+'.log'))
    with log.open('w',encoding='utf-8') as stream:
        solver.log_callback=lambda text:(stream.write(text+'\n'),stream.flush())
        status=solver.Solve(m,Callback())
    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        if status==cp_model.INFEASIBLE and hint is None:
            save(HERE/'proven_infeasible.json',dict(policy=policy,direction=direction,status='INFEASIBLE',response=solver.ResponseStats(),solver_wall_seconds=solver.WallTime()))
        raise RuntimeError(('ENCODING_CONTRADICTS_AUDITED_REFERENCE' if hint else 'REFERENCE_INFEASIBLE',solver.StatusName(status),solver.ResponseStats()))
    if status!=cp_model.OPTIMAL:
        save(HERE/'checkpoints'/f'{policy}_{direction}_unfinished.json',dict(status=solver.StatusName(status),lower_bound=solver.BestObjectiveBound(),seconds=time.perf_counter()-start))
        raise RuntimeError((policy,direction,solver.StatusName(status),'RESUME_REQUIRED'))
    sch=extract(solver.Value);obj=int(round(solver.ObjectiveValue()));lb=solver.BestObjectiveBound()
    check=audit_result(sub,dict(schedule=sch,objective=obj),dm)
    assert check['passed'] and accepted(obj,lb)
    result=dict(policy=policy,direction=direction,status=solver.StatusName(status),objective=obj,lower_bound=lb,
           accepted=True,relative_gap=relative_gap(obj,lb),proven_exact=(math.ceil(lb-1e-6)==obj),schedule=sch,audit=check,mask=dm,history=history,seconds=time.perf_counter()-start,
           solver_wall_seconds=solver.WallTime(),process_cpu_seconds_at_completion=time.process_time(),solver='CP-SAT',audited_hint_upper_bound=hint_upper,version=ortools.__version__,workers=12,branches=solver.NumBranches(),conflicts=solver.NumConflicts(),
           response=solver.ResponseStats(),valid_relaxation_lower_bound_source=certified_bound_source,
           source_sha256=LOADED_SOURCE_SHA256,
           parameters=dict(workers=solver.parameters.num_search_workers,seed=solver.parameters.random_seed,
                linearization_level=solver.parameters.linearization_level,num_full_subsolvers=solver.parameters.num_full_subsolvers,
                subsolvers=list(solver.parameters.subsolvers),proof_slice_seconds=None,relative_gap_limit=GAP_LIMIT))
    save(output,result)
    print(json.dumps(dict(done=True,policy=policy,direction=direction,objective=obj,seconds=result['seconds'])),flush=True)
    return result



def main():
    global BLIND_BOUNDS
    started=time.perf_counter();job=read(HERE/'job.json')
    raw=read(HERE/'raw_input.json')['data'];mask=read(HERE/'mask.json');ref=read(HERE/'reference.json')
    assert audit_result(raw,ref,mask)['passed']
    assert hashlib.sha256((HERE/'reference.json').read_bytes()).hexdigest()==job['reference_sha256']
    BLIND_BOUNDS={}
    if job['stage']=='B':
        from fair_bound import ensure
        for direction in ['F','R']:BLIND_BOUNDS[direction]=ensure(Path(job.get('bound_jobdir',str(HERE))),direction)
    hint=ref['schedule']
    if job.get('refinement'):
        assert job['variant']=='FULL'
        hint=read(HERE/'refinement_hint.json')['schedule']
        assert audit_result(raw,dict(schedule=hint,objective=job['hint_objective']),mask)['passed']
    pieces=[]
    for direction in ['F','R']:
        pieces.append(solve_direction(raw,mask,job['policy'],direction,hint))
    sch={t:v for p in pieces for t,v in p['schedule'].items()};obj=sum(p['objective'] for p in pieces);lb=sum(p['lower_bound'] for p in pieces)
    result=dict(case=job['case'],stage=job['stage'],variant=job['variant'],objective=obj,lower_bound=lb,
        accepted=accepted(obj,lb),relative_gap=relative_gap(obj,lb),proven_exact=(math.ceil(lb-1e-6)==obj),schedule=sch,audit=audit_result(raw,dict(schedule=sch,objective=obj),mask),
        directional=[{k:p[k] for k in ['direction','objective','lower_bound','seconds','parameters']} for p in pieces],
        solver_source_sha256=LOADED_SOURCE_SHA256,reference_sha256=job['reference_sha256'],mask_sha256=job['mask_sha256'],
        bound_proofs=list(BLIND_BOUNDS.values()),cross_variant_data_read=bool(job.get('refinement')),refinement=bool(job.get('refinement')),target_optimum_read=False,worker_seconds=time.perf_counter()-started)
    assert result['audit']['passed'] and result['accepted']
    save(HERE/'result.json',result);print('ACCEPTED_FAIR_RESULT',job['case'],job['stage'],job['variant'],obj,flush=True)
if __name__=='__main__':main()
