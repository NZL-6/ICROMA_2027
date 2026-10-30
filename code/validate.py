"""Independent state, allocation and split checks; three real pilot cases retained."""
import copy,collections,datetime,sys
from common import *
bootstrap_engine()
from jobs import base_results,costs
from audit import audit_result

def state_checks():
    manifest=read(ROOT/'case_manifest.json');base=[s for s in manifest if s['cohort']=='base'];transfer=[s for s in manifest if s['cohort']=='transfer']
    assert len(base)==75 and len(transfer)==36
    for sc in manifest:
        if not sc['valid']:continue
        f=ROOT/'cases'/sc['case'];raw=read(f/'common_input.json')['data'];em=read(f/'emergency_state.json')
        expected=set()
        for t in raw['trains']:
            for n,r in enumerate(t['runs']):
                if raw['running_resources'][r['resource']]['physical_section']==sc['section']:
                    entry=t['visits'][n]['plan_departure'];exit=t['visits'][n+1]['plan_arrival']
                    assert not entry<=sc['start_minutes']<exit
                    if entry<sc['end_minutes'] and exit>sc['start_minutes']:expected.add(t['id'])
        assert expected==set(em['directly_affected'])=={e['train'] for e in em['records']}
        if sc['cohort']=='base':assert expected
        if sc['section']!='H1--C':assert len(expected)<len(raw['trains']),('not local',sc['case'])
    folds=read(ROOT/'folds.json');seen=[]
    for f in folds[:5]:
        assert not set(f['train'])&set(f['test']);seen+=f['test']
        for inner in f['inner']:
            assert not set(inner['train'])&set(inner['validation'])
            assert set(inner['train'])|set(inner['validation'])==set(f['train'])
    assert sorted(seen)==sorted(s['case'] for s in base if s['valid']) and len(seen)==len(set(seen))
    if all(s['valid'] for s in base):
        assert all(len(f['train'])==60 and len(f['test'])==15 for f in folds[:5])
        assert all(len(i['train'])==45 and len(i['validation'])==15 for f in folds[:5] for i in f['inner'])
    for s in transfer:
        matching=[b for b in base if b['section']==s['section']]
        if s['group'] in ['new_onset','joint']:assert s['start_minutes'] not in {b['start_minutes'] for b in matching}
        if s['group'] in ['new_duration','joint']:assert s['duration_minutes'] not in {b['duration_minutes'] for b in matching}
    save(ROOT/'validation/design_and_split.json',dict(passed=True,registered=111,local_stopping=True,no_scenario_leakage=True))

def native_track_checks():
    sys.modules['numexpr']=None;sys.modules['bottleneck']=None
    sys.path.insert(0,str(ENGINE/'cp_deps'));from ortools.sat.python import cp_model
    # A fixed track occupied until 20; a second movement arrives BEFORE reopening
    # at 10 and must select track 2. Capacity alone cannot honor preassignments.
    m=cp_model.CpModel();x=m.NewBoolVar('track1');y=m.NewBoolVar('track2');m.AddExactlyOne(x,y)
    first=m.NewIntervalVar(0,20,20,'fixed1');second1=m.NewOptionalIntervalVar(10,5,15,x,'choice1');second2=m.NewOptionalIntervalVar(10,5,15,y,'choice2')
    m.AddNoOverlap([first,second1]);m.AddNoOverlap([second2]);sol=cp_model.CpSolver();sol.parameters.num_search_workers=12;sol.parameters.random_seed=SEED
    assert sol.Solve(m)==cp_model.OPTIMAL and sol.Value(y)==1
    m.Add(x==1);assert sol.Solve(m)==cp_model.INFEASIBLE
    # Half-open clearance intervals may touch without an overlap.
    m=cp_model.CpModel();m.AddNoOverlap([m.NewIntervalVar(0,10,10,'a'),m.NewIntervalVar(10,5,15,'b')]);assert sol.Solve(m)==cp_model.OPTIMAL
    save(ROOT/'validation/native_tracks.json',dict(passed=True,pre_reopening_variable_assignment=True,preassigned_collision_rejected=True,half_open_boundary=True))

def pilots():
    manifest={r['case']:r for r in read(ROOT/'case_manifest.json')};records=[]
    for case in ['D001','D026','D051']:
        assert base_results(manifest[case]),case
        raw=read(ROOT/'cases'/case/'common_input.json')['data']
        for policy in ['P1','P2','P3']:
            r=read(ROOT/'cases'/case/f'{policy}_optimal.json');mask=read(ROOT/'cases'/case/'masks'/f'{policy}.json')
            assert audit_result(raw,r,mask)['passed']
            bad=copy.deepcopy(r);t=raw['trains'][0];bad['schedule'][t['id']][0]['departure']=-1
            assert not audit_result(raw,bad,mask)['passed']
            records.append(dict(case=case,policy=policy,objective=r['objective'],gap=r['relative_gap'],passed=True))
        p1=read(ROOT/'cases'/case/'P1_optimal.json');p2=read(ROOT/'cases'/case/'P2_optimal.json');p3=read(ROOT/'cases'/case/'P3_optimal.json')
        assert p3['lower_bound']<=p2['objective'] and p2['lower_bound']<=p1['objective']
    save(ROOT/'validation/pilot_results.json',dict(passed=True,records=records,real_case_results_retained=True))
    save(ROOT/'preflight_complete.json',dict(passed=True,at=datetime.datetime.now().astimezone().isoformat(),pilot_cases=['D001','D026','D051'],initial_target_jobs_completed=6,reference_jobs_completed=3))
if __name__=='__main__':
    state_checks();native_track_checks();pilots();print('PREFLIGHT_COMPLETE',flush=True)
