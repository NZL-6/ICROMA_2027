"""Timing-only wrapper around the byte-identical existing physical solver."""
import json
import os
import sys
import time
import traceback
from common import *

def main():
    assert_ready();verify_frozen();run=Path(sys.argv[1]).resolve();job=read(run/'job.json')
    native=module(ROOT/'solver_worker.py','bc_native_solver')
    measure=dict(schema=1,native_solver_sha256=sha(ROOT/'solver_worker.py'),wrapper_sha256=sha(__file__),
        directions=[],audits=[],writes=[],preprocessing=[],nested_costs_do_not_add_to_solver_time=True)
    active={};original_solve=native.cp_model.CpSolver.Solve;original_preprocess=native.preprocess
    original_audit=native.audit_result;original_save=native.save;original_direction=native.solve_direction
    def timed_solve(solver,*args,**kwargs):
        at=time.perf_counter();cpu=time.process_time();active['in_solve']=True
        try:return original_solve(solver,*args,**kwargs)
        finally:
            active['in_solve']=False
            active['solve_call_wall_seconds']=time.perf_counter()-at
            active['solve_call_cpu_seconds']=time.process_time()-cpu
            active['native_solver_wall_seconds']=solver.WallTime()
            measure.setdefault('solve_calls',[]).append(dict(direction=active.get('direction'),native_wall_seconds=solver.WallTime(),wall_seconds=active['solve_call_wall_seconds'],cpu_seconds=active['solve_call_cpu_seconds']))
            save(run/'worker_measurements.json',measure)
    def timed_preprocess(*args,**kwargs):
        at=time.perf_counter();cpu=time.process_time()
        try:return original_preprocess(*args,**kwargs)
        finally:measure['preprocessing'].append(dict(direction=active.get('direction'),wall_seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu))
    def timed_audit(*args,**kwargs):
        at=time.perf_counter();cpu=time.process_time()
        try:return original_audit(*args,**kwargs)
        finally:measure['audits'].append(dict(direction=active.get('direction'),inside_solver_callback=active.get('in_solve',False),wall_seconds=time.perf_counter()-at,cpu_seconds=time.process_time()-cpu))
    def timed_save(path,value):
        at=time.perf_counter()
        try:return original_save(path,value)
        finally:measure['writes'].append(dict(path=str(Path(path).relative_to(run)),inside_solver_callback=active.get('in_solve',False),wall_seconds=time.perf_counter()-at))
    def timed_direction(raw,mask,policy,direction,hint=None):
        active.clear();active['direction']=direction;at=time.perf_counter();cpu=time.process_time()
        result=original_direction(raw,mask,policy,direction,hint)
        elapsed=time.perf_counter()-at
        measure['directions'].append(dict(direction=direction,total_wall_seconds=elapsed,total_cpu_seconds=time.process_time()-cpu,
            solve_call_wall_seconds=active.get('solve_call_wall_seconds',0.),solve_call_cpu_seconds=active.get('solve_call_cpu_seconds',0.),
            native_solver_wall_seconds=active.get('native_solver_wall_seconds',0.),
            construction_and_direction_audit_wall_seconds=elapsed-active.get('solve_call_wall_seconds',0.),
            cached_without_new_optimizer_call='solve_call_wall_seconds' not in active))
        return result
    native.cp_model.CpSolver.Solve=timed_solve;native.preprocess=timed_preprocess
    native.audit_result=timed_audit;native.save=timed_save;native.solve_direction=timed_direction
    started=time.perf_counter();cpu=time.process_time()
    try:
        if job.get('reference_generation'):
            raw=read(run/'raw_input.json')['data'];mask=read(run/'mask.json');native.BLIND_BOUNDS={}
            parts=[timed_direction(raw,mask,job['policy'],d,None) for d in ('F','R')]
            result=dict(case=job['case'],stage=job['stage'],variant=job['variant'],policy=job['policy'],name=job['policy'],
                schedule={t:s for p in parts for t,s in p['schedule'].items()},objective=sum(p['objective'] for p in parts),
                lower_bound=sum(p['lower_bound'] for p in parts),accepted=True,
                proven_exact=all(p['proven_exact'] for p in parts),
                directional=[{k:p[k] for k in ['direction','objective','lower_bound','seconds','parameters']} for p in parts],
                bound_proofs=[],reference_generation=True,solver_source_sha256=sha(ROOT/'solver_worker.py'))
            result['relative_gap']=native.relative_gap(result['objective'],result['lower_bound'])
            result['audit']=original_audit(raw,result,mask);assert result['audit']['passed']
            original_save(run/'result.json',result)
        else:native.main()
        measure['passed']=True
    except BaseException as exc:
        measure.update(passed=False,error=repr(exc));raise
    finally:
        measure.update(worker_wall_seconds=time.perf_counter()-started,worker_cpu_seconds=time.process_time()-cpu)
        save(run/'worker_measurements.json',measure)

if __name__=='__main__':main()
