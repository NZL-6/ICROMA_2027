"""Preserve actual training/solver overlap without adding overlapping phase clocks."""
import csv,datetime
from common import *

def seconds(s):return datetime.datetime.fromisoformat(s).timestamp()

def audit_schedule():
    if not (ROOT/'transfer_precompute_control.json').exists():return dict(enabled=False,passed=True)
    assert read(ROOT/'transfer_full_complete.json')['complete']
    phases=read(ROOT/'phase_ledger.json')
    training=next(p for p in phases if p['phase']=='training' and p['returncode']==0)
    end=seconds(training['ended_at']);begin=end-training['seconds']
    finish=seconds(read(ROOT/'transfer_full_complete.json')['completed_at'])
    rows=[];known=[];main_after_transfer=[]
    for path in (ROOT/'runs').glob('*/attempts.json'):
        job=read(path.parent/'job.json')
        for attempt in read(path):
            if attempt.get('interrupted'):
                rows.append(dict(case=job['case'],key=job['key'],attempt=attempt['attempt'],timing_uncertain=True,
                    process_seconds=attempt['process_wall_upper_seconds'],training_overlap_seconds=None,
                    process_start=None,process_end=None));continue
            last=seconds(attempt['ended_at']);duration=attempt['process_wall_seconds'];first=last-duration
            overlap=max(0.,min(last,end)-max(first,begin))
            row=dict(case=job['case'],key=job['key'],attempt=attempt['attempt'],timing_uncertain=False,
                process_seconds=duration,training_overlap_seconds=overlap,
                process_start=datetime.datetime.fromtimestamp(first).astimezone().isoformat(),process_end=attempt['ended_at'])
            rows.append(row);known.append((first,last,job['case'],job['key']))
            if job.get('method') in ['STAT','ML']:main_after_transfer.append(first)
    # Whole worker intervals include build/audit and helpers, so non-overlap is stronger than native-search non-overlap.
    known.sort();overlaps=[]
    for earlier,later in zip(known,known[1:]):
        if earlier[1]-later[0]>.05:overlaps.append(dict(earlier=earlier,later=later,seconds=earlier[1]-later[0]))
    assert not overlaps,('unexpected simultaneous solver workers',overlaps)
    assert not main_after_transfer or min(main_after_transfer)>=finish-.05
    target=ROOT/'reports/execution_schedule.csv';target.parent.mkdir(exist_ok=True)
    with target.open('w',encoding='utf-8',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=['case','key','attempt','timing_uncertain','process_seconds','training_overlap_seconds','process_start','process_end'])
        writer.writeheader();writer.writerows(rows)
    result=dict(enabled=True,passed=True,training_start=datetime.datetime.fromtimestamp(begin).astimezone().isoformat(),
        training_end=training['ended_at'],transfer_full_completed_at=read(ROOT/'transfer_full_complete.json')['completed_at'],
        attempts_overlapping_training=sum(r['training_overlap_seconds'] is not None and r['training_overlap_seconds']>0 for r in rows),
        process_overlap_training_seconds=sum(r['training_overlap_seconds'] or 0. for r in rows),
        exact_attempts_checked=len(known),uncertain_attempts=sum(r['timing_uncertain'] for r in rows),
        no_simultaneous_target_solver_workers=True,restricted_solver_phase_waited=True,
        interval_scope='Worker process interval reconstructed from durable ended_at minus measured process_wall_seconds; not allocated native solver time.',
        accounting='All native and process measurements remain actual, without subtraction or normalization. Overlapping training and transfer phase durations must not be added as elapsed wall time.',
        gate_waits=read(ROOT/'transfer_full_gate_waits.json') if (ROOT/'transfer_full_gate_waits.json').exists() else [])
    save(ROOT/'reports/scheduling_overlap_audit.json',result);return result
