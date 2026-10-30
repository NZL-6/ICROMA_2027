"""Independent physical audits and label extraction for the reduction study.

This module deliberately imports neither gurobipy nor the formulation module.
Only exported physical schedules are inspected. Auxiliary MILP values are never
treated as physical labels. Time intervals use [start, end) throughout.
"""
from __future__ import annotations

import itertools
import math
import time


def _index(data):
    visits, runs = {}, {}
    for train in data["trains"]:
        for n, visit in enumerate(train["visits"]):
            visits[f"{train['id']}:{n}"] = (train, n, visit)
        for n, run in enumerate(train["runs"]):
            runs[f"{train['id']}:{n}"] = (train, n, run)
    return visits, runs


def _bounds(data, train, n, phase):
    visit = train["visits"][n]
    lo, hi = visit.get(f"{phase}_bounds", (data["now"], data["horizon"]))
    if f"fixed_{phase}" in visit:
        return (visit[f"fixed_{phase}"],) * 2
    lo, hi = max(lo, data["now"]), min(hi, data["horizon"])
    if n == 0 or phase == "departure" and visit.get("commercial", False):
        lo = max(lo, visit["plan_departure"])
    return lo, hi


def _modes(train, n):
    visit = train["visits"][n]
    if n in (0, len(train["visits"]) - 1) or visit.get("stop") == "forbidden":
        choices = [0]
    elif visit.get("stop") == "required":
        choices = [1]
    else:
        choices = [0, 1]
    return [v for v in choices if "fixed_stop" not in visit or v == visit["fixed_stop"]]


def _tracks(data, visit):
    choices = visit.get("tracks", data["stations"][visit["station"]]["tracks"])
    return [v for v in choices if "fixed_track" not in visit or v == visit["fixed_track"]]


def audit_restrictions(data, restrictions=None):
    """Reject masks that enlarge domains, override history, or have no meaning.

    Empty intersections are valid reductions but are flagged as immediately
    infeasible. Malformed/inconsistent masks are audit errors, not solver failure.
    """
    restrictions = restrictions or {}
    visits, runs = _index(data)
    errors, empty = [], []
    for ident, value in restrictions.get("stops", {}).items():
        if ident not in visits:
            errors.append(f"Unknown stop key: {ident}")
            continue
        train, n, visit = visits[ident]
        if value not in (0, 1):
            errors.append(f"Stop mask is not binary: {ident}")
        elif "fixed_stop" in visit and value != visit["fixed_stop"]:
            errors.append(f"Stop mask overrides immutable state: {ident}")
        elif value not in _modes(train, n):
            empty.append(f"stop:{ident}")
    for ident, value in restrictions.get("tracks", {}).items():
        if ident not in visits:
            errors.append(f"Unknown track key: {ident}")
            continue
        train, n, visit = visits[ident]
        if n in (0, len(train["visits"]) - 1):
            errors.append(f"Boundary visit has no track decision: {ident}")
        elif "fixed_track" in visit and value != visit["fixed_track"]:
            errors.append(f"Track mask overrides immutable state: {ident}")
        elif value not in _tracks(data, visit):
            empty.append(f"track:{ident}")
    for event, interval in restrictions.get("bounds", {}).items():
        try:
            ident, phase = event.rsplit(":", 1)
            train, n, visit = visits[ident]
            if phase not in ("arrival", "departure"):
                raise KeyError(phase)
        except (ValueError, KeyError):
            errors.append(f"Unknown event-bound key: {event}")
            continue
        if not isinstance(interval, (tuple, list)) or len(interval) != 2 or any(type(x) is not int for x in interval):
            errors.append(f"Noninteger/malformed bound mask: {event}")
            continue
        lo, hi = _bounds(data, train, n, phase)
        if max(lo, interval[0]) > min(hi, interval[1]):
            empty.append(f"bound:{event}")
    seen = {}
    for field,index in (('dwell_bounds',visits),('running_bounds',runs)):
        for ident,interval in restrictions.get(field,{}).items():
            if ident not in index or not isinstance(interval,(list,tuple)) or len(interval)!=2 or any(type(x) is not int for x in interval):
                errors.append(f'Malformed duration mask: {field}:{ident}')
            elif interval[0]>interval[1]:
                empty.append(f'{field}:{ident}')
    for record in restrictions.get("orders", []):
        try:
            kind, resource = record["kind"], record["resource"]
            first, second = record["first"], record["second"]
        except (KeyError, TypeError):
            errors.append(f"Malformed order record: {record}")
            continue
        relevant = False
        if kind == "run" and first in runs and second in runs:
            ta, _, ra = runs[first]
            tb, _, rb = runs[second]
            relevant = ta["id"] != tb["id"] and ra["resource"] == rb["resource"] == resource
        elif kind == "station" and first in visits and second in visits:
            ta, na, va = visits[first]
            tb, nb, vb = visits[second]
            if ta["id"] != tb["id"] and va["station"] == vb["station"]:
                if na not in (0, len(ta["visits"]) - 1) and nb not in (0, len(tb["visits"]) - 1):
                    relevant = any(resource == f"{va['station']}:{track}" for track in set(_tracks(data, va)) & set(_tracks(data, vb)))
        if not relevant:
            errors.append(f"Order has no common physical decision: {record}")
            continue
        pair = (kind, resource, *sorted((first, second)))
        direction = (first, second)
        if pair in seen and seen[pair] != direction:
            errors.append(f"Contradictory order records: {record}")
        seen[pair] = direction
    return {"passed": not errors, "errors": errors, "empty_domains": empty,
            "immediately_infeasible": bool(empty)}


def _safe_run(data, first, second):
    resource = data["running_resources"][first["resource"]]
    if first["direction"] == second["direction"]:
        return (first["departure"] + resource["entry_headway"] <= second["departure"]
                and first["arrival"] + resource["exit_headway"] <= second["arrival"])
    return first["arrival"] + resource["opposite_clearance"] <= second["departure"]


def _records(data, schedule):
    """Construct physical use intervals, not formulation-specific auxiliaries."""
    runs, occupancy = [], []
    for train in data["trains"]:
        tid = train["id"]
        values = schedule[tid]
        for n, actual in enumerate(values):
            if n in (0, len(values) - 1):
                continue
            ident = f"{tid}:{n}"
            station_name = train["visits"][n]["station"]
            station = data["stations"][station_name]
            occupancy.append({"train": tid, "id": ident, "kind": "station",
                              "resource": f"{station_name}:{actual['track']}",
                              "start": actual["arrival"], "end": actual["departure"] + station["clearance"]})
            for phase in ("arrival", "departure"):
                for j, lock in enumerate(station.get(f"{phase}_locks", [])):
                    anchor = actual[phase]
                    occupancy.append({"train": tid, "id": f"{ident}:{phase}:{j}", "kind": "throat",
                                      "resource": lock["resource"], "start": anchor + lock["start_offset"],
                                      "end": anchor + lock["end_offset"]})
        for n, run in enumerate(train["runs"]):
            runs.append(dict(run, train=tid, id=f"{tid}:{n}",
                             departure=values[n]["departure"], arrival=values[n + 1]["arrival"]))
    return runs, occupancy


def audit_result(data, result, restrictions=None):
    """Audit schedule, initial-state commitments, masks and objective independently.

    ``feasibility`` is feasible / proven_infeasible / unresolved / invalid.
    No incumbent from a time-limited solve is never called infeasible.
    """
    started = time.perf_counter()
    restrictions = restrictions or {}
    domain_audit = audit_restrictions(data, restrictions)
    errors = list(domain_audit["errors"])
    schedule = result.get("schedule", {})
    if errors:
        return {"passed": False, "schedule_valid": False, "feasibility": "invalid",
                "errors": errors, "recomputed_objective": None,
                "domain_audit": domain_audit, "validation_seconds": time.perf_counter() - started}
    if not schedule:
        infeasible = result.get("status") == 3 or domain_audit["immediately_infeasible"]
        return {"passed": not errors, "schedule_valid": False,
                "feasibility": "invalid" if errors else "proven_infeasible" if infeasible else "unresolved",
                "errors": errors, "recomputed_objective": None,
                "domain_audit": domain_audit, "validation_seconds": time.perf_counter() - started}
    expected = {t["id"] for t in data["trains"]}
    if set(schedule) != expected:
        errors.append("Exported train set does not match input")
    structure_ok = True
    for train in data["trains"]:
        tid = train["id"]
        values = schedule.get(tid, [])
        if len(values) != len(train["visits"]):
            errors.append(f"Wrong visit count: {tid}")
            structure_ok = False
            continue
        for n, (visit, actual) in enumerate(zip(train["visits"], values)):
            ident = f"{tid}:{n}"
            if any(field not in actual for field in ("station", "arrival", "departure", "stop", "track")):
                errors.append(f"Missing physical schedule field: {ident}")
                structure_ok = False
                continue
            if actual["station"] != visit["station"]:
                errors.append(f"Wrong station: {ident}")
            if any(not isinstance(actual[p], (int, float)) or not math.isfinite(actual[p]) or abs(actual[p] - round(actual[p])) > 1e-6
                   for p in ("arrival", "departure")):
                errors.append(f"Event outside integer grid: {ident}")
                structure_ok = False
                continue
            for phase in ("arrival", "departure"):
                lo, hi = _bounds(data, train, n, phase)
                event = f"{ident}:{phase}"
                if event in restrictions.get("bounds", {}):
                    rlo, rhi = restrictions["bounds"][event]
                    lo, hi = max(lo, rlo), min(hi, rhi)
                if not lo <= actual[phase] <= hi:
                    errors.append(f"Event violates original or reduced bounds: {event}")
            a, d, mode = actual["arrival"], actual["departure"], actual["stop"]
            for interval in (visit.get('duration_bounds'),restrictions.get('dwell_bounds',{}).get(ident)):
                if interval is not None and not interval[0]<=d-a<=interval[1]:
                    errors.append(f'Dwell duration violation: {ident}')
            if mode not in _modes(train, n) or mode != restrictions.get("stops", {}).get(ident, mode):
                errors.append(f"Invalid stop mode: {ident}")
            if mode == 0 and a != d:
                errors.append(f"Unmarked waiting: {ident}")
            if n in (0, len(values) - 1):
                if a != d or actual["track"] is not None:
                    errors.append(f"Invalid boundary visit: {ident}")
            else:
                station = data["stations"][visit["station"]]
                if mode == 1 and d - a < visit.get("minimum_dwell", station["minimum_dwell"]):
                    errors.append(f"Insufficient dwell: {ident}")
                if actual["track"] not in _tracks(data, visit) or actual["track"] != restrictions.get("tracks", {}).get(ident, actual["track"]):
                    errors.append(f"Invalid track or immutable track change: {ident}")
        if structure_ok:
            for n, run in enumerate(train["runs"]):
                start, end = values[n], values[n + 1]
                addition = run.get("stop_depart_add", 0) * start["stop"] + run.get("stop_arrive_add", 0) * end["stop"]
                duration = end["arrival"] - start["departure"]
                interval=restrictions.get('running_bounds',{}).get(f'{tid}:{n}')
                if interval is not None and not interval[0]<=duration<=interval[1]:
                    errors.append(f'Restricted running duration violation: {tid}:{n}')
                if not run["minimum"] + addition <= duration <= run["maximum"] + addition:
                    errors.append(f"Running-time violation: {tid}:{n}")
                for closure in data.get("closures", []):
                    if closure["resource"] == run["resource"] and start["departure"] < closure["end"] and end["arrival"] > closure["start"]:
                        errors.append(f"Closure overlap: {tid}:{n}")
    objective = None
    if structure_ok:
        runs, occupancy = _records(data, schedule)
        order_lookup = {(r["kind"], r["resource"], r["first"], r["second"]) for r in restrictions.get("orders", [])}
        for first, second in itertools.combinations(runs, 2):
            if first["train"] == second["train"] or first["resource"] != second["resource"]:
                continue
            safe_ab, safe_ba = _safe_run(data, first, second), _safe_run(data, second, first)
            if not (safe_ab or safe_ba):
                errors.append(f"Running-resource conflict: {first['id']} / {second['id']}")
            if ("run", first["resource"], first["id"], second["id"]) in order_lookup and not safe_ab:
                errors.append(f"Reduced run order violated: {first['id']} / {second['id']}")
            if ("run", first["resource"], second["id"], first["id"]) in order_lookup and not safe_ba:
                errors.append(f"Reduced run order violated: {second['id']} / {first['id']}")
        for first, second in itertools.combinations(occupancy, 2):
            if first["train"] == second["train"] or (first["kind"], first["resource"]) != (second["kind"], second["resource"]):
                continue
            if max(first["start"], second["start"]) < min(first["end"], second["end"]):
                errors.append(f"Physical resource overlap: {first['kind']}:{first['resource']} / {first['id']} / {second['id']}")
            if first["kind"] == "station":
                if ("station", first["resource"], first["id"], second["id"]) in order_lookup and first["end"] > second["start"]:
                    errors.append(f"Reduced station order violated: {first['id']} / {second['id']}")
                if ("station", first["resource"], second["id"], first["id"]) in order_lookup and second["end"] > first["start"]:
                    errors.append(f"Reduced station order violated: {second['id']} / {first['id']}")
        objective = sum(t.get("weight", 1) * max(0, schedule[t["id"]][-1]["arrival"] - t["visits"][-1]["plan_arrival"])
                        for t in data["trains"])
        if result.get("objective") is None or abs(objective - result["objective"]) > 1e-6:
            errors.append(f"Exported objective mismatch: {objective} vs {result.get('objective')}")
        if '_offline_upper_cutoff' in data and objective>data['_offline_upper_cutoff']+1e-6:
            errors.append('Offline incumbent improvement constraint violated')
    if result.get("proven_optimal") and result.get("status") != 2:
        errors.append("Optimality claim inconsistent with solver status")
    if result.get("status") == 3:
        errors.append("Solver reports infeasible but schedule is exported")
    return {"passed": not errors, "schedule_valid": not errors,
            "feasibility": "feasible" if not errors else "invalid", "errors": errors,
            "recomputed_objective": objective, "domain_audit": domain_audit,
            "validation_seconds": time.perf_counter() - started}


def derive_physical_decisions(data, schedule):
    """Return restriction-shaped labels for a physically validated schedule.

    Station order labels exist only when both trains actually use that track.
    ``throat_orders`` and ``closure_sides`` are metadata: baseline solvers do not
    expose restriction interfaces for those choices.
    """
    result = {"stops": {}, "tracks": {}, "orders": [], "events": {},
              "throat_orders": [], "closure_sides": []}
    for train in data["trains"]:
        tid = train["id"]
        for n, actual in enumerate(schedule[tid]):
            ident = f"{tid}:{n}"
            for phase in ("arrival", "departure"):
                result["events"][f"{ident}:{phase}"] = actual[phase]
            if n not in (0, len(train["visits"]) - 1):
                result["stops"][ident] = actual["stop"]
                result["tracks"][ident] = actual["track"]
    runs, occupancy = _records(data, schedule)
    for first, second in itertools.combinations(runs, 2):
        if first["train"] == second["train"] or first["resource"] != second["resource"]:
            continue
        if not _safe_run(data, first, second):
            first, second = second, first
        if not _safe_run(data, first, second):
            raise ValueError("Cannot label unsafe physical running order")
        result["orders"].append({"kind": "run", "resource": first["resource"], "first": first["id"], "second": second["id"]})
    for first, second in itertools.combinations(occupancy, 2):
        if first["train"] == second["train"] or (first["kind"], first["resource"]) != (second["kind"], second["resource"]):
            continue
        if first["start"] > second["start"]:
            first, second = second, first
        if first["end"] > second["start"]:
            raise ValueError("Cannot label overlapping physical occupancy")
        record = {"kind": first["kind"], "resource": first["resource"], "first": first["id"], "second": second["id"]}
        result["orders" if first["kind"] == "station" else "throat_orders"].append(record)
    for run in runs:
        for n, closure in enumerate(data.get("closures", [])):
            if run["resource"] == closure["resource"]:
                side = "before" if run["arrival"] <= closure["start"] else "after" if run["departure"] >= closure["end"] else "overlap"
                result["closure_sides"].append({"operation": run["id"], "closure": n, "side": side})
    return result


def objective_comparison(reference, reduced):
    """Separate certified reduction loss from incumbent-only comparisons."""
    ref_obj, red_obj = reference.get("objective"), reduced.get("objective")
    both_optimal = bool(reference.get("proven_optimal") and reduced.get("proven_optimal"))
    output = {"reference_proven_optimal": bool(reference.get("proven_optimal")),
              "reduced_proven_optimal": bool(reduced.get("proven_optimal")),
              "exact_reduction_loss": None, "incumbent_difference": None,
              "reduction_loss_lower_bound": None, "reduction_loss_upper_bound": None}
    if ref_obj is not None and red_obj is not None:
        output["incumbent_difference"] = red_obj - ref_obj
        if both_optimal:
            output["exact_reduction_loss"] = red_obj - ref_obj
    ref_lb, red_lb = reference.get("lower_bound"), reduced.get("lower_bound")
    if red_lb is not None and math.isfinite(red_lb) and ref_obj is not None:
        output["reduction_loss_lower_bound"] = max(0, red_lb - ref_obj)
    if red_obj is not None and ref_lb is not None and math.isfinite(ref_lb):
        output["reduction_loss_upper_bound"] = max(0, red_obj - ref_lb)
    return output


def _self_check():
    """Audit stored formulation checks plus adversarial scientific safeguards."""
    import copy
    import json
    from pathlib import Path

    here = Path(__file__).resolve().parent
    base = here.parent / "common_dispatch_baseline_20261023"
    checked = []
    for result_path in sorted((base / "results").glob("*.json")):
        case = json.loads((base / "cases" / result_path.name).read_text(encoding="utf-8"))
        results = json.loads(result_path.read_text(encoding="utf-8"))
        for formulation, result in results.items():
            audit = audit_result(case["data"], result, case["restrictions"])
            assert audit["passed"], (result_path.stem, formulation, audit)
            checked.append({"case": result_path.stem, "formulation": formulation,
                            "feasibility": audit["feasibility"], "passed": audit["passed"]})
    case = json.loads((base / "cases" / "two_track_overtaking.json").read_text(encoding="utf-8"))
    data = case["data"]
    result = json.loads((base / "results" / "two_track_overtaking.json").read_text(encoding="utf-8"))["big_m"]
    gates = []
    altered = copy.deepcopy(result)
    altered["schedule"]["F"][1]["track"] = altered["schedule"]["S"][1]["track"]
    gates.append(("reject_same_track_overlap", not audit_result(data, altered)["schedule_valid"]))
    altered = copy.deepcopy(result)
    altered["objective"] += 1
    gates.append(("reject_wrong_objective", not audit_result(data, altered)["schedule_valid"]))
    altered = copy.deepcopy(result)
    altered["schedule"]["S"][0]["arrival"] += 1
    altered["schedule"]["S"][0]["departure"] += 1
    gates.append(("reject_changed_immutable_event", not audit_result(data, altered)["schedule_valid"]))
    current = json.loads((base / "cases" / "initial_station_occupancy_and_track.json").read_text(encoding="utf-8"))["data"]
    gates.append(("reject_immutable_track_override", not audit_restrictions(current, {"tracks": {"S:1": "B2"}})["passed"]))
    conflict = {"orders": [{"kind": "run", "resource": "BC", "first": "S:1", "second": "F:1"},
                           {"kind": "run", "resource": "BC", "first": "F:1", "second": "S:1"}]}
    gates.append(("reject_contradictory_order_masks", not audit_restrictions(data, conflict)["passed"]))
    timeout = {"status": 9, "schedule": {}, "objective": None, "proven_optimal": False}
    gates.append(("timeout_without_incumbent_is_unresolved", audit_result(data, timeout)["feasibility"] == "unresolved"))
    gates.append(("empty_domain_is_proven_infeasible", audit_result(data, timeout, {"bounds": {"S:1:departure": [50, 60]}})["feasibility"] == "proven_infeasible"))
    approximate = dict(result, proven_optimal=False, status=9)
    gates.append(("incumbent_is_not_an_optimum", objective_comparison(approximate, result)["exact_reduction_loss"] is None))
    decisions = derive_physical_decisions(data, result["schedule"])
    station_labels = [x for x in decisions["orders"] if x["kind"] == "station"]
    gates.append(("inactive_station_order_has_no_label", not station_labels))
    label_masks = {k: decisions[k] for k in ("stops", "tracks", "orders")}
    gates.append(("physical_labels_retain_source_schedule", audit_result(data, result, label_masks)["schedule_valid"]))
    for name, passed in gates:
        assert passed, name
    output = {"source": "Independent exported-schedule audit; no formulation helpers imported",
              "stored_results_checked": len(checked), "all_passed": all(r["passed"] for r in checked),
              "stored_checks": checked, "adversarial_gates": [{"name": k, "passed": v} for k, v in gates]}
    (here / "audit_verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stored_results_checked": len(checked), "adversarial_gates": len(gates), "passed": True}))


if __name__ == "__main__":
    _self_check()
