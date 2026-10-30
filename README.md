# ICROMA: indicator analysis and progressive decision space reduction

Reproduction materials for the railway rescheduling study on a fixed network and planned timetable. The experiment contains 75 cases for stratified training/testing and 36 cases testing new disruption start times and durations on the same three blockage locations. Each case has eight initial target models, giving 888 initial target solves, plus Strategy-1 reference generation, private bound models and any required full-model refinements.

The release contains executable experiment sources, the original initial inputs, archived result tables, and the supplementary material. The code and results from the completed experiment have not been rerun or changed to prepare this package.

## Contents

- [`supplementary/supplementary.pdf`](supplementary/supplementary.pdf): indicator definitions and complete association results.
- [`supplementary/indicator_catalog_complete.csv`](supplementary/indicator_catalog_complete.csv): all 267 catalog items, including unavailable or constant items, with the reasons they are not estimable.
- [`results/reports/indicator_associations.csv`](results/reports/indicator_associations.csv): complete machine-readable association responses; join to the catalog by the indicator fields explained in the supplementary material.
- `results/reports/`: prediction, quality, computation-time, sensitivity and cost tables from the completed experiment. These are archived evidence and are never used as fresh-run target inputs.
- `data/`: baseline network/timetable, approved case design, case-level cross-validation folds, candidate catalog and the initial inputs for all 111 cases.
- `code/`: experiment implementation and four required source snapshots.
- `environment/`: pinned package versions and the separate learning/solver stacks.
- `provenance/`: source and result hashes, documented portability adapters, and verification evidence.

## Inspect without running experiments

Python's standard library is sufficient:

```text
python reproduce.py --verify
python reproduce.py --help
```

The verifier checks source/data/result hashes, Python syntax, all 999 JSON input archive members, and isolation of the five outer and four inner folds. It imports no experiment modules and invokes no solver or training code. It does not certify model correctness by rerunning optimization.

## Prepare the software environment

Use Python 3.12, preferably on Windows x64, matching the original platform. For example in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r environment/requirements-learning.txt
python -m pip install --no-deps --target .solver_deps -r environment/requirements-solver.txt
python reproduce.py --check-environment --solver-deps .solver_deps
```

On Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`. Cross-platform execution is not claimed to have been tested. Read [`environment/README.md`](environment/README.md) for the intentionally separate numerical stacks and the documented `typing-extensions` metadata mismatch in the archived environment.

## Recompute in a separate fresh directory

The following command is an explicit full experimental run. It has **not** been executed during release preparation:

```text
python reproduce.py --new-run ../icroma-fresh-run --solver-deps .solver_deps --execute
```

Omit `--execute` to prepare the directory without starting any calculations. The destination must not exist and must be outside this repository. The runner copies the source, baseline and initial case inputs, creates fresh hash locks, and copies the separate solver packages into the expected runtime location. It never copies target solutions, trained models, completion markers or archived timing tables into the fresh run. Thus nothing is skipped by importing historical completion state.

The pipeline first validates the formulation (its three formal pilot cases remain part of the experiment), computes full-model references, derives indicators and descriptive associations, fits the nested models, evaluates the full-reference and chained settings, freezes the final base-trained models, evaluates transfer cases, and produces audited tables and a separate evidence archive. Settings remain 12 CP-SAT workers, seed 20261026, no time limit, relative gap strictly below 0.001, and 50% fixing within each eligible decision class. Optional additional stops and associated dwell durations stay free.

Do not run `code/validate.py` as a quick check: it performs actual optimization. The source snapshots contain historical helper entry points that are not standalone release commands. Use `reproduce.py` as the entry point.

The fresh pipeline is sequential. In the reported run, independent transfer full-model calculations overlapped offline training; neither algorithm nor reference information is changed by the sequential release entry point. Archived timing/cost records preserve the measured run. A fresh run must report its own timing, and cannot recreate the original hardware scheduling or an accidental failed attempt.

If execution is interrupted, retain the whole run directory. The pipeline contains durable completed-job and phase markers, but an interrupted active attempt must be reviewed and its elapsed cost recovered before restarting. There is deliberately no blind automatic resume command that could silently omit interrupted costs or run a duplicate solver.

## Reading the comparisons

Strategy-2 changes dwell/running durations with train order retained. Strategy-3 changes durations and order jointly. Its direct full-model baseline starts from Strategy-1. Restricted Strategy-3 uses either a full Strategy-2 reference (the full-reference setting) or the same prediction method's restricted Strategy-2 reference (the chained setting). Both settings use the same trained Strategy-3 prediction model.

The published time tables include cumulative native solver time, prediction plus solver time, and complete online time. Acquiring the upstream Strategy-2 timetable is charged once in each corresponding setting. Common Strategy-1 generation is included once in complete online time. Offline training and full-label generation are recorded separately; `results/training_costs.json`, `results/phase_ledger.json` and the reference-cost tables retain these costs. `results/recovery_preparation_costs.json` retains the failed-preparation timing ledger while replacing machine-specific exception text and paths with release-relative metadata. No numeric costs are altered.

Accepted targets satisfy the prescribed tolerance; labels from one representative timetable need not uniquely identify necessary adjustments. A multithreaded fresh run can yield another accepted representative and different prediction labels or wall times. Reproducing the computational procedure is distinct from guaranteeing bit-identical solutions or elapsed times.

## Preservation and release checks

Twenty-eight of the 30 archived Python files are byte-identical to the experiment. Only `common.py` (relative paths, current interpreter and rebased hash checks) and `package.py` (fresh-run evidence destination) have portability adaptations. The solver, learning, feature and statistical algorithms are unchanged. Unified diffs and both original/release hashes identify these changes. The historical `validate_features.py` depended on a separate earlier experiment and is omitted because the released pipeline never calls it. The `.gitattributes` file disables line-ending conversion so these archived hashes survive Git clones on different platforms.

The archived result tables are byte-identical to the originals except the explicitly documented nonnumeric projection of recovery metadata. Raw solver logs, every target timetable, binary environments, and the large historical evidence archive are not bundled here; a future full run creates its own complete evidence package. The original full evidence remains preserved by the authors.

No full experiment or clean-environment installation was performed for this release. Static validation and archive integrity checks are reported in `provenance/release_validation.json`. A reuse license has not been assigned; see [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md).

This package accompanies the manuscript at the public repository <https://github.com/NZL-6/ICROMA_2027>. The manuscript's supplementary-material statement uses this address. The release-validation records describe the checks performed before publication.
