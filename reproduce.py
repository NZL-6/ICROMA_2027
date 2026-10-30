"""Portable release entry point. Default verification never imports experiment code."""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return p


def verify():
    """Read-only, standard-library checks; no model import, training, or solving."""
    manifest = read(REPO / "provenance/computational_manifest.json")
    for row in manifest["files"]:
        p = REPO / safe_member(row["path"])
        if not p.is_file() or digest(p) != row["sha256"]:
            raise ValueError(f"Release hash mismatch: {row['path']}")
    sources = read(REPO / "provenance/source_manifest.json")["files"]
    for row in sources:
        p = REPO / row["path"]
        ast.parse(p.read_text(encoding="utf-8-sig"), filename=row["path"])
        if digest(p) != row["sha256"]:
            raise ValueError(f"Source manifest mismatch: {row['path']}")
        if row["adaptation"] is None and row["sha256"] != row["original_sha256"]:
            raise ValueError(f"Undocumented source adaptation: {row['path']}")
    ast.parse(Path(__file__).read_text(encoding="utf-8"), filename="reproduce.py")
    initial = read(REPO / "data/input_cases_manifest.json")
    expected = {r["path"]: r for r in initial["files"]}
    with zipfile.ZipFile(REPO / "data/input_cases.zip") as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(expected):
            raise ValueError("Initial-input archive is incomplete or corrupt")
        for name in archive.namelist():
            safe_member(name)
            if "optimal" in name or "complete" in name or "result" in name:
                raise ValueError(f"Target result/marker in fresh inputs: {name}")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"Initial-input hash mismatch: {name}")
            json.loads(data)
    cases = read(REPO / "data/case_manifest.json")
    base = {c["case"] for c in cases if c["cohort"] == "base"}
    transfer = {c["case"] for c in cases if c["cohort"] == "transfer"}
    if (len(base), len(transfer), len(cases), len(expected)) != (75, 36, 111, 999):
        raise ValueError("Unexpected design size")
    folds = read(REPO / "data/folds.json")
    seen = []
    for fold in folds[:5]:
        train, test = set(fold["train"]), set(fold["test"])
        if len(train) != 60 or len(test) != 15 or train & test or train | test != base:
            raise ValueError("Outer-fold isolation failed")
        for inner in fold["inner"]:
            a, b = set(inner["train"]), set(inner["validation"])
            if len(a) != 45 or len(b) != 15 or a & b or a | b != train:
                raise ValueError("Inner-fold isolation failed")
        seen.extend(test)
    if collections.Counter(seen) != collections.Counter({c: 1 for c in base}):
        raise ValueError("A base case is not tested exactly once")
    if set(folds[-1]["train"]) != base or folds[-1]["test"]:
        raise ValueError("Final frozen model is not base-only")
    for row in read(REPO / "provenance/published_results_manifest.json")["byte_identical_files"]:
        if digest(REPO / row["path"]) != row["sha256"]:
            raise ValueError(f"Archived result changed: {row['path']}")
    result = {"passed": True, "source_files": len(sources), "initial_cases": 111,
              "input_files": len(expected), "base_folds": 5,
              "planned_initial_targets": 888, "experiment_executed": False}
    print(json.dumps(result, indent=2))
    return result


def distributions(path=None):
    packages = importlib.metadata.distributions(path=[str(path)]) if path else importlib.metadata.distributions()
    found = {}
    for distribution in packages:
        # Respect sys.path precedence when a venv inherits system packages.
        found.setdefault(distribution.metadata["Name"].lower().replace("_", "-"), distribution.version)
    return found


def check_environment(solver_deps):
    """Distribution metadata only; does not import NumPy, PyTorch, or CP-SAT."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 (the archived interpreter was 3.12.3)")
    spec = read(REPO / "environment/environment.json")
    checks = []
    for label, actual, expected in [("learning", distributions(), spec["learning_packages"]),
                                    ("solver", distributions(solver_deps), spec["solver_packages"])]:
        for name, version in expected.items():
            found = actual.get(name)
            checks.append({"stack": label, "package": name, "expected": version, "found": found})
            if found != version:
                raise RuntimeError(f"{label}: {name} expected {version}, found {found}")
    overlay = distributions(solver_deps).get("typing-extensions")
    if overlay not in (None, "4.12.2"):
        raise RuntimeError("Unexpected solver typing-extensions overlay")
    print(json.dumps({"passed": True, "metadata_checks": checks,
                      "solver_typing_extensions_overlay": overlay,
                      "solver_or_training_called": False}, indent=2))


def prepare_run(destination, solver_deps):
    target = Path(destination).expanduser().resolve()
    if target == REPO or REPO in target.parents:
        raise ValueError("Choose a new run directory outside this repository")
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing directory")
    deps = Path(solver_deps).expanduser().resolve()
    check_environment(deps)
    shutil.copytree(REPO / "code", target)
    for folder in ["runs", "models", "features", "reports", "validation", "history", "phase_logs", "predictions"]:
        (target / folder).mkdir(exist_ok=True)
    shutil.copytree(REPO / "data/baseline", target / "baseline")
    for name in ["protocol.json", "case_manifest.json", "folds.json", "candidate_catalog.json"]:
        shutil.copy2(REPO / "data" / name, target / name)
    with zipfile.ZipFile(REPO / "data/input_cases.zip") as archive:
        for name in archive.namelist():
            p = target / "cases" / safe_member(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(archive.read(name))
    shutil.copytree(deps, target / "vendor/cp_deps", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_files = [{"path": r["path"].removeprefix("code/"), "sha256": r["sha256"]}
                    for r in read(REPO / "provenance/source_manifest.json")["files"]]
    write(target / "source_lock.json", {"scope": "Portable release source", "files": source_files})
    write(target / "execution_lock.json", {"files": source_files})
    inputs = list((target / "baseline").rglob("*.json"))
    inputs += [target / n for n in ["protocol.json", "folds.json", "candidate_catalog.json"]]
    # Mutable input_validation/case_manifest status records are not frozen. Physical
    # inputs, masks, timetable and scenario definitions are immutable during a run.
    inputs += [p for p in (target / "cases").rglob("*.json") if p.name != "input_validation.json"]
    write(target / "input_lock.json", {"files": [{"path": str(p), "sha256": digest(p)} for p in inputs]})
    write(target / "release_run.json", {"release_manifest_sha256": digest(REPO / "provenance/computational_manifest.json"),
          "fresh_inputs_only": True, "initial_targets": 888, "reference_jobs": 111,
          "sequential_pipeline": True, "published_results_copied_into_run": False})
    (target / "README.md").write_text("Fresh reproduction run. Run pipeline.py only from this directory with the prepared Python environment.\n", encoding="utf-8")
    print(f"Prepared a fresh run at {target}; no experiment has been executed.")
    return target


def execute(target):
    lock = target / "release_driver.lock"
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid()}, f)
    try:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        print("Starting the full experiment: no solve time limit is imposed.", flush=True)
        return subprocess.call([sys.executable, "-B", "-u", str(target / "pipeline.py")], cwd=target, env=env)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Read-only hashes, input archive and split checks (default)")
    parser.add_argument("--check-environment", action="store_true", help="Check package metadata without importing model code")
    parser.add_argument("--solver-deps", type=Path, default=REPO / ".solver_deps")
    parser.add_argument("--new-run", type=Path, help="Prepare a NONEXISTENT directory outside the repository")
    parser.add_argument("--execute", action="store_true", help="Explicitly execute the full experiment; requires --new-run")
    args = parser.parse_args()
    if args.execute and not args.new_run:
        parser.error("--execute requires --new-run; interrupted jobs require evidence-aware recovery")
    verify()
    if args.check_environment:
        check_environment(args.solver_deps)
    if args.new_run:
        target = prepare_run(args.new_run, args.solver_deps)
        if args.execute:
            return execute(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
