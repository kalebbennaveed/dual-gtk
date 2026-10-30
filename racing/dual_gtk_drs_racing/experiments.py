"""Paired, resumable paper experiments. Randomness is in fixed episode friction.

Every trial executes the real closed-loop simulator and its independent audit.
Raw records (including rejected candidates) are retained as compressed JSON.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
import traceback
import numpy as np
from .simulation import simulate
from .validation import audit_record
from .scenarios import get_scenario

METHODS = ("proposed", "passive", "weighted", "robust")
NUMERICAL_FILES = ("model.py", "fast_model.py", "expressions.py", "safety.py", "gatekeeper.py", "belief.py",
                   "simulation.py", "track.py", "scenarios.py", "dynamics.py", "mppi.py", "control_safety.py", "assets/barcelona.csv")
PROTOCOL = 3


def implementation_signature():
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / p).read_bytes() for p in NUMERICAL_FILES)).hexdigest()


def design(trials=100, seed=20261007, seconds=360., grid=True, weights=True, scenario="traction"):
    if trials < 1 or seconds <= 0:
        raise ValueError("Use at least one independent trial and a positive horizon")
    rng = np.random.default_rng(seed)
    cases = []
    def add(group, trial, mu, method, weight=.4):
        cases.append(dict(id=f"{group}-{trial:03d}-{method}", group=group, trial=trial,
            method=method, true_mu=float(mu), seed=seed + trial, seconds=float(seconds),
            scenario=scenario, information_weight=2. if method=="proposed" else float(weight) if method=="weighted" else 0., max_cells=256, start=0.))
    # IID draws, never a grid or deterministic repeated seeds labelled Monte Carlo.
    for i, mu in enumerate(rng.uniform(.45, 1.05, trials)):
        for method in METHODS:
            add("monte_carlo", i, mu, method)
    if grid:
        for i, mu in enumerate(np.linspace(.45, 1.05, 13)):
            for method in METHODS:
                add("mu_grid", i, mu, method)
    if weights:
        for i, (mu, weight) in enumerate((mu, w) for mu in (.65, .85, 1.05) for w in (0., .5, 1., 2., 4., 8.)):
            add("weight_sweep", i, mu, "weighted", weight)
    return cases


def first_time(t, condition):
    idx = np.flatnonzero(condition)
    return float(t[idx[0]]) if len(idx) else None


def metrics(record, audit):
    f = record["frames"]
    t = np.array([v["t"] for v in f])
    x = np.array([v["state"] for v in f])
    mu = np.array([v["mu"] for v in f])
    s = record["summary"]
    truth = s["true_mu"]
    width = mu[:, 1] - mu[:, 0]
    containment = np.minimum(truth - mu[:, 0], mu[:, 1] - truth)
    assert np.all(containment >= -1e-9), "SMID excluded truth"
    assert np.all(np.diff(mu[:, 0]) >= -1e-9) and np.all(np.diff(mu[:, 1]) <= 1e-9), "SMID sets are not nested"
    dt = np.diff(t)
    u = np.array([v["control"] for v in f[:-1]]).reshape(-1,3)
    target = record["scenario"]["racing_speed"]
    components = dict(time_cost=float(dt.sum()),
        speed_cost=float(np.sum(dt * .004 * (target - x[:-1, 3])**2)),
        lateral_cost=float(np.sum(dt * .08 * x[:-1, 1]**2)),
        heading_cost=float(np.sum(dt * .2 * x[:-1, 2]**2)),
        force_cost=float(np.sum(dt * .002 * ((u[:, 0] + u[:, 1]) / 720.)**2)),
        steering_cost=float(np.sum(dt * .08 * u[:, 2]**2)))
    assert abs(sum(components.values()) - s["cost"]) < 1e-7
    certificates = [e["selected"]["certificate"] for e in record["events"] if e["status"] == "committed"]
    tire = [v["traction_margin"] for v in f if "traction_margin" in v]
    checks=list({(e['time'],c['sample_seed']):c for e in record['events'] for c in e.get('certification_checks',[])}.values())
    accepted=[c for c in checks if c['accepted']]
    return dict(**s, final_width=float(width[-1]), peak_speed=float(x[:, 3].max()),
        mean_speed=float(np.sum(dt * x[:-1, 3]) / t[-1]) if t[-1] else 0.,
        width_integral=float(np.sum(dt * width[:-1])),
        time_to_90pct_contraction=first_time(t, width <= .1*width[0]),
        time_to_99pct_contraction=first_time(t, width <= .01*width[0]),
        first_mission_time=first_time(t, np.array([v["mode"] == "mission" for v in f])),
        first_fifty_time=first_time(t, x[:, 3] >= 50.),
        containment_violations=int(np.sum(containment < -1e-9)),
        containment_margin=float(containment.min()), frames=len(f),
        nested_set_violations=int(np.sum((np.diff(mu[:, 0]) < -1e-9) | (np.diff(mu[:, 1]) > 1e-9))),
        minimum_tire_margin=float(min(tire)) if tire else None,
        accepted_certificate_minimum=float(min(c["margin"] for c in certificates)) if certificates else None,
        accepted_certificate_cells_max=max((c["cells"] for c in certificates), default=0),
        certification_checks=len(checks),certification_rollouts=sum(c['evaluated'] for c in checks),
        certificate_upper_max=max((c['failure_upper'] for c in accepted),default=None),
        accepted_sample_failures=sum(c['failures'] for c in accepted),
        epsilon=record['scenario']['epsilon'],
        replay_error=audit["max_replay_error"], audit_passed=audit["passed"],
        **components)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def run_case(case, directory, signature):
    directory = Path(directory)
    row = dict(case=case, implementation=signature, protocol=PROTOCOL)
    begin = time.perf_counter()
    record = None
    try:
        record = simulate(**{k: v for k, v in case.items() if k not in ("id", "group", "trial")})
        if record["implementation"] != signature:
            raise RuntimeError("Numerical source changed during the experiment")
        audit = audit_record(record)
        row.update(status="ok", metrics=metrics(record, audit), audit=audit)
    except Exception as exc:
        # Failed verification, unsafe states and estimator inconsistencies are
        # never discarded, rerolled, or reported as successful safe episodes.
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    if record is not None:
        raw = directory / "records" / (case["id"] + ".json.gz")
        with gzip.open(raw.with_suffix(".gz.tmp"), "wt", compresslevel=3) as stream:
            json.dump(record, stream, allow_nan=False, separators=(",", ":"))
        raw.with_suffix(".gz.tmp").replace(raw)
        row["record"] = str(raw.relative_to(directory))
    row["total_wall_seconds"] = time.perf_counter() - begin
    atomic_json(directory / "runs" / (case["id"] + ".json"), row)
    return row


def run_suite(directory, trials=100, seed=20261007, seconds=360., workers=4,
              grid=True, weights=True, scenario="traction"):
    directory = Path(directory).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    cases = design(trials, seed, seconds, grid, weights, scenario)
    signature = implementation_signature()
    for name in ("records", "runs"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    import scipy, numba
    manifest = dict(protocol=PROTOCOL, implementation=signature, trials=trials, seed=seed,
        seconds=seconds, distribution="IID Uniform[0.45, 1.05], fixed within each episode",
        pairing="Same true_mu and initial state for all four methods in each trial",
        sensor="Deterministic quantized yaw acceleration. Fresh independent certification batches; paired episode seeds across methods.",
        scenario=asdict(get_scenario(scenario)), cases=cases,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                         platform=platform.platform(), workers=workers, cpu_count=os.cpu_count()),
        inference="Binomial intervals use independent Monte Carlo episodes per method only; grid/weight tests are separate",
        safety_scope="Sampled DRS, beta=0, assumed conditional uniform friction and robustly invariant backup (alpha=0). Epsilon/delta are per decision.")
    old_path = directory / "manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if any(old.get(k) != manifest[k] for k in ("protocol", "implementation", "cases")):
            raise ValueError("Existing results use a different design or implementation; choose a new output directory")
    else:
        atomic_json(old_path, manifest)
    todo = []
    for case in cases:
        path = directory / "runs" / (case["id"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["implementation"] != signature or row["case"] != case:
                raise ValueError(f"Incompatible cached run: {path}")
        else:
            todo.append(case)
    print(f"{len(cases)} episodes; {len(cases)-len(todo)} cached; {workers} workers", flush=True)
    # Limit BLAS in spawned processes. Compile/load the numerical kernels once
    # before launching workers so a cold cache is not raced by every process.
    if todo:
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[key] = "1"
        warm = simulate("proposed", seconds=.1, scenario=scenario)
        audit_record(warm)
        begin = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {executor.submit(run_case, case, str(directory), signature): case for case in todo}
            done = len(cases) - len(todo)
            errors = 0
            for future in as_completed(pending):
                row = future.result()
                done += 1
                errors += row["status"] != "ok"
                if done % 8 == 0 or row["status"] != "ok" or done == len(cases):
                    print(f"{done}/{len(cases)} | {time.perf_counter()-begin:.0f}s wall | errors {errors} | {row['case']['id']} {row['status']}", flush=True)
    return directory


def load_rows(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        path = directory / "runs" / (case["id"] + ".json")
        if not path.exists():
            raise ValueError(f"Experiment is incomplete: missing {case['id']}")
        row = json.loads(path.read_text())
        if row["case"] != case or row["implementation"] != manifest["implementation"]:
            raise ValueError(f"Run does not match manifest: {case['id']}")
        rows.append(row)
    return manifest, rows


def load_record(directory, row):
    with gzip.open(Path(directory) / row["record"], "rt") as stream:
        return json.load(stream)
