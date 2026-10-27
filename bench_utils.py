"""
Benchmarking Utilities for fashion-mnist-torch-vs-jax.
Provides subprocess isolation, repeated execution (N=5) with statistical summaries,
and JSON artifact persistence.
"""
import os
import sys
import json
import subprocess
import statistics
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_isolated(fn_module, fn_name, **kwargs):
    """Run a training function in a completely isolated Python subprocess."""
    code = (
        f"import sys, os; sys.path.insert(0, {SCRIPT_DIR!r}); "
        f"import json, {fn_module} as m; "
        f"h = m.{fn_name}(**{kwargs!r}); print('__RESULT__' + json.dumps(h))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(proc.stdout.split("__RESULT__", 1)[1])

def repeat(fn_module, fn_name, n=5, **kwargs):
    """Repeat an isolated benchmark run N times and compute median, min, max."""
    runs = []
    for i in range(1, n + 1):
        print(f"    [Run {i}/{n}] executing {fn_module}.{fn_name}...")
        r = run_isolated(fn_module, fn_name, **kwargs)
        runs.append(r)

    ss = [r["throughputs"][-1] for r in runs]
    warm = [r["epoch_times"][0] - statistics.median(r["epoch_times"][1:]) for r in runs]
    
    return {
        "runs": runs,
        "steady_median": statistics.median(ss),
        "steady_min": min(ss),
        "steady_max": max(ss),
        "warmup_median_s": statistics.median(warm),
        "peak_mem_mb": max(r.get("peak_mem_mb", 0.0) for r in runs),
        "resident_state_mb": max(r.get("resident_state_mb", 0.0) for r in runs),
    }

def save_results(name, payload):
    """Save benchmark results dictionary to results/<name>_<YYYY-MM-DD>.json."""
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(results_dir, f"{name}_{today}.json")
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[✓] Saved benchmark results to: {filepath}")
    return filepath
