"""Compare the evaluator with repository source at a fixed revision.

Run from the repository root: python3 benchmark/performance_audit.py
Requires only NumPy; creates all benchmark CSVs in temporary directories.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark.voice_identity_evaluator import VoiceIdentityEvaluator


def measure(call, repeats=5):
    call()  # Warm up imports and numerical routines outside measurements.
    elapsed = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - started)
    gc.collect()
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"median_ms": statistics.median(elapsed) * 1000,
            "peak_traced_bytes": peak, "repeats": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="f59066ad19c810b6ab4a3fbe5f308a31a4218a88")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline], text=True).strip()
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/benchmark/voice_identity_evaluator.py"], text=True)
    original = types.ModuleType("original_evaluator")
    exec(compile(source, "original_evaluator.py", "exec"), original.__dict__)
    # Extract the original scan verbatim to benchmark the modified kernel alone.
    scan = source[source.index("        thresholds = np.linspace"):
                  source.index("        # 2. Export Pairs CSV")]
    exec("def original_eer(same_arr, diff_arr):\n" + scan +
         "        return eer, eer_thresh\n", original.__dict__)
    results = {"baseline_revision": revision, "python": platform.python_version(),
               "numpy": np.__version__, "platform": platform.platform(), "cases": {}}
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as output:
        for singers in (12, 100):
            # Same dimensions/song count as the existing runner; also test scale.
            corpus = {}
            for i in range(singers):
                base = rng.standard_normal(192).astype(np.float32)
                base /= np.linalg.norm(base)
                corpus[str(i)] = {}
                for j in range(3):
                    vector = base + rng.standard_normal(192).astype(np.float32) * 0.18
                    vector /= np.linalg.norm(vector)
                    corpus[str(i)][str(j)] = vector.tolist()
            before_dir, after_dir = str(Path(output) / "before"), str(Path(output) / "after")
            before = lambda: original.VoiceIdentityEvaluator.evaluate_corpus(corpus, before_dir)
            after = lambda: VoiceIdentityEvaluator.evaluate_corpus(corpus, after_dir)
            assert before() == after()
            for filename in ("pairwise_similarities.csv", "voice_identity_benchmark.json"):
                assert (Path(before_dir) / filename).read_bytes() == (Path(after_dir) / filename).read_bytes()
            results["cases"][f"corpus_{singers}_singers"] = {
                "pairs": (singers * 3) * (singers * 3 - 1) // 2,
                "before": measure(before), "after": measure(after),
                "artifacts_byte_identical": True,
            }
        for count in (630, 1000000):
            same, different = rng.uniform(-1, 1, count), rng.uniform(-1, 1, count)
            assert original.original_eer(same, different) == VoiceIdentityEvaluator._equal_error_rate(same, different)
            results["cases"][f"eer_{count}_scores_per_class"] = {
                "before": measure(lambda: original.original_eer(same, different)),
                "after": measure(lambda: VoiceIdentityEvaluator._equal_error_rate(same, different)),
                "exact_result_match": True,
            }
    for case in results["cases"].values():
        case["latency_speedup"] = case["before"]["median_ms"] / case["after"]["median_ms"]
        case["peak_allocation_reduction"] = case["before"]["peak_traced_bytes"] / case["after"]["peak_traced_bytes"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
