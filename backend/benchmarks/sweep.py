"""Live-policy sweep on cached browser-quality features (run from backend/;
see docs/performance.md). Runs benchmarks/replay.py's scenarios for every
combination of window / stride / threshold / debounce and prints one JSON
line per configuration. Tune on the validation split only.

Usage: python benchmarks/sweep.py <cache.npz> <split> [model_type]
"""
import itertools
import os
import subprocess
import sys

cache, split = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "transformer"
replay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay.py")

for window, stride, threshold, debounce in itertools.product([32, 40, 48], [2, 4, 8], [0.5, 0.6, 0.7], [1, 2, 3, 4]):
    out = subprocess.run(
        [sys.executable, replay, cache, split, model,
         f"realtime_window_frames={window}", f"sequence_stride={stride}",
         f"confidence_threshold={threshold}", f"debounce_repeats={debounce}"],
        capture_output=True, text=True, check=True,
    ).stdout
    print(out[out.index("{"):].replace("\n", " "), flush=True)
