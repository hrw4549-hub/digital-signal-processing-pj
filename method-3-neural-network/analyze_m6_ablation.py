import csv
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

package = Path(sys.argv[1])
inference = Path(sys.argv[2])
output = Path(sys.argv[3])
sys.path.insert(0, str(package))

import numpy as np
from DATAPY.common import CONFIG, CHANNELS, counts, filename, loadnpz, readcsv
from DATAPY.reservoir import apply

selection = json.loads((package / "weights/selection.json").read_text(encoding="utf-8"))["choices"]
reference = {(int(r["seed"]), int(r["case"])): r for r in readcsv(package / "expected/per_capture.csv")}
rows = []

for seed in CONFIG["seeds"]:
    for case in CONFIG["captures"]:
        params = loadnpz(package / "weights/rc" / (filename(seed, case) + ".npz"))
        caps = [np.load(package / "data/cap" / f"case{case:02d}_{ch}.npy") for ch in CHANNELS]
        _, _, branches = apply(case, seed, caps, package / "weights/rc")
        choices = selection[f"{seed}_{(case - 1) % 5}"]
        weight = np.array([choices["PBS_CH1_weight"]])
        out = loadnpz(inference / "outputs" / (filename(seed, case) + ".npz"))
        candidates = {
            "linear_rc": [branches[f"linear_{ch}"][19999:51200] for ch in CHANNELS],
            "nonlinear_rc": [branches[f"nonlinear_{ch}"][19999:51200] for ch in CHANNELS],
            "m6_neural": [out[ch] for ch in CHANNELS],
        }
        errors = {name: int(counts(*signal, weight, 20000, 51200)[0]) for name, signal in candidates.items()}
        expected = int(reference[seed, case]["test_errors"])
        if errors["m6_neural"] != expected:
            raise RuntimeError(f"M6 mismatch for seed={seed}, case={case}")
        rows.append({"seed": seed, "case": case, **{f"{k}_errors": v for k, v in errors.items()}})

output.mkdir(parents=True, exist_ok=True)
with (output / "per_seed_capture.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

models = ("linear_rc", "nonlinear_rc", "m6_neural")
by_case = []
for case in CONFIG["captures"]:
    group = [row for row in rows if row["case"] == case]
    by_case.append({"case": case, **{name: np.mean([r[f"{name}_errors"] for r in group]) / 187200 for name in models}})

rng = np.random.default_rng(20261109)
summary = {"records": len(rows), "test_bits_per_record": 187200, "by_model": {}}
for name in models:
    case_bers = np.array([r[name] for r in by_case])
    draws = rng.integers(0, len(case_bers), size=(20000, len(case_bers)))
    boot = case_bers[draws].mean(axis=1)
    summary["by_model"][name] = {
        "mean_BER_over_95": float(np.mean([r[f"{name}_errors"] for r in rows]) / 187200),
        "mean_case_BER": float(case_bers.mean()),
        "case_bootstrap_95pct": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
    }

for left, right in (("m6_neural", "linear_rc"), ("m6_neural", "nonlinear_rc")):
    diff = np.array([r[left] - r[right] for r in by_case])
    draws = rng.integers(0, len(diff), size=(20000, len(diff)))
    boot = diff[draws].mean(axis=1)
    summary["by_model"][f"{left}_minus_{right}"] = {
        "mean_case_BER_difference": float(diff.mean()),
        "captures_improved": int((diff < 0).sum()),
        "captures_worsened": int((diff > 0).sum()),
        "captures_tied": int((diff == 0).sum()),
        "case_bootstrap_95pct": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
    }

(output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
