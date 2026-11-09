import csv
import json
import sys
from pathlib import Path

import numpy as np

folder = Path(sys.argv[1])
files = sorted(folder.glob("aligned_case*.json"), key=lambda p: tuple(int(v) for v in p.stem.replace("aligned_case", "").replace("_seed", "_").split("_")))
per_seed = []
for path in files:
    result = json.loads(path.read_text(encoding="utf-8"))
    capture = result["protocol"]["capture"]
    seed = result["protocol"].get("seed", 100)
    result["protocol"]["capture"] = capture
    if "_seed" not in path.stem:
        result["selection"]["weighted_sum_alpha"] = 1 - result["selection"]["weighted_sum_alpha"]
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    per_seed.append({"capture": capture, "seed": seed, **{name: result["models"][name]["test_BER"] for name in ("linear_fir", "nonlinear_esn", "weighted_sum", "output_feature_ridge", "dual_mlp", "linear_only_mlp")}})

captures = sorted({r["capture"] for r in per_seed})
names = ("linear_fir", "nonlinear_esn", "weighted_sum", "output_feature_ridge", "dual_mlp", "linear_only_mlp")
rows = [{"capture": c, "seeds": len([r for r in per_seed if r["capture"] == c]), **{name: float(np.mean([r[name] for r in per_seed if r["capture"] == c])) for name in names}} for c in captures]

rng = np.random.default_rng(20261109)
summary = {"captures": len(rows), "total_runs": len(per_seed), "seeds_per_capture": sorted({r["seeds"] for r in rows}), "models": {}}
for name in ("linear_fir", "nonlinear_esn", "weighted_sum", "output_feature_ridge", "dual_mlp", "linear_only_mlp"):
    values = np.array([r[name] for r in rows])
    boot = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    summary["models"][name] = {"mean_BER": float(values.mean()), "median_BER": float(np.median(values)), "capture_bootstrap_95pct": [float(x) for x in np.quantile(boot, [0.025, 0.975])]}

for left, right in (("weighted_sum", "linear_fir"), ("output_feature_ridge", "linear_fir"), ("dual_mlp", "linear_only_mlp"), ("nonlinear_esn", "linear_fir")):
    diff = np.array([r[left] - r[right] for r in rows])
    boot = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(axis=1)
    summary["models"][f"{left}_minus_{right}"] = {
        "mean_paired_BER_difference": float(diff.mean()),
        "captures_better": int((diff < 0).sum()),
        "captures_worse": int((diff > 0).sum()),
        "captures_tied": int((diff == 0).sum()),
        "paired_capture_bootstrap_95pct": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }

with (folder / "per_seed_capture.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=per_seed[0].keys())
    writer.writeheader()
    writer.writerows(per_seed)
with (folder / "per_capture.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
(folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
