"""Summarize per-capture RC/MLP scores against the MATLAB symbol-LMS baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from run_nltcp_nn import bit_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result_path in sorted(args.run_dir.glob("case*_seed*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        capture = int(result["protocol"]["capture"])
        seed = int(result["protocol"]["seed"])
        data = loadmat(args.data_dir / f"aligned_case{capture:02d}.mat")
        target = data["NLTCP_qamdata"].reshape(-1)
        avp = float(np.sqrt(np.mean(np.abs(target) ** 2)))
        ref_i = data["decimal_PAM4_1"].reshape(-1).astype(np.int64)
        ref_q = data["decimal_PAM4_2"].reshape(-1).astype(np.int64)
        test_start = result["protocol"]["train_symbols"] + result["protocol"]["validation_symbols"]
        errors, _ = bit_errors(data["symbol1"].reshape(-1)[test_start:],
                               ref_i[test_start:], ref_q[test_start:], avp, trim=25)
        _, bits = bit_errors(data["symbol1"].reshape(-1)[test_start:],
                             ref_i[test_start:], ref_q[test_start:], avp, trim=25)
        denominator_scale = 1.5
        rc = [result["rc_ch1"], result["rc_ch2"]]
        selected = min(rc, key=lambda item: item["val_ber"])
        best_rc = selected["errors"] / (selected["bits"] / denominator_scale)
        mlp = result["dual_channel_mlp"]["errors"] / (result["dual_channel_mlp"]["bits"] / denominator_scale)
        solo = result["linear_output_mlp_ablation"]["errors"] / (result["linear_output_mlp_ablation"]["bits"] / denominator_scale)
        lms = errors / bits
        rows.append({"capture": capture, "seed": seed, "best_single_rc_test_ber": best_rc,
                     "dual_mlp_test_ber": mlp, "single_output_mlp_test_ber": solo,
                     "symbol_lms_test_ber": lms,
                     "mlp_vs_single_rc_reduction_pct": 100 * (best_rc - mlp) / best_rc,
                     "mlp_vs_symbol_lms_reduction_pct": 100 * (lms - mlp) / lms})

    csv_path = args.output_dir / "per_capture_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    names = ["best_single_rc_test_ber", "dual_mlp_test_ber", "single_output_mlp_test_ber", "symbol_lms_test_ber"]
    summary = {name: float(np.mean([row[name] for row in rows])) for name in names}
    summary.update({
        "capture_count": len(rows),
        "seed_count": len(set(row["seed"] for row in rows)),
        "mean_paired_mlp_vs_single_rc_reduction_pct": float(np.mean([row["mlp_vs_single_rc_reduction_pct"] for row in rows])),
        "mean_paired_dual_mlp_vs_single_output_mlp_reduction_pct": float(np.mean([
            100 * (row["single_output_mlp_test_ber"] - row["dual_mlp_test_ber"]) / row["single_output_mlp_test_ber"] for row in rows
        ])),
        "mean_paired_mlp_vs_symbol_lms_reduction_pct": float(np.mean([row["mlp_vs_symbol_lms_reduction_pct"] for row in rows])),
        "captures_mlp_beats_single_rc": sum(row["mlp_vs_single_rc_reduction_pct"] > 0 for row in rows),
        "runs_dual_mlp_beats_single_output_mlp": sum(row["single_output_mlp_test_ber"] > row["dual_mlp_test_ber"] for row in rows),
        "captures_mlp_beats_symbol_lms": sum(row["mlp_vs_symbol_lms_reduction_pct"] > 0 for row in rows),
        "warning": "Exploratory historical captures; one seed is not a final generalization estimate."
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
