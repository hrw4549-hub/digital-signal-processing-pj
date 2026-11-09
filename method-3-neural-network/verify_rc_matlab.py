"""Prepare/compare a deterministic MATLAB-Python RC numerical parity case."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from rc_single import fit_readout, predict, reservoir_states


def prepare(path):
    rng = np.random.RandomState(20261109)
    features = rng.normal(size=(100, 257))
    target = rng.normal(size=(2, 257))
    recurrent = rng.uniform(-1, 1, (80, 80))
    recurrent *= 0.9 / np.max(np.abs(np.linalg.eigvals(recurrent)))
    input_weights = rng.uniform(-1, 1, (80, 101))
    savemat(path, {"features": features, "target": target, "recurrent": recurrent,
                   "input_weights": input_weights, "train_stop": 206,
                   "leak": 0.7, "regularization": 1e-5})


def compare(input_path, matlab_path):
    source = loadmat(input_path)
    result = loadmat(matlab_path)
    x = source["features"]
    states = reservoir_states(x.T, source["recurrent"], source["input_weights"], 0.7).T
    design = np.vstack((states, x, np.ones((1, x.shape[1]))))
    readout = fit_readout(design, source["target"], 206, 1e-5)
    predicted = predict(readout, design)
    comparisons = {
        "reservoir_states_max_abs": float(np.max(np.abs(states - result["R"]))),
        "design_matrix_max_abs": float(np.max(np.abs(design - result["O"]))),
        "readout_max_abs": float(np.max(np.abs(readout - result["Wout"]))),
        "prediction_max_abs": float(np.max(np.abs(predicted - result["prediction"]))),
    }
    print(comparisons)
    if max(comparisons.values()) > 1e-8:
        raise SystemExit("MATLAB/Python RC parity tolerance exceeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "compare"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.input)
    else:
        compare(args.input, args.output)
