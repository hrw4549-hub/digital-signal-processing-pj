"""NLTCP/PAM4 exploration: single-RC reproduction and physical dual-channel MLP."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

from rc_single import design_features, fit_readout, make_reservoir, predict


def decode_pam4(values):
    six = np.argmin(np.abs(values[:, None] / 2 + 2.5 - np.arange(6)[None, :]), axis=1)
    decoded = np.zeros_like(six)
    decoded[1:] = np.abs(six[1:] - six[:-1] // 2)
    return decoded


def bit_errors(values, ref_i, ref_q, avp, trim):
    i = decode_pam4((values * avp).real)
    q = decode_pam4((values * avp).imag)
    sl = slice(trim, len(values) - trim)
    lut = np.array([int(k).bit_count() for k in range(8)])
    errors_i = int(lut[np.bitwise_xor(i[sl], ref_i[sl])].sum())
    errors_q = int(lut[np.bitwise_xor(q[sl], ref_q[sl])].sum())
    total_bits = 4 * len(i[sl])
    return errors_i + errors_q, total_bits


class SmallMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        return self.layers(x)


def train_mlp(x_train, y_train, x_valid, y_valid, seed, epochs=500, patience=50):
    torch.manual_seed(seed)
    model = SmallMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    xv = torch.as_tensor(x_valid, dtype=torch.float32)
    yv = torch.as_tensor(y_valid, dtype=torch.float32)
    best_state, best_loss, wait = None, float("inf"), 0
    rng = np.random.default_rng(seed + 1)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        for start in range(0, len(order), 256):
            ids = order[start : start + 256]
            loss = nn.functional.mse_loss(model(xt[ids]), yt[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = nn.functional.mse_loss(model(xv), yv).item()
        if valid_loss < best_loss - 1e-9:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def standardize(train, valid, test):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1
    return (train - mean) / scale, (valid - mean) / scale, (test - mean) / scale


def complex_features(z):
    return np.column_stack((z.real, z.imag))


def eval_candidate(z, labels_i, labels_q, avp, trim, valid_start, test_start):
    val_err, val_bits = bit_errors(z[valid_start:test_start], labels_i[valid_start:test_start], labels_q[valid_start:test_start], avp, trim=0)
    test_err, test_bits = bit_errors(z[test_start:], labels_i[test_start:], labels_q[test_start:], avp, trim=25)
    return val_err / val_bits, test_err / test_bits, test_err, test_bits


def run_capture(path, seed):
    data = loadmat(path)
    waves = [data["datarx1"].reshape(-1), data["datarx2"].reshape(-1)]
    target = data["NLTCP_qamdata"].reshape(-1).astype(np.complex128)
    ref_i = data["decimal_PAM4_1"].reshape(-1).astype(np.int64)
    ref_q = data["decimal_PAM4_2"].reshape(-1).astype(np.int64)
    avp = float(np.sqrt(np.mean(np.abs(target) ** 2)))
    target_scaled = target / avp
    n = len(target)
    train_stop = min(10000, n)
    valid_stop = min(20000, n)
    test_start = valid_stop
    labels = np.vstack((target_scaled.real, target_scaled.imag)).T

    branch_outputs = []
    for channel, wave in enumerate(waves):
        recurrent, input_weights = make_reservoir(seed=seed + channel, input_size=50)
        feature, _ = design_features(wave, recurrent, input_weights, samples_per_symbol=1)
        readout = fit_readout(feature, labels.T, train_stop=train_stop, regularization=1e-5)
        estimate = predict(readout, feature)
        branch_outputs.append(estimate[0] + 1j * estimate[1])

    results = {}
    trim = 25
    for channel, z in enumerate(branch_outputs, start=1):
        val_ber, test_ber, errors, bits = eval_candidate(z, ref_i, ref_q, avp, trim, train_stop, test_start)
        results[f"rc_ch{channel}"] = {"val_ber": val_ber, "test_ber": test_ber, "errors": errors, "bits": bits}

    y_train = labels[:train_stop]
    y_valid = labels[train_stop:valid_stop]
    y_test = labels[test_start:]
    train_pair = np.column_stack([complex_features(branch_outputs[0][:train_stop]), complex_features(branch_outputs[1][:train_stop])])
    valid_pair = np.column_stack([complex_features(branch_outputs[0][train_stop:valid_stop]), complex_features(branch_outputs[1][train_stop:valid_stop])])
    test_pair = np.column_stack([complex_features(branch_outputs[0][test_start:]), complex_features(branch_outputs[1][test_start:])])
    train_pair, valid_pair, test_pair = standardize(train_pair, valid_pair, test_pair)
    model, val_loss = train_mlp(train_pair, y_train, valid_pair, y_valid, seed)
    with torch.no_grad():
        pred_pair = model(torch.as_tensor(test_pair, dtype=torch.float32)).numpy()
    fused = pred_pair[:, 0] + 1j * pred_pair[:, 1]
    val_pred = model(torch.as_tensor(valid_pair, dtype=torch.float32)).detach().numpy()
    val_complex = val_pred[:, 0] + 1j * val_pred[:, 1]
    val_errors, val_bits = bit_errors(val_complex, ref_i[train_stop:valid_stop], ref_q[train_stop:valid_stop], avp, trim=0)
    test_errors, test_bits = bit_errors(fused, ref_i[test_start:], ref_q[test_start:], avp, trim=trim)
    results["dual_channel_mlp"] = {"val_ber": val_errors / val_bits, "test_ber": test_errors / test_bits, "errors": test_errors, "bits": test_bits, "val_mse": val_loss}

    ablation_results = []
    for channel, branch in enumerate(branch_outputs):
        xtr = complex_features(branch[:train_stop])
        xva = complex_features(branch[train_stop:valid_stop])
        xte = complex_features(branch[test_start:])
        xtr, xva, xte = standardize(xtr, xva, xte)
        solo, solo_loss = train_mlp(xtr, y_train, xva, y_valid, seed + 200 + channel)
        with torch.no_grad():
            pred = solo(torch.as_tensor(xte, dtype=torch.float32)).numpy()
        est = pred[:, 0] + 1j * pred[:, 1]
        err, nbits = bit_errors(est, ref_i[test_start:], ref_q[test_start:], avp, trim)
        ablation_results.append({"channel": channel + 1, "test_ber": err / nbits, "errors": err, "bits": nbits, "val_mse": solo_loss})
    best_ablation = min(ablation_results, key=lambda row: row["val_mse"])
    results["linear_output_mlp_ablation"] = best_ablation
    results["protocol"] = {"seed": seed, "train_symbols": train_stop, "validation_symbols": valid_stop - train_stop, "test_symbols": n - test_start, "rc_nodes": 80, "window_symbols": 25, "samples_per_symbol": 1, "leak": 0.7, "spectral_radius": 0.9, "input": "CAP-demodulated symbol-rate complex I/Q"}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 200, 300])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for capture in range(1, 20):
        path = args.data_dir / f"aligned_case{capture:02d}.mat"
        for seed in args.seeds:
            result = run_capture(path, seed)
            result["protocol"]["capture"] = capture
            out = args.output_dir / f"case{capture:02d}_seed{seed}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            row = {"capture": capture, "seed": seed}
            row.update({key: value["test_ber"] for key, value in result.items() if isinstance(value, dict) and "test_ber" in value})
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (args.output_dir / "per_capture_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
