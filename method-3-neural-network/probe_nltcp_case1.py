"""One-capture NLTCP/PAM4 screening experiment; not a final result."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn


def windows(x, taps):
    half = taps // 2
    return np.lib.stride_tricks.sliding_window_view(np.pad(x, (half, half)), taps)


def ridge(x, y, alpha):
    gram = x.conj().T @ x / len(x)
    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, x.conj().T @ y / len(x))


def decode_pam4(z):
    six = np.clip(np.floor(z / 2 + 3.0), 0, 5).astype(np.int64)
    pam4 = np.zeros_like(six)
    pam4[1:] = np.abs(six[1:] - six[:-1] // 2)
    return pam4


def ber_pair(z, ref_i, ref_q, avp, trim):
    i = decode_pam4((z * avp).real)
    q = decode_pam4((z * avp).imag)
    mask = slice(trim, len(z) - trim)
    lut = np.array([int(v).bit_count() for v in range(8)])
    ei = lut[np.bitwise_xor(i[mask], ref_i[mask])].sum()
    eq = lut[np.bitwise_xor(q[mask], ref_q[mask])].sum()
    bits = 6 * len(i[mask])
    return float((ei + eq) / bits), int(ei), int(eq)


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(x, y, xv, yv, seed, epochs=300):
    torch.manual_seed(seed)
    model = MLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    xval, yval = torch.tensor(xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)
    best, best_loss, patience = None, float("inf"), 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        for start in range(0, len(order), 256):
            ids = order[start : start + 256]
            loss = nn.functional.mse_loss(model(xt[ids]), yt[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = nn.functional.mse_loss(model(xval), yval).item()
        if score < best_loss - 1e-8:
            best_loss, best, patience = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 30:
                break
    model.load_state_dict(best)
    return model, best_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--taps", type=int, default=115)
    args = ap.parse_args()
    torch.set_num_threads(1)
    a = loadmat(args.mat)
    rx = a["datarx1"].ravel().astype(np.complex128)
    target = a["NLTCP_qamdata"].ravel().astype(np.complex128)
    ref_i = a["decimal_PAM4_1"].ravel().astype(np.int64)
    ref_q = a["decimal_PAM4_2"].ravel().astype(np.int64)
    avp = np.sqrt(np.mean(np.abs(target) ** 2))
    target /= avp
    rx /= np.sqrt(np.mean(np.abs(rx[:10000]) ** 2))
    X = windows(rx, args.taps)
    X = np.column_stack((X, np.ones(len(X))))

    train = slice(args.taps // 2, 10000)
    valid = slice(10000, 20000)
    test = slice(20000, len(rx))
    y = target
    wlin = ridge(X[train], y[train], 1e-3)
    linear = X @ wlin

    rng = np.random.default_rng(args.seed)
    nstate = 32
    recurrent = rng.uniform(-1, 1, (nstate, nstate))
    recurrent *= 0.9 / np.max(np.abs(np.linalg.eigvals(recurrent)))
    iq = np.column_stack((X.real, X.imag))
    win = rng.uniform(-1, 1, (nstate, iq.shape[1])) / np.sqrt(iq.shape[1])
    state = np.zeros(nstate)
    states = np.empty((len(X), nstate))
    for i, row in enumerate(iq):
        state = 0.7 * state + 0.3 * np.tanh(recurrent @ state + win @ row)
        states[i] = state
    nonlinear_features = np.column_stack((iq, states, states**2, np.ones(len(X))))
    wnon = ridge(nonlinear_features[train], (y - linear)[train], 1e-2)
    nonlinear = linear + nonlinear_features @ wnon

    av_tr = np.column_stack((linear[train].real, linear[train].imag, nonlinear[train].real, nonlinear[train].imag))
    av_va = np.column_stack((linear[valid].real, linear[valid].imag, nonlinear[valid].real, nonlinear[valid].imag))
    av_te = np.column_stack((linear[test].real, linear[test].imag, nonlinear[test].real, nonlinear[test].imag))
    one_tr, one_va, one_te = av_tr[:, :2], av_va[:, :2], av_te[:, :2]
    y2 = np.column_stack((y.real, y.imag))

    def standardize(tr, va, te):
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd < 1e-8] = 1
        return (tr - mu) / sd, (va - mu) / sd, (te - mu) / sd

    av_tr, av_va, av_te = standardize(av_tr, av_va, av_te)
    one_tr, one_va, one_te = standardize(one_tr, one_va, one_te)
    model_dual, val_dual = train_mlp(av_tr, y2[train], av_va, y2[valid], args.seed)
    model_single, val_single = train_mlp(one_tr, y2[train], one_va, y2[valid], args.seed)
    alpha_grid = np.linspace(0, 1, 101)
    alpha_scores = [ber_pair(alpha * linear[valid] + (1 - alpha) * nonlinear[valid], ref_i[valid], ref_q[valid], avp, 1)[0] for alpha in alpha_grid]
    alpha = float(alpha_grid[int(np.argmin(alpha_scores))])
    weighted = alpha * linear + (1 - alpha) * nonlinear

    feature_train = np.column_stack((linear[train].real, linear[train].imag, nonlinear[train].real, nonlinear[train].imag))
    feature_valid = np.column_stack((linear[valid].real, linear[valid].imag, nonlinear[valid].real, nonlinear[valid].imag))
    feature_test = np.column_stack((linear[test].real, linear[test].imag, nonlinear[test].real, nonlinear[test].imag))
    feature_train, feature_valid, feature_test = standardize(feature_train, feature_valid, feature_test)
    feature_train = np.column_stack((feature_train, np.ones(len(feature_train))))
    feature_valid = np.column_stack((feature_valid, np.ones(len(feature_valid))))
    feature_test = np.column_stack((feature_test, np.ones(len(feature_test))))
    target2 = np.column_stack((y.real, y.imag))
    readout_candidates = []
    for reg in (1e-6, 1e-4, 1e-2, 1.0):
        coeff = ridge(feature_train, target2[train], reg)
        pred_valid = feature_valid @ coeff
        mse = float(np.mean((pred_valid - target2[valid]) ** 2))
        readout_candidates.append((mse, reg, coeff))
    _, selected_reg, feature_coeff = min(readout_candidates, key=lambda item: item[0])
    feature_prediction = feature_test @ feature_coeff
    output_features = feature_prediction[:, 0] + 1j * feature_prediction[:, 1]
    with torch.no_grad():
        pred_dual = model_dual(torch.tensor(av_te, dtype=torch.float32)).numpy()
        pred_single = model_single(torch.tensor(one_te, dtype=torch.float32)).numpy()
    dual = pred_dual[:, 0] + 1j * pred_dual[:, 1]
    single = pred_single[:, 0] + 1j * pred_single[:, 1]
    start = test.start
    capture = int(args.mat.stem.replace("aligned_case", "")) if args.mat.stem.startswith("aligned_case") else 1
    results = {"protocol": {"capture": capture, "seed": args.seed, "taps": args.taps, "train": [0, 9999], "validation": [10000, 19999], "test": [20000, len(rx)-1], "trim": args.taps}, "selection": {"weighted_sum_alpha": alpha, "output_feature_ridge": selected_reg}, "models": {}}
    for name, z in (("linear_fir", linear[test]), ("nonlinear_esn", nonlinear[test]), ("weighted_sum", weighted[test]), ("output_feature_ridge", output_features), ("dual_mlp", dual), ("linear_only_mlp", single)):
        ber, err_i, err_q = ber_pair(z, ref_i[start:], ref_q[start:], avp, args.taps)
        results["models"][name] = {"test_BER": ber, "I_bit_errors": err_i, "Q_bit_errors": err_q}
    results["validation_mse"] = {"output_feature_ridge": min(readout_candidates, key=lambda item: item[0])[0], "dual_mlp": val_dual, "linear_only_mlp": val_single}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
