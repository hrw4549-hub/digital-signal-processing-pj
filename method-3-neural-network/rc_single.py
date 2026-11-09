"""按报告中的助教单支 RC 参数实现波形到符号的 Python 版本。"""
from __future__ import annotations

import numpy as np


def build_symbol_windows(waveform, samples_per_symbol=4, taps=25):
    """以每个符号起点为中心，取 taps 个符号的波形采样。"""
    x = np.asarray(waveform).reshape(-1)
    half_left = (taps // 2) * samples_per_symbol
    width = taps * samples_per_symbol
    centers = np.arange(len(x) // samples_per_symbol) * samples_per_symbol
    padded = np.pad(x, (half_left, width - half_left - samples_per_symbol))
    offsets = centers[:, None] + np.arange(width)[None, :]
    windows = padded[offsets]
    return np.column_stack((windows.real, windows.imag))


def make_reservoir(size=80, input_size=200, seed=100, spectral_radius=0.9):
    """生成固定储备池；Fortran 顺序对应 MATLAB rand(M,N) 填充顺序。"""
    rng = np.random.RandomState(seed)
    recurrent = rng.rand(size * size).reshape((size, size), order="F").astype(np.float64) * 2 - 1
    input_weights = rng.rand(size * (input_size + 1)).reshape((size, input_size + 1), order="F").astype(np.float64) * 2 - 1
    eigen_radius = np.max(np.abs(np.linalg.eigvals(recurrent)))
    recurrent *= spectral_radius / eigen_radius
    return recurrent, input_weights


def reservoir_states(windows, recurrent, input_weights, leak=0.7):
    x = np.asarray(windows, dtype=np.float64)
    states = np.zeros((len(x), recurrent.shape[0]), dtype=np.float64)
    state = np.zeros(recurrent.shape[0], dtype=np.float64)
    for i, row in enumerate(x):
        drive = recurrent @ state + input_weights @ np.r_[row, 1.0]
        state = (1 - leak) * state + leak * np.tanh(drive)
        states[i] = state
    return states


def design_features(waveform, recurrent, input_weights, taps=25, samples_per_symbol=4, leak=0.7):
    normalized = np.asarray(waveform).reshape(-1)
    rms = np.sqrt(np.mean(np.abs(normalized) ** 2))
    if rms == 0:
        raise ValueError("输入波形 RMS 为零")
    normalized = normalized / rms
    windows = build_symbol_windows(normalized, samples_per_symbol, taps)
    states = reservoir_states(windows, recurrent, input_weights, leak)
    features = np.vstack((states.T, windows.T, np.ones((1, len(windows)))))
    return features.astype(np.float64, copy=False), rms


def fit_readout(features, target, train_stop=4096, regularization=1e-5):
    """岭回归，和 Wout = Y O' (O O' + lambda I)^-1 等价。"""
    x = np.asarray(features[:, :train_stop], dtype=np.float64)
    y = np.asarray(target[:, :train_stop], dtype=np.float64)
    gram = x @ x.T
    gram.flat[:: len(gram) + 1] += regularization
    return np.linalg.solve(gram, x @ y.T).T


def predict(readout, features):
    return np.asarray(readout) @ np.asarray(features)
