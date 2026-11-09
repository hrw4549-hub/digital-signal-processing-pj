# NLTCP/PAM4 19-Capture Screening

Status: local exploratory result in `mywork`; not pushed to GitHub. The numerical result comes from the supplied historical captures and three reservoir/network seeds. Treat it as a route-selection screen, not a final performance claim.

## Data path and scoring check

The supplied MATLAB receiver script was run from an isolated working copy, not from the source data directory. For each of 19 captures it performed the existing 33/43-tap waveform Volterra-LMS, CAP demodulation, and 115-tap symbol LMS chain, then saved the CAP symbol stream, NLTCP reference symbols, and PAM4 component labels. MATLAB's original PBS BER was reproduced capture by capture from the resulting `berall.txt` output. The Python BER routine was checked against MATLAB `biterr` on capture 1: I/Q bit errors were exactly 193/160 and the average BER was 1.1543e-3.

## Screening protocol

- Captures: 1-19; seeds 100, 101, and 102 for the fixed ESN and neural network.
- Train: symbols 1-10000; validation: 10001-20000; test: 20001-51200.
- Input: CAP-demodulated symbol-rate stream from CH1 after the supplied waveform Volterra-LMS stage; 115-symbol centered input window.
- Target: transmitted complex NLTCP symbols, scaled by the known reference RMS.
- Test score: existing NLTCP differential decode, PAM4 I/Q bit errors, average BER; trim 115 symbols at the ends.
- Validation only: Method 1 weight searches 101 values from 0 to 1 by decoded BER. Method 2 selects ridge strength from `{1e-6, 1e-4, 1e-2, 1}` by validation I/Q MSE after training-only feature standardization. MLP uses validation MSE for early stopping (patience 30).
- Method 1: `alpha * linear_FIR + (1-alpha) * nonlinear_ESN`.
- Method 2: standardized `[I/Q linear output, I/Q nonlinear output]` features with a linear ridge readout.
- Method 3: two-layer 32-32 MLP on both branch outputs. The ablation is a same-width MLP using only the linear output.
- Nonlinear ESN: 32 tanh states, 115-symbol I/Q windows as joint input, spectral radius 0.9 and leak 0.3; quadratic state features plus direct I/Q window; readout fits the linear-branch residual.

## Results

Mean BER across 57 capture/seed test runs:

| Candidate | Mean BER |
|---|---:|
| Linear 115-tap FIR | 7.0725e-4 |
| Nonlinear ESN branch | 7.6229e-4 |
| Method 1 validation-weighted sum | 7.2472e-4 |
| Method 2 output-feature ridge | 7.2802e-4 |
| Method 3 dual-output MLP | 7.5738e-4 |
| Method 3 linear-only MLP ablation | 7.4530e-4 |

Metrics and paired intervals first average the three seeds within each capture, then bootstrap the 19 captures. The tanh ESN was worse than linear FIR in 17 captures and better in 2. Its paired mean BER difference was +5.50e-5, with a capture-bootstrap 95% interval `[3.74e-5, 7.18e-5]`.

Method 1's validation-selected weighted sum was worse than linear FIR in 13 captures, better in 4, and tied in 2. Its paired BER difference was +1.75e-5, with interval `[4.15e-6, 3.20e-5]`.

Method 2 output-feature ridge was worse than linear FIR in 13 captures and better in 6. Its paired difference was +2.08e-5, with interval `[5.85e-6, 3.46e-5]`.

Method 3 dual-output MLP was worse than the linear-only MLP in 12 captures, better in 6, and tied in 1. Its paired difference was +1.21e-5, with interval `[-8.69e-6, 3.35e-5]`, which includes zero. Its mean BER was also above the linear FIR baseline. The validation MSE often slightly preferred the linear-only MLP, so current evidence does not show a reliable nonlinear-branch contribution.

## Interpretation and limitations

The most consistent observation is that this 32-node tanh ESN branch is not useful by itself under the tested 115-symbol-window setup. Validation-selected direct weighting usually falls back to the linear endpoint. Output-feature ridge is sensitive to branch correlation and regularization. The dual MLP sometimes exploits the nonlinear output, but it does not do so consistently enough to establish an incremental gain.

All 19 captures are historical captures already used in earlier receiver work. The capture bootstrap describes variation among these 19 records; it does not turn them into a new independent lockbox. Centered input windows can use unlabeled neighboring received samples across split boundaries. Normalization is fitted from the training prefix, but the current reference and original receiver conventions should still be checked by the group before treating the score as a final project BER.

The current screen is one-channel after waveform Volterra-LMS. It does not yet evaluate joint CH1/CH3 RC output fusion, which may be more promising because the two optical receive branches are the system's natural diversity inputs. The next experiment should compare independent per-channel linear estimates with a joint-channel linear baseline first, then add a small neural reliability/gating model over the two channel outputs. Evaluate on held-out captures or a new measurement batch before making a performance claim.

## Files

- `probe_nltcp_case1.py`: one-capture train/validation/test implementation; reusable for every aligned MATLAB `.mat` capture.
- `summarize_nltcp_probes.py`: per-capture paired means and bootstrap intervals.
- `case_results/`: per-capture model scores and selected validation settings.
- `case_results/per_capture.csv`: compact comparison table.
- `case_results/per_seed_capture.csv`: all 57 capture/seed records.
- `case_results/summary.json`: aggregate results.
