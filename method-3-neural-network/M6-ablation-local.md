# Local M6 Ablation Study

Status: exploratory local analysis; not pushed. This reproduces and probes the supplied historical M6 package. It is not a new NLTCP/PAM4 result.

## Data and protocol

- Source: the supplied standalone M6 reproducibility archive, with 19 historical CH1/CH3 captures and seeds 100-104.
- Labels: the package's 64APSK reference and historical six-bit scoring convention.
- Test interval: symbols 20001-51200 (187200 scored bits per record).
- PBS weight: the original validation-selected weight for each seed/fold, held fixed for all three candidates below.
- Neural output: package-frozen M6 checkpoints and original validation-selected checkpoint and branch correction strengths. No test result was used to tune a parameter.
- Validation correction-strength selection: all 25 seed/fold combinations chose checkpoint 4 (the first saved checkpoint); CH1 correction was zero in 11/25 choices, while CH3 correction was zero in 18/25 and 1.5 in 7/25. This shows the nonlinear/neural correction was not consistently selected across channels.
- RC outputs: recomputed from the package's frozen CAP and RC coefficients. The standalone M6 inference outputs matched the archived outputs within 9.7e-7 maximum absolute difference. The package integrity check passed for all 430 manifested files, and the official scorer reproduced all 95 error counts.

The analysis script and compact per-record results are stored alongside this note. The script regenerates the frozen RC branch endpoints, checks the M6 output error counts against the archive, and writes a paired comparison without changing validation-selected settings.

## Paired results

Mean across the 95 capture/seed records:

| Candidate | Mean BER |
|---|---:|
| Linear RC outputs | 8.19276e-5 |
| Nonlinear RC outputs | 8.17027e-5 |
| M6 neural residual outputs | 8.15340e-5 |

The neural residual minus linear-output endpoint averaged -3.94e-7 BER across the 19 capture-level means. It improved 5 captures, worsened 2, and tied 12. A capture-level paired bootstrap 95% interval was [-1.07e-6, 1.69e-7], which includes zero. Against the nonlinear RC endpoint, the mean difference was -1.69e-7; it improved 9 captures, worsened 8, and tied 2, with a 95% interval [-3.71e-6, 3.09e-6].

Interpretation: in this historical M6 setup, the learned residual makes a very small numerical change over the linear endpoint, with no clear capture-level evidence of a reliable gain. Direct nonlinear RC output is similarly close. These data do not support presenting neural fusion or the nonlinear branch as the main source of the reported end-to-end BER improvement. The package report's stronger comparison is joint linear reception/FIR against the old independent receiver.

## Scope and caveats

- This comparison isolates the output correction on top of the package's joint-input linear RC. It does not retrain an M6 without the nonlinear branch, does not compare matched-compute alternatives, and does not establish causal contribution under a new lockbox.
- The package is historical 64APSK data, not the team's NLTCP/PAM4 experiment. Its output corrections and PBS are already selected by the published validation protocol, and its historical test interval has been used during prior development.
- The available NN-data folder does not contain the `afterLMS` arrays expected by the supplied 64APSK feasibility script. The provided MATLAB reference does give a route to generate the project-specific signal stages, but the NLTCP inputs, alignment, training split, and scored BER convention still need to be made explicit before porting this M6 architecture.

## Next experiment for our branch

1. Generate and freeze aligned symbol-rate arrays from the same NLTCP/PAM4 captures: Volterra-LMS waveform output, CAP-demodulated samples, transmitted reference symbols/bits, and the symbol-LMS baseline output. Save a manifest with capture IDs, channel names, sample/symbol offsets, edge trimming, normalization, and checksum.
2. Agree on one chronological train/validation/test split per capture and one NLTCP decoder/BER routine. Fit all normalization and RC/MLP parameters on train only; select checkpoints and fusion strength on validation only.
3. Compare linear RC, nonlinear RC, fixed fusion, linear-only MLP residual, and dual-input MLP residual under the same features, symbols, seeds, and parameter budget. Report the incremental dual-input residual against the linear-only MLP ablation, not just against LMS.
4. Keep an untouched capture or newly acquired measurement set for the final decision. Retain per-capture results and paired uncertainty; do not select architecture using that final set.

The immediate blocker for a project-specific experiment is producing and freezing the aligned NLTCP/PAM4 symbol arrays and evaluation contract. Once those exist, the MLP output-fusion ablation is the smallest useful first experiment; copying the larger BiLSTM M6 model is not justified by this ablation.
