# Neural-network branch

Scope: replace or augment the symbol-level equalizer/fusion stage in the NLTCP receiver. The project signal chain to preserve is waveform LMS/Volterra -> CAP demodulation -> symbol equalization -> NLTCP decode. Report both PAM4 component BER and the project's agreed aggregate BER; do not compare against 64APSK results as if they were the same task.

## Evidence reviewed

- The supplied `dual_branch_nn` feasibility experiment uses 64APSK `afterLMS41/42/48`, 25-symbol windows, M=80 reservoirs, and a 4096/1024/remaining train/validation/test split. Its results show `nn_linear_only` beats `nn_fuse_outputs` in all five reported capture/seed settings. The tanh-only branch is also substantially worse than the linear-only model. This suggests that the tested tanh reservoir adds little useful complementary information under that setup.
- The supplied M6 report uses 19 historical captures, training symbols 1-10000, validation 10001-20000, and test 20001-51200. It reports that joint linear FIR has lower mean test BER than its selected dual-RC BiLSTM candidate (7.7598e-5 vs 8.1534e-5). The report itself says this does not establish an independent gain from the nonlinear RC or neural fusion. These are retrospective captures, not a new blind test.
- The available NN data folder contains MATLAB scripts and raw captures, but not the `Data/Data/afterLMS*` inputs referenced by the feasibility script. Therefore those published NN results could not be independently rerun from the supplied NN folder alone. The 64APSK and M6 experiments are useful design evidence, not direct NLTCP/PAM4 validation.

## Proposed experiment

Use the same input captures, waveform preprocessing, symbol alignment, label range, training/validation/test indices, and NLTCP decoder for every candidate. Keep the test interval untouched until model and hyperparameters are fixed on validation data. First establish the MATLAB LMS/Volterra plus symbol-LMS baseline and a linear FIR/RC baseline. Then compare, at matched training symbols and approximately matched parameter counts:

1. Linear branch only.
2. Nonlinear branch only.
3. Fixed-weight linear/nonlinear fusion.
4. Small MLP residual fusion: `y = y_linear + delta([y_linear, y_nonlinear])`.
5. Linear-only MLP residual ablation with the same width and training budget.

Fit normalization on training data only. Select early stopping and fusion strength on validation only. Repeat across capture conditions and reservoir/network seeds, record BER/SER and confidence intervals by capture, and retain negative results. The nonlinear branch earns its complexity only if the fused model reliably beats the linear-only ablation on held-out captures, not merely one seed or the receiver's overall baseline.

## Current status

Evidence review and experiment protocol are recorded; no new NLTCP neural BER claim is made because the aligned `afterLMS` inputs and a runnable project-specific dataset manifest are not present in the supplied NN data directory. Add scripts and compact result tables here once the team confirms those inputs and the shared evaluation convention.
