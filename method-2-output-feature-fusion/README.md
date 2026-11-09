# Method 2: Output-Feature Fusion

Use the linear and nonlinear branch outputs as features, for example `[Re(y_linear), Im(y_linear), Re(y_nonlinear), Im(y_nonlinear)]`, and fit a linear readout to the transmitted symbol I/Q target. This lets the readout learn a global combination while keeping the model smaller and easier to diagnose than an MLP.

Fit feature normalization and readout coefficients on training symbols only. Select regularization on validation data only. Include linear-only and nonlinear-only readouts as ablations, and report held-out BER by capture and seed.

Keep the shared preprocessing, symbol alignment, NLTCP decoder, and BER calculation identical to the other method folders.
