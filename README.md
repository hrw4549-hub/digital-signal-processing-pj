# Digital Signal Processing Project

Project: high-speed visible-light communication using NLTCP coding and a dual-branch reservoir-computing (RC) equalizer.

## Three method branches

- [`method-1-weighted-sum/`](method-1-weighted-sum/README.md): directly combine the linear and nonlinear branch outputs with fixed or validation-selected weights.
- [`method-2-output-feature-fusion/`](method-2-output-feature-fusion/README.md): use both branch outputs as features for a learned linear/readout fusion.
- [`method-3-neural-network/`](method-3-neural-network/README.md): learn a neural fusion or residual correction from the branch outputs.

All three methods use the same receiver inputs, symbol alignment, train/validation/test split, NLTCP decoder, and BER scoring convention. Keep raw captures, derived signal arrays, model checkpoints, and meeting notes out of Git. See `.gitignore` before adding experiment outputs.
