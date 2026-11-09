# Method 3: Neural-Network Fusion

Use a compact MLP to fuse the two branch outputs, preferably as a residual correction around the stronger single branch:

`y = y_linear + delta([y_linear, y_nonlinear])`

Compare it with an equal-capacity linear-only MLP residual ablation. The key question is whether the nonlinear branch contributes complementary information beyond what a nonlinear mapping of the linear output already provides. Normalize from training data only; choose architecture, early stopping, and correction strength on validation data only.

The [local M6 ablation](M6-ablation-local.md) checks a related historical 64APSK package. It found only a small, uncertain residual gain and is not direct validation of this project's NLTCP/PAM4 chain. A separate [19-capture NLTCP/PAM4 screen](NLTCP-case-screening.md) now compares the three assigned methods and neural ablations on the supplied project captures. It uses historical data and is exploratory, not a final held-out result.

Use the shared preprocessing, symbol alignment, NLTCP decoder, and BER calculation documented at the repository root. Do not claim a neural-branch gain unless it beats the linear-only ablation on held-out captures.
