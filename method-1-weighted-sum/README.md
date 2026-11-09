# Method 1: Weighted Sum

Combine the linear and nonlinear RC branch symbol estimates directly:

`y = alpha * y_linear + (1 - alpha) * y_nonlinear`

Start with the existing fixed system weight as the baseline. If tuning `alpha`, choose it from the validation interval only and freeze it before testing. Record the selected value per capture and seed. Compare against both branches alone so a useful branch is not mistaken for a useful fusion.

Keep the shared preprocessing, symbol alignment, NLTCP decoder, and BER calculation identical to the other method folders.
