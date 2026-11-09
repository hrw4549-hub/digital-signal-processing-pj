# RC equalizer branch

Own the linear/nonlinear reservoir definitions, state generation, readout fitting, and comparison against symbol-level LMS and linear FIR. Record taps, reservoir size, spectral radius, leak rate, regularization, random seed, and exact training symbols for every result.

The neural branch consumes aligned symbol estimates or states from this work area. Keep the existing MATLAB chain as the reference until a Python implementation is numerically checked against it. Store source and small configuration files here; keep captures, derived arrays, and checkpoints outside Git.
