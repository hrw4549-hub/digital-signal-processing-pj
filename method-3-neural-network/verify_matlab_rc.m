function verify_matlab_rc(inputMat, outputMat)
% Independent MATLAB calculation used to check the Python RC matrix math.
S = load(inputMat);
X = S.features;
Y = S.target;
A = S.recurrent;
Win = S.input_weights;
trainStop = S.train_stop;
leak = S.leak;
lambda = S.regularization;
M = size(A, 1);
N = size(X, 2);
R = zeros(M, N);
r = zeros(M, 1);
for n = 1:N
    r = (1-leak)*r + leak*tanh(A*r + Win*[X(:,n); 1]);
    R(:,n) = r;
end
O = [R; X; ones(1,N)];
Wout = Y(:,1:trainStop)*O(:,1:trainStop)' / ...
    (O(:,1:trainStop)*O(:,1:trainStop)' + lambda*eye(size(O,1)));
prediction = Wout*O;
save(outputMat, 'R', 'O', 'Wout', 'prediction', '-v7');
end
