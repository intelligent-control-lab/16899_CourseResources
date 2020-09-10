T = 1;
A = [1 T; 0 1];
B = [T^2/2; T];
Q = [1 0; 0 0];
R = 1;

N = 10;
P = cell(N);
P{N} = [1 0;0 1];
p11 = zeros(1,N);p11(N) = P{N}(1,1);
p12 = zeros(1,N);p12(N) = P{N}(1,2);
p21 = zeros(1,N);p21(N) = P{N}(2,1);
p22 = zeros(1,N);p22(N) = P{N}(2,2);
%% Solving Riccati Equation
for i = N-1:-1:1
    P{i} = Q + A' * P{i+1} * A - A' * P{i+1} * B * inv(R + B' * P{i+1} * B) * B' * P{i+1} * A;
    p11(i) = P{i}(1,1);
    p12(i) = P{i}(1,2);
    p21(i) = P{i}(2,1);
    p22(i) = P{i}(2,2);
end
control_gain = inv(R + B' * P{1} * B) * B' * P{1} * A
%% Plot
figure; hold on;
plot(p11)
plot(p12)
plot(p21)
plot(p22)
legend("p_{11}", "p_{12}", "p_{21}", "p_{22}")
%% LQR
[k, p] = dlqr(A, B, Q, R);