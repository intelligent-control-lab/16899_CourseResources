% Example for Kalman Filter
% Setting up the dynamics
% x' = x + u + w;
% y = x + v
A = 0.5;
B = 1;
Bw = 1;
C = 1;
x0 = 0;
N = 101;
u = sin((1:N)./(N/2/pi));
x = zeros(1,N);x(1) = x0;
X0 = 1;
W = 0.01;
V = 0.0001;
xhat = zeros(1,N);xhat(1) = x0 + randn * sqrt(X0);%A posteriori
Z = zeros(1,N);Z(1) = X0;
M = zeros(1,N);

%% KF
for k = 1:N-1
    % Dynamics
    x(k+1) = A*x(k) + B*u(k)+ Bw * randn * sqrt(W);
    y(k+1) = C*x(k+1) + randn * sqrt(V);
    % Dynamic update
    x_pri = A*xhat(k) + B*u(k);
    M(k+1) = A*Z(k)*A' + Bw*W*Bw';
    % Measurement update
    KF_gain = M(k+1)*C'*inv(V+C*M(k+1)*C');
    error = y(k+1) - C*x_pri;
    xhat(k+1) = x_pri + KF_gain * error;
    Z(k+1) = M(k+1) - M(k+1)*C'*inv(V+C*M(k+1)*C')*C*M(k+1);
end
measurement_std = norm(y-x)
estimate_std = norm(xhat-x)
figure(1);clf;hold on;
plot(x,'k')
plot(y,'r')
plot(xhat,'b')
legend("Ground truth x", "Measurement y", "A posteriori estimate xhat")
figure(2);clf;hold on;
plot(Z)
plot(M)
legend("A posteriori variance","A priori variance")
%% Steady state KF 
Ms = dare(A', C', Bw*W*Bw', V)
Fs = Ms*C'*inv(V+C*Ms*C')
