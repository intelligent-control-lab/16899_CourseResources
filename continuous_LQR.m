% Continuous Model
A = [0 1; 0 0];
B = [0;1];
Q = [1 0; 0 0];
R = 0.1;
x0 = [1;1];
horizon = 10;
[K, P] = lqr(A, B, Q, R);
dyn = @(t,x) (A-B*K)*x;
[t, x] = ode45(dyn, [0,10],x0);
figure;hold on;
plot(x(:,1), x(:,2),'k')
%%
% Discrete Model
T = 0.01; N = horizon/T;
Ad = [1 T; 0 1];
Bd = [T^2/2; T];

[Kd, Pd] = dlqr(Ad, Bd, Q, R);
xd = zeros(2, N+1);
xd(:,1) = x0;
for k = 2:N
    xd(:,k) = (Ad - Bd*Kd)*xd(:,k-1);
end
hold on
plot(xd(1,:), xd(2,:),'r')