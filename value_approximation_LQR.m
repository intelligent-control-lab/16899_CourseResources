% Value approximation in LQR

x0 = 1;
nx = 1;
nu = 1;
A = 1; B = 0.5;
Q = 1; R = 1;
p.A = A; p.B = B; p.Q = Q; p.R = R;
p.terminal = 0.01; p.nx = nx; p.nu = nu; p.x0 = x0;
% LQR solution
[K, P, CLP] = dlqr(A,B,Q,R);
x = x0; k = 0; k_max = 20;
traj_opt = zeros(1,nx*k_max);
cost = 0;
while ~terminal(x, p) && k < k_max
    u = - K*x;
    [xnew, l] = dynamics(x, u, p);
    cost = cost + l;
    x = xnew;
    k = k+1;
    traj_opt(1,nx*(k-1)+1:nx*k) = x;
end

W_gt = [Q+A'*P*A A'*P*B; B'*P*A B'*P*B+R];
%% Learning paramters
W_init = [4 1; 1 4];
n_ep = 100;
epsilon = 0.1;
delta = 1;
alpha = 1;
k_max = 100;
%% Monte Carlo
W = W_init; % Qudratic parameterization
traj_mc = zeros(n_ep,nx*k_max);
rms_mc = zeros(1,n_ep);
figure; hold on;
for episode = 1:n_ep
    x = x0; k = 0; 
    u = greedy(x, W, epsilon);
    dW = zeros(nx+nu);
    x_list = x;
    u_list = u;
    l_list = [];
    while ~terminal(x, p) && k < k_max
        [x, l] = dynamics(x, u, p);
        u = greedy(x, W, epsilon);
        x_list = [x_list;x];
        u_list = [u_list;u];
        l_list = [l_list;l];
        k = k+1;
        traj_mc(episode,nx*(k-1)+1:nx*k) = x;
    end
    for kk = 1:k
        G = sum(l_list(kk:end));
        x = x_list(kk);
        u = u_list(kk);
        dW = dW + alpha * (G - [x;u]'*W*[x;u]/2) * 0.5*[x;u]*[x;u]';
    end
    plot(0:k,[x0 traj_mc(episode,1:nx*k)],'color',[0.2+0.8*episode/n_ep, 1-0.8*episode/n_ep, 1-0.9*episode/n_ep])
    W = W + dW;
    rms_mc(episode) = norm(W - W_gt);
end
plot(0:10,[x0 traj_opt(1,1:nx*10)],'--k')
box on;
%% SARSA
W = W_init; % Qudratic parameterization
traj_sarsa = zeros(n_ep,nx*k_max);
rms_sarsa = zeros(1,n_ep);
figure; hold on;
for episode = 1:n_ep
    x = x0; k = 0; 
    u = greedy(x, W, epsilon);
    while ~terminal(x, p) && k < k_max
        [xnew, l] = dynamics(x, u, p);
        unew = greedy(xnew, W, epsilon);
        W = W + alpha * (l + delta * [xnew;unew]'*W*[xnew;unew]/2 - [x;u]'*W*[x;u]/2) * 0.5*[x;u]*[x;u]';
        x = xnew;
        u = unew;
        k = k+1;
        traj_sarsa(episode,nx*(k-1)+1:nx*k) = x;
    end
    plot(0:k,[x0 traj_sarsa(episode,1:nx*k)],'color',[0.2+0.8*episode/n_ep, 1-0.8*episode/n_ep, 1-0.9*episode/n_ep])
    rms_sarsa(episode) = norm(W - W_gt);
end
plot(0:10,[x0 traj_opt(1,1:nx*10)],'--k')
box on;
%% Q-Learning
W = W_init; % Qudratic parameterization
traj_q = zeros(n_ep,nx*k_max);
rms_q = zeros(1,n_ep);
figure; hold on;
for episode = 1:n_ep
    x = x0; k = 0;
    while ~terminal(x, p) && k < k_max
        u = greedy(x, W, epsilon);
        xnew = dynamics(x, u, p);
        W = W + alpha * (l + delta * x'*min_Q(W,p)*x/2 - [x;u]'*W*[x;u]/2) * 0.5*[x;u]*[x;u]';
        x = xnew;
        u = unew;
        k = k+1;
        traj_q(episode,nx*(k-1)+1:nx*k) = x;
    end
    plot(0:k,[x0 traj_q(episode,1:nx*k)],'color',[0.2+0.8*episode/n_ep, 1-0.8*episode/n_ep, 1-0.9*episode/n_ep])
    rms_q(episode) = norm(W - W_gt);
    %epsilon = epsilon * (1-episode/n_ep);
end
plot(0:10,[x0 traj_opt(1,1:nx*10)],'--k')
box on;
%% Visualize SARSA Q-Learning
figure;
hold on
plot(rms_sarsa)
plot(rms_q)
plot(rms_mc)
legend("SARSA", "Q-Learning","Monte Carlo")
box on;
%% Functions
function V = min_Q(W,p)
nx = p.nx;
W11= W(1:nx,1:nx);
W12= W(1:nx,nx+1:end);
W21= W(nx+1:end, 1:nx);
W22= W(nx+1:end, nx+1:end);
V = W11 - W12 * inv(W22) * W21;
end
function [xnew, cost] = dynamics(x,u,p)
cost = (x'*p.Q*x + u'*p.R*u)/2;
xnew = p.A*x + p.B*u;
end

function flag = terminal(x,p)
if norm(x) < p.terminal
    flag = true;
else
    flag = false;
end
end

function u = greedy(x, W, epsilon)
sample = rand(1);
nx = length(x);
nu = size(W,1)-nx;
u = - inv(W(nx+1:end, nx+1:end))*(W(nx+1:end, 1:nx) + W(1:nx, nx+1:end))/2*x;
if sample < epsilon
    u = -rand(nu)*x;
end
end