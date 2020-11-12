% Policy gradient for LQR
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
%% Learning paramters
n_ep = 200;
delta = 1;
alpha = 0.01; %policy
alpha_baseline = 1; %value
k_max = 10;
W_init = 3;
%% REINFORCE
[mu, sigma] = init(nx,nu);
traj_reinforce = zeros(n_ep,nx*k_max);
rms_reinforce = zeros(1,n_ep);
std_reinforce = zeros(1,n_ep);
figure(1);clf;subplot(311);hold on;
for episode = 1:n_ep
    x = x0; k = 0; 
    u = mu*x + randn * sigma*x;
    dtheta = zeros(nx*2, nu);
    x_list = x;
    u_list = u;
    l_list = [];
    while ~terminal(x, p) && k < k_max
        [x, l] = dynamics(x, u, p);
        u = mu*x + randn * sigma*x;
        x_list = [x_list;x];
        u_list = [u_list;u];
        l_list = [l_list;l];
        k = k+1;
        traj_reinforce(episode,nx*(k-1)+1:nx*k) = x;
    end
    for kk = 1:k
        G = sum(l_list(kk:end));
        x = x_list(kk);
        u = u_list(kk);
        D = policy_gradient_Gaussian(mu, sigma, x, u);
        dtheta = dtheta - alpha * G * D;
    end
    plot(0:k,[x0 traj_reinforce(episode,1:nx*k)],'color',[0.2+0.8*episode/n_ep, 1-0.8*episode/n_ep, 1-0.9*episode/n_ep])
    mu = mu + dtheta(1:nx, :);
    sigma = sigma + dtheta(nx+1:end,:);
    rms_reinforce(episode) = norm(mu + K);
    std_reinforce(episode) = sigma;
end
plot(0:10,[x0 traj_opt(1,1:nx*10)],'--k')
title("REINFORCE")
box on;
%% REINFORCE with Baseline
[mu, sigma] = init(nx,nu);
traj_reinforce_baseline = zeros(n_ep,nx*k_max);
rms_reinforce_baseline = zeros(1,n_ep);
std_reinforce_baseline = zeros(1,n_ep);
W = W_init;
subplot(312);hold on;
for episode = 1:n_ep
    x = x0; k = 0; 
    u = mu*x + randn * sigma*x;
    dtheta = zeros(nx*2, nu);
    dW = zeros(nx);
    x_list = x;
    u_list = u;
    l_list = [];
    while ~terminal(x, p) && k < k_max
        [x, l] = dynamics(x, u, p);
        u = mu*x + randn * sigma*x;
        x_list = [x_list;x];
        u_list = [u_list;u];
        l_list = [l_list;l];
        k = k+1;
        traj_reinforce_baseline(episode,nx*(k-1)+1:nx*k) = x;
    end
    for kk = 1:k
        G = sum(l_list(kk:end));
        x = x_list(kk);
        u = u_list(kk);
        D = policy_gradient_Gaussian(mu, sigma, x, u);
        d = G - x'*W*x/2;
        dtheta = dtheta - alpha * d * D;
        dW = dW + alpha_baseline * d * 0.5* x^2;
    end
    plot(0:k,[x0 traj_reinforce_baseline(episode,1:nx*k)],'color',[0.2+0.8*episode/n_ep, 1-0.8*episode/n_ep, 1-0.9*episode/n_ep])
    mu = mu + dtheta(1:nx, :);
    sigma = sigma + dtheta(nx+1:end,:);
    W = W + dW;
    rms_reinforce_baseline(episode) = norm(mu + K);
    std_reinforce_baseline(episode) = sigma;
end
plot(0:10,[x0 traj_opt(1,1:nx*10)],'--k')
title("REINFORCE with Baseline")
box on;
%% Actor Critic
[mu, sigma] = init(nx,nu);
traj_ac = zeros(n_ep,nx*k_max);
rms_ac = zeros(1,n_ep);
std_ac = zeros(1,n_ep);
W = W_init;
subplot(313);hold on;
for episode = 1:n_ep
    x = x0; k = 0; 
    while ~terminal(x, p) && k < k_max
        u = mu*x + randn * sigma*x;
        [xnew, l] = dynamics(x, u, p);
        d = l + xnew'*W*xnew/2 - x'*W*x/2;
        D = policy_gradient_Gaussian(mu, sigma, x, u);
        mu = mu - alpha * d * D(1:nx, :);
        sigma = sigma - alpha * d * D(nx+1:end,:);
        W = W + alpha_baseline * d * 0.5*x^2;
        k = k+1;
        x = xnew;
        traj_ac(episode,nx*(k-1)+1:nx*k) = x;
    end
    plot(0:k,[x0 traj_ac(episode,1:nx*k)],'color',[0.2+0.8*episode/n_ep, 1-0.8*episode/n_ep, 1-0.9*episode/n_ep])
    rms_ac(episode) = norm(mu + K);
    std_ac(episode) = sigma;
end
plot(0:10,[x0 traj_opt(1,1:nx*10)],'--k')
title("Actor Critic")
box on;
%%
figure(2);clf;hold on
plot(rms_reinforce,'b');
plot(rms_reinforce_baseline,'r');
plot(rms_ac,'k')
% plot(rms_reinforce + std_reinforce,'--b');
% plot(rms_reinforce - std_reinforce,'--b');
% plot(rms_reinforce_baseline + std_reinforce_baseline,'--r');
% plot(rms_reinforce_baseline - std_reinforce_baseline,'--r');
% plot(rms_ac + std_ac,'--k');
% plot(rms_ac - std_ac,'--k');
legend("REINFORCE", "REINFORCE with Baseline", "Actor Critic")
box on;
%% Functions
function D = policy_gradient_Gaussian(mu, sigma, x, u)
    D = [(u-mu*x)/(sigma*x)^2*x;
        -x + (u-mu*x)^2/(sigma*x)^2*x];
end

function flag = terminal(x,p)
if norm(x) < p.terminal
    flag = true;
else
    flag = false;
end
end

function [xnew, cost] = dynamics(x,u,p)
cost = (x'*p.Q*x + u'*p.R*u)/2;
xnew = p.A*x + p.B*u;
end

function [mu, sigma] = init(nx,nu)
    mu = -0.1*eye(nu,nx);
    sigma = 0.1*eye(nu,nx);
end