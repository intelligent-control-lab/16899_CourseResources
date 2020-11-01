% Simple MDP example
% Contains implementation of 
%     Value Iteration
%     Policy Iteration
%     Tabular SARSA
%     Tabular Q-Learning

map = [ 8 9 10 11; ...
        5 -1 6  7; ...
        1 2 3  4];

nx = 11;
nu = 4;
terminal = [0 0 0 0 0 0 1 0 0 0 -1];
action = [1,2,3,4]; % left, up, down, right
d = zeros(nx,nu,nx);
imax = 3;
jmax = 4;

% Dynamics
for x = 1:nx
    if terminal(x) == 0
        [i,j] = find(map==x);
        if x == 6
            dd = 1;
        end
        % left
        xx = rec(x, map(i, max(j-1,1)));
        d(x, 1, xx) = d(x, 1, xx) + 0.8;
        xx = rec(x, map(max(i-1,1), j));
        d(x, 1, xx) = d(x, 1, xx) + 0.1;
        xx = rec(x, map(min(i+1,3), j));
        d(x, 1, xx) = d(x, 1, xx) + 0.1;
        % up
        xx = rec(x, map(max(i-1,1), j));
        d(x, 2, xx) = d(x, 2, xx) + 0.8;
        xx = rec(x, map(i, max(j-1,1)));
        d(x, 2, xx) = d(x, 2, xx) + 0.1;
        xx = rec(x, map(i, min(j+1,jmax)));
        d(x, 2, xx) = d(x, 2, xx) + 0.1;
        % down
        xx = rec(x, map(min(i+1,imax), j));
        d(x, 3, xx) = d(x, 3, xx) + 0.8;
        xx = rec(x, map(i, max(j-1,1)));
        d(x, 3, xx) = d(x, 3, xx) + 0.1;
        xx = rec(x, map(i, min(j+1,jmax)));
        d(x, 3, xx) = d(x, 3, xx) + 0.1;
        % right
        xx = rec(x, map(i, min(j+1,4)));
        d(x, 4, xx) = d(x, 4, xx) + 0.8;
        xx = rec(x, map(max(i-1,1), j));
        d(x, 4, xx) = d(x, 4, xx) + 0.1;
        xx = rec(x, map(min(i+1,imax), j));
        d(x, 4, xx) = d(x, 4, xx) + 0.1;
    else
        for a = 1:4
            d(x,a,x) = 1;
        end
    end
end

%% Value Iteration
N = 100;
V = zeros(nx,N);
action = zeros(1,4);
l = 0.1;delta = 0.8;
V(7, 1) = 1; V(11, 1) = -1;
for k = 2:N
    action = policy_improve(d,V(:,k-1),l,delta);
    for x = 1:nx
        if terminal(x) == 0
            V(x, k) = action_eval(x,action(x),d,V(:, k-1),l,delta);
        else
            V(x, k) = terminal(x);
        end
    end
    if norm(V(:,k)- V(:,k-1)) < 0.001
        break;
    end
end
Vopt = V(:,k);

%% Policy Iteration
action = 4*ones(1,11); % Initial policy
N = 100;
l = 0.1;delta = 0.8;
for k = 2:N
    for x = 1:nx
        if terminal(x) == 0
            V(x, k) = action_eval(x,action(x),d,V(:, k-1),l,delta);
        else
            V(x, k) = terminal(x);
        end
    end
    if norm(V(:,k)- V(:,k-1)) < 0.001
        break;
    end
end
action = policy_improve(d,V(:,k),l,delta);

%% SARSA
Q = init_Q(nx, nu, terminal);
n_ep = 500;
epsilon = 0.1;
alpha = 0.1;
rms_sarsa = zeros(1,n_ep);
for episode = 1:n_ep
    x = init(nx, terminal);
    u = greedy(Q(x,:), epsilon);
    while terminal(x) == 0
        xnew = dynamics(x, u, d);
        unew = greedy(Q(xnew,:), epsilon);
        Q(x,u) = Q(x,u) + alpha * (l + delta * Q(xnew, unew) - Q(x,u));
        x = xnew;
        u = unew;
    end
    rms_sarsa(episode) = norm(min(Q') - Vopt);
end

%% Q-Learning
Q = init_Q(nx, nu, terminal);
n_ep = 500;
epsilon = 0.1;
alpha = 0.1;
rms_q = zeros(1,n_ep);
for episode = 1:n_ep
    x = init(nx, terminal);
    while terminal(x) == 0
        u = greedy(Q(x,:), epsilon);
        xnew = dynamics(x, u, d);
        Q(x,u) = Q(x,u) + alpha * (l + delta * min(Q(xnew, :)) - Q(x,u));
        x = xnew;
        u = unew;
    end
    rms_q(episode) = norm(min(Q') - Vopt);
end

%% Visualize SARSA Q-Learning
figure;
hold on
plot(rms_sarsa)
plot(rms_q)
legend("SARSA", "Q-Learning")
box on;
%% Functions
function xnew = dynamics(x, u, d)
sample = rand(1);
cum = 0;
for i = 1:size(d,3)
    cum = cum + d(x,u,i);
    if cum > sample
        break;
    end
end
xnew = i;
end

function Q = init_Q(nx, nu, terminal)
Q = 10*ones(nx,nu);
for x = 1:nx
    if terminal(x) ~= 0
        for u = 1:nu
            Q(x,u) = terminal(x);
        end
    end
end
end
function u = greedy(Q, epsilon)
sample = rand(1);
nu = length(Q);
if sample > epsilon
    u = ceil(rand(1) * nu);
else
    [~, u] = min(Q);
    u = u(1);
end
end
function x = init(nx, terminal)
x = ceil(rand(1) * nx);
while terminal(x) ~= 0
    x = ceil(rand(1) * nx);
end
end
function [action, Vnew] = policy_improve(d,V,l,delta)
Vnew = V;
for x = 1:size(d,1)
    q = zeros(1,4);
    for a = 1:4
        q(a) = action_eval(x,a,d,V,l,delta);
    end
    Vnew(x) = min(q);
    possible_action = find(q == Vnew(x));
    if length(possible_action) > 1
        action(x) = possible_action(1); % or -1
    else
        action(x) = possible_action;
    end
end
end

function q = action_eval(x, a, d, V, l, delta)
q = 0;
for xx = 1:size(d,1)
    q = q + d(x,a,xx)*(l + delta * V(xx));
end
end

function y = rec(x, xx)
if xx == -1
    y = x;
else
    y = xx;
end
end