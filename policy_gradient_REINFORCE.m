%% policy gradient system initialization
clear;

steps = 50;
iters = 500;
freq = 1000; % sampling frequency

A = 1/2;
B = 1;
x0 = 1;

N = 50; % dimension of policy network
% policy network
theta_mu = zeros(N+1,1);
theta_sigma = zeros(N+1,1);
Theta_mu = theta_mu;
Theta_sigma = theta_sigma;
rbf_center_mu = linspace(-1,1,N);
radius_mu = .8;
rbf_center_sigma = linspace(-0.5,0.5,N);
radius_sigma = 0.1;

alpha = linspace(1e-6,1e-7,iters); % learning rate

X = [];
U = [];
Mu = [];
Sigma = [];

%u0 = 0*0.5*sin(2*linspace(0,2*pi,steps));
f_signal = 50; % u0 signal frequency
u0 = 0.1.*(sqrt(1:steps) .* sin(2*pi*f_signal*(1:steps)/freq))'; %sin(2*pi*f_signal*(1:steps)/freq)';%+sin(pi*f_signal*(1:steps)/freq)';

cost = zeros(1,steps);
figure(1); clf; hold on;
for i = 1:iters
    % band-limited noise
    noise = 0.3*randn(steps, 1);
    for k = 1:2
    for j = 2:steps-1
        noise(j) = 0.1*noise(j-1) + 0.8*noise(j-1) + 0.1*noise(j+1);
    end
    end
    %fs = 200; % band width
    %sys = c2d(tf((2*pi*fs)^2,[1, 2*0.7*2*pi*fs, (2*pi*fs)^2]),1/freq);
    % apply to filter to yield bandlimited noise
    %noise = filtfilt(sys.num{1},sys.den{1},noise);
    
    % system dynamics
    [x, u, mu, sigma, F_mu, F_sigma] = sys_dyn(A, B, x0, theta_mu, theta_sigma, steps, radius_mu, radius_sigma, rbf_center_mu, rbf_center_sigma, noise, u0);
    X = [X,x];
    U = [U,u];
    Mu = [Mu,mu];
    Sigma = [Sigma, sigma];
    
    % policy gradient parameter update
    [theta_mu_new, theta_sigma_new] = policy_gradient(x, u, mu, sigma, F_mu, F_sigma, theta_mu, theta_sigma, steps, alpha(i));
    
    theta_mu = theta_mu_new;
    theta_sigma = theta_sigma_new;
    
    Theta_mu = [Theta_mu,theta_mu_new];
    Theta_sigma = [Theta_sigma,theta_sigma_new];
    
    plot(0:steps,X(:,i),'color',[i/iters, 1- i/iters, 1-i/iters],'LineWidth',2);
    
    cost(i) = norm(X(:,i))^2 + norm(U(:,i))^2;
end

list = [];
for i = 1:iters
    list = [list, strcat("Episode ", num2str(i))];
end
%legend(list);
xlabel('Time step'); ylabel('states'); title(['REINFORCE (',num2str(iters),' episodes)']);
box on;

% Plot final learned mean policy
[x, u, mu, sigma, F_mu, F_sigma] = sys_dyn_mean(A, B, x0, theta_mu, theta_sigma, steps, radius_mu, radius_sigma, rbf_center_mu, rbf_center_sigma, noise, u0);
plot(0:steps,x,'k','LineWidth',2);
% figure(2); hold on;
% plot(X(:,1)); hold on; plot(X(:,ceil(iters/2))); plot(X(:,ceil(2*iters/3))); plot(X(:,end));
% grid on;
% legend('0 iter',[num2str(ceil(iters/3)),' iter'],[num2str(ceil(2*iters/3)),' iter'],[num2str(iters),' iter']);
% xlabel('steps'); ylabel('states'); title('policy gradient (partial iterations)');
figure(4);clf;
plot(cost)
%% function of system dynamics
function [X, U, Mu, Sigma, F_mu, F_sigma] = sys_dyn(A, B, x0, theta_mu, theta_sigma, steps, radius_mu, radius_sigma, rbf_center_mu, rbf_center_sigma, noise, u0)
    X = x0;
    xk = x0;
    U = [];
    Mu = [];
    Sigma = [];
    F_mu = [];
    F_sigma = [];
    for k = 1:steps
        f_mu = [exp(-radius_mu*(xk - rbf_center_mu).^2) 1]';
        f_sigma = [exp(-radius_sigma*(xk - rbf_center_sigma).^2) 1]';
        mu = theta_mu'*f_mu;
        sigma = exp(theta_sigma'*f_sigma);
        u = normrnd(mu,sigma);
        u = u + u0(k);
        if length(noise) > 1
            xk1 = A*xk + B*u + noise(k);
        else
            xk1 = A*xk + B*u + noise*xk;
        end

        xk = xk1;
        X = [X;xk1];
        U = [U;u];
        Mu = [Mu;mu];
        Sigma = [Sigma;sigma];
        F_mu = [F_mu,f_mu];
        F_sigma = [F_sigma,f_sigma];
    end
end

function [X, U, Mu, Sigma, F_mu, F_sigma] = sys_dyn_mean(A, B, x0, theta_mu, theta_sigma, steps, radius_mu, radius_sigma, rbf_center_mu, rbf_center_sigma, noise, u0)
    X = x0;
    xk = x0;
    U = [];
    Mu = [];
    Sigma = [];
    F_mu = [];
    F_sigma = [];
    for k = 1:steps
        f_mu = [exp(-radius_mu*(xk - rbf_center_mu).^2) 1]';
        f_sigma = [exp(-radius_sigma*(xk - rbf_center_sigma).^2) 1]';
        mu = theta_mu'*f_mu;
        sigma = exp(theta_sigma'*f_sigma);
        u = mu;
        u = u + u0(k);
        if length(noise) > 1
            xk1 = A*xk + B*u + noise(k);
        else
            xk1 = A*xk + B*u + noise*xk;
        end

        xk = xk1;
        X = [X;xk1];
        U = [U;u];
        Mu = [Mu;mu];
        Sigma = [Sigma;sigma];
        F_mu = [F_mu,f_mu];
        F_sigma = [F_sigma,f_sigma];
    end
end


%% function of policy gradient update
function [theta_mu_new, theta_sigma_new] = policy_gradient(X, U, Mu, Sigma, F_mu, F_sigma, theta_mu, theta_sigma, steps, alpha)
    L = X.^2 + [U;0].^2;

    D_theta_mu = F_mu;
    tmp = (U - Mu)./(Sigma.^2);
    for i = 1:steps
        D_theta_mu(:,i) = D_theta_mu(:,i) * tmp(i);
    end
    
    D_theta_sigma = F_sigma;
    tmp = ((U - Mu).^2)./(Sigma.^2) - 1;
    for i = 1:steps
        D_theta_sigma(:,i) = D_theta_sigma(:,i) * tmp(i);
    end
    
    grad_theta_mu = D_theta_mu * triu(ones(steps,steps+1)) * L;
    grad_theta_sigma = D_theta_sigma * triu(ones(steps,steps+1)) * L;
    
    theta_mu_new = theta_mu - alpha * grad_theta_mu;
    theta_sigma_new = theta_sigma - alpha/10 * grad_theta_sigma;
end