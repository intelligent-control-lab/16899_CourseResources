% LQR vs MPC
%% Specify problem
problem = 2;
switch problem
    case 1
        ndim = 1; % Dimension of the system
        dt = 0.5; % Sampling time
        nx = ndim;
        nu = ndim;
        A = 1;
        B = 1;
        Q = eye(nx);
        R = 10;
        S = 100*eye(nx);
        x0 = [10];
        N = 10;
    case 2
        ndim = 1; % Dimension of the system
        dt = 0.5; % Sampling time
        nx = 2 * ndim;
        nu = ndim;

        A = [eye(ndim) eye(ndim).*dt;
            zeros(ndim) eye(ndim)];
        B = [eye(ndim).*(dt^2/2);
            eye(ndim).*dt];
        
        Q = [1 0; 0 0];
        R = 1;
        S = 10*eye(nx);
        x0 = [10; 0];
        N = 10;
end


%% Finite time LQR
P = zeros(nx, nx, N+1);
P(:,:,N+1) = S;
K = zeros(nu, nx, N);
for i = N:-1:1
    K(:,:,i) = inv(R + B' * P(:,:,i+1) * B) * B' * P(:,:,i+1) * A;
    P(:,:,i) = Q + A' * P(:,:,i+1) * A - A' * P(:,:,i+1) * B * K(:,:,i);
end

%% Infinite time LQR
[K_inf, P_inf, ~] = dlqr(A, B, Q, R, zeros(nx, nu));

%% Compare the P matrices
figure; hold on
for i = 1:nx
    for j = 1:nx
        plot(0:N,permute(P(i,j,1:N+1),[3,1,2]),"LineStyle","-","Marker","*")
    end
end
for i = 1:nx
    for j = 1:nx
        plot(0:N,P_inf(i,j)*ones(1,N+1),"LineStyle","--")
    end
end
box on
legend("p_{11}","p_{12}","p_{21}","p_{22}")

%% Compare the gain K
figure; hold on
for i = 1:nu
    for j = 1:nx
        plot(0:N-1,permute(K(i,j,1:N),[3,1,2]),"LineStyle","-","Marker","*")
    end
end
for i = 1:nu
    for j = 1:nx
        plot(0:N-1,K_inf(i,j)*ones(1,N),"LineStyle","--")
    end
end
box on
legend("k_{11}","k_{12}")
%% Visualiza trajectory

x_lqr_f = zeros(nx, N+1); x_lqr_f(:,1) = x0;
x_lqr_inf = zeros(nx, N+1); x_lqr_inf(:,1) = x0;
x_lqr_mp = zeros(nx, N+1); x_lqr_mp(:,1) = x0;
for t = 1:N
    x_lqr_f(:,t+1) = (A - B * K(:,:,t)) * x_lqr_f(:,t);
    x_lqr_inf(:,t+1) = (A - B * K_inf) * x_lqr_inf(:,t);
    x_lqr_mp(:,t+1) = (A - B * K(:,:,1)) * x_lqr_mp(:,t);
end

figure; hold on
plot(0:N, x_lqr_f(1,:));
plot(0:N, x_lqr_inf(1,:));
plot(0:N, x_lqr_mp(1,:));
plot(0:N, zeros(1,N+1),"--k")
box on
legend("Finite LQR", "Infinite LQR", "MPC")

%% MPC with Different Preview Horizon
x_lqr_mps = zeros(nx, N+1, N);
for i = 1:N
    x_lqr_mps(:, 1, i) = x0;
end
for t = 1:2*N
    x_lqr_inf(:,t+1) = (A - B * K_inf) * x_lqr_inf(:,t);
    for i = 1:N
        x_lqr_mps(:,t+1,i) = (A - B * K(:,:,i)) * x_lqr_mps(:,t,i);
    end
end

figure; hold on
plot(0:N, x_lqr_f(1,:));
plot(0:2*N, x_lqr_inf(1,:));
for i = 1:N
    plot(0:2*N, x_lqr_mps(1,:,i), "color", [1-i/N, i/N, 1-i/N]);
end
plot(0:2*N, zeros(1,2*N+1),"--k")
box on
legend("Finite LQR", "Infinite LQR", "MPC-10","MPC-9","MPC-8","MPC-7","MPC-6","MPC-5","MPC-4","MPC-3","MPC-2","MPC-1")

%% Predicted Output
x_lqr_mp_steps = zeros(nx, N+1, N+1);
x_lqr_mp_steps(:, 1, 1) = x0;
for i = 1:N
    x_lqr_mp_steps(:,1,i+1) = (A - B * K(:,:,i)) * x_lqr_mp_steps(:,1,i);
end
for t = 1:N
    x_lqr_mp_steps(:,t+1,1) = (A - B * K(:,:,1)) * x_lqr_mp_steps(:,t,1);
    for i = 1:N
        x_lqr_mp_steps(:,t+1,i+1) = (A - B * K(:,:,i)) * x_lqr_mp_steps(:,t+1,i);
    end
end
figure; hold on
plot(0:N, x_lqr_f(1,:));
plot(0:2*N, x_lqr_inf(1,:));
for t = 1:N
    plot(t-1:t+N-1, permute(x_lqr_mp_steps(1,t,:),[1,3,2]), "color", [1-t/N, t/N, 1-t/N]);
end
box on
legend("Finite LQR", "Infinite LQR", "MPC Time 0","MPC Time 1","MPC Time 2","MPC Time 3","MPC Time 4","MPC Time 5","MPC Time 6","MPC Time 7","MPC Time 8","MPC Time 9")
