n = 2; m = 2;
% x' = Ax + Bu
A = rand(n);
B = rand(m);
% target system x' = Astar * x
Astar = 0*eye(2);

x0 = [1;-1];
F = 10*eye(n+m) + ones(n+m);
N = 29;
xstar = zeros(n,N);xstar(:,1) = x0;
x = zeros(n,N);x(:,1) = x0;
u = zeros(m,N);
Ahat = zeros(n,n,N); Ahat(:,:,1) = eye(n);
Bhat = zeros(n,m,N); Bhat(:,:,1) = eye(m);
V = zeros(1,N);
error_pri = zeros(n,N);error_pri(:,1) = x0;
error_pos = zeros(n,N);error_pos(:,1) = x0;
for k = 1:N
    if k > 1
        error = x(:,k) - xstar(:,k);
        phi = [x(:,k-1);u(:,k-1)];
        F = inv(inv(F) + phi*phi');
        dtheta = [Ahat(:,:,k-1) Bhat(:,:,k-1)] + error*phi'*F;
        Ahat(:,:,k) = dtheta(1:n, 1:n); Bhat(:,:,k) = dtheta(1:n, n+1:end);
        ABtilde = [Ahat(:,:,k) - A Bhat(:,:,k) -  B];
        error_pri(:,k) = error;
        error_pos(:,k) = (1-phi'*F*phi)*error;
        V(k) = (1-phi'*F*phi)^2*norm(error)^2 + trace(ABtilde * inv(F) * ABtilde');
    else
        ABtilde = [Ahat(k) - A Bhat(k) -  B];
        V(k) = norm(x(k))^2 + trace(ABtilde * inv(F) * ABtilde');
    end
    K = pinv(Bhat(:,:,k))*(Astar - Ahat(:,:,k));
    u(:,k) = K*x(:,k);
    x(:,k+1) = A*x(:,k)+B*u(:,k);
    xstar(:,k+1) = Astar * xstar(:,k);
end
clf;hold on;
plot(x(:,:)','k')
plot(V,'r')
%plot(error_pri','--b')
plot(error_pos','-*b')
box on;
legend(["x1", "x2", "value function", "a posteriori error x1", "a posteriori error x2"])