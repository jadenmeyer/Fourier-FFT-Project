% Simple example to solve u_xx - u = f using Newton's method and GMRES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%cool comment 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Problem parameters
L = 10;                  % Length of the domain (0, L)
n = 100;                 % Number of grid points
x = linspace(0, L, n)';  % Grid points (make sure x is a column vector)
dx = x(2) - x(1);       % Grid spacing
f = sin(2*pi*x);        % Example forcing term f(x) = sin(2*pi*x)

% Initial guess (u0), we assume it's a rough guess
u0 = zeros(n, 1);

% Set up the finite difference matrix for u_xx
D2 = diag(-2*ones(n,1)) + diag(ones(n-1,1), 1) + diag(ones(n-1,1), -1);
D2 = D2 / dx^2;  % Second derivative matrix (scaled by dx^2)

% Define the operator for the PDE
L = @(u) D2*u - u; % L(u) = u_xx - u

% Define the residual function F(u) = L(u) - f
F = @(u) L(u) - f;  % Ensure F(u) returns a column vector

% Newton's method parameters
tol = 1e-6;        % Convergence tolerance
maxIter = 50;      % Maximum number of iterations
u = u0;            % Start with initial guess u0
residual = norm(F(u));  % Initial residual

% Newton's method loop
for iter = 1:maxIter
    if residual < tol
        fprintf('Converged to tolerance after %d iterations.\n', iter);
        break;
    end
    
    % Compute the Jacobian of F(u), which is the matrix of second derivatives
    % For the linear case, the Jacobian is just D2 - I (where I is the identity matrix)
    J = D2 - eye(n);

    % Solve the linear system J * du = -F(u) using GMRES
    rhs = F(u);  % Ensure rhs is a column vector
    if size(rhs, 1) ~= n
        error('Right-hand side vector does not have the correct size');
    end
    
    % GMRES with right-hand side rhs
    [du, flag] = gmres(J, rhs, 20, tol, 50);  % GMRES with right-hand side rhs

    % Check if GMRES converged
    if flag ~= 0
        fprintf('GMRES did not converge, flag = %d\n', flag);
        break;
    end
    
    % Update the solution
    u = u + du;
    
    % Compute the new residual
    residual = norm(F(u));
    
    fprintf('Iteration %d: Residual = %.6e\n', iter, residual);
end

% Plot the final solution
figure;
plot(x, u, '-o', 'LineWidth', 2);
title('Solution of u_{xx} - u = f');
xlabel('x');
ylabel('u(x)');
grid on;
