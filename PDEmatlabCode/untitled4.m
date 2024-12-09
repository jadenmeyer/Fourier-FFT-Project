% Parameters
siz = 1024;
n = siz * 1024; % discretization size
M = siz * 10; % size of domain in units of 2pi
L = M * pi;
x = linspace(0, L, n+1)'; % domain
x = x(1:end-1); % cut out last point of domain for n points
dx = 2 * L / n; % grid spacing

% Fourier space
k = [[0:n/2], [-n/2+1:-1]]' / M;  % Wave numbers (Fourier space)
k2 = k.^2;
k4 = k2.^2;

% Define coefficient function a(x) and matrix A (diagonal)
a = @(x) 1 + 0.1 * sin(2 * pi * x / L);  % a(x) function
A = a(x);  % diagonal elements of matrix A, which will be treated as a vector

% Define operator L(u)
L_u = @(u) ifft(-k.^2 .* fft(u)) - A .* u; % Fourier and element-wise multiplication

% Initial guess (Zero Initialization or a simple sinusoidal function)
%u = cos(x).^4;  % Zero initialization
% Alternatively, try this instead:
 u = sin(x);  % Simple sinusoidal guess

% Newton's Method Parameters
tol = 1e-6; % tolerance
maxiter = 200; % increased number of iterations
niter = 1; % current iteration count
nitermax = 35; % max number of iterations
nincr = ones(n+1, 1); % step size (could be updated)
minstep = -1e-8; % minimum step size

% Initial residual
residual = L_u(u); % Initial residual: L(u) - f, assuming f = 0 initially
rhs = 0; % Assuming f = 0 for simplicity

% Residual norm
residual_norm = norm(residual);
disp(['Initial residual norm: ', num2str(residual_norm)])

% GMRES tolerance
gmrestol = 1e-6; % less strict tolerance
tic;
% Main iterative loop (Newton's method)
while niter <= nitermax && residual_norm > tol
    % Solve the linear system L(u) = rhs using GMRES
    [u_solution, flag, ~, ~, ~] = gmres(@(u) L_u(u), residual, [], gmrestol, maxiter);
    
    if flag ~= 0
        disp('GMRES did not converge');
        break;
    end
    
    % Update u using the solution from GMRES
    u = u - u_solution;  % This updates the solution

    % Compute the new residual
    residual = L_u(u);  % New residual after update

    % Compute the residual norm
    residual_norm = norm(residual);
    disp(['Iteration: ', num2str(niter), ', Residual norm: ', num2str(residual_norm)])

    % Update iteration counter
    niter = niter + 1;
    
    % Optionally, check for convergence in terms of steps
    if residual_norm < tol
        disp('Convergence achieved.');
        break;
    end
end
toc;
% Plot the final solution u
figure;
plot(x, u);
title('Solution u(x)');
xlabel('x');
ylabel('u(x)');
