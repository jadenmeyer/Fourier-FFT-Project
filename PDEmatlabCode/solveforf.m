% Parameters for the problem
par.n = 256; % Discretization size
M = 1; % Size of domain in units of 2pi
par.L = M*pi;
par.x = linspace(-par.L, par.L, par.n+1)'; par.x = par.x(1:end-1);
par.dx = 2*par.L/par.n;
par.mu = 1.25; % Example system parameter, can be adjusted as needed

% Define a(x) and f(x)
a = @(x) 1 + 0.5 * sin(x); % Example function a(x), modify as needed
f_true = @(x) sin(x); % Example source function f(x), modify as needed

% Derivative vectors (in Fourier space)
k = ([[0:par.n/2] [-par.n/2+1: -1]]./M)'; % Fourier wave numbers
k2 = k.^2; % k^2 for second derivative
k4 = k2.^2; % For higher derivatives if needed

% Construct the Fourier-space differential operator
par.Dxx = -k2; % Fourier space representation of u_xx

% Preconditioner (similar to Swift-Hohenberg)
pc = @(du) ifft(fft(du)./(par.Dxx - a(par.x))); % Preconditioner based on Dxx and a(x)

% Initial guess for u
u = zeros(par.n, 1); % Initial guess for u, modify as needed

% Define objective function (residual)
obj_fun = @(u, par) real(ifft(par.Dxx .* fft(u))) - a(par.x) .* u - f_true(par.x);

% Newton's method setup
npar.tol = 1e-6; % Tolerance
npar.maxiter = 50; % Maximum iterations
npar.niter = 1; % Iteration counter
npar.nitermax = 15; % Maximum iterations for the Newton solver
npar.nincr = ones(par.n, 1); % Increment vector
npar.minstep = 1e-8; % Minimum step size
npar.rhs = 0; % Right-hand side

% Compute initial residual
npar.rhs = obj_fun(u, par); % Compute residual
npar.residual = norm(npar.rhs); % Estimate residual

% Newton iteration loop
while (npar.residual > npar.tol)
    % Define the Jacobian for the Newton method (derivative of the objective function)
    Dobj_fun = @(u, du, par) real(ifft(par.Dxx .* fft(du))) - a(par.x) .* du - f_true(par.x);
    
    % Solve the linear system using GMRES
    [npar.nincr, flag] = gmres(@(du) Dobj_fun(u, du, par), npar.rhs, 20, 1e-12, 25, pc);
    
    % Update the solution
    u = u - npar.nincr;
    
    % Update the iteration count
    npar.niter = npar.niter + 1;
    
    % Recompute the residual
    npar.rhs = obj_fun(u, par); % Compute residual
    npar.residual = norm(npar.rhs); % Estimate residual
    
    % Check for convergence conditions
    if npar.niter > npar.nitermax
        disp(['Maximal number of iterations reached, giving up; residual is ' num2str(npar.residual)]);
        break
    end
    
    if norm(npar.nincr) < npar.minstep
        disp(['Newton step is ineffective, giving up; residual is ' num2str(npar.residual)]);
        break
    end
end

% At this point, u is the solution to the PDE

% Compute u_xx in Fourier space
u_xx = real(ifft(par.Dxx .* fft(u)));

% Compute f(x) = u_xx - a(x)*u
f_computed = u_xx - a(par.x).*u;

% Plot the computed f(x)
figure;
plot(par.x, f_computed, 'LineWidth', 2);
xlabel('x');
ylabel('f(x)');
title('Computed Source Term f(x)');
grid on;

% Plot the solution u(x)
figure;
plot(par.x, u, 'LineWidth', 2);
xlabel('x');
ylabel('u(x)');
title('Solution of u_{xx} - a(x)u = f(x)');
grid on;
