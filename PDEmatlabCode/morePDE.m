%  Define parameters and set up the problem

par.n = 256;             % Discretization size
M = 1;                   % Size of domain in units of 2pi
par.L = M*pi;            % Length of the domain
par.x = linspace(-par.L, par.L, par.n+1)'; 
par.x = par.x(1:end-1);  % Spatial grid points
par.dx = 2 * par.L / par.n;  % Grid spacing
par.mu = 1.25;           % System parameter (unused, but kept for consistency)
par.ell = 1.1;           % Wavenumber for Swift-Hohenberg (not used in this case)

% Define the wavenumbers (Fourier modes)
k = [0:par.n/2, -par.n/2+1:-1] / M; 
k = k';                  % Wavenumber vector
k2 = k.^2;               % k^2 for second derivative in Fourier space

% Define the differential operator for u_xx in Fourier space
par.D = -k2;             % Second derivative in Fourier space

% Preconditioner function (same as before)
pc = @(du) ifft(fft(du)./(par.D - (1+par.mu)*ones(par.n, 1))); 

% Define the function a(x), which multiplies u
a_x = @(x) 1 + 0.5 * sin(x);  % Example: a(x) = 1 + 0.5 * sin(x)

% Define the forcing term f(x) (can be any function)
f = @(x) 0.5 * sin(2*x);  % Example: f(x) = 0.5 * sin(2*x)

% Initial guess for u (can be an arbitrary initial guess)
u = zeros(par.n, 1);  % Initial guess: zero solution

% Define the objective function for the residual of the PDE
SHrolla = @(u, par) real(ifft(par.D .* fft(u))) - a_x(par.x) .* u - f(par.x);
SHroll = @(u) SHrolla(u, par);

% Define the Jacobian of the objective function
DSHrolla = @(u, du, par) real(ifft(par.D .* fft(du))) - a_x(par.x) .* du;
DSHroll = @(du) DSHrolla(u, du, par);

% Newton solver setup
npar.tol = 1e-6;          % Convergence tolerance for residual
npar.maxiter = 50;        % Max iterations in Newton's method
npar.niter = 1;           % Iteration counter
npar.nitermax = 15;       % Max number of Newton iterations
npar.nincr = ones(par.n, 1);  % Step size increment
npar.minstep = 1e-8;      % Minimum step size to avoid small updates
npar.rhs = 0;             % Right-hand side for residual
npar.gmrestol = 1e-12;    % Tolerance for GMRES solver

% Compute the initial residual
npar.rhs = SHroll(u);  % Compute residual
npar.residual = norm(npar.rhs);  % Compute norm of residual
tic;
% Newton method loop for solving the PDE
while npar.residual > npar.tol
    % Solve the linear system using GMRES with preconditioner
    [npar.nincr, flag] = gmres(DSHroll, npar.rhs, 20, npar.gmrestol, 25, pc);
    
    % Update u with the new increment
    u = u - npar.nincr;
    
    % Increment the number of Newton iterations
    npar.niter = npar.niter + 1;
    
    % Recompute the residual
    npar.rhs = SHroll(u);
    npar.residual = norm(npar.rhs);
    
    % Check for convergence
    if npar.niter > npar.nitermax
        disp(['Maximal number of iterations reached, giving up; residual is ' num2str(npar.residual)]);
        break;
    end
    
    % Check if the Newton step is too small (no progress)
    if norm(npar.nincr) < npar.minstep
        disp(['Newton step is ineffective, giving up; residual is ' num2str(npar.residual)]);
        break;
    end
end
toc;
% Output final result
disp(['Solution found with residual ' num2str(npar.residual)]);

plot(x, n.par);