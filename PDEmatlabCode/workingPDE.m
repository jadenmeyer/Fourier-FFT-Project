%  Define parameters and set up the problem

par.n = 256;             % Discretization size
M = 1;                   % Size of domain in units of 2pi
par.L = M*pi;            % Length of the domain
par.x = linspace(-par.L, par.L, par.n+1)'; 
par.x = par.x(1:end-1);  % Spatial grid points
par.dx = 2 * par.L / par.n;  % Grid spacing

% Define the wavenumbers (Fourier modes)
k = [0:par.n/2, -par.n/2+1:-1] / M; 
k = k';                  % Wavenumber vector
k2 = k.^2;               % k^2 for second derivative in Fourier space

% Define the differential operator for u_xx in Fourier space
par.D = -k2;             % Second derivative in Fourier space

% Define the function a(x), which multiplies u
a_x = @(x) 1 + 0.5 * sin(x);  % Example: a(x) = 1 + 0.5 * sin(x)

% Initial guess for u (can be an arbitrary initial guess)
u_init = zeros(par.n, 1);  % Initial guess: zero solution

% Move u and other arrays to the GPU
u_gpu = gpuArray(u_init);
k_gpu = gpuArray(k);
k2_gpu = gpuArray(k2);
par.D_gpu = gpuArray(par.D);
par.x_gpu = gpuArray(par.x);
a_x_gpu = gpuArray(a_x(par.x));

% Define the GMRES solver
% Preconditioner (using the GPU)
pc = @(du) ifft(fft(du) ./ (par.D_gpu - (1+par.mu)*ones(par.n, 1)), 'symmetric');

% Right-hand side f(x) (this can be any forcing function or data you're given)
f_gpu = gpuArray(sin(par.x));  % Example: set f(x) to be sin(x)

% Define the operator A(u)
A = @(u) real(ifft(par.D_gpu .* fft(u))) - a_x_gpu .* u;

% Solve the system A(u) = f using GMRES
% GMRES requires the right-hand side (f_gpu) and the operator A
tic;
[sol_gpu, flag] = gmres(A, f_gpu, 20, 1e-12, 25);
toc;
% Move the solution back to CPU and plot the result
sol_cpu = gather(sol_gpu);

% Plot the result
figure;
plot(par.x, sol_cpu);
title('Solution u(x) computed from f(x) using GMRES on GPU');
xlabel('x');
ylabel('u(x)');
