% Setup the Fourier grid
N = 100;  % Number of grid points
Lx = 10;  % Length of the domain
dx = Lx / N; % Grid spacing
x = (0:N-1)' * dx; % Grid points (column vector)

% Define wave numbers (for the Fourier transform)
k = (2*pi/Lx) * [0:(N/2-1), -N/2:-1]'; % Wave numbers as column vector

% Define the coefficient function a(x) and create A
a = @(x) sin(x);      % Example: a(x) = sin(x) (make sure it's periodic)
a_values = a(x);      % Values of a(x) at grid points
A = diag(a_values);   % Create a diagonal matrix with a(x) values

% Define the function f
f = sin(x);  % Example forcing term (column vector)

% Compute the second derivative using FFT
D2u = @(u) real(ifft(-k.^2 .* fft(u)));  % Second derivative in spectral space

% Define the operator L(u) = D2u - A*u
L = @(u) D2u(u) - A * u;

% Solve the system using GMRES
tol = 1e-6;    % Tolerance
maxit = 100;   % Maximum iterations
[u, flag, relres, iter] = gmres(@(v) L(v), f, [], tol, maxit);

% Output results
if flag == 0
    disp('GMRES converged successfully.');
else
    disp('GMRES did not converge.');
end

disp(['Relative residual: ', num2str(relres)]);
disp(['Iterations: ', num2str(iter)]);
