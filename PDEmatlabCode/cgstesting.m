% Parameters
N = 100;             % Number of grid points
L = 10;              % Length of the domain
dx = L / N;          % Grid spacing
x = (0:N-1)' * dx;   % Grid points (column vector)

% Define wave numbers (for the Fourier transform)
k = (2 * pi / L) * [0:(N/2-1), -N/2:-1]';  % Ensure k is a column vector

% Define the matrix A (diagonal with values of a(x_j))
a = @(x) cos(2*pi*x);       % Example: a(x) = sin(x) (make sure it's periodic)
A = diag(a(x));        % Diagonal matrix

% Define the function f (ensure f is a column vector)
f = sawtooth(x, 0.5);            % Example forcing term
f = f(:);              % Ensure f is a column vector

% Compute the second derivative using FFT
D2u = @(u) ifft(-k.^2 .* fft(u), 'symmetric');  % Second derivative in spectral space

% Define the operator L (linear system: L(u) = D^2u - A*u)
L = @(u) D2u(u) - A * u;  % Function that applies the operator L

% Initial guess for the solution, ensure u0 is a column vector
u0 = zeros(N, 1);       % Initial guess, column vector

% Define a wrapper for L to use with CGS (cgs expects function handles)
L_cgs = @(u) L(u);      % Function handle for the operator

% Solve the system using Conjugate Gradient Squared (CGS)
[u, flag, relres, iter] = cgs(L_cgs, f, 1e-6, 100);

% Check if the solver converged
if flag == 0
    disp('CGS converged successfully.');
elseif flag == 1
    disp('CGS iterated max number of times without converging.');
else
    disp('CGS failed for another reason.');
end

% Display results
% Plot the solution
figure;
plot(x, u, 'b-', 'LineWidth', 2);
xlabel('x');
ylabel('u(x)');
title('Solution of the PDE');
grid on;
