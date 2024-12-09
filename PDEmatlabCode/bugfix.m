n = 128;  % Discretization size
M = 1;  % Size of domain in units of 2pi
L = M * pi;  % High end of the domain
x = linspace(0, L, n+1)'; 
x = x(1:end-1);  % Define the linear space in real domain over n points
dx = 2 * L / n;  % Step size

% System parameter
a = @(x) tan(x);  % Coefficient function a(x)
a_vals = a(x);
a_hat = fft(a_vals);

% Derivative vectors
k = ([[0:n/2] [-n/2+1:-1]]./M)';  % Wave vector
k2 = k.^2;  % k^2
k4 = k2.^2;  % k^4

% Initial shape of the PDE
u0 = cos(x);  % Initial condition
uh = fft(u0);

% Fourier space matrix for the operator
A_k = -k.^2 + a_hat;  % Diagonal matrix in Fourier space

% Set tolerances for residuals
tol = 1e-6;  % Max tolerance for the convergence of the PDE
max_iterations = 100;  % Reasonable max iterations for GMRES

init_guess = uh;

% Start GMRES
[u_hat, flag] = gmres(@(x) apply_operator(x, A_k), init_guess, [], tol, max_iterations);

% Transform solution back to real space
u = ifft(u_hat);

% Plot solution
plot(x, u, '-o');
title('Solution of PDE');
xlabel('x');
ylabel('u(x)');

% GMRES operator function
function out = apply_operator(x, A_k)
    out = A_k .* x;  % Simple multiplication in Fourier space
end

% Display GMRES status
disp('GMRES flag:');
disp(flag);
