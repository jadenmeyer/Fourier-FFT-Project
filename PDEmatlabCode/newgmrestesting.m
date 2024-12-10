%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup the Fourier grid
%N = 50000;  % Number of grid points
%Lx = 10;  % Length of the domain
%dx = Lx / N; % Grid spacing
%x = (0:N-1)' * dx; % Grid points 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

siz = 256;
n = siz * 256;  % Discretization size
M = siz * 10;  % Size of domain in units of 2pi
L = M * pi;  % Length of the domain
x = linspace(0, L, n+1)';  % Grid points
x = x(1:end-1);  % Remove the last point to keep it consistent with n points
dx = 2 * L / n;  % Grid spacing

% Define the known solution u(x)
u_exact = @(x) sawtooth(x, 0.5);  % Example: sawtooth wave

% Define the coefficient function a(x)
a = @(x) cos(x);  % Example: a(x) = cos(x)

% Compute wave numbers (for the Fourier transform)
k = (2 * pi / L) * [0:(n/2-1), -n/2:-1]';  % Wave numbers as column vector
%pc = @(x) 1./a(x); %preconditioner

% Compute the second derivative using FFT
D2u = @(u) real(ifft(-k.^2 .* fft(u)));  % Second derivative in spectral space

% Compute the forcing term f(x) = u_xx - a(x)*u(x)
f = D2u(u_exact(x)) - a(x) .* u_exact(x);

% Define the operator L(u) = D2u(u) - a(x)*u
L = @(u) D2u(u) - (a(x) .* u);

% Solve the system using GMRES
tol = 1e-8;    % Tolerance
maxit = 1000;   % Maximum iterations

tic;
[u, flag, relres, iter] = gmres(@(v) L(v), f, [], tol, maxit);
toc;

% Check convergence status
if flag == 0
    disp('GMRES converged successfully!');
elseif flag == 1
    disp('GMRES did not converge within the maximum number of iterations.');
    disp(['Number of iterations: ', num2str(iter)]);
elseif flag == 2
    disp('GMRES encountered a breakdown during the iterations.');
    disp(['Number of iterations: ', num2str(iter)]);
else
    disp('Unknown error with GMRES solver.');
end

% Compare numerical and exact solutions
disp('Comparison of numerical and exact solutions:');
disp(['Relative residual: ', num2str(relres)]);
disp(['Iterations: ', num2str(iter)]);

% Optional: Plot the results
figure;
plot(x, u, 'r-', 'DisplayName', 'Numerical Solution');
hold on;
plot(x, u_exact(x), 'b--', 'DisplayName', 'Exact Solution');
legend;
xlabel('x');
ylabel('u(x)');
title('Comparison of Numerical and Exact Solutions');
