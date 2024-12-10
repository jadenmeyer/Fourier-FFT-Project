% Setup the Fourier grid and domain
siz = 256;
n = siz * 256;         % Discretization size
M = siz * 10;           % Size of domain in units of 2pi
L = M * pi; 
x = linspace(0, L, n+1)'; 
x = x(1:end-1);         % Remove the last point
dx = 2 * L / n;         % Grid spacing

% Define the known solution u(x)
u_exact = @(x) sawtooth(x, 0.5);  % Example: sawtooth wave

% Define the coefficient function a(x)
a = @(x) cos(x);            % Example: a(x) = cos(x)

% Compute f(x) = u_xx - a(x)*u(x)
% Define wave numbers for Fourier transform (same size as u)
k = (2 * pi / L) * [0:(n/2-1), -n/2:-1]'; % Wave numbers as column vector
k = k(:);  % Ensure k is a column vector for proper multiplication

% Preconditioner: Element-wise inverse of a(x) (diagonal preconditioner)
M_inv = @(x_gpu) 1 ./ a(x_gpu);  % Inverse of a(x) at each point

% Compute the second derivative in Fourier space
D2u_gpu = @(u) real(ifft(-k.^2 .* fft(u)));  % Second derivative in spectral space

% Define the operator L(u) = D2u(u) - A*u
L_gpu = @(u) D2u_gpu(u) - a(x) .* u;

% Compute the forcing term f_gpu = D2u(u_exact(x)) - a(x) * u_exact(x)
f_gpu = D2u_gpu(u_exact(x)) - a(x) .* u_exact(x);

% Solve the system using GMRES with a preconditioner
tol = 1e-12;    % Tolerance for convergence
maxit = 1000;   % Maximum iterations for GMRES
tic;
[u_gpu, flag, relres, iter] = gmres(@(v) L_gpu(v), f_gpu, [], tol, maxit, [], @(v) M_inv(v));
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

% Display residual information
disp(['Relative residual: ', num2str(relres)]);
disp(['Iterations: ', num2str(iter)]);

% Transfer the solution back to CPU for plotting
u = gather(u_gpu);  % Transfer result back to CPU for visualization

% Optional: Plot the results
figure;
plot(x, u, 'r-', 'DisplayName', 'Numerical Solution');
hold on;
plot(x, gather(u_exact(x)), 'b--', 'DisplayName', 'Exact Solution');
legend;
xlabel('x');
ylabel('u(x)');
title('Comparison of Numerical and Exact Solutions');
