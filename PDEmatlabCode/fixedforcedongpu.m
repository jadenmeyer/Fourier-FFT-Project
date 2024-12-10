siz = 256;
n = siz * 256;  % Discretization size
M = siz * 10;  % Size of domain in units of 2pi
L = M * pi;  % Length of the domain
% Define the grid points on the CPU
x = linspace(0, L, n+1)';  % Grid points
x = x(1:end-1);  % Remove the last point to keep it consistent with n points
dx = 2 * L / n;  % Grid spacing

% Transfer x to GPU
x_gpu = gpuArray(x);  % Transfer the grid points to the GPU

% Define the known solution u(x) on the GPU by evaluating on GPU
u_exact_gpu = @(x_gpu) sawtooth(gather(x_gpu), 0.5);  % Apply to x_gpu after gathering

% Define the coefficient function a(x) on the GPU
a_gpu = @(x_gpu) cos(gather(x_gpu));  % Apply to x_gpu after gathering

% Compute wave numbers (for the Fourier transform) on the GPU
k_gpu = (2 * pi / L) * [0:(n/2-1), -n/2:-1]';  % Wave numbers as column vector
k_gpu = gpuArray(k_gpu);  % Transfer k to GPU

% Compute the second derivative using FFT on the GPU
D2u_gpu = @(u) real(ifft(-k_gpu.^2 .* fft(u)));  % Second derivative in spectral space

% Compute the forcing term f(x) on the GPU
f_gpu = D2u_gpu(u_exact_gpu(x_gpu)) - a_gpu(x_gpu) .* u_exact_gpu(x_gpu);

% Define the operator L(u) = D2u(u) - a(x)*u on the GPU
L_gpu = @(u) D2u_gpu(u) - a_gpu(x_gpu) .* u;

% Solve the system using GMRES on the GPU
tol = 1e-8;    % Tolerance
maxit = 1000;   % Maximum iterations
tic;
[u_gpu, flag, relres, iter] = gmres(@(v) L_gpu(v), f_gpu, [], tol, maxit);
toc;

% Transfer the solution back to CPU for plotting
u = gather(u_gpu);  % Transfer result back to CPU for visualization

% Compare numerical and exact solutions
disp('Comparison of numerical and exact solutions:');
disp(['Relative residual: ', num2str(relres)]);
disp(['Iterations: ', num2str(iter)]);

% Optional: Plot the results
figure;
plot(x, u, 'r-', 'DisplayName', 'Numerical Solution');
hold on;
plot(x, gather(u_exact_gpu(x_gpu)), 'b--', 'DisplayName', 'Exact Solution');
legend;
xlabel('x');
ylabel('u(x)');
title('Comparison of Numerical and Exact Solutions');
