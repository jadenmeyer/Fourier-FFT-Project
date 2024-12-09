% Number of Fourier modes (N modes)
N = 256;  % You can adjust N based on your desired resolution

% Length of the domain (periodicity)
L = 1;

% Fourier coefficients of u(x) (these are given or computed)
U = rand(2*N+1, 1);  % Example Fourier coefficients of u(x) (replace with your actual coefficients)

% Fourier coefficients of a(x) (the coefficient function a(x))
a_n = rand(2*N+1, 1);  % Example Fourier coefficients of a(x) (replace with your actual coefficients)

% Construct the diagonal matrix A
A = zeros(2*N+1, 2*N+1);  % Initialize the matrix A
for n = -N:N
    idx = n + N + 1;  % Mapping index from [-N, N] to [1, 2N+1]
    A(idx, idx) = ( (n * 2 * pi / L)^2 - a_n(idx) );
end

% Use GMRES to solve for the Fourier coefficients of f(x)
[F, flag] = gmres(@(x) A * x, U);

% Check if GMRES converged (flag = 0 means success)
if flag == 0
    disp('GMRES converged successfully.');
else
    disp(['GMRES did not converge. Flag: ', num2str(flag)]);
end

% Reconstruct the forcing function f(x) using the Fourier coefficients F
x = linspace(0, L, 1000);  % Define the x grid for plotting
f_x = zeros(size(x));  % Initialize the vector for f(x)

% Reconstruct f(x) from its Fourier series
for n = -N:N
    f_x = f_x + F(n + N + 1) * exp(1i * n * 2 * pi / L * x);
end

% Plot the resulting f(x)
figure;
plot(x, real(f_x), 'LineWidth', 2);
xlabel('x');
ylabel('f(x)');
title('Reconstructed Forcing Function f(x)');
grid on;
