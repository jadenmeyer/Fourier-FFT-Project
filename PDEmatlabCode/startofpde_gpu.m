% Function to solve PDE using Newton's method in Fourier space
%function sh
    n = 1024; % Discretization size
    M = 1; % Size of domain in units of 2pi
    L = M * pi; % High end of the domain
    x = linspace(0, L, n + 1)'; 
    x = x(1:end-1); % Define the linear space in real domain over n points
    dx = 2 * L / n; % Step size

    % System parameters
    a = @(x) (x); % Dependent on position in domain (nonlinear PDE)
    a_vals = a(x);
    a_hat = fft(a_vals);

    % Derivative vectors
    k = ([[0:n/2] [-n/2+1: -1]]./M)'; % Wave vector
    k2 = k.^2; % k^2
    k4 = k2.^2; % k^4

    % Initial shape of the PDE (initial guess for u)
    u0 = cos(x).^4; % Consider this the initial shape (u0)
    uh = fft(u0); % Fourier transform of u0

    % Fourier space matrix for the operator
    A_k = -k.^2 + a_hat; % Diagonal matrix in Fourier space

    % Set tolerances for residuals
    tol = 1e-8; % Max tolerance for the convergence of the PDE
    max_iterations = 1000; % Maximum number of solver iterations

    init_guess = uh; % Initial guess for Newton's method
tic;
% Start Newton's method
for iter = 1:max_iterations
    % Compute the residual F(u) and Jacobian J(u) at the current u_hat
    F = residual(init_guess, A_k, a_hat);
    J = jacobian(init_guess, A_k, a_hat);
    
    % Solve the linear system J * delta_u = -F using GMRES
    delta_u_hat = gmres(@(x) apply_operator(x, J), -F, [], tol, max_iterations);
    
    % Update the solution
    init_guess = init_guess + delta_u_hat;
    
    % Check for convergence (using the norm of the residual)
    if norm(F) < tol
        fprintf('Converged in %d iterations\n', iter);
        break;
    end
end
toc;
% Convert the solution back to real space
u = ifft(init_guess);

% Plot the solution (real part only)
plot(x, real(u));
title('Solution of the PDE');
xlabel('x');
ylabel('u(x)');

% Function to compute the residual (F(u) = 0)
function F = residual(u_hat, A_k, a_hat)
    % The residual function is simply the equation you're solving in Fourier space
    conv = A_k .* u_hat;  % Apply the operator (A_k * u_hat)
    F = conv - a_hat;     % The residual (you can adjust this depending on your PDE)
end

% Function to compute the Jacobian (J(u)) of the residual
function J = jacobian(u_hat, A_k, a_hat)
    % The Jacobian is the derivative of the residual with respect to u_hat
    % For simplicity, assume it's just the diagonal of A_k (for linear problems)
    % In case of nonlinear PDEs, this may need adjustment
    J = A_k;  % For linear problems, the Jacobian is just A_k
end

% Function to apply the operator in Fourier space (multiplication)
function opEnd = apply_operator(x, A_k)
    opEnd = A_k .* x;  % Simple multiplication in Fourier space
end
