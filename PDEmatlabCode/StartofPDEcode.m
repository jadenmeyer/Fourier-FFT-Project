%  function sh
n=128;% discretization size
M = 1; % size of domain in units of 2pi
L = M*pi; %gives the high end of the domain
x = linspace(0, L, n+1)'; 
x=x(1:end-1); %define the linear space in real domain over n points
dx=2*L/n; %acts similar to a step size

%system parameter
a = @(x) tan(x); %dependent on position in domain changes PDE like so
a_vals = a(x);
a_hat = fft(a_vals);

%derivative vectors
k = ([[0:n/2] [-n/2+1: -1]]./M)'; % wave vector
k2 = k.^2; %define the square k
k4 = k2.^2 % define the k to the fourth


%initial shape of the PDE:
u0 = cos(x); %consider this the temp shape
uh = fft(u0);

%convolution in Fourier Space do piecewise mult in real
conv = a_hat.*uh;
conv_hat = fft(conv);

%diff_op = ifft(a_hat *)


% Fourier space matrix for the operator
A_k = -k.^2 + a_hat;  % Diagonal matrix in Fourier space

%set tolerances for residuals
tol = 1e-6; %max tolerance for the convergence of the PDE
max_iterations = 1000 %call out maximum number of solver iters

init_guess = uh;

%start Gmres
[u_hat, flag] = gmres(@(x) apply_operator(x, A_k), init_guess, [], tol, max_iterations);

u = ifft(u_hat);

plot(u);

function opEnd = apply_operator(x, A_k)
    opEnd = A_k .* x;  % Simple multiplication in Fourier space
end