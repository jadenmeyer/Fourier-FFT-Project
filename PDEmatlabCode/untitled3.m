siz=1024;
n=siz*1024;% discretization size
M = siz*10; % size of domain in units of 2pi
L = M*pi; 
x = linspace(0, L, n+1)'; %domain
x=x(1:end-1);%cut out last point of domain for n points
dx=2*L/n;
s=0;
%derivative vectors (in Fourier space)
k = ([[0:n/2] [-n/2+1: -1]]./M)';
k2=k.^2;
k4 = k2.^2;


L_u = @(u) ifft(-k.^2 .* fft(u)) - A * u; %differential operator

%initial shpaing
u = cos(x).^6;
u = [u:0];


% Newton parameters
tol=1e-6;
maxiter = 50;
niter = 1;
nitermax= 35;
nincr=ones(n+1,1);
minstep=-1e-8;
rhs=0;


% initial res
residual = L_u(u) %initial guess of the function since we dont know f
gmrestol=1e-8;

% Residual norm
residual_norm = norm(residual);
disp(['Initial residual norm: ', num2str(residual_norm)]);

while(residual > tol)
    [u_solution, flag] = gmres(@(u) L_u(u), residual, [], gmrestol, maxiter);

    if flag==1
        sprintf(['gmres did not converge, residual is ' num2str(residual) ' after ' num2str(niter) ' iterations'])
        break
    end
    u=u-nincr;                         % Newton step
    niter=niter+1;  
end