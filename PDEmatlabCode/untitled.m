% Define grid %
n = 64; % Size of matrix
size = 1; % Size of domain (units of 2pi)
D = size*pi;
x = linspace(-D, D, n+1)'; x=x(1:end-1);
dx = 2*D/n; % Grid spacing
sp = 1.5; % System parameter
wn = 1; % Wave number
% Construct matrix %
v = ([[0:n/2] [-n/2+1:-1]]./size)'; % vector construction for matrix
v2 = v.^2;
v4 = v2.^2;
par.s = 0 % unsure what this does?
DSH = -(wn^4*v4 - 2*wn^2*v2 + 1 + par.s*i*v - sp) % constructs differential operator
PC = @(du)[ifft(fft(du(1:end-1))./(DSH-(1+sp)*ones(n,1)));du(end)]; % preconditioner
IG = sqrt(4*(sp-(wn-1)^2)/3)*cos(x); % initial guess
IG = [IG;0]; % append
ftIG = fft(IG) % FT of IG (needed for finding residual in Fourier space)
% Discretize uxx? %
% Define residual function + Jacobian (linearization) %
SHRolla=@(IG,parr)[real(ifft(DSH.*fft(IG(1:end-1))))- IG(1:end-1).^3;sin(x)'*IG(1:end-1)];
SHRoll= @(IG) SHRolla(IG,par);
DSHRolla= @(IG,du,par) ...
 [real(ifft(DSH.*fft(du(1:end-1))))-3*IG(1:end-1).^2.*du(1:end-1)+ ...
 + real(ifft(-i*k.*fft(IG(1:end-1))))*du(end);...
 sin(x)'*du(1:end-1)];
% Newton parameters (pulled from GetRoll.m) %
npar.tol=1e-6;
npar.maxiter = 50;
npar.niter = 1;
npar.nitermax= 35;
npar.nincr=ones(n+1,1);
npar.minstep=-1e-8;
npar.rhs=0;
% Compute FT of u
function F=ResSH(ftIG,par)
 F = DSH.*ftIG-fft(real(ifft(ftIG)).^3);
end
% Compute residuals 1
npar.residual=norm(SHRoll(IG)); % initial residual
npar.gmrestol=1e-8;
npar.rhs=SHRoll(IG); % computes residual
npar.residual=norm(npar.rhs); % estimates residual
tic
% Newton loop (pulled from GetRoll.m) %
while (npar.residual>npar.tol)
%
 DSHRoll=@(du) DSHRolla(IG,du,par); % form Jacobian
 [npar.nincr,flag]=gmres(DSHRoll,npar.rhs,20,npar.gmrestol,25,PC);
% [npar.nincr,flag]=gmres(DSHroll,npar.rhs,20,npar.gmrestol,25,pc);
 % gmres solve for increment
 if flag==1
 sprintf(['gmres did not converge, residual is ' num2str(npar.residual) ' after ' num2str(npar.niter) ' iterations'])
 break
 end
 IG=IG-npar.nincr; % Newton step
 npar.niter=npar.niter+1; %
keep track of number of iterations
 %
 % recompute residual
 npar.rhs=SHRoll(IG); % compute residual
 npar.residual=norm(npar.rhs); % estimate residual
 %
 if npar.niter>npar.nitermax
 sprintf(['Maximal number of iterations reached, giving up; residual is ' num2str(npar.residual)])
 break
 end
 %
 if norm(npar.nincr)<npar.minstep
 sprintf(['Newton step is ineffective, giving up; residual is ' num2str(npar.residual)])
 break
 end
%
end
toc