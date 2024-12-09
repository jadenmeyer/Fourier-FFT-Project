    % Setup parameters
    par.n = 20000000;        % Large number of points in discretization
    M = 100;              % Large domain in units of 2π
    par.L = M * pi;      % Physical domain size
    par.x = linspace(-par.L, par.L, par.n+1)'; 
    par.x = par.x(1:end-1); % Spatial grid (periodic domain)
    par.dx = 2 * par.L / par.n;
    
    % Move data to GPU
    u = gpuArray(sawtooth(par.x,0.5));          % Known solution u(x)
    a = gpuArray(1 + 0.5 * cos(par.x)); % Known coefficient a(x)
    
    % Fourier space setup (on GPU)
    k = gpuArray(([0:par.n/2 -par.n/2+1:-1]' / M)); % Wave numbers
    k2 = k.^2;                                      % k^2 for second derivative
    
    % Initial guess for f (on GPU)
    %f = gpuArray.zeros(size(u));
    f=gpuArray(u);
    % Define residual function on GPU
    R = @(f) f - real(ifft(-k2 .* fft(u))) + a .* u;
    
    % Preconditioner (on GPU)
    %pc = @(df) ifft(fft(df) ./ (1 + k2)); % Simple preconditioner
    pc = @(df) df ./ a;

    % Newton-GMRES parameters
    tol = 1e-6;          % Convergence tolerance
    max_iter = 100;       % Maximum Newton iterations
    gmres_tol = 1e-8;   % GMRES tolerance
    gmres_maxit = 100;    % GMRES max iterations
    
    % Newton loop (on GPU)
    tic;
    for iter = 1:max_iter
        % Compute residual
        residual = R(f);
        norm_residual = gather(norm(residual)); % Transfer residual norm to CPU for logging
        fprintf('Iteration %d: Residual norm = %.6e\n', iter, norm_residual);
    
        % Check convergence
        if norm_residual < tol
            fprintf('Converged to solution with residual %.6e\n', norm_residual);
            break;
        end
    
        % Solve for update using GMRES (GPU-enabled)
        [df, flag] = gmres(@(x) x, -residual, 40, gmres_tol, gmres_maxit, pc);
    
        if flag ~= 0
            error('GMRES failed to converge.');
        end
    
        % Update solution
        f = f + df;
    end
    toc;
    % Transfer result back to CPU for visualization
    f = gather(f);
    
    % Plot results
    figure;
    plot(par.x, f, 'LineWidth', 1.5);
    title('Computed f(x) (GPU Accelerated)');
    xlabel('x'); ylabel('f(x)');
