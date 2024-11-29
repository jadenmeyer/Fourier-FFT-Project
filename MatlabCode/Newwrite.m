g = gpuDevice;


% Display total and free GPU memory
disp(['Total GPU Memory: ', num2str(g.TotalMemory / 1e9), ' GB']);
disp(['Free GPU Memory: ', num2str(g.FreeMemory / 1e9), ' GB']);

Gsize = 0:10000:1000000000;  %max size vectorization

timeVal = zeros(size(Gsize));  %zero vector for time

for index = 1:length(Gsize)
    N = Gsize(index);

    A = single(rand(N, 1));

    success = false;
    attempts = 3;
    attempt = 1;
    try 
        
        disp(['Processing GPU: Column vector size:', num2str(N)])

        A_gpu = gpuArray(A);
        tic;
        fft(A_gpu);
        timeVal(index) = toc;
        success = true;
        
        if timeVal(index) > 4
            avgvec = [];
            disp(['GPU took too long recalculating'])
            for avg = 1:20
                tic;
                fft(A_gpu);
                avgvec(avg) = toc;
            end
            timeVal(index) = mean(avgvec);
        end

    catch ME
        disp(['ERROR OCCURED RETRYING:', num2str(index)])
        if attempt < attempts
            reset(g);
            pause(3);
            attempt = attempt + 1;
        end
    end

    if ~success
        disp(['Failed to process size ', num2str(N), ' after ', num2str(attempts), ' attempts']);
        continue;  % Skip to next iteration if all attempts fail
    end

    if mod(N, 100000000) == 0
        clear("A_gpu");
    end
end


totalGPUtime = sum(timeVal);
disp(['Total GPU computation time: ', num2str(totalGPUtime)]);


Csize = 0:10000:1000000000;  % Define CPU sizes
CtimeVal = zeros(size(Csize));  % Preallocate time array for CPU

for idx = 1:length(Csize)
    M = Csize(idx);
    B = single(rand(M, 1));  % Create random vector

    disp(['Processing CPU: Column vector size = ', num2str(M)]);
    tic;
    fft(B);  % Perform FFT on the CPU
    CtimeVal(idx) = toc;  % Store time taken
end

totalGPUtime = sum(timeVal)
totalCPUtime = sum(CtimeVal)

figure;
plot(Csize, CtimeVal, '-o')
xlabel('Matrix Size (M)');
ylabel('Time (seconds)');
title('Time to compute FFT on CPU');
grid on;


figure;
plot(Gsize, timeVal, '-o')
xlabel('Matrix Size (N)');
ylabel('Time (seconds)');
title('Time to compute FFT on GPU');
grid on;

figure;

plot(Csize, CtimeVal, '-o')
hold on;
plot(Gsize, timeVal, '-o')

% Labels and title for the combined plot
xlabel('Matrix Size');
ylabel('Time (seconds)');
title('Comparison of FFT Computation Time on CPU and GPU');
grid on;
legend('CPU', 'GPU');
% Turn off the hold to stop adding to this plot
hold off;