gpuDevice(1);  % Select GPU device (if you have multiple GPUs)
g = gpuDevice();  % Get the current GPU device information

% Display total and free GPU memory
disp(['Total GPU Memory: ', num2str(g.TotalMemory / 1e9), ' GB']);
disp(['Free GPU Memory: ', num2str(g.FreeMemory / 1e9), ' GB']);

% Declare the range of matrix sizes (e.g., from 0 to 1 billion elements, in steps of 150,000)
Gsize = 0:1000000:1000000000;  % Example sizes: 0, 150000, 300000, ..., up to 1 billion

timeVal = zeros(size(Gsize));  % To store the time values for each iteration

% Loop through different matrix sizes
for index = 1:length(Gsize)
    N = Gsize(index);  % Get the current size
    
    % Check memory before allocation
    memoryUsed = g.TotalMemory - g.FreeMemory;  % Used memory
    disp(['Memory used before iteration ', num2str(index), ': ', num2str(memoryUsed / 1e9), ' GB']);
    
    % Create a random matrix of size N on the CPU
    A = single(rand(N, 1));  % Create a column vector of size N
    
    % Try-catch block for handling GPU memory issues
    success = false;  % Flag to check if operation was successful
    attempts = 3;  % Number of retry attempts
    
    for attempt = 1:attempts
        try
            % Transfer the data to the GPU
            A_gpu = gpuArray(A);  % Transfer the matrix to GPU
            
            % Perform FFT on the GPU only if the transfer is successful
            tic;  % Start the timer here (only for successful operations)
            fft(A_gpu);  % Perform FFT directly on the GPU array
            timeVal(index) = toc;  % Record the time for FFT computation
            
            % If successful, break out of the retry loop
            success = true;
            break;  % Exit retry loop if operation is successful
            
        catch ME
            % If an error occurs (likely memory-related), display the error
            disp(['Error during iteration ', num2str(index), ' (attempt ', num2str(attempt), '): ', ME.message]);
            
            % If not the last attempt, clear GPU memory and try again
            if attempt < attempts
                disp('Clearing GPU memory and retrying...');
                reset(g);  % Reset GPU device to free memory
                pause(2);  % Pause for a brief moment before retrying
            end
        end
    end
    
    if ~success
        disp(['Failed to process size ', num2str(N), ' after ', num2str(attempts), ' attempts']);
        continue;  % Skip to next iteration if all attempts fail
    end
    
    % Clear the GPU array to free memory after each iteration
    clear A_gpu;  % Explicitly clear the GPU array
    
    % Recheck memory after clearing
    g = gpuDevice();  % Re-query GPU device for updated memory info
    disp(['Free GPU Memory after iteration ', num2str(index), ': ', num2str(g.FreeMemory / 1e9), ' GB']);
end

% Calculate the total GPU computation time
totalGPUtime = sum(timeVal);
disp(['Total GPU computation time: ', num2str(totalGPUtime)]);


Csize = 0:1000000:1000000000;         % Define the sizes (from 0 to 1 million with step of 2)
CtimeVal = zeros(size(Csize));  % Preallocate the time array to store results

maxC = max(Csize);
B = zeros(maxC, 1, 'single'); 

for idx = 1:length(Csize)
    M = Csize(idx);            % Current size of the column vector
    
    % Only create the vector of size M
    B = single(rand(M, 1));    % Create a column vector (M x 1)
    
    disp(['Processing CPU: Column vector size = ', num2str(M)]);
    
    % Perform FFT and measure the time
    tic;
    fft(B);                    % Perform FFT on the vector
    CtimeVal(idx) = toc;       % Store the time taken for this iteration
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