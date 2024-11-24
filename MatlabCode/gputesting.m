gpuDevice(1);
if gpuDeviceCount > 0
    disp('GPU is available.');
else
    disp('GPU is not available.');
    return;
end

Gsize = 0:15000000:1000000000; 
timeVal = zeros(size(Gsize));

% Pre-allocate a large enough GPU array to hold the largest matrix
maxSize = max(Gsize);  
A_gpu = gpuArray.zeros(maxSize, 1, 'single');  % Pre-allocate GPU memory for column vector

% Loop over different matrix sizes
for index = 1:length(Gsize)
    N = Gsize(index);
    
    % Create a random matrix of size N on the CPU
    A = single(rand(N, 1));  % Create a column vector of size N
    
    % Transfer the data to the GPU (only the relevant part)
    A_gpu(1:N) = gpuArray(A);  % Assign to the first N elements of the column vector
    disp(['Processing GPU: Matrix size = ', num2str(N)]);
    % Perform FFT on the GPU (now it's just a vector computation)
    tic;
    fft(A_gpu(1:N));  % Perform FFT on the relevant portion of the GPU array (N x 1 vector)
    timeVal(index) = toc;  % Record the time
end


totalGPUtime = sum(timeVal)

Csize = 0:15000000:1000000000;         % Define the sizes (from 0 to 1 million with step of 2)
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