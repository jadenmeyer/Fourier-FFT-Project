#include <stdio.h>
#include <complex>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono> 
#include <fstream>  // For writing to a file


//weird ass macro since CUDA is horrible
#ifndef C_PI
#define C_PI 3.14159265358979323846f
#endif

// DFT Kernel for GPU
__global__ void DFT(cuFloatComplex* a, cuFloatComplex* b, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x; //this call tells us what like position the GPU is on (its weird)

    if (index < N) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

        // Perform the DFT 
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * C_PI * index * n / N; //CUDA C++ doesn't have e so we do this and use macros for pi
            cuFloatComplex w = make_cuFloatComplex(cos(angle), sin(angle));  // e^(-2*pi*i*k*n/N) yeah this
            cuFloatComplex input_value = a[n];

            sum = cuCaddf(sum, cuCmulf(input_value, w));
        }

        // Store the result in the output array
        b[index] = sum;
    }
}

// CPU DFT need complex since Fourier so good 
void cpuDFT(std::complex<float>* a, std::complex<float>* b, int N) {
    for (int k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * C_PI * k * n / N; //this annoyance again
            std::complex<float> w(cos(angle), sin(angle));  // e^(-2*pi*i*k*n/N)
            sum += a[n] * w;
        }
        b[k] = sum;  // Store the result in the output array
    }
}

//this is a function call to automate things I hate
void clearAndReinitMemory(cuFloatComplex** d_array, cuFloatComplex** h_array, int maxSize) {
    // Allocate the largest required array size for both host and device
    if (*d_array == nullptr) {
        cudaMalloc((void**)d_array, maxSize * sizeof(cuFloatComplex));
    }

    *h_array = new cuFloatComplex[maxSize]; // Allocate host memory (host is RAM not GDDRAM)

    // Initialize the array on the host with random RE and IM values
    for (int i = 0; i < maxSize; ++i) {
        float real = rand() % 1000;  // Initialize random RE part
        float imag = rand() % 1000;  // Initialize random IM part
        (*h_array)[i] = make_cuFloatComplex(real, imag);
    }

    // Copy the data from host to device (NEED TO DO OR GPU HAS NOTHING!!!)
    cudaMemcpy(*d_array, *h_array, maxSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
}

// Function to write GPU times to a file
void writeGpuTimesToFile(const std::vector<float>& DFTtimes, const std::vector<int>& sizes) {
    std::ofstream gpuFile("gpu_times_seconds.txt");  // File to store GPU times in seconds
    for (size_t i = 0; i < DFTtimes.size(); ++i) {
        gpuFile << sizes[i] << " " << DFTtimes[i] / 1000.0f << "\n"; // Convert ms to seconds
    }
    gpuFile.close();
}

// Function to write CPU times to a file
void writeCpuTimesToFile(const std::vector<float>& cpuTimes, const std::vector<int>& sizes) {
    std::ofstream cpuFile("cpu_times_seconds.txt");  // File to store CPU times in seconds
    for (size_t i = 0; i < cpuTimes.size(); ++i) {
        cpuFile << sizes[i] << " " << cpuTimes[i] << "\n";  // Store CPU time in seconds
    }
    cpuFile.close();
}

int main() {
    srand(time(0));

    int maxSize = 5000;  // Maximum size for arrays
    int stepSize = 1000; // Step size for array size
    int numArrays = maxSize / stepSize; // Number of arrays to create

    std::vector<float> DFTtimes;
    std::vector<float> cpuTimes;  // Vector to store CPU times
    std::vector<int> sizes;  // Vector to store sizes for plotting

    // Declare host and device pointers for GPU DFT
    cuFloatComplex* h_a = nullptr;
    cuFloatComplex* h_b = nullptr;
    cuFloatComplex* d_a = nullptr;
    cuFloatComplex* d_b = nullptr;

    // Declare host arrays for CPU DFT
    std::complex<float>* h_a_cpu = nullptr;
    std::complex<float>* h_b_cpu = nullptr;

    // Declare CUDA stream for parallel execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Event timers for CUDA kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for the largest possible array
    clearAndReinitMemory(&d_a, &h_a, maxSize);
    clearAndReinitMemory(&d_b, &h_b, maxSize);

    // Allocate memory for CPU arrays with memory fill
    h_a_cpu = new std::complex<float>[maxSize];
    h_b_cpu = new std::complex<float>[maxSize];

    for (int i = 0; i < numArrays; ++i) {
        int gpuSize = (i + 1) * stepSize; // Size of the current array for DFT
        float DFTtime = 0;

        // Print the current vector size before launching the kernel
        printf("Processing DFT for vector size: %d\n", gpuSize);

        // Copy the relevant portion of the array for the current size to the GPU
        cudaMemcpy(d_a, h_a, gpuSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // Set up the number of threads and blocks for the kernel
        int threadsinBlock = 256;
        int blocksPerGrid = (gpuSize + threadsinBlock - 1) / threadsinBlock;

        // Start the timer for GPU DFT
        cudaEventRecord(start, 0);

        // Launch the kernel for DFT (GPU) asynchronously using a stream
        DFT << < blocksPerGrid, threadsinBlock, 0, stream >> > (d_a, d_b, gpuSize);

        // Perform the CPU DFT and time it
        auto cpu_start = std::chrono::high_resolution_clock::now();
        printf("Processing DFT for vector size: %d (CPU)\n", gpuSize);

        // init the CPU array
        for (int j = 0; j < gpuSize; ++j) {
            h_a_cpu[j] = std::complex<float>(rand() % 1000, rand() % 1000);
        }

        // CPU DFT
        cpuDFT(h_a_cpu, h_b_cpu, gpuSize);

        // Stop the CPU timing
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
        cpuTimes.push_back(cpu_duration.count());  // Store CPU time in seconds
        printf("CPU DFT computation time: %.6f seconds\n", cpu_duration.count());

        // Synchronize the stream and stop the timer for GPU DFT
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Get time for the GPU DFT and store it
        cudaEventElapsedTime(&DFTtime, start, stop);
        DFTtimes.push_back(DFTtime);  // Store GPU time in ms
        sizes.push_back(gpuSize);  // Store the current vector size for plotting
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    delete[] h_a_cpu;
    delete[] h_b_cpu;

    // Destroy the stream
    cudaStreamDestroy(stream);

    // Write the data to separate files
    writeGpuTimesToFile(DFTtimes, sizes);  // Write GPU times in seconds
    writeCpuTimesToFile(cpuTimes, sizes);  // Write CPU times in seconds

    // Print total GPU computation time
    printf("Total GPU Computation Time: %f seconds\n", std::accumulate(DFTtimes.begin(), DFTtimes.end(), 0));

    return 0;
}
