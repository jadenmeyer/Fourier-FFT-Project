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
#include <fstream>

// Weird ass macro since CUDA is horrible
#ifndef C_PI
#define C_PI 3.14159265358979323846f
#endif

// DFT Kernel for GPU
__global__ void DFT(cuFloatComplex* a, cuFloatComplex* b, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

        // Perform the DFT 
        for (int n = 0; n < N; ++n) {
            // Print which iteration (index for frequency bin, n for time domain sample)
            

            float angle = -2.0f * C_PI * index * n / N;
            cuFloatComplex w = make_cuFloatComplex(cos(angle), sin(angle));
            cuFloatComplex input_value = a[n];

            sum = cuCaddf(sum, cuCmulf(input_value, w));
        }

        // Store the result in the output array
        //b[index] = sum;
        printf("Thread %d finished processing element %d\n", index, index);
    }
}

// CPU DFT
void cpuDFT(std::complex<float>* a, std::complex<float>* b, int N) {
    for (int k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * C_PI * k * n / N;
            std::complex<float> w(cos(angle), sin(angle));
            sum += a[n] * w;
        }
        b[k] = sum;
        std::cout << "number of iterations:" << k << std::endl;
    }
}

// Initialize memory
void clearAndReinitMemory(cuFloatComplex** d_array, cuFloatComplex** h_array, int maxSize) {
    if (*d_array == nullptr) {
        cudaMalloc((void**)d_array, maxSize * sizeof(cuFloatComplex));
    }

    *h_array = new cuFloatComplex[maxSize];

    // Initialize the array with random values
    for (int i = 0; i < maxSize; ++i) {
        float real = rand() % 1000;
        float imag = rand() % 1000;
        (*h_array)[i] = make_cuFloatComplex(real, imag);
    }

    cudaMemcpy(*d_array, *h_array, maxSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
}

// Write GPU times to a file
void writeGpuTimesToFile(const std::vector<float>& DFTtimes, const std::vector<int>& sizes) {
    std::ofstream gpuFile("gpu_times_seconds.txt");
    for (size_t i = 0; i < DFTtimes.size(); ++i) {
        gpuFile << sizes[i] << " " << DFTtimes[i] / 1000.0f << "\n"; // Convert ms to seconds
    }
    gpuFile.close();
}

// Write CPU times to a file
void writeCpuTimesToFile(const std::vector<float>& cpuTimes, const std::vector<int>& sizes) {
    std::ofstream cpuFile("cpu_times_seconds.txt");
    for (size_t i = 0; i < cpuTimes.size(); ++i) {
        cpuFile << sizes[i] << " " << cpuTimes[i] << "\n";
    }
    cpuFile.close();
}

// Handle CUDA FFT
void cudaFFT(cufftComplex* d_data, int N) {
    cufftHandle plan;
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create a 1D complex-to-complex FFT plan
    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Plan creation failed!" << std::endl;
        return;
    }

    cudaEventRecord(start, 0);

    // Execute cuFFT
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT execution failed!" << std::endl;
        return;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Get time for cuFFT execution
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "CUDA FFT execution time: " << elapsedTime << " ms" << std::endl;

    cufftDestroy(plan);
}

int main() {
    srand(time(0));

    int maxSize = 10000000;
    int stepSize = 1000000;
    int numArrays = maxSize / stepSize;

    std::vector<float> DFTtimes, fftTimes, cpuTimes;
    std::vector<int> sizes;

    cuFloatComplex* h_a = nullptr;
    cuFloatComplex* h_b = nullptr;
    cuFloatComplex* d_a = nullptr;
    cuFloatComplex* d_b = nullptr;

    std::complex<float>* h_a_cpu = nullptr;
    std::complex<float>* h_b_cpu = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    clearAndReinitMemory(&d_a, &h_a, maxSize);
    clearAndReinitMemory(&d_b, &h_b, maxSize);

    h_a_cpu = new std::complex<float>[maxSize];
    h_b_cpu = new std::complex<float>[maxSize];

    for (int i = 0; i < numArrays; ++i) {
       /* if (i >= 100) {
            stepSize = 1000000;
        }*/

        int gpuSize = (i + 1) * stepSize;

        float DFTtime = 0;
        float fftTime = 0;

        printf("Processing DFT for vector size: %d\n", gpuSize);

        // Handle device memory
        cudaMemcpy(d_a, h_a, gpuSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // Configure kernel launch
        int threadsPerBlock = 1024;
        int blocksPerGrid = (gpuSize + threadsPerBlock - 1) / threadsPerBlock;
        std::cout << "successful output of 1024 threads" << std::endl;

        // Timing GPU DFT
        cudaEventRecord(start, 0);
        printf("DFT set to go to kernel");
        DFT << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_a, d_b, gpuSize);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&DFTtime, start, stop);

        // CPU DFT
        std::cout << "going to CPU DFT now" << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < gpuSize; ++j) {
            h_a_cpu[j] = std::complex<float>(rand() % 1000, rand() % 1000);
        }
        cpuDFT(h_a_cpu, h_b_cpu, gpuSize);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
        cpuTimes.push_back(cpu_duration.count());
        printf("CPU DFT computation time: %.6f seconds\n", cpu_duration.count());

        // Sync stream and get GPU DFT time
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&DFTtime, start, stop);
        DFTtimes.push_back(DFTtime);
        sizes.push_back(gpuSize);

        std::cout << "CUDA DFT execution time: " << DFTtimes.at(i) << " ms" << std::endl;

        // Timing cuFFT
        cudaEventRecord(start, 0);
        cufftComplex* d_data = reinterpret_cast<cufftComplex*>(d_a);
        cudaFFT(d_data, gpuSize);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&fftTime, start, stop);
        fftTimes.push_back(fftTime);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    delete[] h_a_cpu;
    delete[] h_b_cpu;
    cudaStreamDestroy(stream);

    writeGpuTimesToFile(DFTtimes, sizes);
    writeCpuTimesToFile(cpuTimes, sizes);
    writeGpuTimesToFile(fftTimes, sizes);

    printf("Total GPU DFT Time: %.6f seconds\n", std::accumulate(DFTtimes.begin(), DFTtimes.end(), 0.0f) / 1000.0f);
    printf("Total CPU DFT Time: %.6f seconds\n", std::accumulate(cpuTimes.begin(), cpuTimes.end(), 0.0f));

    return 0;
}
