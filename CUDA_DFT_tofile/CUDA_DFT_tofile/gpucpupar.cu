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


//weird ass macro since CUDA is horrible
#ifndef C_PI
#define C_PI 3.14159265358979323846f
#endif

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

int main() {
    srand(time(0));

    int maxSize = 500000;  // Maximum size for arrays
    int stepSize = 100000; // Step size for array size
    int numArrays = maxSize / stepSize; // Number of arrays to create

    std::vector<float> DFTtimes;

    // Declare host and device pointers for GPU DFT
    cuFloatComplex* h_a = nullptr;
    cuFloatComplex* h_b = nullptr;
    cuFloatComplex* d_a = nullptr;
    cuFloatComplex* d_b = nullptr;
    //calling nullptr for memory 

    // Declare host arrays for CPU DFT
    std::complex<float>* h_a_cpu = nullptr;
    std::complex<float>* h_b_cpu = nullptr;
    //this makes more sense since we are talking about


    // Declare CUDA stream for parallel execution
    /**************************************************************************
    * cuda stream allows for GPU to work alongside the CPU and is just a class
    * so we instantiated an object of stream
    ***************************************************************************/
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Event timers for CUDA kernel execution
    //also another class
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for the largest possible array
    clearAndReinitMemory(&d_a, &h_a, maxSize);
    clearAndReinitMemory(&d_b, &h_b, maxSize);

    // Allocate memory for CPU arrays with memory fill
    // got to be careful calling new! might cause leak
    h_a_cpu = new std::complex<float>[maxSize];
    h_b_cpu = new std::complex<float>[maxSize];

    for (int i = 0; i < numArrays; ++i) {
        int gpuSize = (i + 1) * stepSize; // Size of the current array for DFT (will be called in terminal)
        float DFTtime = 0;

        // Print the current vector size before launching the kernel
        printf("Processing DFT for vector size: %d\n", gpuSize);

        // Copy the relevant portion of the array for the current size to the GPU
        cudaMemcpy(d_a, h_a, gpuSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        // Set up the number of threads and blocks for the kernel
        int threadsinBlock = 256;
        int blocksPerGrid = (gpuSize + threadsinBlock - 1) / threadsinBlock;
        //want this to be dependent on data set size

        // Start the timer for GPU DFT (hence why we called the object)
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
        // chrono is the first library that showed up this seems to work
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
        printf("CPU DFT computation time: %.6f seconds\n", cpu_duration.count());

        // synch the stream and stop the timer for GPU DFT
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // get time for the GPU DFT and store it
        cudaEventElapsedTime(&DFTtime, start, stop);
        DFTtimes.push_back(DFTtime);
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    //remember we had special ones just for CPU
    delete[] h_a_cpu;
    delete[] h_b_cpu;

    // destructorize the stream
    cudaStreamDestroy(stream);

    // Print total GPU computation time
    //accumulate from numeric library
    printf("Total GPU Computation Time: %f seconds\n", std::accumulate(DFTtimes.begin(), DFTtimes.end(), 0.0f) / 1000);
    return 0;
}
