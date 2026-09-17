#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix_sort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 128;

        __device__ unsigned int sortKey(int value) {
            return static_cast<unsigned int>(value) ^ 0x80000000u;
        }

        __global__ void kernMarkZeros(
            int n, int size, int bit,
            const int *input, int *indices) {

            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= size) {
                return;
            }

            if (i < n) {
                unsigned int key = sortKey(input[i]);
                indices[i] = ((key >> bit) & 1u) == 0u ? 1 : 0;
            } else {
                indices[i] = 0;
            }
        }

        __global__ void kernScatterBit(
            int n, int bit, const int *input,
            const int *indices, int *output) {

            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            unsigned int key = sortKey(input[i]);
            bool isZero = ((key >> bit) & 1u) == 0u;

            unsigned int lastKey = sortKey(input[n - 1]);
            int lastIsZero = ((lastKey >> bit) & 1u) == 0u ? 1 : 0;
            int totalZeros = indices[n - 1] + lastIsZero;

            int destination;
            if (isZero) {
                destination = indices[i];
            } else {
                destination = totalZeros + i - indices[i];
            }

            output[destination] = input[i];
        }

        /**
         * Sorts signed integers in ascending order using scan-based radix sort.
         */
        void sort(int n, int *odata, const int *idata) {
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            int size = 1 << ilog2ceil(n);
            int blocks = (n + blockSize - 1) / blockSize;
            int paddedBlocks = (size + blockSize - 1) / blockSize;

            int *devA;
            int *devB;
            int *devIndices;

            cudaMalloc((void**)&devA, n * sizeof(int));
            checkCUDAError("Allocate radix input failed");

            cudaMalloc((void**)&devB, n * sizeof(int));
            checkCUDAError("Allocate radix output failed");

            cudaMalloc((void**)&devIndices, size * sizeof(int));
            checkCUDAError("Allocate radix indices failed");

            cudaMemcpy(devA, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);
            checkCUDAError("Copy radix input failed");

            int *input = devA;
            int *output = devB;

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; ++bit) {
                kernMarkZeros<<<paddedBlocks, blockSize>>>(
                    n, size, bit, input, devIndices);
                checkCUDAError("Mark radix bits failed");

                Efficient::scanDevice(size, devIndices);

                kernScatterBit<<<blocks, blockSize>>>(
                    n, bit, input, devIndices, output);
                checkCUDAError("Scatter radix bits failed");

                int *temp = input;
                input = output;
                output = temp;
            }

            timer().endGpuTimer();
            checkCUDAError("Radix sort failed");

            cudaMemcpy(odata, input, n * sizeof(int),
                       cudaMemcpyDeviceToHost);
            checkCUDAError("Copy radix output failed");

            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devIndices);
            checkCUDAError("Free radix buffers failed");
        }
    }
}