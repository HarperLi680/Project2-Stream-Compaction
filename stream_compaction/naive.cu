#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 512;

        __global__ void kernShift(int n, int *output, const int *input) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            output[i] = i == 0 ? 0 : input[i - 1];
        }

        __global__ void kernScanStep(
            int n, int offset, int *output, const int *input) {

            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            output[i] = input[i];

            if (i >= offset) {
                output[i] += input[i - offset];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            int *devA;
            int *devB;

            cudaMalloc((void**)&devA, n * sizeof(int));
            checkCUDAError("Allocate devA failed");

            cudaMalloc((void**)&devB, n * sizeof(int));
            checkCUDAError("Allocate devB failed");

            cudaMemcpy(devA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Copy input failed");

            int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            kernShift<<<blocks, blockSize>>>(n, devB, devA);
            checkCUDAError("Shift failed");

            int *input = devB;
            int *output = devA;

            int levels = ilog2ceil(n);

            for (int d = 0; d < levels; ++d) {
                int offset = 1 << d;

                kernScanStep<<<blocks, blockSize>>>(
                    n, offset, output, input);
                checkCUDAError("Scan step failed");

                int *temp = input;
                input = output;
                output = temp;
            }

            timer().endGpuTimer();
            checkCUDAError("Naive scan failed");

            cudaMemcpy(odata, input, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy output failed");

            cudaFree(devA);
            cudaFree(devB);
            checkCUDAError("Free buffers failed");
        }
    }
}