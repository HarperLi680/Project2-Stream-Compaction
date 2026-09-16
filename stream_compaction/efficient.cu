#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 128;

        __global__ void kernUpsweep(int count, int stride, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= count) {
                return;
            }

            int right = (i + 1) * stride - 1;
            int left = right - stride / 2;

            data[right] += data[left];
        }

        __global__ void kernResetRoot(int size, int *data) {
            data[size - 1] = 0;
        }

        __global__ void kernDownsweep(int count, int stride, int *data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= count) {
                return;
            }

            int right = (i + 1) * stride - 1;
            int left = right - stride / 2;

            int temp = data[left];
            data[left] = data[right];
            data[right] += temp;
        }

        void scanDevice(int size, int *data) {
            int levels = ilog2ceil(size);

            for (int d = 0; d < levels; ++d) {
                int stride = 1 << (d + 1);
                int count = size / stride;
                int blocks = (count + blockSize - 1) / blockSize;

                kernUpsweep<<<blocks, blockSize>>>(count, stride, data);
                checkCUDAError("Upsweep failed");
            }

            kernResetRoot<<<1, 1>>>(size, data);
            checkCUDAError("Reset root failed");

            for (int d = levels - 1; d >= 0; --d) {
                int stride = 1 << (d + 1);
                int count = size / stride;
                int blocks = (count + blockSize - 1) / blockSize;

                kernDownsweep<<<blocks, blockSize>>>(count, stride, data);
                checkCUDAError("Downsweep failed");
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

            int size = 1 << ilog2ceil(n);
            int *devData;

            cudaMalloc((void**)&devData, size * sizeof(int));
            checkCUDAError("Allocate scan buffer failed");

            cudaMemset(devData, 0, size * sizeof(int));
            checkCUDAError("Clear scan buffer failed");

            cudaMemcpy(devData, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);
            checkCUDAError("Copy scan input failed");

            timer().startGpuTimer();

            scanDevice(size, devData);

            timer().endGpuTimer();
            checkCUDAError("Efficient scan failed");

            cudaMemcpy(odata, devData, n * sizeof(int),
                       cudaMemcpyDeviceToHost);
            checkCUDAError("Copy scan output failed");

            cudaFree(devData);
            checkCUDAError("Free scan buffer failed");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return 0;
            }

            int size = 1 << ilog2ceil(n);
            int blocks = (n + blockSize - 1) / blockSize;

            int *devInput;
            int *devOutput;
            int *devFlags;
            int *devIndices;

            cudaMalloc((void**)&devInput, n * sizeof(int));
            checkCUDAError("Allocate input failed");

            cudaMalloc((void**)&devOutput, n * sizeof(int));
            checkCUDAError("Allocate output failed");

            cudaMalloc((void**)&devFlags, size * sizeof(int));
            checkCUDAError("Allocate flags failed");

            cudaMalloc((void**)&devIndices, size * sizeof(int));
            checkCUDAError("Allocate indices failed");

            cudaMemcpy(devInput, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);
            checkCUDAError("Copy compaction input failed");

            cudaMemset(devFlags, 0, size * sizeof(int));
            checkCUDAError("Clear flags failed");

            timer().startGpuTimer();

            Common::kernMapToBoolean<<<blocks, blockSize>>>(
                n, devFlags, devInput);
            checkCUDAError("Map failed");

            cudaMemcpy(devIndices, devFlags, size * sizeof(int),
                       cudaMemcpyDeviceToDevice);
            checkCUDAError("Copy flags failed");

            scanDevice(size, devIndices);

            Common::kernScatter<<<blocks, blockSize>>>(
                n, devOutput, devInput, devFlags, devIndices);
            checkCUDAError("Scatter failed");

            timer().endGpuTimer();
            checkCUDAError("Efficient compaction failed");

            int count;
            cudaMemcpy(&count, devIndices + n - 1, sizeof(int),
                       cudaMemcpyDeviceToHost);
            checkCUDAError("Copy count failed");

            count += idata[n - 1] != 0 ? 1 : 0;

            if (count > 0) {
                cudaMemcpy(odata, devOutput, count * sizeof(int),
                           cudaMemcpyDeviceToHost);
                checkCUDAError("Copy compaction output failed");
            }

            cudaFree(devInput);
            cudaFree(devOutput);
            cudaFree(devFlags);
            cudaFree(devIndices);
            checkCUDAError("Free compaction buffers failed");

            return count;
        }
    }
}