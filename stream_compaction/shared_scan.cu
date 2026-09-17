#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "common.h"
#include "efficient.h"
#include "shared_scan.h"

namespace StreamCompaction {
    namespace SharedScan {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Each block scans two elements per thread
        __global__ void kernNaive(
            int n, int *output, const int *input,
            int *blockSums) {

            extern __shared__ int data[];

            int thread = threadIdx.x;
            int tileSize = 2 * blockDim.x;
            int base = blockIdx.x * tileSize;

            int read = 0;
            int write = tileSize;

            for (int j = thread; j < tileSize; j += blockDim.x) {
                int source = base + j - 1;

                data[j] = (j > 0 && source < n)
                    ? input[source]
                    : 0;
            }

            __syncthreads();

            for (int offset = 1; offset < tileSize; offset *= 2) {
                for (int j = thread; j < tileSize; j += blockDim.x) {
                    int value = data[read + j];

                    if (j >= offset) {
                        value += data[read + j - offset];
                    }

                    data[write + j] = value;
                }

                __syncthreads();

                int temp = read;
                read = write;
                write = temp;
            }

            for (int j = thread; j < tileSize; j += blockDim.x) {
                if (base + j < n) {
                    output[base + j] = data[read + j];
                }
            }

            if (thread == 0) {
                int last = base + tileSize - 1;
                int lastValue = last < n ? input[last] : 0;

                blockSums[blockIdx.x] =
                    data[read + tileSize - 1] + lastValue;
            }
        }

        template <bool padded>
        __device__ int sharedIndex(int index) {
            return padded
                ? index + (index >> 5) + (index >> 10)
                : index;
        }

        template <bool padded>
        __global__ void kernEfficient(
            int n, int *output, const int *input,
            int *blockSums) {

            extern __shared__ int data[];

            int thread = threadIdx.x;
            int tileSize = 2 * blockDim.x;
            int base = blockIdx.x * tileSize;

            // Fill the last partial tile with zeros
            for (int j = thread; j < tileSize; j += blockDim.x) {
                data[sharedIndex<padded>(j)] =
                    base + j < n ? input[base + j] : 0;
            }

            __syncthreads();

            // Upsweep
            for (int stride = 2; stride <= tileSize; stride *= 2) {
                int count = tileSize / stride;

                if (thread < count) {
                    int right = (thread + 1) * stride - 1;
                    int left = right - stride / 2;

                    data[sharedIndex<padded>(right)] +=
                        data[sharedIndex<padded>(left)];
                }

                __syncthreads();
            }

            if (thread == 0) {
                int root = sharedIndex<padded>(tileSize - 1);

                blockSums[blockIdx.x] = data[root];
                data[root] = 0;
            }

            __syncthreads();

            // Downsweep
            for (int stride = tileSize; stride >= 2; stride /= 2) {
                int count = tileSize / stride;

                if (thread < count) {
                    int right = sharedIndex<padded>(
                        (thread + 1) * stride - 1);

                    int left = sharedIndex<padded>(
                        (thread + 1) * stride - 1 - stride / 2);

                    int temp = data[left];
                    data[left] = data[right];
                    data[right] += temp;
                }

                __syncthreads();
            }

            for (int j = thread; j < tileSize; j += blockDim.x) {
                if (base + j < n) {
                    output[base + j] =
                        data[sharedIndex<padded>(j)];
                }
            }
        }

        __global__ void kernAddBlockOffsets(
            int n, int *output, const int *blockOffsets) {

            int tileSize = 2 * blockDim.x;
            int base = blockIdx.x * tileSize;
            int offset = blockOffsets[blockIdx.x];

            for (int j = threadIdx.x; j < tileSize; j += blockDim.x) {
                if (base + j < n) {
                    output[base + j] += offset;
                }
            }
        }

        // 0 = naive, 1 = efficient, 2 = efficient with padding
        static void scan(
            int n, int *odata, const int *idata,
            int blockSize, int mode) {

            if (blockSize < 1 || blockSize > 1024 ||
                (blockSize & (blockSize - 1)) != 0) {

                throw std::invalid_argument(
                    "Block size must be a power of two from 1 to 1024");
            }

            if (n <= 0) {
                timer().startGpuTimer();
                timer().endGpuTimer();
                return;
            }

            int tileSize = 2 * blockSize;
            int blocks = (n - 1) / tileSize + 1;
            int sumsSize = 1 << ilog2ceil(blocks);

            int *devInput;
            int *devOutput;
            int *devSums;

            cudaMalloc((void**)&devInput, n * sizeof(int));
            checkCUDAError("Allocate shared scan input failed");

            cudaMalloc((void**)&devOutput, n * sizeof(int));
            checkCUDAError("Allocate shared scan output failed");

            cudaMalloc((void**)&devSums, sumsSize * sizeof(int));
            checkCUDAError("Allocate block sums failed");

            cudaMemcpy(
                devInput, idata, n * sizeof(int),
                cudaMemcpyHostToDevice);
            checkCUDAError("Copy shared scan input failed");

            cudaMemset(devSums, 0, sumsSize * sizeof(int));
            checkCUDAError("Initialize block sums failed");

            timer().startGpuTimer();

            if (mode == 0) {
                size_t sharedBytes = 2 * tileSize * sizeof(int);

                kernNaive<<<blocks, blockSize, sharedBytes>>>(
                    n, devOutput, devInput, devSums);
            } else if (mode == 1) {
                size_t sharedBytes = tileSize * sizeof(int);

                kernEfficient<false>
                    <<<blocks, blockSize, sharedBytes>>>(
                        n, devOutput, devInput, devSums);
            } else {
                int last = tileSize - 1;
                int sharedElements =
                    tileSize + (last >> 5) + (last >> 10);

                size_t sharedBytes = sharedElements * sizeof(int);

                kernEfficient<true>
                    <<<blocks, blockSize, sharedBytes>>>(
                        n, devOutput, devInput, devSums);
            }

            checkCUDAError("Shared scan kernel failed");

            if (blocks > 1) {
                Efficient::scanDevice(sumsSize, devSums);

                kernAddBlockOffsets<<<blocks, blockSize>>>(
                    n, devOutput, devSums);

                checkCUDAError("Add block offsets failed");
            }

            timer().endGpuTimer();
            checkCUDAError("Shared scan failed");

            cudaMemcpy(
                odata, devOutput, n * sizeof(int),
                cudaMemcpyDeviceToHost);
            checkCUDAError("Copy shared scan output failed");

            cudaFree(devInput);
            cudaFree(devOutput);
            cudaFree(devSums);
            checkCUDAError("Free shared scan buffers failed");
        }

        void scanNaive(
            int n, int *odata, const int *idata,
            int blockSize) {

            scan(n, odata, idata, blockSize, 0);
        }

        void scanEfficient(
            int n, int *odata, const int *idata,
            int blockSize) {

            scan(n, odata, idata, blockSize, 1);
        }

        void scanEfficientPadded(
            int n, int *odata, const int *idata,
            int blockSize) {

            scan(n, odata, idata, blockSize, 2);
        }
    }
}