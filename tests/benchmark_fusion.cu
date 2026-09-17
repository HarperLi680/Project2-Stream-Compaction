#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include "stream_compaction/efficient.h"

void checked(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        return 1;
    }

    int n = std::atoi(argv[1]);
    int mode = std::atoi(argv[2]);

    if (n <= 0 || n > (1 << 22) || mode < 0 || mode > 1) {
        return 1;
    }

    int size = 1;
    while (size < n) {
        size *= 2;
    }

    std::vector<int> input(size, 0);
    std::vector<int> expected(size);
    std::vector<int> output(size);

    for (int i = 0; i < n; ++i) {
        input[i] = (i * 7LL + 3) % 10;
    }

    std::exclusive_scan(
        input.begin(), input.end(), expected.begin(), 0);

    int *data;
    checked(cudaMalloc((void**)&data, size * sizeof(int)));

    cudaEvent_t start, stop;
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));

    const int warmup = 5;
    const int samples = 20;
    std::vector<float> times;

    for (int trial = 0; trial < warmup + samples; ++trial) {
        // Restore the original input before each in-place scan.
        checked(cudaMemcpy(
            data, input.data(), size * sizeof(int),
            cudaMemcpyHostToDevice));
        checked(cudaDeviceSynchronize());

        checked(cudaEventRecord(start));

        if (mode == 0) {
            StreamCompaction::Efficient::scanDeviceBaseline(size, data);
        } else {
            StreamCompaction::Efficient::scanDevice(size, data);
        }

        checked(cudaEventRecord(stop));
        checked(cudaEventSynchronize(stop));

        float milliseconds;
        checked(cudaEventElapsedTime(&milliseconds, start, stop));

        // Verify every result outside the timing interval.
        checked(cudaMemcpy(
            output.data(), data, size * sizeof(int),
            cudaMemcpyDeviceToHost));

        if (output != expected) {
            fprintf(stderr, "FAIL: n=%d mode=%d trial=%d\n",
                    n, mode, trial);
            return 1;
        }

        if (trial >= warmup) {
            times.push_back(milliseconds);
        }
    }

    std::sort(times.begin(), times.end());
    float median = (times[9] + times[10]) * 0.5f;

    printf("%s,%d,%d,%.6f\n",
           mode == 0 ? "Baseline" : "Fused",
           n, samples, median);

    checked(cudaEventDestroy(start));
    checked(cudaEventDestroy(stop));
    checked(cudaFree(data));
    return 0;
}
