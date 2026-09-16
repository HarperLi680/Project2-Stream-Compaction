#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "stream_compaction/cpu.h"
#include "stream_compaction/naive.h"
#include "stream_compaction/efficient.h"
#include "stream_compaction/thrust.h"

#ifndef BENCH_BLOCK_SIZE
#define BENCH_BLOCK_SIZE 128
#endif

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: benchmark array_size mode\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int mode = std::atoi(argv[2]);

    if (n <= 0 || mode < 0 || mode > 3) {
        return 1;
    }

    using namespace StreamCompaction;

    const char *names[] = {
        "CPU", "Naive", "Efficient", "Thrust"
    };

    std::vector<int> input(n);
    std::vector<int> output(n);
    std::vector<int> expected(n);

    for (int i = 0; i < n; ++i) {
        input[i] = (i * 7LL + 3) % 10;
    }

    std::exclusive_scan(
        input.begin(), input.end(), expected.begin(), 0);

    auto step = [&]() {
        switch (mode) {
            case 0:
                CPU::scan(n, output.data(), input.data());
                return CPU::timer().getCpuElapsedTimeForPreviousOperation();
            case 1:
                Naive::scan(n, output.data(), input.data());
                return Naive::timer().getGpuElapsedTimeForPreviousOperation();
            case 2:
                Efficient::scan(n, output.data(), input.data());
                return Efficient::timer().getGpuElapsedTimeForPreviousOperation();
            default:
                Thrust::scan(n, output.data(), input.data());
                return Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        }
    };

    auto verify = [&]() {
        if (output != expected) {
            std::cerr << names[mode] << ": incorrect scan result\n";
            std::exit(1);
        }
    };

    const int warmup = 5;
    const int repeats = 20;

    for (int i = 0; i < warmup; ++i) {
        step();
        verify();
    }

    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
        float elapsed = step();
        verify();
        times.push_back(elapsed);
    }

    std::sort(times.begin(), times.end());
    float median = (times[repeats / 2 - 1] + times[repeats / 2]) / 2;

    std::cout << names[mode] << ","
              << n << ","
              << BENCH_BLOCK_SIZE << ","
              << repeats << ","
              << std::fixed << std::setprecision(6)
              << median << "\n";

    return 0;
}
