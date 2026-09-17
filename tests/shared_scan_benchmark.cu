#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "stream_compaction/shared_scan.h"
#include "stream_compaction/efficient.h"

using ScanFunction = void (*)(int, int *, const int *, int);

struct Configuration {
    const char *name;
    int blockSize;
    ScanFunction scan;
    bool shared;
};

static void scanFused(
    int n, int *output, const int *input, int blockSize) {

    StreamCompaction::Efficient::scan(n, output, input);
}

static double benchmark(
    const Configuration& config,
    const std::vector<int>& input,
    const std::vector<int>& expected) {

    int n = static_cast<int>(input.size());
    std::vector<int> output(n);
    std::vector<float> times;

    for (int call = 0; call < 25; ++call) {
        config.scan(
            n, output.data(), input.data(), config.blockSize);

        float elapsed = config.shared
            ? StreamCompaction::SharedScan::timer()
                .getGpuElapsedTimeForPreviousOperation()
            : StreamCompaction::Efficient::timer()
                .getGpuElapsedTimeForPreviousOperation();

        // Check every call, outside the timed interval.
        if (output != expected) {
            std::fprintf(
                stderr,
                "FAIL: %s, n=%d, block=%d, call=%d\n",
                config.name, n, config.blockSize, call);

            std::exit(EXIT_FAILURE);
        }

        if (call >= 5) {
            times.push_back(elapsed);
        }
    }

    std::sort(times.begin(), times.end());
    return (static_cast<double>(times[9]) + times[10]) / 2.0;
}

int main() {
    std::vector<Configuration> configurations;

    for (int blockSize : {64, 128, 256, 512}) {
        configurations.push_back({
            "SharedNaive",
            blockSize,
            StreamCompaction::SharedScan::scanNaive,
            true
        });

        configurations.push_back({
            "SharedEfficient",
            blockSize,
            StreamCompaction::SharedScan::scanEfficient,
            true
        });

        configurations.push_back({
            "SharedPadded",
            blockSize,
            StreamCompaction::SharedScan::scanEfficientPadded,
            true
        });
    }

    // The existing Efficient implementation uses 128 threads/block.
    configurations.push_back({
        "FusedEfficient", 128, scanFused, false
    });

    const std::vector<int> sizes = {
        256,
        4096,
        65536,
        262144,
        1048576,
        4194304
    };

    std::printf("run,mode,array_size,block_size,samples,median_ms\n");

    for (int run = 1; run <= 3; ++run) {
        std::vector<int> orderedSizes = sizes;

        if (run == 2) {
            std::reverse(orderedSizes.begin(), orderedSizes.end());
        }

        for (int n : orderedSizes) {
            // Use the same input for every configuration and run.
            std::mt19937 generator(42);
            std::uniform_int_distribution<int> distribution(-10, 10);

            std::vector<int> input(n);
            std::vector<int> expected(n);

            int sum = 0;
            for (int i = 0; i < n; ++i) {
                input[i] = distribution(generator);
                expected[i] = sum;
                sum += input[i];
            }

            std::vector<Configuration> order = configurations;

            // Change configuration order to reduce ordering bias.
            if (run == 2) {
                std::reverse(order.begin(), order.end());
            } else if (run == 3) {
                std::rotate(order.begin(), order.begin() + 5, order.end());
            }

            for (const Configuration& config : order) {
                double elapsed = benchmark(config, input, expected);

                std::printf(
                    "%d,%s,%d,%d,20,%.6f\n",
                    run, config.name, n, config.blockSize, elapsed);

                std::fflush(stdout);
            }
        }
    }

    return 0;
}
