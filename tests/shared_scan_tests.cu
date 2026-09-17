#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <vector>
#include "stream_compaction/shared_scan.h"

using ScanFunction = void (*)(int, int *, const int *, int);

static int totalTests = 0;
static int failedTests = 0;

static bool check(
    const char *caseName,
    const std::vector<int>& input,
    int blockSize,
    const char *mode,
    ScanFunction scan) {

    int n = static_cast<int>(input.size());

    std::vector<int> expected(n);
    std::vector<int> output(n > 0 ? n : 1, -123456);

    int sum = 0;
    for (int i = 0; i < n; ++i) {
        expected[i] = sum;
        sum += input[i];
    }

    scan(n, output.data(), input.data(), blockSize);

    cudaError_t error = cudaDeviceSynchronize();
    bool passed = error == cudaSuccess;

    if (!passed) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    if (passed && n == 0) {
        passed = output[0] == -123456;
    }

    if (passed) {
        for (int i = 0; i < n; ++i) {
            if (output[i] != expected[i]) {
                std::printf(
                    "Mismatch at index %d: expected %d, got %d\n",
                    i, expected[i], output[i]);

                passed = false;
                break;
            }
        }
    }

    ++totalTests;
    if (!passed) {
        ++failedTests;
    }

    std::printf(
        "%s | %s | n=%d | block=%d | %s\n",
        caseName, mode, n, blockSize,
        passed ? "PASS" : "FAIL");

    return passed;
}

static void checkAll(
    const char *caseName,
    const std::vector<int>& input,
    int blockSize) {

    check(
        caseName, input, blockSize, "Shared Naive",
        StreamCompaction::SharedScan::scanNaive);

    check(
        caseName, input, blockSize, "Shared Efficient",
        StreamCompaction::SharedScan::scanEfficient);

    check(
        caseName, input, blockSize, "Shared Padded",
        StreamCompaction::SharedScan::scanEfficientPadded);
}

int main() {
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> distribution(-10, 10);

    for (int blockSize : {64, 128, 256, 512}) {
        int tileSize = 2 * blockSize;

        checkAll("Empty", {}, blockSize);
        checkAll("Single zero", {0}, blockSize);
        checkAll("Single negative", {-7}, blockSize);
        checkAll("Example", {3, 1, 0, 2}, blockSize);
        checkAll("Mixed signs", {-3, 5, -2, 0, 7, -1}, blockSize);

        checkAll(
            "All zeros",
            std::vector<int>(tileSize + 1, 0),
            blockSize);

        checkAll(
            "All ones",
            std::vector<int>(2 * tileSize + 3, 1),
            blockSize);

        const int sizes[] = {
            tileSize - 1,
            tileSize,
            tileSize + 1,
            2 * tileSize - 1,
            2 * tileSize,
            2 * tileSize + 1,
            10000,
            1048573,
            1048576
        };

        for (int n : sizes) {
            std::vector<int> input(n);

            for (int& value : input) {
                value = distribution(generator);
            }

            checkAll("Random signed integers", input, blockSize);
        }
    }

    std::printf(
        "\nTests: %d | Passed: %d | Failed: %d\n",
        totalTests, totalTests - failedTests, failedTests);

    std::printf(
        "%s\n",
        failedTests == 0
            ? "ALL SHARED SCAN TESTS PASSED"
            : "SHARED SCAN TESTS FAILED");

    return failedTests == 0 ? 0 : 1;
}
