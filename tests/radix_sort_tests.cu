#include <algorithm>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "stream_compaction/radix_sort.h"

bool check(const char *name, const std::vector<int>& input) {
    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());

    // Keep a valid output pointer even for an empty input.
    std::vector<int> output(
        std::max<std::size_t>(input.size(), 1), 123456789);

    StreamCompaction::RadixSort::sort(
        static_cast<int>(input.size()), output.data(), input.data());

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return false;
    }

    bool passed = std::equal(
        expected.begin(), expected.end(), output.begin());

    if (input.empty()) {
        passed = passed && output[0] == 123456789;
    }

    printf("%s | n=%zu | %s\n",
           name, input.size(), passed ? "PASS" : "FAIL");

    if (!passed) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (output[i] != expected[i]) {
                printf("First mismatch at %zu: expected %d, got %d\n",
                       i, expected[i], output[i]);
                break;
            }
        }
    }

    if (input.size() == 8) {
        printf("Sorted output:");
        for (std::size_t i = 0; i < input.size(); ++i) {
            printf(" %d", output[i]);
        }
        printf("\n");
    }

    return passed;
}

int main() {
    bool passed = true;

    passed = check("Empty", {}) && passed;
    passed = check("Single zero", {0}) && passed;
    passed = check("Single negative", {-7}) && passed;
    passed = check("All equal", std::vector<int>(1000, -3)) && passed;
    passed = check("Sorted", {-9, -4, 0, 1, 7, 20}) && passed;
    passed = check("Reversed", {20, 7, 1, 0, -4, -9}) && passed;
    passed = check("Example", {5, -3, 0, 128, 5, 7, -1000, 2}) && passed;

    passed = check("Integer limits", {
        std::numeric_limits<int>::max(),
        0,
        std::numeric_limits<int>::min(),
        -1,
        1,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    }) && passed;

    std::mt19937 generator(42);
    std::uniform_int_distribution<int> distribution(
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max());

    for (int n : {127, 128, 129, 255, 256, 257,
                  511, 512, 513, 10000, 1048573, 1048576}) {
        std::vector<int> input(n);

        for (int& value : input) {
            value = distribution(generator);
        }

        passed = check("Random signed integers", input) && passed;
    }

    std::vector<int> duplicates(10000);
    for (int i = 0; i < 10000; ++i) {
        duplicates[i] = i % 5 - 2;
    }
    passed = check("Repeated keys", duplicates) && passed;

    printf("%s\n", passed
        ? "ALL RADIX SORT TESTS PASSED"
        : "RADIX SORT TESTS FAILED");

    return passed ? 0 : 1;
}
