#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "stream_compaction/cpu.h"
#include "stream_compaction/naive.h"
#include "stream_compaction/efficient.h"
#include "stream_compaction/thrust.h"

int main() {
    using namespace StreamCompaction;

    std::vector<std::vector<int>> cases = {
        {},
        {0},
        {7},
        {0, 0, 0, 0, 0},
        {1, 2, 3, 4, 5},
        {1, 5, 0, 1, 2, 0, 3},
        {0, -2, 0, 3, -1, 0}
    };

    for (int n : {127, 128, 129, 255, 256, 257, 10000}) {
        std::vector<int> input(n);
        for (int i = 0; i < n; ++i) {
            input[i] = i % 7 - 3;
        }
        cases.push_back(input);
    }

    using ScanFunction = void (*)(int, int*, const int*);
    ScanFunction scans[] = {
        CPU::scan, Naive::scan, Efficient::scan, Thrust::scan
    };
    const char *scanNames[] = {
        "CPU", "Naive", "Efficient", "Thrust"
    };

    using CompactFunction = int (*)(int, int*, const int*);
    CompactFunction compacts[] = {
        CPU::compactWithoutScan,
        CPU::compactWithScan,
        Efficient::compact
    };
    const char *compactNames[] = {
        "CPU without scan", "CPU with scan", "Efficient"
    };

    bool allPassed = true;

    for (int c = 0; c < static_cast<int>(cases.size()); ++c) {
        const auto &input = cases[c];
        int n = static_cast<int>(input.size());

        std::vector<int> expectedScan(n);
        std::exclusive_scan(
            input.begin(), input.end(), expectedScan.begin(), 0);

        std::vector<int> expectedCompact;
        std::copy_if(
            input.begin(), input.end(),
            std::back_inserter(expectedCompact),
            [](int value) { return value != 0; });

        for (int mode = 0; mode < 4; ++mode) {
            std::vector<int> output(std::max(n, 1), -999);
            scans[mode](n, output.data(), input.data());

            bool passed = std::equal(
                expectedScan.begin(), expectedScan.end(), output.begin());

            std::cout << "Case " << c << " n=" << n
                      << " scan " << scanNames[mode]
                      << ": " << (passed ? "PASS" : "FAIL") << '\n';

            allPassed = allPassed && passed;
        }

        for (int mode = 0; mode < 3; ++mode) {
            std::vector<int> output(std::max(n, 1), -999);
            int count = compacts[mode](n, output.data(), input.data());

            bool passed =
                count == static_cast<int>(expectedCompact.size()) &&
                std::equal(expectedCompact.begin(), expectedCompact.end(),
                           output.begin());

            std::cout << "Case " << c << " n=" << n
                      << " compact " << compactNames[mode]
                      << ": " << (passed ? "PASS" : "FAIL") << '\n';

            allPassed = allPassed && passed;
        }
    }

    std::cout << (allPassed ? "ALL EDGE TESTS PASSED"
                           : "EDGE TESTS FAILED") << '\n';

    return allPassed ? 0 : 1;
}
