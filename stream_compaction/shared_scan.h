#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedScan {
        Common::PerformanceTimer& timer();

        void scanNaive(
            int n, int *odata, const int *idata,
            int blockSize = 256);

        void scanEfficient(
            int n, int *odata, const int *idata,
            int blockSize = 256);

        void scanEfficientPadded(
            int n, int *odata, const int *idata,
            int blockSize = 256);
    }
}