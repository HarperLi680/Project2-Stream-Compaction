#include <cstdio>
#include <vector>
#include "cpu.h"
#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */

        static void scan(int n, int *odata, const int *idata, bool timed) {
            if (timed) {
                timer().startCpuTimer();
            }

            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
            }

            if (timed) {
                timer().endCpuTimer();
            }
        }

        void scan(int n, int *odata, const int *idata) {
            scan(n, odata, idata, true);
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    ++count;
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                timer().startCpuTimer();
                timer().endCpuTimer();
                return 0;
            }

            std::vector<int> flags(n);
            std::vector<int> indices(n);

            timer().startCpuTimer();

            for (int i = 0; i < n; ++i) {
                flags[i] = idata[i] != 0 ? 1 : 0;
            }

            scan(n, indices.data(), flags.data(), false);

            for (int i = 0; i < n; ++i) {
                if (flags[i] != 0) {
                    odata[indices[i]] = idata[i];
                }
            }

            int count = indices[n - 1] + flags[n - 1];

            timer().endCpuTimer();
            return count;
        }

    }
}
