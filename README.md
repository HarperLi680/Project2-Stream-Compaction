CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

- Qingying Li
  - [LinkedIn](https://www.linkedin.com/in/harper-li-292730373/)
- Tested on: NVIDIA H200 server, Ubuntu 22.04 Docker container, CUDA Toolkit 12.4, CMake 3.22.1.
- GPU benchmarks were compiled with optimization enabled and targeted `sm_90`.

### Implementation

This project implements exclusive prefix sum and stream compaction.

Four scan implementations are included:

- **CPU:** A sequential loop that keeps a running sum.
- **Naive:** A GPU scan that adds values from increasing offsets using two alternating buffers.
- **Efficient:** A work-efficient GPU scan with up-sweep and down-sweep phases. Non-power-of-two inputs are padded with zeros.
- **Thrust:** A scan using `thrust::exclusive_scan`.

Stream compaction removes zeros while keeping the remaining elements in
their original order. The CPU implementation includes a direct version
and a version using the map, scan, and scatter steps. GPU compaction
uses the work-efficient scan.

### Correctness Tests

The provided tests passed for power-of-two and non-power-of-two inputs,
including arrays with 1,048,576 and 1,048,573 elements.

Additional tests covered empty arrays, single elements, all-zero inputs,
negative values, and lengths around thread-block boundaries.

The extra tests compared scan results with `std::exclusive_scan` and
compaction results with `std::copy_if`. All 98 checks passed.

An excerpt from the provided test output is shown below:

~~~text
==== naive scan, power-of-two ====
    passed
==== naive scan, non-power-of-two ====
    passed
==== work-efficient scan, power-of-two ====
    passed
==== work-efficient scan, non-power-of-two ====
    passed
==== thrust scan, power-of-two ====
    passed
==== thrust scan, non-power-of-two ====
    passed
==== work-efficient compact, power-of-two ====
    passed
==== work-efficient compact, non-power-of-two ====
    passed
~~~

### Benchmark Method

CPU scans were measured with the CPU timer, and GPU scans were measured
with CUDA events. Initial GPU memory allocation and input/output
transfers were outside the GPU timing interval. Work performed inside
the scan call, including Thrust's internal allocation, remained timed.

Each configuration used 5 warm-up calls followed by 20 measured calls.
The experiment was repeated three times.

Each plotted point is the median of the three run medians. Shading
shows the minimum and maximum of those three values.

Raw data and the plotting script are saved in `results/`.

### Block Size

![Block size comparison](img/performance-block-size.png)

This experiment used 1,048,576 elements and tested 64, 128, 256, and
512 threads per block. Lower time is better.

| Threads per block | Naive (ms) | Efficient (ms) |
|---|---:|---:|
| 64 | 0.254176 | 0.105728 |
| 128 | 0.150656 | 0.100752 |
| 256 | 0.103008 | 0.104448 |
| 512 | 0.101232 | 0.113200 |

Naive improved as the block size increased, although the difference
between 256 and 512 was small. Efficient was fastest at 128 threads
per block and became slightly slower with larger blocks.

The array-size comparison used 512 threads per block for Naive
and 128 for Efficient, based on the block-size experiment.

### Array Size

![Scan performance by array size](img/performance-array-size.png)

This experiment tested arrays from 256 to 4,194,304 elements.
Both axes use logarithmic scales, and lower time is better.

CPU scan was fastest for small inputs. Small GPU tasks still need
kernel launches and synchronization, so parallel execution did not
provide an advantage at those sizes.

Naive was faster than Efficient for several smaller sizes. Although
Efficient performs less total work, its up-sweep and down-sweep require
more kernel launches. At larger sizes, reducing the total work becomes
more useful.

At 4,194,304 elements, the measured times were:

| Implementation | Scan time (ms) |
|---|---:|
| CPU | 1.593361 |
| Naive | 0.296864 |
| Efficient | 0.170128 |
| Thrust | 0.180624 |

Efficient was fastest at this size, with about 1.74 times the speed of
Naive within the measured timing intervals.

GPU timings exclude initial and final CPU–GPU data transfers.

### Thrust Profiling

Thrust was fast for smaller inputs, but its measured time increased
sharply between 65,536 and 262,144 elements.

Nsight Systems showed one allocation, two kernel launches, one stream
synchronization, and one deallocation inside each timed Thrust scan.

| CPU API operation | 65,536 elements (µs) | 262,144 elements (µs) |
|---|---:|---:|
| Allocation | 1.45 | 73.28 |
| Deallocation | 2.85 | 82.18 |

These values are median CPU API durations per measured scan.
The main scan kernel's average GPU execution time increased only
from about 3.23 to 4.33 microseconds.

This suggests that temporary-memory management contributed to the
increase in scan time. The trace does not establish the exact reason
the allocation and deallocation became slower.

Nsight adds profiling overhead, so the performance graphs use the
separate measurements collected without profiling.

The profiling summary is saved in `results/nsight/scan-analysis.txt`.

### Build Notes

A typo in `stream_compaction/CMakeLists.txt` was corrected:
`stream_compaction}` was changed to `stream_compaction` in the
`set_target_properties` call.

The project was built with:

~~~bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_FLAGS="-arch=sm_90"

cmake --build build -j2

./build/bin/cis5650_stream_compaction_test
~~~

The `sm_90` setting targets the H200 used for these experiments.