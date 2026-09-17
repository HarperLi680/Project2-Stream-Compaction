#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$0")/.."

cmake --build build -j2

/usr/local/cuda/bin/nvcc \
    -std=c++17 -O3 -arch=sm_90 \
    -I. \
    tests/benchmark_fusion.cu \
    build/lib/libstream_compaction.a \
    -o build/bin/benchmark_fusion

mkdir -p results/extra-credit

{
    echo "run,mode,array_size,samples,median_ms"

    for run in 1 2 3; do
        if [ "$run" -eq 2 ]; then
            sizes="4194304 1048576 262144 65536 16384 4096 256"
            modes="1 0"
        else
            sizes="256 4096 16384 65536 262144 1048576 4194304"
            modes="0 1"
        fi

        for n in $sizes; do
            for mode in $modes; do
                result=$(./build/bin/benchmark_fusion "$n" "$mode")
                printf "%s,%s\n" "$run" "$result"
            done
        done
    done
} | tee results/extra-credit/fusion-benchmark.csv
