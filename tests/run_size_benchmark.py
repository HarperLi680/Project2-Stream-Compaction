import csv
import subprocess
from pathlib import Path

project = Path("/workspace/Project2-Stream-Compaction-main")
bench = Path("/workspace/diagnostics/project2-blocksize")
output = project / "results" / "array-size-benchmark.csv"

sizes = [
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
]

settings = {
    0: ("CPU", 128),
    1: ("Naive", 512),
    2: ("Efficient", 128),
    3: ("Thrust", 128),
}

header = [
    "run", "mode", "array_size",
    "block_size", "samples", "median_ms",
]

with output.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    print(",".join(header), flush=True)

    for run in range(1, 4):
        size_order = sizes if run != 2 else list(reversed(sizes))
        mode_order = [(i + run - 1) % 4 for i in range(4)]

        for n in size_order:
            for mode in mode_order:
                name, block_size = settings[mode]
                executable = bench / str(block_size) / "benchmark"

                result = subprocess.run(
                    [str(executable), str(n), str(mode)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                fields = next(csv.reader([result.stdout.strip()]))

                if len(fields) != 5 or fields[0] != name:
                    raise RuntimeError(
                        f"Unexpected benchmark output: {result.stdout}"
                    )

                if mode in (0, 3):
                    fields[2] = "NA"

                row = [str(run)] + fields
                writer.writerow(row)
                file.flush()
                print(",".join(row), flush=True)

print(f"Saved: {output}", flush=True)
