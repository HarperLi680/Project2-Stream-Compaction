import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[1]
output = project / "img"

colors = {
    "CPU": "#666666",
    "Naive": "#D55E00",
    "Efficient": "#009E73",
    "Thrust": "#0072B2",
}
markers = {
    "CPU": "s",
    "Naive": "o",
    "Efficient": "^",
    "Thrust": "D",
}


def read_results(filename):
    with (project / "results" / filename).open() as file:
        return list(csv.DictReader(file))


def draw(ax, rows, mode, x_column):
    groups = defaultdict(list)

    for row in rows:
        if row["mode"] == mode:
            groups[int(row[x_column])].append(float(row["median_ms"]))

    x = sorted(groups)
    y = [median(groups[value]) for value in x]
    low = [min(groups[value]) for value in x]
    high = [max(groups[value]) for value in x]

    ax.plot(
        x, y,
        marker=markers[mode],
        color=colors[mode],
        label=mode,
    )
    ax.fill_between(x, low, high, color=colors[mode], alpha=0.15)


def finish(fig, ax, filename):
    ax.set_ylabel("Scan time (ms; lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(
        0.5, 0.02,
        "Lines: median of 3 run medians. Shading: min–max across runs.\n"
        "Each run: 5 warm-up calls, then 20 measured calls.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path = output / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("Generated:", path)


blocks = read_results("blocksize-benchmark.csv")
fig, ax = plt.subplots(figsize=(8, 5))

for mode in ["Naive", "Efficient"]:
    draw(ax, blocks, mode, "block_size")

ax.set_xticks([64, 128, 256, 512])
ax.set_ylim(bottom=0)
ax.set_xlabel("Threads per block")
ax.set_title("Block size comparison | 1,048,576 elements | NVIDIA H200")
finish(fig, ax, "performance-block-size.png")


sizes = read_results("array-size-benchmark.csv")
fig, ax = plt.subplots(figsize=(9, 5.5))

for mode in ["CPU", "Naive", "Efficient", "Thrust"]:
    draw(ax, sizes, mode, "array_size")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Number of input elements (log scale)")
ax.set_title(
    "Exclusive scan performance | NVIDIA H200 GPU tests\n"
    "Naive: 512 threads/block | Efficient: 128 threads/block"
)
finish(fig, ax, "performance-array-size.png")
