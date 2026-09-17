import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
folder = root / 'results' / 'extra-credit'
out = root / 'img'
out.mkdir(exist_ok=True)
styles = {
    'Baseline': ('#777777', 's'),
    'Fused': ('#333333', 'D'),
    'FusedEfficient': ('#333333', 'D'),
    'SharedNaive': ('#D55E00', 'o'),
    'SharedEfficient': ('#009E73', '^'),
    'SharedPadded': ('#0072B2', 's'),
}

def read(name, expected):
    with (folder / name).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == expected, (name, len(rows))
    groups = defaultdict(list)
    for r in rows:
        key = (r['mode'], int(r['array_size']), int(r.get('block_size', 128)))
        groups[key].append((int(r['run']), float(r['median_ms'])))
        assert int(r['samples']) == 20
    result = {}
    for key, values in groups.items():
        assert sorted(r for r, _ in values) == [1, 2, 3], key
        times = [t for _, t in values]
        assert min(times) > 0
        result[key] = (median(times), min(times), max(times))
    return result

def line(ax, data, mode, block=None, size=None):
    points = sorted((n if size is None else b, v)
                    for (m, n, b), v in data.items()
                    if m == mode and (block is None or b == block)
                    and (size is None or n == size))
    x = [p[0] for p in points]
    y, lo, hi = zip(*(p[1] for p in points))
    color, marker = styles[mode]
    label = mode + (f' ({block} threads)' if block is not None else '')
    ax.plot(x, y, color=color, marker=marker, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15)

def save(fig, axes, name):
    for ax in axes:
        ax.set_ylabel('Scan time (ms; lower is better)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.text(0.5, 0.02,
             'Median of 3 run medians; shading: min-max across runs (not a confidence interval).\n'
             'Each configuration: 5 warm-ups + 20 measured calls per run. NVIDIA H200.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out / name, dpi=180)
    plt.close(fig)
    print('Generated:', out / name)

fusion = read('fusion-benchmark.csv', 42)
shared = read('shared-scan-benchmark.csv', 234)
with (folder / 'summary.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['experiment','mode','array_size','block_size','median_ms','min_ms','max_ms'])
    for name, data in [('fusion', fusion), ('shared', shared)]:
        for key, values in sorted(data.items()):
            writer.writerow([name, *key, *[f'{v:.6f}' for v in values]])

fig, ax = plt.subplots(figsize=(9, 5.5))
for mode in ['Baseline', 'Fused']:
    line(ax, fusion, mode)
ax.set(xscale='log', yscale='log', xlabel='Input elements (log scale)',
       title='Global-memory scan: baseline and fused | 128 threads/block')
save(fig, [ax], 'extra-fusion.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, n in zip(axes, [1048576, 4194304]):
    for mode in ['SharedNaive', 'SharedEfficient', 'SharedPadded']:
        line(ax, shared, mode, size=n)
    ax.set(title=f'{n:,} elements', xlabel='Threads per block', ylim=(0, 0.14))
    ax.set_xticks([64, 128, 256, 512])
save(fig, axes, 'extra-shared-block-size.png')

fig, ax = plt.subplots(figsize=(9, 5.5))
for mode in ['SharedNaive', 'SharedEfficient', 'SharedPadded', 'FusedEfficient']:
    line(ax, shared, mode, block=128)
ax.set(xscale='log', yscale='log', xlabel='Input elements (log scale)',
       title='Shared-memory and fused scans | fixed 128 threads/block')
save(fig, [ax], 'extra-shared-array-size.png')
