import sqlite3
from pathlib import Path
from collections import defaultdict
from statistics import median

folder = Path(__file__).resolve().parent

for n in [65536, 262144]:
    connection = sqlite3.connect(folder / f"thrust-{n}.sqlite")

    calls = connection.execute("""
        SELECT r.start, r.end, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        JOIN StringIds AS s ON r.nameId = s.id
        ORDER BY r.start
    """).fetchall()

    events = [
        call for call in calls
        if call[2].startswith("cudaEventRecord")
    ]

    print(f"\n===== Thrust: {n} elements =====")

    if len(events) != 50:
        raise RuntimeError(
            f"Expected 50 timer records, found {len(events)}"
        )

    totals = defaultdict(list)
    counts = defaultdict(list)

    # Skip the first five warm-up scans.
    for scan in range(5, 25):
        begin = events[2 * scan][1]
        end = events[2 * scan + 1][0]
        inside = [
            call for call in calls
            if call[0] >= begin and call[1] <= end
        ]

        durations = defaultdict(float)
        numbers = defaultdict(int)

        for start, stop, name in inside:
            durations[name] += (stop - start) / 1000
            numbers[name] += 1

        for name in durations:
            totals[name].append(durations[name])
            counts[name].append(numbers[name])

        if scan == 5:
            print("First measured scan: CPU API timeline")
            print("Offset(us)  Duration(us)  API")
            for start, stop, name in inside:
                print(
                    f"{(start - begin) / 1000:10.3f}  "
                    f"{(stop - start) / 1000:12.3f}  {name}"
                )

    print("\nMedian CPU API time per scan (microseconds):")
    for name in sorted(totals):
        times = totals[name] + [0.0] * (20 - len(totals[name]))
        number = counts[name] + [0] * (20 - len(counts[name]))
        print(
            f"{name}: {median(times):.3f} us, "
            f"{median(number):g} calls/scan"
        )

    connection.close()
