from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CICIoT2023 metrics files.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="runs/summary.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.runs_root).glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": path.parent.name,
                "experiment": metrics.get("experiment", ""),
                "architecture": metrics.get("architecture", ""),
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
                "best_val_macro_f1": metrics.get("best_val_macro_f1", 0.0),
                "parameter_count": metrics.get("parameter_count", 0),
                "training_seconds": metrics.get("training_seconds", 0.0),
                "latency_ms_per_sample": metrics.get("latency_ms_per_sample", ""),
                "throughput_samples_per_second": metrics.get("throughput_samples_per_second", ""),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {output}")
    _print_group_summary(rows)
    return 0


def _print_group_summary(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["architecture"], []).append(row)
    for architecture, group_rows in sorted(groups.items()):
        if len(group_rows) < 2:
            continue
        macro_values = [float(row["macro_f1"]) for row in group_rows]
        acc_values = [float(row["accuracy"]) for row in group_rows]
        print(
            f"{architecture}: "
            f"accuracy={mean(acc_values):.4f}+/-{stdev(acc_values):.4f}, "
            f"macro_f1={mean(macro_values):.4f}+/-{stdev(macro_values):.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
