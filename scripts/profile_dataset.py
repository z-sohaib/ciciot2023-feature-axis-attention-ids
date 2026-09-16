from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ciciot2023_ids.data import normalize_label


LABEL_CANDIDATES = [
    "label",
    "Label",
    "class",
    "Class",
    "Attack",
    "attack",
    "Attack_type",
    "attack_type",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile CICIoT2023 CSV files.")
    parser.add_argument("--input", default="data/raw", help="CSV file or directory containing CSV files.")
    parser.add_argument("--output-dir", default="runs/profile")
    parser.add_argument("--sample-rows", type=int, default=200_000)
    parser.add_argument("--exact-label-counts", action="store_true")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = _csv_files(input_path)
    if not csv_files:
        raise SystemExit(f"No CSV files found at {input_path}")

    first = pd.read_csv(csv_files[0], nrows=5)
    label_column = _detect_label_column(first.columns)
    summary: dict[str, Any] = {
        "input": str(input_path),
        "file_count": len(csv_files),
        "files": [str(path) for path in csv_files],
        "detected_label_column": label_column,
        "sample_rows_per_file": args.sample_rows,
        "exact_label_counts": args.exact_label_counts,
        "columns": list(first.columns),
    }

    total_rows = 0
    missing_by_column: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    dtypes: dict[str, str] = {}

    for path in csv_files:
        sample = pd.read_csv(path, nrows=args.sample_rows, low_memory=False)
        total_rows += _count_rows(path)
        for column, value in sample.dtypes.astype(str).items():
            dtypes.setdefault(column, value)
        for column, value in sample.isna().sum().items():
            missing_by_column[column] = missing_by_column.get(column, 0) + int(value)
        if label_column and not args.exact_label_counts:
            counts = sample[label_column].astype(str).value_counts()
            for label, count in counts.items():
                class_counts[label] = class_counts.get(label, 0) + int(count)
                category = normalize_label(label, "category")
                category_counts[category] = category_counts.get(category, 0) + int(count)
        elif label_column:
            counts = _count_labels_exact(path, label_column, args.chunksize)
            for label, count in counts.items():
                class_counts[label] = class_counts.get(label, 0) + count
                category = normalize_label(label, "category")
                category_counts[category] = category_counts.get(category, 0) + count

    summary["estimated_total_rows"] = total_rows
    summary["sampled_missing_values"] = missing_by_column
    summary["dtypes"] = dtypes
    summary["sampled_class_counts"] = class_counts
    summary["category_counts"] = category_counts
    summary["feature_count_excluding_label"] = (
        len(first.columns) - 1 if label_column and label_column in first.columns else len(first.columns)
    )

    (output_dir / "dataset_profile.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_class_distribution(output_dir / "class_distribution.csv", class_counts, args.exact_label_counts)
    _write_distribution(output_dir / "category_distribution.csv", category_counts, "category", args.exact_label_counts)
    print(f"Wrote {output_dir / 'dataset_profile.json'}")
    print(f"Wrote {output_dir / 'class_distribution.csv'}")
    print(f"Wrote {output_dir / 'category_distribution.csv'}")
    if not label_column:
        print("Warning: no label column detected. Check dataset_profile.json columns.")
    return 0


def _csv_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return []


def _detect_label_column(columns: pd.Index) -> str | None:
    normalized = {str(column).strip(): str(column) for column in columns}
    lower_lookup = {key.lower(): value for key, value in normalized.items()}
    for candidate in LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def _count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _write_class_distribution(path: Path, class_counts: dict[str, int], exact: bool) -> None:
    _write_distribution(path, class_counts, "label", exact)


def _write_distribution(path: Path, counts: dict[str, int], name_column: str, exact: bool) -> None:
    count_column = "count" if exact else "sampled_count"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name_column, count_column])
        writer.writeheader()
        for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({name_column: name, count_column: count})


def _count_labels_exact(path: Path, label_column: str, chunksize: int) -> dict[str, int]:
    class_counts: dict[str, int] = {}
    for chunk in pd.read_csv(path, usecols=[label_column], chunksize=chunksize, low_memory=False):
        counts = chunk[label_column].astype(str).value_counts()
        for label, count in counts.items():
            class_counts[label] = class_counts.get(label, 0) + int(count)
    return class_counts


if __name__ == "__main__":
    raise SystemExit(main())
