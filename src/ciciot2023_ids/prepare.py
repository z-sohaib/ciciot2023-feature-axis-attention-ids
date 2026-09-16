from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ciciot2023_ids.config import load_config
from ciciot2023_ids.data import load_capped_dataset, make_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a fixed CICIoT2023 processed split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    prepare(args.config, args.output)
    return 0


def prepare(config_path: str | Path, output: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    processed_path = Path(output or config["data"].get("processed_path", "data/processed/dataset.npz"))
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_capped_dataset(config)
    x_train, x_val, x_test, y_train, y_val, y_test, class_names = make_splits(frame, config)

    np.savez_compressed(
        processed_path,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        class_names=np.asarray(class_names),
    )

    summary = {
        "config": str(config_path),
        "processed_path": str(processed_path),
        "target_granularity": config["data"].get("target_granularity"),
        "sample_strategy": config["data"].get("sample_strategy"),
        "max_rows_per_class": config["data"].get("max_rows_per_class"),
        "input_features_after_preprocessing": int(x_train.shape[1]),
        "preprocessing": config.get("preprocessing", {}),
        "class_names": class_names,
        "rows": {
            "sampled_total": int(len(frame)),
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "class_distribution": {
            "sampled": _distribution(frame["target"].astype(str).tolist()),
            "train": _encoded_distribution(y_train, class_names),
            "validation": _encoded_distribution(y_val, class_names),
            "test": _encoded_distribution(y_test, class_names),
        },
        "preprocessing_protocol": [
            "Loaded all CSV shards recursively from data.raw_path.",
            "Dropped rows with missing labels.",
            "Mapped CICIoT2023 fine-grained labels to category labels.",
            "Applied stratified random per-category sampling before splitting.",
            "Created stratified train/validation/test splits.",
            "Fitted scaling/encoding only on the training split.",
            "Applied feature filtering/selection using training data only when enabled.",
            "Transformed validation and test splits with the training-fitted preprocessing pipeline.",
        ],
    }
    summary_path = processed_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved processed data to {processed_path}")
    print(f"Saved preprocessing summary to {summary_path}")
    print(f"rows train={len(y_train)} val={len(y_val)} test={len(y_test)} features={x_train.shape[1]}")
    return summary


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))


def _encoded_distribution(values: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {class_names[index]: int(counts.get(index, 0)) for index in range(len(class_names))}


if __name__ == "__main__":
    raise SystemExit(main())
