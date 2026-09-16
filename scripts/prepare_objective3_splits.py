from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ciciot2023_ids.config import load_config
from ciciot2023_ids.data import apply_feature_filters, build_preprocessor, detect_label_column, normalize_label


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare CICIoT2023 Objective 3 robustness/generalization splits."
    )
    parser.add_argument("--base-config", required=True, help="Existing CICIoT2023 TOML config.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["unseen_holdout", "shard_shift"],
        help="Objective 3 protocol to prepare.",
    )
    parser.add_argument(
        "--heldout-family",
        default="Web",
        help="Attack family removed from training for unseen_holdout mode.",
    )
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    args = parser.parse_args()

    config = load_config(args.base_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "unseen_holdout":
        summary = prepare_unseen_holdout(config, args.heldout_family, output_path)
    else:
        summary = prepare_shard_shift(config, output_path)

    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved Objective 3 split to {output_path}")
    print(f"Saved summary to {summary_path}")
    return 0


def prepare_unseen_holdout(config: dict[str, Any], heldout_family: str, output_path: Path) -> dict[str, Any]:
    print(f"[objective3] Loading capped CICIoT2023 sample for held-out family={heldout_family}", flush=True)
    frame = _load_file_group(_csv_files(Path(config["data"]["raw_path"])), config)
    if "target" not in frame.columns:
        raise ValueError("Expected the loader to create a target column.")

    heldout_mask = frame["target"].astype(str) == heldout_family
    if not heldout_mask.any():
        available = sorted(frame["target"].astype(str).unique().tolist())
        raise ValueError(f"Held-out family {heldout_family!r} not found. Available: {available}")

    known = frame.loc[~heldout_mask].copy()
    heldout = frame.loc[heldout_mask].copy()
    print(
        f"[objective3] Loaded rows={len(frame)} known={len(known)} heldout={len(heldout)}",
        flush=True,
    )
    seed = int(config["experiment"].get("seed", 42))
    test_size = float(config["data"].get("test_size", 0.2))
    val_size = float(config["data"].get("val_size", 0.15))

    train_val, known_test = train_test_split(
        known,
        test_size=test_size,
        random_state=seed,
        stratify=known["target"].astype(str),
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val["target"].astype(str),
    )
    test = pd.concat([known_test, heldout], ignore_index=True).sample(frac=1.0, random_state=seed)

    arrays, summary = _transform_splits(
        train=train,
        validation=validation,
        test=test,
        config=config,
        seed=seed,
    )
    _save_npz(output_path, arrays)
    summary.update(
        {
            "mode": "unseen_holdout",
            "heldout_family": heldout_family,
            "protocol": [
                "Capped dataset is loaded with the base configuration.",
                "The held-out family is removed from training and validation.",
                "The held-out family is appended only to the final test split.",
                "Preprocessing and feature selection are fitted only on known-family training rows.",
            ],
        }
    )
    return summary


def prepare_shard_shift(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    raw_path = Path(config["data"]["raw_path"])
    csv_files = _csv_files(raw_path)
    if len(csv_files) < 3:
        raise ValueError("Shard-shift mode requires at least three CSV files.")

    seed = int(config["experiment"].get("seed", 42))
    rng = np.random.default_rng(seed)
    shuffled = [csv_files[index] for index in rng.permutation(len(csv_files))]
    train_end = max(1, int(len(shuffled) * 0.70))
    val_end = max(train_end + 1, int(len(shuffled) * 0.85))
    train_files = shuffled[:train_end]
    validation_files = shuffled[train_end:val_end]
    test_files = shuffled[val_end:]
    if not test_files:
        test_files = [validation_files.pop()]

    print(
        f"[objective3] Shard split files train={len(train_files)} val={len(validation_files)} test={len(test_files)}",
        flush=True,
    )
    train = _load_file_group(train_files, config)
    validation = _load_file_group(validation_files, config)
    test = _load_file_group(test_files, config)

    arrays, summary = _transform_splits(
        train=train,
        validation=validation,
        test=test,
        config=config,
        seed=seed,
    )
    _save_npz(output_path, arrays)
    summary.update(
        {
            "mode": "shard_shift",
            "train_files": [str(path) for path in train_files],
            "validation_files": [str(path) for path in validation_files],
            "test_files": [str(path) for path in test_files],
            "protocol": [
                "CSV shards are split by file before preprocessing.",
                "Training, validation, and test rows come from disjoint shard groups.",
                "Preprocessing and feature selection are fitted only on training shards.",
                "This is a distribution-shift diagnostic, not the primary benchmark protocol.",
            ],
        }
    )
    return summary


def _transform_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray | list[str]], dict[str, Any]]:
    class_names = sorted(
        set(train["target"].astype(str))
        | set(validation["target"].astype(str))
        | set(test["target"].astype(str))
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    x_train_df = _features(train, config)
    x_val_df = _features(validation, config)
    x_test_df = _features(test, config)
    y_train = label_encoder.transform(train["target"].astype(str))
    y_val = label_encoder.transform(validation["target"].astype(str))
    y_test = label_encoder.transform(test["target"].astype(str))

    preprocessor = build_preprocessor(x_train_df)
    print(
        f"[objective3] Fitting preprocessing on train={len(y_train)} rows, "
        f"validation={len(y_val)}, test={len(y_test)}",
        flush=True,
    )
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_val = preprocessor.transform(x_val_df).astype(np.float32)
    x_test = preprocessor.transform(x_test_df).astype(np.float32)
    print("[objective3] Applying configured feature filters/selection", flush=True)
    x_train, x_val, x_test = apply_feature_filters(
        x_train,
        x_val,
        x_test,
        y_train,
        config.get("preprocessing", {}),
        seed,
    )

    arrays: dict[str, np.ndarray | list[str]] = {
        "x_train": x_train.astype(np.float32),
        "x_val": x_val.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "y_val": y_val.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "class_names": class_names,
    }
    summary = {
        "base_seed": seed,
        "class_names": class_names,
        "input_features_after_preprocessing": int(x_train.shape[1]),
        "rows": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "class_distribution": {
            "train": _distribution(train["target"].astype(str)),
            "validation": _distribution(validation["target"].astype(str)),
            "test": _distribution(test["target"].astype(str)),
        },
        "preprocessing": config.get("preprocessing", {}),
    }
    return arrays, summary


def _features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    drop_columns = set(config["data"].get("drop_columns", []))
    label_column = config["data"].get("label_column") or detect_label_column(frame.columns)
    if label_column:
        drop_columns.add(label_column)
    drop_columns.add("target")
    features = frame.drop(columns=[column for column in drop_columns if column in frame.columns])
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def _load_file_group(paths: list[Path], config: dict[str, Any]) -> pd.DataFrame:
    label_column = config["data"].get("label_column") or None
    target_granularity = config["data"].get("target_granularity", "category")
    max_rows_per_class = int(config["data"].get("max_rows_per_class") or 0)
    chunksize = int(config["data"].get("chunksize") or 200_000)
    kept_counts: Counter[str] = Counter()
    frames: list[pd.DataFrame] = []

    print(f"[objective3] Reading {len(paths)} CSV file(s)", flush=True)
    for file_index, path in enumerate(paths, start=1):
        print(f"[objective3] Reading file {file_index}/{len(paths)}: {path.name}", flush=True)
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            if label_column is None:
                label_column = detect_label_column(chunk.columns)
            if label_column is None or label_column not in chunk.columns:
                raise ValueError("Could not detect label column. Set data.label_column in the config.")
            chunk = chunk.loc[chunk[label_column].notna()].copy()
            chunk["target"] = chunk[label_column].map(lambda value: normalize_label(value, target_granularity))
            if max_rows_per_class > 0:
                chunk = _cap_by_class(chunk, kept_counts, max_rows_per_class)
            if not chunk.empty:
                kept_counts.update(chunk["target"].astype(str))
                frames.append(chunk)
        if kept_counts:
            compact_counts = ", ".join(f"{key}={value}" for key, value in sorted(kept_counts.items()))
            print(f"[objective3] Kept counts after {path.name}: {compact_counts}", flush=True)
    if not frames:
        raise ValueError("No rows were loaded for one shard group.")
    return pd.concat(frames, ignore_index=True)


def _cap_by_class(chunk: pd.DataFrame, kept_counts: Counter[str], max_rows_per_class: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, group in chunk.groupby("target", sort=False):
        remaining = max_rows_per_class - kept_counts[str(label)]
        if remaining > 0:
            parts.append(group.head(remaining))
    if not parts:
        return chunk.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def _distribution(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in Counter(values.tolist()).items()}


def _save_npz(path: Path, arrays: dict[str, np.ndarray | list[str]]) -> None:
    np.savez_compressed(
        path,
        x_train=np.asarray(arrays["x_train"], dtype=np.float32),
        x_val=np.asarray(arrays["x_val"], dtype=np.float32),
        x_test=np.asarray(arrays["x_test"], dtype=np.float32),
        y_train=np.asarray(arrays["y_train"], dtype=np.int64),
        y_val=np.asarray(arrays["y_val"], dtype=np.int64),
        y_test=np.asarray(arrays["y_test"], dtype=np.int64),
        class_names=np.asarray(arrays["class_names"]),
    )


def _csv_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return []


if __name__ == "__main__":
    raise SystemExit(main())
