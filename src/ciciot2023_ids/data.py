from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


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


def load_capped_dataset(config: dict[str, Any]) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    csv_files = _csv_files(raw_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at {raw_path}")

    label_column = config["data"].get("label_column") or None
    max_rows_per_class = int(config["data"].get("max_rows_per_class") or 0)
    chunksize = int(config["data"].get("chunksize") or 200_000)
    target_granularity = config["data"].get("target_granularity", "category")
    sample_strategy = config["data"].get("sample_strategy", "stratified_random")
    seed = int(config["experiment"].get("seed", 42))

    if max_rows_per_class > 0 and sample_strategy == "stratified_random":
        return _load_stratified_random_sample(
            csv_files=csv_files,
            label_column=label_column,
            target_granularity=target_granularity,
            max_rows_per_class=max_rows_per_class,
            chunksize=chunksize,
            seed=seed,
        )

    frames: list[pd.DataFrame] = []
    kept_counts: Counter[str] = Counter()

    for path in csv_files:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            if label_column is None:
                label_column = detect_label_column(chunk.columns)
            if label_column is None or label_column not in chunk.columns:
                raise ValueError("Could not detect label column. Set data.label_column in the config.")

            chunk = chunk.loc[chunk[label_column].notna()].copy()
            chunk["target"] = chunk[label_column].map(lambda value: normalize_label(value, target_granularity))
            if max_rows_per_class > 0:
                chunk = _cap_chunk_by_class(chunk, kept_counts, max_rows_per_class)
            if not chunk.empty:
                frames.append(chunk)
                kept_counts.update(chunk["target"].astype(str))

    if not frames:
        raise ValueError("No rows were loaded. Check labels and max_rows_per_class.")
    return pd.concat(frames, ignore_index=True)


def _load_stratified_random_sample(
    csv_files: list[Path],
    label_column: str | None,
    target_granularity: str,
    max_rows_per_class: int,
    chunksize: int,
    seed: int,
) -> pd.DataFrame:
    target_counts: Counter[str] = Counter()
    detected_label_column = label_column
    for path in csv_files:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            if detected_label_column is None:
                detected_label_column = detect_label_column(chunk.columns)
            if detected_label_column is None or detected_label_column not in chunk.columns:
                raise ValueError("Could not detect label column. Set data.label_column in the config.")
            chunk = chunk.loc[chunk[detected_label_column].notna()]
            targets = chunk[detected_label_column].map(lambda value: normalize_label(value, target_granularity))
            target_counts.update(targets.astype(str))

    sample_fraction = {
        target: min(1.0, max_rows_per_class / count)
        for target, count in target_counts.items()
        if count > 0
    }

    frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for path in csv_files:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = chunk.loc[chunk[detected_label_column].notna()].copy()
            chunk["target"] = chunk[detected_label_column].map(lambda value: normalize_label(value, target_granularity))
            probabilities = chunk["target"].astype(str).map(sample_fraction).fillna(0.0).to_numpy()
            mask = rng.random(len(chunk)) < probabilities
            sampled = chunk.loc[mask]
            if not sampled.empty:
                frames.append(sampled)

    if not frames:
        raise ValueError("No rows were sampled. Check max_rows_per_class and label mapping.")

    frame = pd.concat(frames, ignore_index=True)
    capped_parts = []
    for target, group in frame.groupby("target", sort=False):
        if len(group) > max_rows_per_class:
            capped_parts.append(group.sample(n=max_rows_per_class, random_state=seed))
        else:
            capped_parts.append(group)
    return pd.concat(capped_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_splits(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    seed = int(config["experiment"].get("seed", 42))
    test_size = float(config["data"].get("test_size", 0.2))
    val_size = float(config["data"].get("val_size", 0.15))
    drop_columns = set(config["data"].get("drop_columns", []))
    preprocessing_config = config.get("preprocessing", {})
    label_column = config["data"].get("label_column") or detect_label_column(frame.columns)
    if label_column:
        drop_columns.add(label_column)
    drop_columns.add("target")

    y_text = frame["target"].astype(str)
    feature_frame = frame.drop(columns=[column for column in drop_columns if column in frame.columns])
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = list(label_encoder.classes_)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        feature_frame,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    preprocessor = build_preprocessor(x_train)
    x_train_np = preprocessor.fit_transform(x_train).astype(np.float32)
    x_val_np = preprocessor.transform(x_val).astype(np.float32)
    x_test_np = preprocessor.transform(x_test).astype(np.float32)
    x_train_np, x_val_np, x_test_np = apply_feature_filters(
        x_train_np,
        x_val_np,
        x_test_np,
        y_train,
        preprocessing_config,
        seed,
    )
    return x_train_np, x_val_np, x_test_np, y_train, y_val, y_test, class_names


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_columns),
            ("cat", Pipeline([("onehot", encoder)]), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def apply_feature_filters(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variance_threshold = config.get("variance_threshold")
    if variance_threshold is not None:
        selector = VarianceThreshold(threshold=float(variance_threshold))
        x_train = selector.fit_transform(x_train)
        x_val = selector.transform(x_val)
        x_test = selector.transform(x_test)

    method = str(config.get("feature_selection", "none")).lower()
    if method in {"none", "", "false"}:
        return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)

    if method != "mutual_info":
        raise ValueError(f"Unsupported feature_selection method: {method}")

    k = int(config.get("feature_selection_k", x_train.shape[1]))
    k = min(k, x_train.shape[1])
    selector = SelectKBest(
        score_func=lambda features, target: mutual_info_classif(
            features,
            target,
            discrete_features=False,
            random_state=seed,
        ),
        k=k,
    )
    x_train = selector.fit_transform(x_train, y_train)
    x_val = selector.transform(x_val)
    x_test = selector.transform(x_test)
    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


def detect_label_column(columns: pd.Index) -> str | None:
    normalized = {str(column).strip(): str(column) for column in columns}
    lower_lookup = {key.lower(): value for key, value in normalized.items()}
    for candidate in LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def normalize_label(value: object, granularity: str) -> str:
    label = str(value).strip()
    if granularity == "fine":
        return label
    lowered = label.lower()
    compact = lowered.replace("-", "").replace("_", "").replace(" ", "")
    if lowered in {"benign", "normal"}:
        return "Benign"
    if "ddos" in compact:
        return "DDoS"
    if compact.startswith("dos"):
        return "DoS"
    if "mirai" in compact:
        return "Mirai"
    if "recon" in compact or "scan" in compact or "scanning" in compact:
        return "Recon"
    if "web" in compact or "xss" in compact or "sql" in compact:
        return "Web"
    if "commandinjection" in compact or "browserhijacking" in compact:
        return "Web"
    if "backdoormalware" in compact or "uploadingattack" in compact:
        return "Web"
    if "brute" in compact or "password" in compact or "dictionary" in compact:
        return "BruteForce"
    if "spoof" in compact or "mitmarp" in compact:
        return "Spoofing"
    return label


def _cap_chunk_by_class(chunk: pd.DataFrame, kept_counts: Counter[str], max_rows_per_class: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, group in chunk.groupby("target", sort=False):
        remaining = max_rows_per_class - kept_counts[str(label)]
        if remaining <= 0:
            continue
        parts.append(group.head(remaining))
    if not parts:
        return chunk.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def _csv_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return []
