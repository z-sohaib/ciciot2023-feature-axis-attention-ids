from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import wandb

from ciciot2023_ids.config import load_config
from ciciot2023_ids.train import run_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one CICIoT2023 W&B trial.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--group", default="ciciot2023_proposed")
    known, unknown = parser.parse_known_args()

    config = load_config(known.base_config)
    overrides = parse_overrides(unknown)
    apply_overrides(config, overrides)

    run = wandb.init(
        project="pfe-thesis-ciciot2023",
        group=known.group,
        config=flatten_dict(config),
    )
    for key, value in dict(wandb.config).items():
        apply_override(config, key, value)

    run_id = run.id if run is not None else "local"
    config["experiment"]["name"] = f"{config['experiment']['name']}_{run_id}"
    config["output"]["run_dir"] = f"runs/wandb/{known.group}/{run_id}"

    metrics = run_config(config)
    wandb.log(
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "best_val_macro_f1": metrics["best_val_macro_f1"],
            "latency_ms_per_sample": metrics.get("latency_ms_per_sample", 0.0),
            "throughput_samples_per_second": metrics.get("throughput_samples_per_second", 0.0),
            "parameter_count": metrics["parameter_count"],
            "training_seconds": metrics["training_seconds"],
        }
    )
    for class_name in metrics["class_names"]:
        report = metrics["classification_report"][class_name]
        wandb.log(
            {
                f"{class_name}/precision": report["precision"],
                f"{class_name}/recall": report["recall"],
                f"{class_name}/f1": report["f1-score"],
                f"{class_name}/fpr": metrics["per_class_false_positive_rate"][class_name],
            }
        )
    wandb.finish()
    return 0


def parse_overrides(args: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for arg in args:
        if not arg.startswith("--") or "=" not in arg:
            continue
        key, raw_value = arg[2:].split("=", 1)
        overrides[key] = parse_value(raw_value)
    return overrides


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        apply_override(config, key, value)


def apply_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if "," in value:
        return [parse_value(part.strip()) for part in value.split(",")]
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def flatten_dict(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = copy.deepcopy(value)
    return flat


if __name__ == "__main__":
    raise SystemExit(main())
