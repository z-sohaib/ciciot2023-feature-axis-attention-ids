from __future__ import annotations

import importlib.util
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
]


def main() -> int:
    print("CICIoT2023 project setup check")
    missing = []
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
            print(f"[missing] {module}")
        else:
            print(f"[ok] {module}")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if csv_files:
        print(f"[ok] found {len(csv_files)} CSV file(s) in {raw_dir}")
        for path in csv_files[:10]:
            print(f"  - {path}")
        if len(csv_files) > 10:
            print(f"  ... {len(csv_files) - 10} more")
    else:
        print(f"[data] no CSV files found in {raw_dir}")

    if missing:
        print("Setup incomplete: install project dependencies.")
        return 1
    print("Setup check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
