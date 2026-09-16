from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ciciot2023_ids.train import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CICIoT2023 model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
