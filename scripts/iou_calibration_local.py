"""Local IoU calibration driver -- Stage 6 pseudo-labels vs 984 doctor boxes.

Uses the 10,022 pseudo-labels from artifacts/kaggle_full_train/ as input.
Writes iou_report.json + thresh_sweep.png to artifacts/iou_calibration/.

No argparse -- paths resolved from configs/config.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from src.pseudo_label.iou_calibration import calibrate_thresholds  # noqa: E402


def main() -> None:
    cfg = Config()
    pseudo_csv = Path(cfg.paths.project_root) / "artifacts" / "kaggle_full_train" \
        / "pseudo_labels" / "pseudo_labels.csv"
    bbox_csv = Path(cfg.paths.bbox_csv)
    out_dir = Path(cfg.paths.output_dir) / "iou_calibration"
    print(f"[driver] pseudo_csv={pseudo_csv}")
    print(f"[driver] bbox_csv={bbox_csv}")
    print(f"[driver] out_dir={out_dir}")
    calibrate_thresholds(pseudo_csv, bbox_csv, out_dir)


if __name__ == "__main__":
    main()
