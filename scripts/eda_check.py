"""Local EDA / data-integrity check for NIH ChestX-ray14.

No argparse. Run:  python scripts/eda_check.py
Verifies: image index count, 14-class label distribution, BBox_List parse.
Imported by nothing; standalone local entry (Stage 3 EDA).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from src.data.dataset import (  # noqa: E402
    DISEASE_CLASSES, build_image_index, labels_to_vector, load_dataframe,
)


def main() -> None:
    cfg = Config()
    print("== Path config ==")
    print("data_dir:", cfg.paths.data_dir)
    print("data_entry_csv exists:", Path(cfg.paths.data_entry_csv).exists())

    print("\n== Building image index (12 folders) ==")
    index = build_image_index(cfg.paths.data_dir)
    print("indexed images on disk:", len(index))

    print("\n== Loading Data_Entry_2017.csv ==")
    df = load_dataframe(cfg.paths.data_entry_csv, index)
    print("rows with resolved path:", len(df))
    print("unique patients:", df["patient_id"].nunique())

    print("\n== 14-class positive counts ==")
    counts = Counter()
    for s in df["labels"].values:
        vec = labels_to_vector(s)
        for i, v in enumerate(vec):
            if v > 0:
                counts[DISEASE_CLASSES[i]] += 1
    for name in DISEASE_CLASSES:
        print(f"  {name:20s} {counts[name]:6d}")
    n_no_finding = int((df["labels"] == "No Finding").sum())
    print(f"  {'No Finding':20s} {n_no_finding:6d}")

    print("\n== BBox_List_2017.csv ==")
    bbox = pd.read_csv(cfg.paths.bbox_csv)
    bbox.columns = [c.strip() for c in bbox.columns]
    print("bbox rows:", len(bbox))
    print("bbox unique images:", bbox["Image Index"].nunique())
    print("bbox labels:", sorted(bbox["Finding Label"].unique().tolist()))


if __name__ == "__main__":
    main()
