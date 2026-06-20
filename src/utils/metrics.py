"""Evaluation metrics. Imported by train_classifier.py and cam_pipeline.py."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def multilabel_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Per-class + mean AUROC for multi-label classification.

    Classes with only one ground-truth value present are skipped (AUROC undefined).
    Returns {"mean": float, "per_class": list[float|nan]}.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    per_class = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.min() == yt.max():  # only one class present
            per_class.append(float("nan"))
            continue
        per_class.append(float(roc_auc_score(yt, y_prob[:, c])))
    valid = [v for v in per_class if not np.isnan(v)]
    mean = float(np.mean(valid)) if valid else float("nan")
    return {"mean": mean, "per_class": per_class}


def bbox_iou(box_a, box_b) -> float:
    """IoU of two boxes in [x, y, w, h] (top-left origin). Used to calibrate
    CAM-derived boxes against the 984 radiologist boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0
