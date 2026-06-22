"""IoU calibration: pseudo-label boxes vs 984 radiologist boxes.

Compares CAM-derived pseudo-labels against `BBox_List_2017.csv` over the 8
overlapping classes (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass,
Nodule, Pneumonia, Pneumothorax). Sweeps `confidence_thresh` post-hoc — does
NOT re-run CAM. `heatmap_thresh` calibration would need a fresh inference run.

Public entry: calibrate_thresholds(pseudo_csv, bbox_csv, out_dir, sweep=None)
              -> (best_thresh, report)

Imported by scripts/iou_calibration_local.py (local driver) and
scripts/kaggle_full_pseudo_label.py (Kaggle full inference kernel).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import BBOX_LABELS, BBOX_NAME_FIX
from src.utils.metrics import bbox_iou


def _load_doctor_boxes(bbox_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(bbox_csv)
    raw = raw.rename(columns={raw.columns[0]: "Image Index",
                              raw.columns[1]: "Finding Label",
                              raw.columns[2]: "x", raw.columns[3]: "y",
                              raw.columns[4]: "w", raw.columns[5]: "h"})
    keep = ["Image Index", "Finding Label", "x", "y", "w", "h"]
    df = raw[keep].copy()
    df["Finding Label"] = df["Finding Label"].replace(BBOX_NAME_FIX)
    return df


def _per_box_best_iou(doctor: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    """For each doctor box, find best-IoU pseudo box of same class+image."""
    pkey = pseudo.groupby(["Image Index", "Finding Label"])
    best, matched = [], []
    for _, drow in doctor.iterrows():
        key = (drow["Image Index"], drow["Finding Label"])
        if key not in pkey.groups:
            best.append(0.0)
            matched.append(0)
            continue
        candidates = pkey.get_group(key)
        ious = [bbox_iou([drow["x"], drow["y"], drow["w"], drow["h"]],
                         [r["x"], r["y"], r["w"], r["h"]])
                for _, r in candidates.iterrows()]
        best.append(max(ious) if ious else 0.0)
        matched.append(1 if ious else 0)
    out = doctor.copy()
    out["best_iou"] = best
    out["matched"] = matched
    return out


def _summarize(matched_df: pd.DataFrame, pseudo_n: int) -> dict:
    per_class = {}
    for cls in BBOX_LABELS:
        cls_norm = BBOX_NAME_FIX.get(cls, cls)
        sub = matched_df[matched_df["Finding Label"] == cls_norm]
        if len(sub) == 0:
            per_class[cls_norm] = {"doctor_boxes": 0, "matched": 0,
                                   "mean_iou": 0.0, "recall@0.3": 0.0,
                                   "recall@0.5": 0.0}
            continue
        iou = sub["best_iou"].values
        per_class[cls_norm] = {
            "doctor_boxes": int(len(sub)),
            "matched": int(sub["matched"].sum()),
            "mean_iou": float(iou.mean()),
            "recall@0.3": float((iou >= 0.3).mean()),
            "recall@0.5": float((iou >= 0.5).mean()),
        }
    iou_all = matched_df["best_iou"].values
    return {
        "n_doctor_boxes": int(len(matched_df)),
        "n_matched": int(matched_df["matched"].sum()),
        "n_pseudo_boxes": int(pseudo_n),
        "mean_iou": float(iou_all.mean()) if len(iou_all) else 0.0,
        "recall@0.3": float((iou_all >= 0.3).mean()) if len(iou_all) else 0.0,
        "recall@0.5": float((iou_all >= 0.5).mean()) if len(iou_all) else 0.0,
        "per_class": per_class,
    }


def _plot_sweep(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thr = [r["confidence_thresh"] for r in rows]
    miou = [r["mean_iou"] for r in rows]
    r03 = [r["recall@0.3"] for r in rows]
    r05 = [r["recall@0.5"] for r in rows]
    pn = [r["n_pseudo_boxes"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(thr, miou, "o-", color="#3b7dd8", label="mean IoU")
    ax1.plot(thr, r03, "s-", color="#2ca02c", label="recall@IoU>=0.3")
    ax1.plot(thr, r05, "^-", color="#d62728", label="recall@IoU>=0.5")
    ax1.set_xlabel("confidence_thresh")
    ax1.set_ylabel("metric")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(thr, pn, "d--", color="gray", alpha=0.7, label="pseudo box count")
    ax2.set_ylabel("pseudo box count (after filter)")
    ax2.legend(loc="upper right")
    plt.title("IoU calibration sweep -- pseudo vs 984 doctor boxes")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def calibrate_thresholds(pseudo_csv: Path, bbox_csv: Path, out_dir: Path,
                         sweep: list[float] | None = None) -> tuple[float, dict]:
    """Sweep confidence_thresh; pick value maximizing (recall@0.3, mean_iou)."""
    sweep = sweep or [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                      0.55, 0.60, 0.65, 0.70]
    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo = pd.read_csv(pseudo_csv)
    doctor_all = _load_doctor_boxes(bbox_csv)
    overlap_imgs = set(pseudo["Image Index"]) & set(doctor_all["Image Index"])
    doctor = doctor_all[doctor_all["Image Index"].isin(overlap_imgs)].reset_index(drop=True)
    print(f"[iou] pseudo rows={len(pseudo)} | doctor rows={len(doctor_all)} "
          f"| overlap imgs={len(overlap_imgs)} | doctor in overlap={len(doctor)}",
          flush=True)

    rows = []
    for thr in sweep:
        sub = pseudo[pseudo["confidence"] >= thr]
        matched_df = _per_box_best_iou(doctor, sub)
        s = _summarize(matched_df, len(sub))
        s["confidence_thresh"] = thr
        rows.append(s)
        print(f"  thr={thr:.2f}  pseudo={len(sub):>6}  "
              f"mean_iou={s['mean_iou']:.3f}  "
              f"R@0.3={s['recall@0.3']:.3f}  R@0.5={s['recall@0.5']:.3f}",
              flush=True)

    best = max(rows, key=lambda r: (r["recall@0.3"], r["mean_iou"]))
    report = {"best_thresh": best["confidence_thresh"],
              "best_summary": best, "sweep": rows,
              "n_overlap_images": len(overlap_imgs)}

    with open(out_dir / "iou_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _plot_sweep(rows, out_dir / "thresh_sweep.png")
    print(f"[iou] best confidence_thresh={best['confidence_thresh']:.2f}  "
          f"mean_iou={best['mean_iou']:.3f}  R@0.3={best['recall@0.3']:.3f}",
          flush=True)
    return best["confidence_thresh"], report
