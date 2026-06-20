"""Grad-CAM(++) pseudo-labeling pipeline for NIH ChestX-ray14.

Turns a trained multi-label classifier into bounding-box pseudo-labels:
  1. forward pass -> per-class probabilities
  2. for each class above prob threshold, compute Grad-CAM(++) heatmap
  3. threshold heatmap -> OpenCV contours -> bounding boxes (x,y,w,h)
  4. confidence = class_prob * heatmap_peak; drop boxes below cam.confidence_thresh
  5. write artifacts/pseudo_labels/pseudo_labels.csv + overlay visualizations

No argparse. Driven by configs/config.py. Calibrated against the 984
radiologist boxes in BBox_List_2017.csv via src.utils.metrics.bbox_iou.

Public entry: generate_pseudo_labels(cfg, ckpt_path, image_names=None, limit=None).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import Config, DISEASE_LABELS  # noqa: E402
from src.data.dataset import (  # noqa: E402
    DISEASE_CLASSES, build_image_index, labels_to_vector,
)
from src.models.classifier import build_classifier, get_cam_target_layer  # noqa: E402

# pytorch-grad-cam (installed as `grad-cam`)
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

_CAM_REGISTRY = {"gradcam": GradCAM, "gradcampp": GradCAMPlusPlus, "xgradcam": XGradCAM}


def _load_model(cfg: Config, ckpt_path: str, device: torch.device):
    model = build_classifier(cfg.model.backbone, len(DISEASE_CLASSES),
                             pretrained=False, drop_rate=0.0)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


def _preprocess(path: str, size: int, mean, std, in_chans: int):
    mode = "RGB" if in_chans == 3 else "L"
    from PIL import Image
    img = Image.open(path).convert(mode)
    orig = np.asarray(img)
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous().unsqueeze(0)
    return tensor, orig


def heatmap_to_boxes(heat: np.ndarray, thresh: float, min_area_ratio: float,
                     max_boxes: int):
    """Binarize a [0,1] heatmap and extract bounding boxes via contours.

    Returns list of (x, y, w, h, peak) in heatmap pixel coords.
    """
    h, w = heat.shape
    binary = (heat >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * h * w
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        peak = float(heat[y:y + bh, x:x + bw].max())
        boxes.append((x, y, bw, bh, peak))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:max_boxes]


def _scale_box(box, from_size: int, to_w: int, to_h: int):
    """Scale (x,y,w,h) from heatmap grid to original image resolution."""
    x, y, bw, bh = box
    sx, sy = to_w / from_size, to_h / from_size
    return [round(x * sx, 2), round(y * sy, 2), round(bw * sx, 2), round(bh * sy, 2)]


def _overlay(orig: np.ndarray, heat: np.ndarray, boxes, labels):
    """Render heatmap + boxes + confidence text onto the original image (BGR)."""
    img = orig if orig.ndim == 3 else cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img.astype(np.uint8), (heat.shape[1], heat.shape[0]))
    cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(img, 0.6, cmap, 0.4, 0)
    for (x, y, bw, bh, conf), name in zip(boxes, labels):
        cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
        cv2.putText(vis, f"{name}:{conf:.2f}", (int(x), max(0, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def generate_pseudo_labels(cfg: Config, ckpt_path: str,
                           image_names: list[str] | None = None,
                           limit: int | None = None, save_viz: bool = True):
    """Main entry: produce confidence-filtered bbox pseudo-labels.

    Returns the output DataFrame and writes CSV + viz PNGs under output_dir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, ckpt_path, device)
    target_layers = get_cam_target_layer(model, cfg.model.backbone)
    cam_cls = _CAM_REGISTRY[cfg.cam.method]

    index = build_image_index(cfg.paths.data_dir)
    names = image_names if image_names is not None else list(index.keys())
    if limit is not None:
        names = names[:limit]

    out_dir = Path(cfg.paths.output_dir) / "pseudo_labels"
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_viz = 0
    with cam_cls(model=model, target_layers=target_layers) as cam:
        for name in names:
            path = index.get(name)
            if path is None:
                continue
            tensor, orig = _preprocess(path, cfg.data.image_size, cfg.data.mean,
                                       cfg.data.std, cfg.data.in_chans)
            tensor = tensor.to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor))[0].cpu().numpy()
            oh, ow = orig.shape[0], orig.shape[1]
            pos_classes = np.where(probs >= cfg.cam.confidence_thresh)[0]
            kept_boxes, kept_labels = [], []
            for ci in pos_classes:
                targets = [ClassifierOutputTarget(int(ci))]
                grayscale = cam(input_tensor=tensor, targets=targets)[0]
                grayscale = (grayscale - grayscale.min()) / (np.ptp(grayscale) + 1e-8)
                raw_boxes = heatmap_to_boxes(grayscale, cfg.cam.heatmap_thresh,
                                             cfg.cam.min_area_ratio,
                                             cfg.cam.max_boxes_per_class)
                cls_name = DISEASE_CLASSES[ci]
                for (x, y, bw, bh, peak) in raw_boxes:
                    confidence = round(float(probs[ci]) * float(peak), 4)
                    if confidence < cfg.cam.confidence_thresh:
                        continue
                    sx, sy, sw, sh = _scale_box((x, y, bw, bh),
                                                cfg.data.image_size, ow, oh)
                    rows.append({"Image Index": name, "Finding Label": cls_name,
                                 "x": sx, "y": sy, "w": sw, "h": sh,
                                 "confidence": confidence, "cam_method": cfg.cam.method})
                    kept_boxes.append((x, y, bw, bh, confidence))
                    kept_labels.append(cls_name)
            if save_viz and kept_boxes and n_viz < cfg.cam.viz_sample_n:
                # recompute a representative heatmap (top class) for the overlay
                top = int(pos_classes[np.argmax(probs[pos_classes])])
                gmap = cam(input_tensor=tensor,
                           targets=[ClassifierOutputTarget(top)])[0]
                gmap = (gmap - gmap.min()) / (np.ptp(gmap) + 1e-8)
                vis = _overlay(orig, gmap, kept_boxes, kept_labels)
                cv2.imwrite(str(viz_dir / f"{Path(name).stem}_cam.png"), vis)
                n_viz += 1

    df = pd.DataFrame(rows, columns=["Image Index", "Finding Label", "x", "y", "w",
                                     "h", "confidence", "cam_method"])
    csv_path = out_dir / "pseudo_labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"[cam] wrote {len(df)} pseudo-boxes for {df['Image Index'].nunique()} "
          f"images -> {csv_path}", flush=True)
    return df, csv_path


if __name__ == "__main__":
    cfg = Config()
    ckpt = Path(cfg.paths.output_dir) / "checkpoints" / cfg.train.ckpt_name
    generate_pseudo_labels(cfg, str(ckpt), limit=200)
