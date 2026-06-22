# Kaggle full pseudo-label + IoU calibration kernel.
# Loads Stage 6 classifier_best.pt (from prior notebook output added via UI),
# runs Grad-CAM++ on all 112K images, then calibrates confidence_thresh
# against the 984 doctor boxes in BBox_List_2017.csv.
#
# Submitted via Kaggle MCP. argparse forbidden.
# UI requirements (manual, per Stage 6 lesson):
#   - Accelerator: GPU T4 x2
#   - Add Input: dataset `nih-chest-xrays/data`
#   - Add Input: Notebook output `xiaobai1221/nih-full-train-t4x2`
#                (provides classifier_best.pt)
#   - Internet: ON (clone repo + pip)
import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

REPO = "https://github.com/NextBai/NIH-Chest-XRay-AutoAnnotation.git"
PROJ = "/kaggle/working/NIH-Chest-XRay-AutoAnnotation"
LOG = "/kaggle/working/full_pseudo_label.log"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _bootstrap():
    if not os.path.exists(PROJ):
        subprocess.run(["git", "clone", "--depth", "1", REPO],
                       cwd="/kaggle/working", check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.27", "grad-cam"], check=True)
    cfg_py = Path(PROJ) / "configs" / "config.py"
    if not cfg_py.exists():
        raise RuntimeError(f"clone incomplete -- {cfg_py} missing; "
                           f"PROJ entries={[p.name for p in Path(PROJ).iterdir()]}")
    os.chdir(PROJ)
    sys.path.insert(0, PROJ)
    os.environ["PYTHONPATH"] = PROJ + os.pathsep + os.environ.get("PYTHONPATH", "")
    log(f"PROJ ready: {sorted(p.name for p in Path(PROJ).iterdir())}")


def _discover_inputs():
    """Prefer paths under /kaggle/input/datasets/ (real dataset mount).

    Stage 8 first-run bug: rglob returned the CSV from the attached prior
    notebook output (which has metadata-only `data/metadata/Data_Entry_2017.csv`
    but no `images_*` dirs) ahead of the real dataset. Filter explicitly.
    """
    root = Path("/kaggle/input")
    log(f"/kaggle/input entries: {[p.name for p in root.iterdir()]}")

    def _prefer_dataset(hits):
        ds = [h for h in hits if "/datasets/" in h.as_posix()]
        return (ds or hits)

    csv_hits = _prefer_dataset(list(root.rglob("Data_Entry_2017.csv")))
    if not csv_hits:
        raise FileNotFoundError("Data_Entry_2017.csv not mounted")
    csv = csv_hits[0]
    data_dir = csv.parent
    # If data_dir has no images_*/images, walk up to find one that does.
    if not list(data_dir.glob("images_*/images")):
        for parent in [data_dir.parent, data_dir.parent.parent]:
            if list(parent.glob("images_*/images")):
                data_dir = parent
                break
    bbox_hits = _prefer_dataset(list(root.rglob("BBox_List_2017.csv")))
    bbox = bbox_hits[0] if bbox_hits else data_dir / "BBox_List_2017.csv"
    ckpt_hits = list(root.rglob("classifier_best.pt"))
    if not ckpt_hits:
        raise FileNotFoundError(
            "classifier_best.pt not mounted -- add prior notebook "
            "xiaobai1221/nih-full-train-t4x2 as Input")
    ckpt = ckpt_hits[0]
    img_dirs = sorted(data_dir.glob("images_*/images"))
    log(f"csv={csv}")
    log(f"data_dir={data_dir}  image dirs={len(img_dirs)}  "
        f"bbox={'found' if bbox_hits else 'MISSING'}  ckpt={ckpt}")
    if len(img_dirs) == 0:
        raise FileNotFoundError(
            f"no images_*/images under {data_dir} -- dataset mount layout "
            f"unexpected. Hits were: {[h.as_posix() for h in csv_hits[:5]]}")
    return data_dir, csv, bbox, ckpt


def _build_cfg(data_dir, csv, bbox, out_dir):
    from configs.config import Config
    cfg = Config()
    cfg.paths.data_dir = data_dir
    cfg.paths.metadata_dir = csv.parent
    cfg.paths.data_entry_csv = csv
    cfg.paths.bbox_csv = bbox
    cfg.paths.output_dir = out_dir
    return cfg


def main():
    log("=== FULL PSEUDO-LABEL + IOU CALIBRATION START ===")
    _bootstrap()
    log("bootstrap done")

    import torch
    data_dir, csv, bbox, ckpt = _discover_inputs()
    out_dir = Path("/kaggle/working/artifacts")
    cfg = _build_cfg(data_dir, csv, bbox, out_dir)

    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    log(f"torch {torch.__version__}  CUDA devices: {n_gpu} {names}")
    log(f"cam.heatmap_thresh={cfg.cam.heatmap_thresh}  "
        f"cam.confidence_thresh={cfg.cam.confidence_thresh}  "
        f"max_boxes/cls={cfg.cam.max_boxes_per_class}")

    from src.pseudo_label.cam_pipeline import generate_pseudo_labels
    log("calling generate_pseudo_labels(limit=None) -- full 112K ...")
    df, csv_path = generate_pseudo_labels(cfg, str(ckpt), limit=None,
                                          save_viz=True)
    n_imgs = df["Image Index"].nunique()
    log(f"pseudo-labels: {len(df)} boxes / {n_imgs} images -> {csv_path}")

    from src.pseudo_label.iou_calibration import calibrate_thresholds
    iou_dir = out_dir / "iou_calibration"
    log(f"running IoU calibration sweep -> {iou_dir} ...")
    best_thresh, report = calibrate_thresholds(csv_path, bbox, iou_dir)
    log(f"IoU calibration done. best confidence_thresh={best_thresh:.2f}  "
        f"mean_iou={report['best_summary']['mean_iou']:.3f}  "
        f"R@0.3={report['best_summary']['recall@0.3']:.3f}  "
        f"R@0.5={report['best_summary']['recall@0.5']:.3f}")

    log("=== SUCCESS ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write("=== TOPLEVEL EXCEPTION ===\n" + tb + "\n")
        except Exception:
            pass
        print(tb, flush=True)
        raise
