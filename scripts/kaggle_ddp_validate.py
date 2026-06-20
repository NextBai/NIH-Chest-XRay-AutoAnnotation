# Kaggle DDP + AMP 驗證腳本 (NIH ChestX-ray14)
# 由 Kaggle MCP 以 script kernel (SaveAndRunAll, GPU on, Internet on) 執行。
# 嚴禁 argparse：所有設定以 Config dataclass 屬性指定。
#
# 重要：所有「會產生副作用」的編排碼都在 `if __name__ == "__main__"` 之下，
# 因為 DDP 用 mp.spawn（start method 'spawn'）會重新 import 主模組。
#
# 全程把進度與例外寫入 /kaggle/working/ddp_run.log（commit 進 output 可取回）。
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO = "https://github.com/NextBai/NIH-Chest-XRay-AutoAnnotation.git"
PROJ = "/kaggle/working/NIH-Chest-XRay-AutoAnnotation"
LOG = "/kaggle/working/ddp_run.log"

sys.path.insert(0, PROJ)
os.environ["PYTHONPATH"] = PROJ + os.pathsep + os.environ.get("PYTHONPATH", "")


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _bootstrap():
    import subprocess
    if not os.path.exists(PROJ):
        subprocess.run(["git", "clone", "--depth", "1", REPO],
                       cwd="/kaggle/working", check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.27", "grad-cam"], check=True)


def _discover_data():
    """在 /kaggle/input 底下找 Data_Entry_2017.csv，回傳 (data_dir, csv, bbox)。
    Kaggle 掛載路徑不固定，故全域 rglob 後以 CSV 所在目錄為 data_dir。"""
    root = Path("/kaggle/input")
    log(f"/kaggle/input entries: {[p.name for p in root.iterdir()] if root.exists() else 'MISSING'}")
    csv_hits = list(root.rglob("Data_Entry_2017.csv"))
    log(f"Data_Entry_2017.csv hits: {[str(p) for p in csv_hits]}")
    if not csv_hits:
        raise FileNotFoundError("Data_Entry_2017.csv not found under /kaggle/input")
    csv = csv_hits[0]
    data_dir = csv.parent
    bbox_hits = list(root.rglob("BBox_List_2017.csv"))
    bbox = bbox_hits[0] if bbox_hits else data_dir / "BBox_List_2017.csv"
    # 影像目錄：優先 images_*/images，否則找任一含 png 的目錄樣本
    img_dirs = sorted(data_dir.glob("images_*/images"))
    if not img_dirs:
        img_dirs = sorted({p.parent for p in list(data_dir.rglob("*.png"))[:50]})
    log(f"data_dir={data_dir}  image dir candidates={len(img_dirs)}  sample={[str(d) for d in img_dirs[:3]]}")
    return data_dir, csv, bbox, img_dirs


def main():
    log("=== START ===")
    _bootstrap()
    log("bootstrap done (clone + pip)")

    import torch
    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    log(f"CUDA devices: {n_gpu} {names} | torch {torch.__version__}")

    from configs.config import Config
    cfg = Config()
    data_dir, csv, bbox, img_dirs = _discover_data()
    cfg.paths.data_dir = data_dir
    cfg.paths.metadata_dir = csv.parent
    cfg.paths.data_entry_csv = csv
    cfg.paths.bbox_csv = bbox
    cfg.paths.output_dir = Path("/kaggle/working/artifacts")

    cfg.runtime.use_ddp = n_gpu > 1            # 2+ GPU → mp.spawn DDP
    cfg.runtime.num_workers = 2
    cfg.runtime.batch_size = 32
    cfg.model.pretrained = False               # 驗證 DDP/AMP 管線，免下載權重
    log(f"use_ddp={cfg.runtime.use_ddp} bs={cfg.runtime.batch_size}")

    from src.train.train_classifier import train_entry
    log("calling train_entry(max_steps=5, subset_n=2000) ...")
    train_entry(cfg, max_steps=5, subset_n=2000)
    log(f"VALIDATION DONE (use_ddp={cfg.runtime.use_ddp}, gpus={n_gpu})")
    log("=== SUCCESS ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write("=== EXCEPTION ===\n" + tb + "\n")
        except Exception:
            pass
        print(tb, flush=True)
        raise
