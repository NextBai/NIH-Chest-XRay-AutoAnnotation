# Kaggle T4x2 DDP + AMP 驗證腳本 (NIH ChestX-ray14)
# 由 Kaggle MCP 以 script kernel (SaveAndRunAll, GPU=T4x2, Internet=on) 執行。
# 嚴禁 argparse：所有設定以 Config dataclass 屬性指定。
#
# 重要：所有「會產生副作用」的編排程式碼都必須放在 `if __name__ == "__main__"`
# 之下。因為 DDP 使用 torch.multiprocessing.spawn（start method = 'spawn'），
# 每個子行程會重新 import 主模組；若編排碼在模組頂層，子行程會重跑 train_entry
# 造成遞迴 spawn 爆炸。子行程只需能 import src.train._worker 即可。
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/NextBai/NIH-Chest-XRay-AutoAnnotation.git"
PROJ = "/kaggle/working/NIH-Chest-XRay-AutoAnnotation"

# 讓主行程與 spawn 子行程都能 import src.*（PYTHONPATH 會被子行程繼承）
sys.path.insert(0, PROJ)
os.environ["PYTHONPATH"] = PROJ + os.pathsep + os.environ.get("PYTHONPATH", "")


def _bootstrap():
    """clone 公開 repo + 安裝必要套件（僅主行程執行一次）。"""
    if not os.path.exists(PROJ):
        subprocess.run(["git", "clone", "--depth", "1", REPO],
                       cwd="/kaggle/working", check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.27", "grad-cam"], check=True)


def _find(base: Path, name: str) -> Path:
    hits = list(base.rglob(name))
    return hits[0] if hits else base / name


def main():
    _bootstrap()

    import torch
    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    print("CUDA devices:", n_gpu, names, flush=True)

    from configs.config import Config
    cfg = Config()
    base = Path("/kaggle/input/data")
    cfg.paths.data_dir = base
    cfg.paths.metadata_dir = base
    cfg.paths.data_entry_csv = _find(base, "Data_Entry_2017.csv")
    cfg.paths.bbox_csv = _find(base, "BBox_List_2017.csv")
    cfg.paths.output_dir = Path("/kaggle/working/artifacts")
    print("data_entry_csv:", cfg.paths.data_entry_csv, flush=True)
    print("image dirs:", len(cfg.paths.image_search_dirs()), flush=True)

    # DDP 驗證：小步數證明 T4x2 spawn + AMP 正常（不報 OOM / I/O 阻塞）
    cfg.runtime.use_ddp = n_gpu > 1
    cfg.runtime.num_workers = 2
    cfg.runtime.batch_size = 32
    cfg.model.pretrained = False  # 驗證 DDP/AMP 管線，免去下載權重的網路波動

    from src.train.train_classifier import train_entry
    train_entry(cfg, max_steps=5, subset_n=2000)
    print(f"KAGGLE DDP VALIDATION DONE (use_ddp={cfg.runtime.use_ddp}, gpus={n_gpu})",
          flush=True)


if __name__ == "__main__":
    main()
