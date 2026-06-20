# Kaggle T4x2 DDP + AMP 驗證腳本 (NIH ChestX-ray14)
# 由 Kaggle MCP 以 script kernel (SaveAndRunAll, GPU=T4x2, Internet=on) 執行。
# 嚴禁 argparse：所有設定以 Config dataclass 屬性指定。
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/NextBai/NIH-Chest-XRay-AutoAnnotation.git"
PROJ = "/kaggle/working/NIH-Chest-XRay-AutoAnnotation"

# 1) 取得專案程式碼（公開 repo）
if not os.path.exists(PROJ):
    subprocess.run(["git", "clone", "--depth", "1", REPO], cwd="/kaggle/working", check=True)
sys.path.insert(0, PROJ)

# 2) 安裝必要套件
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.27", "grad-cam"], check=True)

import torch  # noqa: E402

n_gpu = torch.cuda.device_count()
print("CUDA devices:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)

# 3) 設定路徑指向 Kaggle NIH 公開資料集；CSV 可能在 root 或子目錄，自動偵測
from configs.config import Config  # noqa: E402

cfg = Config()
base = Path("/kaggle/input/data")


def _find(name: str) -> Path:
    hits = list(base.rglob(name))
    return hits[0] if hits else base / name


cfg.paths.data_dir = base
cfg.paths.metadata_dir = base
cfg.paths.data_entry_csv = _find("Data_Entry_2017.csv")
cfg.paths.bbox_csv = _find("BBox_List_2017.csv")
cfg.paths.output_dir = Path("/kaggle/working/artifacts")
print("data_entry_csv:", cfg.paths.data_entry_csv, flush=True)
print("image dirs:", len(cfg.paths.image_search_dirs()), flush=True)

# 4) DDP 驗證：小步數證明 T4x2 spawn + AMP 正常（不報 OOM/I/O 阻塞）
cfg.runtime.use_ddp = n_gpu > 1
cfg.runtime.num_workers = 2
cfg.runtime.batch_size = 32

from src.train.train_classifier import train_entry  # noqa: E402

train_entry(cfg, max_steps=5, subset_n=2000)
print(f"KAGGLE DDP VALIDATION DONE (use_ddp={cfg.runtime.use_ddp}, gpus={n_gpu})", flush=True)
