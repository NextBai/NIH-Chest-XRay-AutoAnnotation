"""集中式設定模組 (NIH ChestX-ray14 自動標註 + DDP/AMP 訓練).

設計準則:
- 嚴禁 argparse / 命令列參數。所有可調項目集中於此, 以 dataclass 表達。
- 環境自動偵測: 本機 RTX 3060 (6GB) vs Kaggle T4x2 / P100, 自動套用對應 batch/worker。
- 任何運算預設開啟 AMP/FP16。

被以下模組 import:
    src/data/dataset.py, src/models/classifier.py,
    src/train/train_classifier.py, src/pseudo_label/cam_pipeline.py, scripts/*.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# 14 類胸腔疾病 (ChestX-ray14 官方順序). "No Finding" 不列入分類輸出。
# --------------------------------------------------------------------------- #
DISEASE_LABELS: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)
NUM_CLASSES: int = len(DISEASE_LABELS)
LABEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(DISEASE_LABELS)}

# BBox_List_2017.csv 僅含 8 類醫師標註框, 用於 IoU 校準。
BBOX_LABELS: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax",
)
# BBox CSV 用 "Infiltrate", Data_Entry 用 "Infiltration"; 統一對映。
BBOX_NAME_FIX: dict[str, str] = {"Infiltrate": "Infiltration"}


def _detect_environment() -> str:
    """偵測執行環境: 'kaggle' | 'local'。"""
    if os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def _detect_project_root() -> Path:
    """專案根目錄: 本機為此檔上兩層; Kaggle 為 dataset 掛載點。"""
    if _detect_environment() == "kaggle":
        for cand in (Path("/kaggle/input/data"), Path("/kaggle/input")):
            if cand.exists():
                return cand
    return Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=_detect_project_root)
    data_dir: Optional[Path] = None
    metadata_dir: Optional[Path] = None
    data_entry_csv: Optional[Path] = None
    bbox_csv: Optional[Path] = None
    train_val_list: Optional[Path] = None
    test_list: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        root = Path(self.project_root)
        self.data_dir = self.data_dir or root / "data"
        self.metadata_dir = self.metadata_dir or self.data_dir / "metadata"
        self.data_entry_csv = self.data_entry_csv or self.metadata_dir / "Data_Entry_2017.csv"
        self.bbox_csv = self.bbox_csv or self.metadata_dir / "BBox_List_2017.csv"
        self.train_val_list = self.train_val_list or self.metadata_dir / "train_val_list.txt"
        self.test_list = self.test_list or self.metadata_dir / "test_list.txt"
        if self.output_dir is None:
            self.output_dir = (
                Path("/kaggle/working/artifacts")
                if _detect_environment() == "kaggle"
                else root / "artifacts"
            )

    def image_search_dirs(self) -> list[Path]:
        """所有 images_xxx/images 子目錄 (建立檔名->路徑索引用)。"""
        return sorted(Path(self.data_dir).glob("images_*/images"))


@dataclass
class DataConfig:
    image_size: int = 224
    in_chans: int = 3
    val_split: float = 0.15
    seed: int = 42
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    backbone: str = "densenet121"  # CheXNet 經典骨幹; CAM 友善。
    pretrained: bool = True
    drop_rate: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 15
    base_lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    use_amp: bool = True
    grad_accum_steps: int = 1
    use_pos_weight: bool = True
    log_interval: int = 50
    ckpt_name: str = "classifier_best.pt"


@dataclass
class RuntimeConfig:
    env: str = field(default_factory=_detect_environment)
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    use_ddp: bool = False
    dist_backend: str = "nccl"
    dist_url: str = "tcp://127.0.0.1:23456"

    def __post_init__(self) -> None:
        if self.env == "kaggle":
            self.batch_size = 64       # T4 16GB/卡
            self.num_workers = 2
            self.use_ddp = True
        else:
            self.batch_size = 16       # RTX 3060 6GB 保守值
            self.num_workers = 4
            self.use_ddp = False


@dataclass
class CAMConfig:
    method: str = "gradcampp"          # gradcam | gradcampp | xgradcam
    heatmap_thresh: float = 0.5
    min_area_ratio: float = 0.005
    confidence_thresh: float = 0.30
    max_boxes_per_class: int = 3
    viz_sample_n: int = 40


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cam: CAMConfig = field(default_factory=CAMConfig)

    def to_dict(self) -> dict:
        def _ser(o):
            if isinstance(o, dict):
                return {k: _ser(v) for k, v in o.items()}
            if isinstance(o, Path):
                return str(o)
            return o
        return _ser(asdict(self))


# 單例式預設設定; 各模組 `from configs.config import CFG` 取用。
CFG = Config()


if __name__ == "__main__":
    import json
    print(json.dumps(CFG.to_dict(), indent=2, ensure_ascii=False))
