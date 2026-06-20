# Kaggle T4x2 完整訓練 kernel (NIH ChestX-ray14 弱監督自動標註)
# 由 Kaggle MCP 以 script kernel(SaveAndRunAll, GPU=Tesla T4 → 自動 2 卡, Internet on)執行。
# 嚴禁 argparse:所有設定以 Config dataclass 屬性覆寫。
#
# 資料來源:官方公開 dataset `nih-chest-xrays/data`(45GB, 1024px PNG + Data_Entry_2017.csv
#           + BBox_List_2017.csv 984 醫師框),不重新上傳。
#
# 流程:
#   1. clone 本專案 public repo(取得 src/ 模組)+ pip 安裝 timm/grad-cam
#   2. 掛載偵測官方 dataset 路徑,覆寫 cfg.paths.*(繞開 config 的本機預設)
#   3. T4x2 DDP + AMP 完整訓練 DenseNet121 多標籤分類器(逐 epoch 存 checkpoint + val AUROC)
#   4. 載入最佳權重,對取樣影像產生 Grad-CAM++ 偽標籤
#   5. 繪製信心分數直方圖(DoD:視覺化的 Confidence Score 篩選機制)
#
# 重要:所有「會產生副作用」的編排碼都在 `if __name__ == "__main__"` 之下,
#       因為 DDP 用 mp.spawn(start method 'spawn')會重新 import 主模組,
#       子行程不可重複 clone/pip。_worker 定義於 src.train,子行程 import 它即可。
#
# 全程寫 /kaggle/working/full_train.log(commit 進 output,ERROR 也能取回)。
import os
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

REPO = "https://github.com/NextBai/NIH-Chest-XRay-AutoAnnotation.git"
PROJ = "/kaggle/working/NIH-Chest-XRay-AutoAnnotation"
LOG = "/kaggle/working/full_train.log"

# 偽標籤取樣量(完整 112K CAM 推論極慢,留待專屬 run;此處取樣足夠畫信心直方圖 + 抽驗)
PSEUDO_SAMPLE_N = 3000


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _bootstrap() -> None:
    """clone repo + pip 安裝額外依賴(僅在主行程執行一次)。"""
    if not os.path.exists(PROJ):
        subprocess.run(["git", "clone", "--depth", "1", REPO],
                       cwd="/kaggle/working", check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.27", "grad-cam"], check=True)
    sys.path.insert(0, PROJ)
    os.environ["PYTHONPATH"] = PROJ + os.pathsep + os.environ.get("PYTHONPATH", "")


def _discover_data():
    """在 /kaggle/input 找官方 dataset。回傳 (data_dir, data_entry_csv, bbox_csv)。
    官方 `nih-chest-xrays/data` 掛載後:CSV 在掛載根目錄,影像在 images_*/images/。"""
    root = Path("/kaggle/input")
    entries = [p.name for p in root.iterdir()] if root.exists() else "MISSING"
    log(f"/kaggle/input entries: {entries}")
    csv_hits = list(root.rglob("Data_Entry_2017.csv"))
    log(f"Data_Entry_2017.csv hits: {[str(p) for p in csv_hits]}")
    if not csv_hits:
        raise FileNotFoundError("Data_Entry_2017.csv 未掛載 — 請確認 dataset nih-chest-xrays/data 已 attach")
    csv = csv_hits[0]
    data_dir = csv.parent
    bbox_hits = list(root.rglob("BBox_List_2017.csv"))
    bbox = bbox_hits[0] if bbox_hits else data_dir / "BBox_List_2017.csv"
    img_dirs = sorted(data_dir.glob("images_*/images"))
    log(f"data_dir={data_dir} | image dirs={len(img_dirs)} | bbox={'found' if bbox_hits else 'MISSING'}")
    return data_dir, csv, bbox


def _build_cfg():
    """建立 Config 並覆寫為 Kaggle 官方 dataset 路徑(繞開 config 本機預設)。"""
    import torch
    from configs.config import Config
    cfg = Config()
    data_dir, csv, bbox = _discover_data()
    cfg.paths.data_dir = data_dir
    cfg.paths.metadata_dir = csv.parent
    cfg.paths.data_entry_csv = csv
    cfg.paths.bbox_csv = bbox
    cfg.paths.output_dir = Path("/kaggle/working/artifacts")

    n_gpu = torch.cuda.device_count()
    cfg.runtime.use_ddp = n_gpu > 1        # 2+ GPU → mp.spawn DDP;1 GPU 自動退單卡
    cfg.runtime.num_workers = 4            # T4x2 instance 4 cores,提升 I/O 吞吐
    cfg.runtime.batch_size = 64            # T4 16GB/卡
    cfg.model.pretrained = True            # CheXNet 風格:ImageNet 預訓練起步
    return cfg, n_gpu


def _plot_confidence_hist(df, out_png: Path, thresh: float) -> None:
    """DoD:視覺化偽標籤信心分數分佈與篩選閾值。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(df):
        ax.hist(df["confidence"].values, bins=40, color="#3b7dd8", edgecolor="white")
    ax.axvline(thresh, color="red", linestyle="--", linewidth=2,
               label=f"confidence_thresh = {thresh}")
    ax.set_title("Pseudo-label Confidence Score Distribution")
    ax.set_xlabel("confidence = class_prob x heatmap_peak")
    ax.set_ylabel("box count")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log(f"  confidence histogram -> {out_png}")


def main():
    log("=== FULL TRAIN START ===")
    _bootstrap()
    log("bootstrap done (clone + pip)")

    import torch
    cfg, n_gpu = _build_cfg()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    log(f"torch {torch.__version__} | CUDA devices: {n_gpu} {names}")
    log(f"use_ddp={cfg.runtime.use_ddp} bs={cfg.runtime.batch_size} "
        f"workers={cfg.runtime.num_workers} epochs={cfg.train.epochs}")

    # --- 完整訓練(DDP T4x2;不帶 max_steps/subset_n → 全資料、全 epoch)---
    from src.train.train_classifier import train_entry
    log("calling train_entry() — full data, full epochs ...")
    train_entry(cfg)                       # 內部 mp.spawn 2 ranks(雙卡 nccl)
    ckpt = Path(cfg.paths.output_dir) / "checkpoints" / cfg.train.ckpt_name
    log(f"training done; checkpoint: {ckpt} (exists={ckpt.exists()})")

    # --- 偽標籤 + 信心直方圖(取樣)---
    from src.pseudo_label.cam_pipeline import generate_pseudo_labels
    log(f"generating pseudo-labels on {PSEUDO_SAMPLE_N} sampled images ...")
    df, csv_path = generate_pseudo_labels(cfg, str(ckpt), limit=PSEUDO_SAMPLE_N,
                                          save_viz=True)
    log(f"pseudo-labels: {len(df)} boxes / {df['Image Index'].nunique()} images -> {csv_path}")
    hist_png = Path(cfg.paths.output_dir) / "pseudo_labels" / "confidence_hist.png"
    _plot_confidence_hist(df, hist_png, cfg.cam.confidence_thresh)

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
