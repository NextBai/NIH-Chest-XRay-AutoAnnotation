# NIH ChestX-ray14 — 弱監督自動標註 + DDP/AMP 訓練

利用 **影像級標籤**(11.2 萬張)訓練多標籤分類器,透過 **Grad-CAM++** 生成疾病邊界框虛擬標籤(Pseudo-labeling),並以 **984 個醫師標註框** 做 IoU 校準。支援 **Kaggle T4×2 DDP + AMP** 大規模訓練。

## 策略概要

ChestX-ray14 僅 984 個醫師框(0.88%),不足以直接訓練偵測器。採弱監督範式:

```
影像級標籤 (112,120) ──► timm DenseNet121 多標籤分類器
                              │
                              ▼ Grad-CAM++ 熱圖
                       閾值化 → OpenCV 輪廓 → bbox
                              │
                              ▼ confidence = class_prob × heatmap_peak
                       信心分數篩選 → 虛擬標籤 CSV + 視覺化
                              │
                              ▼ IoU 校準 (984 醫師框)
                          後續定位/裁切訓練
```

## 專案結構

```
configs/config.py              # 集中 dataclass 設定(無 argparse),環境自動偵測
src/data/dataset.py            # NIH 多標籤 Dataset + I/O 優化 DataLoader + patient-wise split
src/models/classifier.py       # timm DenseNet121 多標籤 + CAM target layer
src/train/train_classifier.py  # DDP(mp.spawn)+ AMP(torch.amp)訓練,無 argparse
src/pseudo_label/cam_pipeline.py  # Grad-CAM++ → bbox → 信心篩選 → CSV + viz
src/utils/{metrics,seed}.py    # AUROC / bbox IoU / 隨機種子
scripts/eda_check.py           # 資料層驗證
scripts/smoke_test_local.py    # RTX 3060 單卡小批次跑通
docs/                          # 實驗追蹤 + 看板(取代 Notion)
```

## 環境

- 本機 RTX 3060 Laptop(**6GB** VRAM):前處理 / EDA / 小樣本除錯。
- Kaggle T4×2:大規模推論標註 + DDP 訓練(`use_ddp=True` 自動 `mp.spawn` 2 進程)。
- PyTorch 2.5.1+cu121,timm 1.0.27,grad-cam 1.5.5。

## 使用(無命令列參數)

```python
from configs.config import Config
from src.train.train_classifier import train_entry
from src.pseudo_label.cam_pipeline import generate_pseudo_labels

cfg = Config()                       # 環境自動偵測(local / kaggle)
train_entry(cfg)                     # 本機單卡 or Kaggle T4×2 DDP
generate_pseudo_labels(cfg, "artifacts/checkpoints/classifier_best.pt")
```

本機冒煙測試:`python scripts/smoke_test_local.py`

## 驗收狀態

- ✅ 本機 RTX 3060 跑通:DataLoader 無阻塞,AMP 訓練 peak 0.58 GiB,CAM peak 1.30 GiB。
- ✅ 自動標註含信心分數篩選 + 視覺化 PNG。
- ✅ DDP + AMP 訓練腳本就緒(本機驗 AMP;DDP 待 Kaggle T4×2 實測)。
