# 實驗追蹤 — NIH ChestX-ray14

> 取代 Notion 的實驗紀錄。每次跑通 / 訓練在此追加一筆。

## EXP-001 — 本機冒煙測試(RTX 3060 Laptop, 6GB)

**日期**:2026-06-20 ｜ **環境**:local / CUDA / PyTorch 2.5.1+cu121

### 設定
- backbone: `densenet121`(pretrained, num_classes=14)
- image_size: 224 ｜ batch_size: 8(smoke)/ 16(預設本機)
- AMP: `torch.amp.autocast('cuda', fp16)` + `GradScaler('cuda')`
- subset_n=300, max_steps=3 ｜ CAM: gradcampp, heatmap_thresh=0.5, confidence_thresh=0.30

### 結果
| 指標 | 值 |
|------|----|
| DataLoader | 4 batches / 13.17s,無阻塞 |
| train loss (step0) | 1.6735 |
| val mAUROC | 0.4552(未訓練,僅 3 step,符合預期) |
| 訓練 peak VRAM | **0.58 GiB** |
| CAM peak VRAM | **1.30 GiB** |
| 偽標籤產出 | 192 boxes / 8 圖 → `artifacts/pseudo_labels/pseudo_labels.csv` |

### 結論
- ✅ 全鏈路通過,VRAM 遠低於 6GB 上限,無 OOM / I/O 阻塞 → 滿足 DoD。
- 修復環境 3 個損壞套件:opencv-python / python-dateutil / huggingface_hub(httpcore)。
- 偽標籤 CSV 欄位:`Image Index, Finding Label, x, y, w, h, confidence, cam_method`。

## EXP-002 —(待辦)Kaggle T4×2 DDP 完整訓練
- 目標:`use_ddp=True` 完整訓練分類器,記錄逐類 AUROC 與 DDP 吞吐。
