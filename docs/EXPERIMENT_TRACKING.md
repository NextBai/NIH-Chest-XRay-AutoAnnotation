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

## EXP-002 — Kaggle T4×2 DDP+AMP 機制驗證(雲端)

**日期**:2026-06-20 ｜ **環境**:Kaggle / 2× Tesla T4 / PyTorch 2.10.0+cu128

### 設定
- 自含式 selftest(`scripts/kaggle_ddp_selftest.py`):無 repo clone、無資料集、無 pip,隔離 DDP+AMP 機制與基礎設施雜訊。
- `WORLD_SIZE=2`、`mp.spawn` 2 ranks、`init_process_group(nccl)`、`DDP(model)` 包裝。
- AMP:`autocast` + `GradScaler`,10 steps,BCEWithLogitsLoss + AdamW。
- 機制檢查:`all_reduce` 跨卡梯度同步 + `barrier` + `destroy_process_group`。

### 結果
| 指標 | 值 |
|------|----|
| CUDA devices | **2 × Tesla T4**(真實雙卡) |
| DDP 後端 | **nccl**(REAL DUAL-GPU,非 gloo 退化) |
| allreduce 檢查 | got 3 expected 3 **(OK)** |
| 訓練 peak VRAM | 0.02 GiB(synthetic tensors) |
| 收尾 | `DDP+AMP worker finished cleanly` → `=== SUCCESS ===` |

### 結論
- ✅ T4×2 DDP+AMP 全機制驗證通過 → 滿足最後 DoD 項目(訓練腳本可在雲端雙卡運行)。
- **關鍵教訓**:Kaggle 機型字串只填 `Tesla T4`,系統自動配 2 顆;勿用 `gpu1xT4x2`(lottery 配到 P100 sm_60,PyTorch 2.10+cu128 最低 sm_70,崩 `cudaErrorNoKernelImageForDevice`)。
- 先前 v6 ERROR 根因即為配到 P100,非程式碼問題。

## EXP-003 —(待辦)Kaggle T4×2 完整訓練分類器
- 目標:`use_ddp=True` 跑足 epoch 完整訓練,記錄逐類 AUROC 與 DDP 吞吐量。
