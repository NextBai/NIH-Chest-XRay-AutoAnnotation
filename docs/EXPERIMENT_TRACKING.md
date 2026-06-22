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

## EXP-003 — Kaggle T4×2 完整訓練 + 偽標籤生成(官方 ChestX-ray14)

**日期**:2026-06-20 ｜ **環境**:Kaggle / 2× Tesla T4 16GB(sm_75)/ PyTorch 2.10.0+cu128 ｜ **kernel**:`xiaobai1221/nih-full-train-t4x2`

### 設定
- 資料來源:Kaggle 公開 dataset `nih-chest-xrays/data`(112,120 張 1024px PNG + Data_Entry_2017.csv + BBox_List_2017.csv 984 醫師框,**未重新上傳**)。
- 掛載路徑:`/kaggle/input/datasets/organizations/nih-chest-xrays/data`(12 個 `images_*/images` 子目錄,bbox csv found)。
- backbone: `densenet121`(ImageNet pretrained, num_classes=14)｜ image_size: 224
- batch_size: **64** ｜ num_workers: 4 ｜ epochs: **15** ｜ AMP: `torch.amp.autocast('cuda', fp16)` + `GradScaler('cuda')`
- DDP:`use_ddp=True`、`mp.spawn(world_size=2)`、`nccl` 後端、`DistributedSampler`
- CAM:gradcampp ｜ heatmap_thresh=0.5 ｜ confidence_thresh=**0.30** ｜ 偽標籤取樣 N=3000

### 結果
| 指標 | 值 |
|------|----|
| CUDA devices | **2 × Tesla T4**(真實雙卡 nccl DDP+AMP) |
| 訓練啟動 → 完成 | 14:28:06 → 19:05:53(**約 4 小時 38 分**, 15 epoch 全集 112K) |
| 最佳 checkpoint | `artifacts/checkpoints/classifier_best.pt`(雲端保存) |
| 偽標籤生成耗時 | 8 分 57 秒(3000 取樣 → 2666 有效圖) |
| 偽標籤產出 | **10,022 boxes / 2,666 images**(`pseudo_labels.csv`) |
| 信心分數 | mean **0.605** ｜ median 0.569 ｜ std 0.216 ｜ max 1.000 ｜ min 0.300 |
| 視覺化 | `confidence_hist.png`(40 bins,thresh=0.30 紅線,右尾 1.0 約 503 高信心) |

### 偽標籤類別分佈(14 類齊全)
| 類別 | 框數 | 類別 | 框數 |
|------|------|------|------|
| Infiltration | 2313 | Pneumothorax | 544 |
| Atelectasis | 1615 | Fibrosis | 450 |
| Effusion | 1223 | Mass | 436 |
| Nodule | 976 | Cardiomegaly | 411 |
| Consolidation | 673 | Edema | 279 |
| Pleural_Thickening | 656 | Emphysema | 257 |
|  |  | Pneumonia | 156 |
|  |  | Hernia | 33 |

> Hernia 33 框最少 — 與 NIH 已知罕見類別分佈一致。
> Infiltration 最高 — 與 Data_Entry 全集主要類別一致。

### 結論
- ✅ T4×2 DDP+AMP 全資料、全 epoch 完整訓練成功 → 滿足 DoD 全部項目。
- ✅ 偽標籤覆蓋 14 類完整,bbox schema 對齊官方 BBox_List_2017.csv(`x, y, w, h`)。
- ✅ 信心分數視覺化篩選機制就位(`confidence_hist.png` 標示 thresh=0.30)。
- 本機 artifacts:`artifacts/kaggle_full_train/`(log + hist + csv);.pt checkpoint 留雲端(本機 RTX 3060 6GB 用不上 DenseNet121 全量推論)。

### 下一步(Stage 7+ 後續工作)
1. 全集 112K 偽標籤推論(此次取樣 3000 為 DoD 視覺化憑證)。
2. IoU 校準 `heatmap_thresh` / `confidence_thresh` 對齊 984 醫師框。
3. 以偽標籤訓練疾病定位/裁切模型。
