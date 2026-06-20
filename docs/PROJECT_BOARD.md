# 專案看板 — NIH ChestX-ray14 自動標註與 DDP 訓練

> 即時更新的 5 階段看板。同步至 GitHub 與 Notion。

| # | 階段 | 狀態 | 產出 |
|---|------|------|------|
| 1 | 研究與 API 確認 | ✅ Done | 弱監督 CAM 範式;timm / grad-cam / torch.amp API 確認 |
| 2 | 專案追蹤初始化 | ✅ Done | repo 結構 + docs 看板 + GitHub + Notion |
| 3 | Kaggle 資料與 I/O 優化 | ✅ Done | I/O 優化 DataLoader,本機 RTX 3060 跑通(無阻塞) |
| 4 | 自動標註管線 | ✅ Done | Grad-CAM++ → bbox + 信心篩選 + 視覺化 |
| 5 | DDP + AMP 訓練腳本 | ✅ Done(雲端實測通過) | **Kaggle Tesla T4×2 真實雙卡 nccl DDP+AMP 驗證成功**(2026-06-20) |

## 關鍵決策

- **資料規模校正**:完整 112,120 張(非「一萬多張」);醫師框僅 984 個 / 880 圖 / 8 類。
- **硬體校正**:本機 RTX 3060 Laptop 為 **6GB**(非 8GB),batch size 保守設 16(smoke 用 8)。
- **策略**:984 框太少,改弱監督 — 影像級標籤訓練分類器 → Grad-CAM++ 轉 bbox → 984 框做 IoU 校準。

## 下一步(Kaggle 雲端)

1. ✅ T4×2 DDP+AMP 機制驗證通過(EXP-002,nccl 雙卡 allreduce OK)。
2. 上傳 dataset + notebook,T4×2 **完整訓練**分類器(`use_ddp=True`,跑足 epoch)。
3. 全集 112K 推論生成虛擬標籤,信心分數分佈分析。
4. IoU 校準 `heatmap_thresh` / `confidence_thresh` 對齊 984 醫師框。
5. 以虛擬標籤訓練定位/裁切模型。

## Kaggle GPU 機型備註

要取得 **T4×2 雙卡**,Kaggle 機型只需指定 **`Tesla T4`**,系統自動配 2 顆。
切勿用 `gpu1xT4x2` 等字串 — 會觸發 lottery 配到 **P100(sm_60)**,而 Kaggle PyTorch 2.10+cu128 最低支援 sm_70,會在 `DDP(model)` 廣播階段崩 `cudaErrorNoKernelImageForDevice`。
