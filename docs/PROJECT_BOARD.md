# 專案看板 — NIH ChestX-ray14 自動標註與 DDP 訓練(取代 Notion)

> 即時更新的 5 階段看板。Notion MCP 不在可用工具鏈中,故以此 markdown 追蹤。

| # | 階段 | 狀態 | 產出 |
|---|------|------|------|
| 1 | 研究與 API 確認 | ✅ Done | 弱監督 CAM 範式;timm / grad-cam / torch.amp API 確認 |
| 2 | 專案追蹤初始化 | ✅ Done | repo 結構 + docs 看板 + GitHub |
| 3 | Kaggle 資料與 I/O 優化 | ✅ Done | I/O 優化 DataLoader,本機 RTX 3060 跑通(無阻塞) |
| 4 | 自動標註管線 | ✅ Done | Grad-CAM++ → bbox + 信心篩選 + 視覺化 |
| 5 | DDP + AMP 訓練腳本 | ✅ Done(本機 AMP 驗證) | T4×2 DDP `mp.spawn` 待雲端實測 |

## 關鍵決策

- **資料規模校正**:完整 112,120 張(非「一萬多張」);醫師框僅 984 個 / 880 圖 / 8 類。
- **硬體校正**:本機 RTX 3060 Laptop 為 **6GB**(非 8GB),batch size 保守設 16(smoke 用 8)。
- **策略**:984 框太少,改弱監督 — 影像級標籤訓練分類器 → Grad-CAM++ 轉 bbox → 984 框做 IoU 校準。

## 下一步(Kaggle 雲端)

1. 上傳 dataset + notebook,T4×2 完整訓練分類器(`use_ddp=True`)。
2. 全集 112K 推論生成虛擬標籤,信心分數分佈分析。
3. IoU 校準 `heatmap_thresh` / `confidence_thresh` 對齊 984 醫師框。
4. 以虛擬標籤訓練定位/裁切模型。
