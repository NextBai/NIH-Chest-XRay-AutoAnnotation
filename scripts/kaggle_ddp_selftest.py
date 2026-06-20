# Kaggle DDP + AMP 自包含驗證腳本 (NIH 專案)
# 目的:在 Kaggle 雲端 GPU 上「純粹」驗證 DDP + AMP 機制是否正確運作,
#       與 src/train/train_classifier.py 完全相同的核心鏈:
#         mp.spawn -> init_process_group -> DDP -> torch.amp.autocast + GradScaler
#         -> loss.backward (梯度 allreduce) -> scaler.step
#
# 設計重點:
# - 零外部依賴:不 clone repo、不掛 45GB 資料集、不 pip install(只用 Kaggle 預裝 torch)。
#   這樣可把「DDP/AMP 程式正確性」與「資料/網路/GPU 配額樂透」等基礎設施雜訊解耦。
# - 合成資料(隨機 tensor)取代影像,專注驗證分散式訓練機制本身。
# - 自動偵測 GPU 數:
#     * n_gpu >= 2 → 真雙卡 DDP(nccl),每 rank 綁一張卡(rank % n_gpu)。
#     * n_gpu == 1 → 仍 spawn 2 個 rank 共用該卡(gloo),驗證多行程 DDP 流程。
# - 全程寫 /kaggle/working/ddp_selftest.log(會 commit 進 output,ERROR 也能取回)。
# - 嚴禁 argparse。
import os
import sys
import traceback
from datetime import datetime

LOG = "/kaggle/working/ddp_selftest.log"
WORLD_SIZE = 2          # 固定 spawn 2 個 rank(雙卡或單卡共用)
STEPS = 10
BATCH = 16
NUM_CLASSES = 14


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _ddp_worker(rank: int, world_size: int, n_gpu: int):
    """每 rank 的訓練流程,鏡像 train_classifier._worker 的 DDP+AMP 核心。"""
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        use_cuda = n_gpu >= 1
        # 雙卡走 nccl;單卡共用走 gloo(nccl 不允許多 rank 綁同一張卡)
        backend = "nccl" if n_gpu >= 2 else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        if n_gpu >= 1:
            device = torch.device("cuda", rank % n_gpu)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        # 簡單 CNN,模擬多標籤分類器輸出
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, NUM_CLASSES),
        ).to(device)
        if n_gpu >= 2:
            model = model.to(memory_format=torch.channels_last)
            ddp_model = DDP(model, device_ids=[rank % n_gpu], output_device=rank % n_gpu)
        else:
            ddp_model = DDP(model)  # 單卡/CPU:不指定 device_ids

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

        peak_mem = 0.0
        for step in range(STEPS):
            imgs = torch.randn(BATCH, 3, 64, 64, device=device)
            if n_gpu >= 2:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = (torch.rand(BATCH, NUM_CLASSES, device=device) > 0.5).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                logits = ddp_model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()   # 觸發梯度 allreduce
            scaler.step(optimizer)
            scaler.update()
            if use_cuda:
                peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            if rank == 0 and step % 5 == 0:
                log(f"  rank{rank} step{step} loss={loss.item():.4f}")

        # 驗證跨 rank 通訊:allreduce 一個 tensor
        t = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if rank == 0:
            expected = sum(range(1, world_size + 1))
            log(f"  allreduce check: got {t.item():.0f} expected {expected} "
                f"({'OK' if int(t.item()) == expected else 'MISMATCH'})")
            log(f"  rank0 peak_mem={peak_mem:.2f} GiB device={device}")
        dist.barrier()
        dist.destroy_process_group()
        if rank == 0:
            log(f"  rank{rank}: DDP+AMP worker finished cleanly")
    except Exception:
        log(f"  rank{rank} EXCEPTION:\n{traceback.format_exc()}")
        raise


def main():
    import torch
    import torch.multiprocessing as mp

    log("=== DDP/AMP SELFTEST START ===")
    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    log(f"torch {torch.__version__} | CUDA devices: {n_gpu} {names}")
    mode = "REAL DUAL-GPU DDP (nccl)" if n_gpu >= 2 else \
           ("single-GPU 2-rank DDP (gloo)" if n_gpu == 1 else "CPU 2-rank DDP (gloo)")
    log(f"world_size={WORLD_SIZE} mode={mode}")

    mp.spawn(_ddp_worker, args=(WORLD_SIZE, n_gpu), nprocs=WORLD_SIZE, join=True)
    log(f"VALIDATION DONE: spawned {WORLD_SIZE} ranks, DDP+AMP exercised on {n_gpu} GPU(s)")
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
