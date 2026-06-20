"""DDP + AMP multi-label classifier training for NIH ChestX-ray14.

Design:
- No argparse. Everything is driven by configs/config.py (Config dataclass).
- Single public entry: train_entry(cfg, max_steps=None, subset_n=None).
    * Kaggle T4x2 (cfg.runtime.use_ddp=True): spawns 2 processes via mp.spawn.
      (DDP defaults to 2 GPUs — no GPU_Num needed, per project convention.)
    * Local RTX 3060: single-GPU path, no process group.
- Mixed precision via torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda')
  (PyTorch 2.5 API; torch.cuda.amp.* is deprecated).
- max_steps/subset_n let the local smoke test run a tiny, fast pass.

Imported by scripts/smoke_test_local.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# Make `configs` / `src` importable when launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.config import Config  # noqa: E402
from src.data.dataset import (  # noqa: E402
    NIHChestDataset, build_image_index, load_dataframe, patient_wise_split,
    make_dataloader, compute_pos_weight,
)
from src.models.classifier import build_classifier, MultiLabelLoss  # noqa: E402
from src.utils.metrics import multilabel_auroc  # noqa: E402
from src.utils.seed import set_seed, seed_worker  # noqa: E402


def _is_main(rank: int) -> bool:
    return rank == 0


def _log(rank: int, msg: str) -> None:
    if _is_main(rank):
        print(msg, flush=True)
def prepare_dataframes(cfg: Config, subset_n: int | None = None):
    """Build train/val dataframes from CSV + on-disk image index."""
    index = build_image_index(cfg.paths.data_dir)
    df = load_dataframe(cfg.paths.data_entry_csv, index)
    if subset_n is not None:
        df = df.sample(n=min(subset_n, len(df)), random_state=cfg.data.seed)
        df = df.reset_index(drop=True)
    train_df, val_df = patient_wise_split(df, cfg.data.val_split, cfg.data.seed)
    return train_df, val_df


def _build_loaders(cfg: Config, train_df, val_df, rank: int, world_size: int):
    train_ds = NIHChestDataset(train_df, cfg.data.image_size, cfg.data.mean,
                               cfg.data.std, train=True, in_chans=cfg.data.in_chans)
    val_ds = NIHChestDataset(val_df, cfg.data.image_size, cfg.data.mean,
                             cfg.data.std, train=False, in_chans=cfg.data.in_chans)
    train_sampler = (DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                        shuffle=True) if world_size > 1 else None)
    train_loader = make_dataloader(
        train_ds, cfg.runtime.batch_size, cfg.runtime.num_workers,
        shuffle=(train_sampler is None), pin_memory=cfg.runtime.pin_memory,
        persistent_workers=cfg.runtime.persistent_workers,
        prefetch_factor=cfg.runtime.prefetch_factor, sampler=train_sampler,
    )
    val_loader = make_dataloader(
        val_ds, cfg.runtime.batch_size, cfg.runtime.num_workers, shuffle=False,
        pin_memory=cfg.runtime.pin_memory,
        persistent_workers=cfg.runtime.persistent_workers,
        prefetch_factor=cfg.runtime.prefetch_factor,
    )
    return train_ds, train_loader, val_loader, train_sampler
@torch.no_grad()
def _evaluate(model, loader, device, max_batches: int | None = None):
    """Return mean AUROC over the validation loader."""
    model.eval()
    ys, ps = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(labels.numpy())
    if not ys:
        return float("nan")
    return multilabel_auroc(np.concatenate(ys), np.concatenate(ps))["mean"]


def _worker(rank: int, world_size: int, cfg: Config,
            max_steps: int | None, subset_n: int | None):
    """Per-process training routine (rank==0 only for single-GPU)."""
    is_ddp = world_size > 1
    if is_ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "23456")
        dist.init_process_group(cfg.runtime.dist_backend, rank=rank,
                                world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    set_seed(cfg.data.seed + rank)

    train_df, val_df = prepare_dataframes(cfg, subset_n=subset_n)
    pos_weight = compute_pos_weight(train_df).to(device) if cfg.train.use_pos_weight else None
    _, train_loader, val_loader, train_sampler = _build_loaders(
        cfg, train_df, val_df, rank, world_size)

    model = build_classifier(cfg.model.backbone, 14, cfg.model.pretrained,
                             cfg.model.drop_rate).to(device)
    if cfg.runtime.env == "kaggle":
        model = model.to(memory_format=torch.channels_last)
    if is_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = MultiLabelLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.base_lr,
                                  weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp)
    global_step = 0
    stop = False
    for epoch in range(cfg.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if cfg.runtime.env == "kaggle":
                imgs = imgs.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.train.use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if global_step % cfg.train.log_interval == 0:
                _log(rank, f"[epoch {epoch} step {global_step}] loss={loss.item():.4f}")
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        if _is_main(rank):
            auroc = _evaluate(model, val_loader, device,
                              max_batches=(2 if max_steps else None))
            _log(rank, f"[epoch {epoch}] val_mAUROC={auroc:.4f}")
            ckpt_dir = Path(cfg.paths.output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            state = (model.module if is_ddp else model).state_dict()
            torch.save({"model": state, "cfg": cfg.to_dict()},
                       ckpt_dir / cfg.train.ckpt_name)
        if stop:
            break
    if is_ddp:
        dist.destroy_process_group()


def train_entry(cfg: Config | None = None, max_steps: int | None = None,
                subset_n: int | None = None) -> None:
    """Public entry. Spawns DDP procs on Kaggle T4x2, else single-GPU.

    No argparse — call this from a notebook/script with a Config instance.
    """
    cfg = cfg or Config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training.")
    world_size = torch.cuda.device_count() if cfg.runtime.use_ddp else 1
    if world_size > 1:
        mp.spawn(_worker, args=(world_size, cfg, max_steps, subset_n),
                 nprocs=world_size, join=True)
    else:
        _worker(0, 1, cfg, max_steps, subset_n)


if __name__ == "__main__":
    train_entry()

