"""Local RTX 3060 smoke test (Stage 3/5 DoD).

No argparse. Run:  python scripts/smoke_test_local.py

Validates end-to-end on a tiny subset WITHOUT VRAM OOM or I/O blocking:
  1. DataLoader iterates a few I/O-optimized batches.
  2. train_entry runs a few AMP steps single-GPU and saves a checkpoint.
  3. CAM pipeline turns the checkpoint into confidence-filtered bbox pseudo-labels.

Prints peak VRAM so we can confirm headroom on the 6GB card.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import Config  # noqa: E402
from src.train.train_classifier import train_entry, prepare_dataframes  # noqa: E402
from src.data.dataset import NIHChestDataset, make_dataloader  # noqa: E402
from src.pseudo_label.cam_pipeline import generate_pseudo_labels  # noqa: E402


def _vram(tag: str) -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[vram] {tag}: peak={peak:.2f} GiB", flush=True)


def main() -> None:
    cfg = Config()
    # keep it tiny and fast for the local card
    cfg.runtime.batch_size = 8
    cfg.runtime.num_workers = 2
    cfg.train.epochs = 1
    cfg.cam.method = "gradcampp"
    print(f"== env={cfg.runtime.env} device={cfg.runtime.device} "
          f"bs={cfg.runtime.batch_size} ==", flush=True)

    # 1) DataLoader I/O check
    train_df, _ = prepare_dataframes(cfg, subset_n=200)
    ds = NIHChestDataset(train_df, cfg.data.image_size, cfg.data.mean,
                         cfg.data.std, train=True, in_chans=cfg.data.in_chans)
    loader = make_dataloader(ds, cfg.runtime.batch_size, cfg.runtime.num_workers,
                             shuffle=True, pin_memory=cfg.runtime.pin_memory,
                             persistent_workers=False, prefetch_factor=2)
    t0 = time.time()
    for i, (imgs, labels, names) in enumerate(loader):
        if i == 0:
            print(f"[io] batch shape={tuple(imgs.shape)} labels={tuple(labels.shape)}",
                  flush=True)
        if i >= 3:
            break
    print(f"[io] 4 batches in {time.time()-t0:.2f}s (no blocking)", flush=True)

    # 2) AMP training smoke (single GPU, few steps)
    cfg.runtime.use_ddp = False
    train_entry(cfg, max_steps=3, subset_n=300)
    _vram("after train")

    # 3) CAM pseudo-label smoke + confidence filter + viz
    ckpt = Path(cfg.paths.output_dir) / "checkpoints" / cfg.train.ckpt_name
    cfg.cam.confidence_thresh = 0.10  # low thresh so the smoke run yields boxes
    df, csv_path = generate_pseudo_labels(cfg, str(ckpt), limit=8, save_viz=True)
    _vram("after cam")
    print(f"[cam] rows={len(df)} csv={csv_path}", flush=True)
    print("SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
