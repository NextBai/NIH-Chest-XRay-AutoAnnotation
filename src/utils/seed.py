"""Reproducibility helpers. Imported by train_classifier.py and smoke_test_local.py."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fix RNG across random / numpy / torch (CPU+CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True lets cuDNN pick fastest kernels for fixed input sizes.
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible multi-worker shuffling."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
