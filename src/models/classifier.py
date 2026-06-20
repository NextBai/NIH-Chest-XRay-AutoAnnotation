"""timm-based multi-label classifier for NIH ChestX-ray14.

Imported by src/train/train_classifier.py and src/pseudo_label/cam_pipeline.py.
No argparse — all behaviour is driven by configs/config.py.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_classifier(
    model_name: str = "densenet121",
    num_classes: int = 14,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Create a timm backbone with a multi-label head (num_classes logits)."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def get_cam_target_layer(model: nn.Module, model_name: str = "densenet121") -> list:
    """Return the last spatial conv layer for Grad-CAM (4D feature map)."""
    if "densenet" in model_name:
        return [model.features.norm5]
    if "resnet" in model_name or "resnext" in model_name:
        return [model.layer4[-1]]
    if "convnext" in model_name:
        return [model.stages[-1]]
    if "efficientnet" in model_name:
        return [model.conv_head]
    raise ValueError(f"No known CAM target layer for model '{model_name}'")


class MultiLabelLoss(nn.Module):
    """BCEWithLogits with optional per-class positive weighting for imbalance."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets)
