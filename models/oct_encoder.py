from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthResolvedOCTEncoder(nn.Module):
    """
    Depth-Resolved OCT Encoder (OCT slices -> depth GRU -> attention pooling)

    输入:
      oct_volume: [B, S, C, H, W]

    输出:
      depth_resolved_feat: [B, embed_dim]
    """

    def __init__(
        self,
        num_slices: int = 20,
        in_channels: int = 3,
        embed_dim: int = 256,
        slice_feat_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.embed_dim = embed_dim

        self.slice_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, slice_feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(slice_feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.slice_proj = nn.Sequential(
            nn.Linear(slice_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self.depth_gru = nn.GRU(
            embed_dim,
            embed_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.depth_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, oct_volume: torch.Tensor) -> torch.Tensor:
        if oct_volume.dim() == 4:
            # 兼容：如果意外传成 [B, S, H, W] 之类，这里不做复杂推断
            oct_volume = oct_volume.unsqueeze(2)

        B, S, C, H, W = oct_volume.shape
        oct_slices = oct_volume.view(B * S, C, H, W)
        slice_feats = self.slice_cnn(oct_slices)  # [B*S, slice_feat_dim, 1, 1]
        slice_feats = slice_feats.view(B, S, -1)
        slice_feats = self.slice_proj(slice_feats)  # [B, S, embed_dim]

        gru_out, _ = self.depth_gru(slice_feats)  # [B, S, embed_dim]
        attn_scores = self.depth_attention(gru_out)  # [B, S, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, S, 1]
        depth_resolved_feat = (gru_out * attn_weights).sum(dim=1)  # [B, embed_dim]

        return depth_resolved_feat

