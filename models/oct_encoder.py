from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceCNN(nn.Module):
    """Original lightweight CNN per OCT slice -> vector [B, out_dim]."""

    def __init__(self, in_channels: int = 3, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.flatten(1)


class SliceCustomViTEncoder(nn.Module):
    """
    Small ViT per slice (train from scratch). Input [B, C, H, W] -> [B, out_dim].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 64,
        patch_size: int = 16,
        vit_dim: int = 256,
        vit_depth: int = 4,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        image_size: int = 224,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.vit_dim = int(vit_dim)
        self.image_size = int(image_size)

        self.patch_embed = nn.Conv2d(
            in_channels,
            self.vit_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        num_patches = (self.image_size // self.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.vit_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_dim,
            nhead=vit_heads,
            dim_feedforward=int(self.vit_dim * vit_mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=vit_depth)
        self.norm = nn.LayerNorm(self.vit_dim)
        self.proj = nn.Linear(self.vit_dim, out_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _get_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = x.shape[1] - 1
        n_ref_tokens = self.pos_embed.shape[1] - 1
        if n_tokens == n_ref_tokens:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        ref_hw = int(n_ref_tokens**0.5)
        tgt_hw = int(n_tokens**0.5)
        patch_pos = patch_pos.reshape(1, ref_hw, ref_hw, self.vit_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(tgt_hw, tgt_hw), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, tgt_hw * tgt_hw, self.vit_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._get_pos_embed(x)
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.proj(x)


class SliceTorchVisionViTAdapter(nn.Module):
    """
    torchvision ViT-B/16 backbone (optional ImageNet weights) -> [B, out_dim].
    Exposes `.vit` for per-group optimizer in training when vit_pretrained=True.
    """

    def __init__(self, out_dim: int = 64, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.vit.hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        tok = self.dropout(x[:, 0])
        return self.proj(tok)


class DepthResolvedOCTEncoder(nn.Module):
    """
    Depth-Resolved OCT Encoder (OCT slices -> per-slice encoder -> depth GRU -> attention pooling)

    encoder_type:
      - cnn: lightweight CNN per slice
      - vit: custom small ViT per slice (from scratch) unless vit_pretrained=True
    """

    def __init__(
        self,
        num_slices: int = 20,
        in_channels: int = 3,
        embed_dim: int = 256,
        slice_feat_dim: int = 64,
        dropout: float = 0.1,
        encoder_type: str = "cnn",
        vit_pretrained: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.embed_dim = embed_dim
        et = str(encoder_type).lower().strip()
        self.encoder_type = et
        self.image_size = int(image_size)

        if et == "cnn":
            self.slice_encoder = SliceCNN(in_channels=in_channels, out_dim=slice_feat_dim)
        elif et == "vit":
            if bool(vit_pretrained):
                self.slice_encoder = SliceTorchVisionViTAdapter(
                    out_dim=slice_feat_dim,
                    pretrained=True,
                    dropout=dropout,
                )
            else:
                self.slice_encoder = SliceCustomViTEncoder(
                    in_channels=in_channels,
                    out_dim=slice_feat_dim,
                    patch_size=16,
                    vit_dim=max(embed_dim, 128),
                    vit_depth=4,
                    vit_heads=8,
                    vit_mlp_ratio=4.0,
                    dropout=dropout,
                    image_size=self.image_size,
                )
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}; expected 'cnn' or 'vit'")

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
            oct_volume = oct_volume.unsqueeze(2)

        B, S, C, H, W = oct_volume.shape
        oct_slices = oct_volume.reshape(B * S, C, H, W)
        slice_feats = self.slice_encoder(oct_slices)
        slice_feats = slice_feats.reshape(B, S, -1)
        slice_feats = self.slice_proj(slice_feats)

        gru_out, _ = self.depth_gru(slice_feats)
        attn_scores = self.depth_attention(gru_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        depth_resolved_feat = (gru_out * attn_weights).sum(dim=1)

        return depth_resolved_feat
