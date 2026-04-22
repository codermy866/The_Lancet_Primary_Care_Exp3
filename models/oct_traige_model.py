from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .oct_encoder import DepthResolvedOCTEncoder


class DualHeadImageEncoder(nn.Module):
    """
    Dual-head:
      - z_causal: 用于分类
      - z_noise: 用于跨中心噪声建模（用于中心对抗 + 反事实一致性）
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.causal_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        self.noise_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, image_features: torch.Tensor):
        proj_feat = self.feature_proj(image_features)
        z_causal = self.causal_head(proj_feat)
        z_noise = self.noise_head(proj_feat)
        return z_causal, z_noise


class NoiseMemoryBank(nn.Module):
    """
    维护每个中心的噪声特征库，用于反事实一致性。
    """

    def __init__(self, num_centers: int, feat_dim: int, capacity: int = 100):
        super().__init__()
        self.num_centers = int(num_centers)
        self.feat_dim = int(feat_dim)
        self.capacity = int(capacity)

        self.register_buffer("bank", torch.randn(self.num_centers, self.capacity, self.feat_dim))
        self.register_buffer("ptr", torch.zeros(self.num_centers, dtype=torch.long))
        self.register_buffer("count", torch.zeros(self.num_centers, dtype=torch.long))

    @torch.no_grad()
    def update(self, z_noise: torch.Tensor, center_ids: torch.Tensor):
        batch_centers = torch.unique(center_ids)
        for c in batch_centers:
            c = int(c.item())
            mask = center_ids == c
            feats = z_noise[mask].detach()
            if feats.numel() == 0:
                continue

            curr_ptr = int(self.ptr[c].item())
            curr_count = int(self.count[c].item())

            if curr_count + feats.shape[0] <= self.capacity:
                end_ptr = curr_ptr + feats.shape[0]
                self.bank[c, curr_ptr:end_ptr] = feats
                self.ptr[c] = (curr_ptr + feats.shape[0]) % self.capacity
                self.count[c] = min(curr_count + feats.shape[0], self.capacity)
            else:
                remaining = self.capacity - curr_ptr
                if remaining > 0:
                    take = min(remaining, feats.shape[0])
                    self.bank[c, curr_ptr : curr_ptr + take] = feats[:take]
                    feats = feats[take:]

                if len(feats) > 0:
                    fill_len = min(len(feats), self.capacity)
                    self.bank[c, :fill_len] = feats[:fill_len]
                    self.ptr[c] = fill_len
                else:
                    self.ptr[c] = 0

                self.count[c] = self.capacity

    def get_counterfactual_noise(self, target_center_ids: torch.Tensor, strategy: str = "random"):
        """
        返回每个样本从 target_center_ids 指定中心采样的反事实噪声。
        """
        B = int(target_center_ids.shape[0])
        device = target_center_ids.device
        out = []
        for i in range(B):
            c = int(target_center_ids[i].item())
            count = int(self.count[c].item())
            if count == 0:
                out.append(torch.randn(self.feat_dim, device=device))
                continue

            if strategy == "mean":
                out.append(self.bank[c, :count].mean(dim=0).to(device))
            else:
                rand_idx = torch.randint(0, count, (1,), device=self.bank.device).item()
                out.append(self.bank[c, rand_idx].to(device))

        return torch.stack(out)  # [B, feat_dim]


class CenterDiscriminator(nn.Module):
    """
    从 z_noise 预测中心ID（用于对抗训练）。
    """

    def __init__(self, feat_dim: int, num_centers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_centers),
        )

    def forward(self, z_noise: torch.Tensor):
        return self.net(z_noise)


class AdversarialLoss(nn.Module):
    """
    Hydra 的裁剪版：中心判别器输出维度为 2 或 num_centers<=2 时用熵/不确定性鼓励。
    """

    def __init__(self, num_centers: int, reduction: str = "mean"):
        super().__init__()
        self.num_centers = int(num_centers)
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, center_logits: torch.Tensor, center_labels: torch.Tensor):
        num_classes = center_logits.size(1)
        if num_classes == 2 or self.num_centers <= 2:
            probs = torch.softmax(center_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            max_entropy = torch.log(
                torch.tensor(num_classes, dtype=torch.float32, device=center_logits.device)
            )
            loss = torch.clamp(max_entropy - entropy.mean(), min=0.0)
            return loss
        return self.criterion(center_logits, center_labels)


class OrthogonalLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, z_causal: torch.Tensor, z_noise: torch.Tensor):
        inner_product = (z_causal * z_noise).sum(dim=-1).abs()
        return inner_product.mean() if self.reduction == "mean" else inner_product.sum()


class CounterfactualConsistencyLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits_orig: torch.Tensor, logits_cf: torch.Tensor):
        return F.mse_loss(logits_orig, logits_cf, reduction=self.reduction)


class OCTTraigeModel(nn.Module):
    """
    OCT-only triage model.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 2,
        oct_num_slices: int = 20,
        dropout: float = 0.3,
        num_centers: int = 5,
        memory_capacity: int = 100,
        alpha_cf: float = 0.3,
        encoder_type: str = "cnn",
        vit_pretrained: bool = False,
        img_size: int = 224,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.num_centers = int(num_centers)
        self.alpha_cf = float(alpha_cf)

        self.oct_encoder = DepthResolvedOCTEncoder(
            num_slices=oct_num_slices,
            in_channels=3,
            embed_dim=embed_dim,
            slice_feat_dim=64,
            dropout=dropout,
            encoder_type=encoder_type,
            vit_pretrained=vit_pretrained,
            image_size=int(img_size),
        )

        self.dual_head = DualHeadImageEncoder(input_dim=embed_dim, embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # 跨中心训练组件
        self.center_discriminator = CenterDiscriminator(feat_dim=embed_dim, num_centers=max(num_centers, 2))
        self.adversarial_loss = AdversarialLoss(num_centers=max(num_centers, 2))
        self.orthogonal_loss = OrthogonalLoss()
        self.consistency_loss = CounterfactualConsistencyLoss()

        self.memory_bank = NoiseMemoryBank(
            num_centers=max(num_centers, 2),
            feat_dim=embed_dim,
            capacity=memory_capacity,
        )

    def forward(
        self,
        oct_images: torch.Tensor,  # [B, S, 3, H, W]
        center_labels=None,
        return_loss_components: bool = False,
    ):
        B = int(oct_images.shape[0])
        f_oct = self.oct_encoder(oct_images)  # [B, embed_dim]
        z_causal, z_noise = self.dual_head(f_oct)

        logits = self.classifier(z_causal)

        output = {
            "pred": logits,
            "logits": logits,
            "z_causal": z_causal,
            "z_noise": z_noise,
        }

        if not return_loss_components:
            return output

        loss_dict: dict[str, torch.Tensor] = {}

        if center_labels is not None:
            # 1) 对抗损失：让 z_noise 的中心可被判别器预测（被模型“反向”时体现域消除）
            center_logits = self.center_discriminator(z_noise)
            loss_dict["L_adv"] = self.adversarial_loss(center_logits, center_labels)

            # 2) 反事实一致性：用其它中心的噪声替换，预测应尽量不变
            self.memory_bank.update(z_noise, center_labels)

            # 构造 target_center：随机 offset，保证尽量换中心
            if self.num_centers >= 2:
                rand = torch.randint(
                    0, max(self.num_centers - 1, 1), (B,), device=center_labels.device
                )
                target_centers = (center_labels + 1 + rand) % self.num_centers
            else:
                target_centers = center_labels

            z_noise_cf = self.memory_bank.get_counterfactual_noise(target_centers, strategy="random")
            z_causal_cf = z_causal + self.alpha_cf * z_noise_cf
            logits_cf = self.classifier(z_causal_cf)
            loss_dict["L_consist"] = self.consistency_loss(logits, logits_cf)

        # 3) 正交损失：鼓励因果/噪声表征解耦
        loss_dict["L_ortho"] = self.orthogonal_loss(z_causal, z_noise)

        output["loss_components"] = loss_dict
        return output

