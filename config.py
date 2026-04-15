"""
OCT_traige 配置

目标：仅使用 OCT（不使用阴道镜/临床/VLM），训练一个可跨中心泛化的 OCT 分类模型。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_data_root() -> str:
    # 允许在新服务器上不改代码切换数据路径
    return os.environ.get(
        "OCT_TRAIGE_DATA_ROOT",
        "/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out",
    )


@dataclass
class OCTTraigeConfig:
    # 数据路径
    data_root: str = ""
    train_csv_name: str = "train_labels.csv"
    val_csv_name: str = "val_labels.csv"

    # OCT 输入
    oct_frames: int = 20
    img_size: int = 224

    # 模型
    embed_dim: int = 256
    dropout: float = 0.3

    # 训练
    batch_size: int = 4
    num_epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    num_workers: int = 4
    pin_memory: bool = True

    # 交叉中心相关损失（基于 Hydra 的 dual-head/center discriminator/反事实一致性裁剪）
    lambda_cls: float = 1.0
    lambda_adv: float = 0.5
    lambda_ortho: float = 0.5
    lambda_consist: float = 0.2
    alpha_cf: float = 0.3  # z_noise_cf 注入到因果表示的系数

    # Memory Bank
    memory_capacity: int = 100

    # 输出
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self):
        if not self.data_root:
            self.data_root = _default_data_root()
        for d in [self.checkpoint_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

