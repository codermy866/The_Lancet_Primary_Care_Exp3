# OCT Triaging 模型架构图

下面为本实验实现的模型架构图（Mermaid）。实现细节参见代码：
- [experiments/OCT_traige/models/oct_encoder.py](experiments/OCT_traige/models/oct_encoder.py)
- [experiments/OCT_traige/models/oct_traige_model.py](experiments/OCT_traige/models/oct_traige_model.py)

```mermaid
flowchart TD
  A[OCT Volume\n(B, S, 3, H, W)] --> B[Per-slice Encoder\n(SliceCNN or SliceViT)]
  B --> C[Per-slice Features\n(S*B, slice_feat_dim) -> reshape -> (B, S, slice_feat_dim)]
  C --> D[Slice Projection\n(Linear -> LayerNorm -> Dropout)\n-> embed_dim]
  D --> E[Depth GRU\n(sequence modeling over S slices)]
  E --> F[Depth Attention Pooling\n(weighted sum over slices)]
  F --> G[Depth-resolved Feature\n(embed_dim)]
  G --> H[Dual-Head Encoder\n(feature_proj -> causal & noise heads)]
  H --> H1[z_causal]
  H --> H2[z_noise]
  H1 --> CLF[Classifier\n(z_causal -> logits_pred)]
  H2 --> DISC[Center Discriminator\n(z_noise -> center_logits)]
  H2 --> MEM[Noise Memory Bank\n(update with z_noise per-center)]
  MEM --> CF[z_noise_cf (counterfactual noise)]
  H1 --> COMB[Compose counterfactual:\nz_causal_cf = z_causal + alpha_cf * z_noise_cf]
  CF --> COMB
  COMB --> CLF2[Classifier\n(z_causal_cf -> logits_cf)]

  %% Losses
  DISC --> L_ADV[Adversarial Loss (L_adv)]
  CLF2 --> L_CONS[Counterfactual Consistency Loss (L_consist)]
  H1 & H2 --> L_ORTH[Orthogonal Loss (L_ortho)]

  classDef comp fill:#f9f,stroke:#333,stroke-width:1px;
  class A,B,D,E,F,G,H,MEM,DISC,CLF,CLF2 comp;

```

**图注（Lancet 风格简短描述）**
- 输入：OCT 体积，形状为 (B, S, 3, H, W)，S 为切片数量（默认 20）。
- 深度分辨编码器（DepthResolvedOCTEncoder）：每切片先由 `SliceCNN` 或 `SliceViT` 编码为切片特征，经线性投影到 `embed_dim`，再通过单层 GRU 建模切片序列，最后用注意力池化得到深度融合特征。实现见 [experiments/OCT_traige/models/oct_encoder.py](experiments/OCT_traige/models/oct_encoder.py)。
- 双头表征（Dual-Head）：将深度融合特征投影为 `z_causal`（用于分类）与 `z_noise`（用于跨中心噪声建模）。
- 分类器：仅使用 `z_causal` 预测三分类/二分类标签（默认二分类）。
- 跨中心鲁棒性模块：`z_noise` 输入中心判别器（对抗损失 L_adv）；同时写入 `NoiseMemoryBank`，用于从其他中心采样反事实噪声 `z_noise_cf`，并构造反事实特征 `z_causal_cf = z_causal + alpha_cf * z_noise_cf`，重算预测并施加一致性损失 L_consist。另有正交损失 L_ortho 以解耦 `z_causal` 与 `z_noise`。

**默认超参数（代码中）**
- `embed_dim=256`, `oct_num_slices=20`, `alpha_cf=0.3`, `memory_capacity=100`。

---

如果需要，我可以：
- 导出为高分辨率 SVG/PNG 并保存到 `experiments/OCT_traige/docs/`；
- 按柳叶刀投稿格式微调图形样式（字体、线宽、颜色、矢量导出）。
