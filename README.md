# OCT_traige (OCT-only multi-center training)

这个目录提供一个**仅使用 OCT** 的跨中心训练方案（基于 Hydra 里的“Depth-Resolved OCT Encoder + 双头因果/噪声解耦 + 中心对抗/反事实一致性”的思路做了裁剪）。

你可以将整个 `OCT_traige/` 目录完整拷贝到其他服务器，然后通过设置环境变量切换数据根目录即可运行。

## 代码入口

训练：

```bash
cd experiments/OCT_traige
export OCT_TRAIGE_DATA_ROOT=/path/to/your/data_root
bash run_train.sh
```

也可以直接：

```bash
cd experiments/OCT_traige
python training/train_oct_traige.py
```

## 数据目录与 CSV 格式（必须）

`OCT_TRAIGE_DATA_ROOT` 下应至少包含：

1. `train_labels.csv`
2. `val_labels.csv`
3. OCT 图片目录：
   - `internal_train/train/oct/{oct_id}/*.png|*.jpg`
   - `internal_train/val/oct/{oct_id}/*.png|*.jpg`

`train_labels.csv/val_labels.csv` 需要包含以下列：

- `OCT`：OCT 样本的 `oct_id`（也支持列名为 `oct_id`）
- `label`：二分类标签（0/1）

可选列（如果你提供了，这个实现会优先使用）：

- `oct_paths`：用 `;` 或 `,` 分隔的一组图片路径（会绕过按 `oct_id` glob 图片）
- `center_id`：中心 ID（如果没有提供，会从 `oct_id` 推断；规则与 Hydra 的基类一致）

## 如何实现“用其他中心训练”

这个实现不强制写死 leave-one-center-out：你只要在 `train_labels.csv` 里放入“要用于训练的中心样本”，在 `val_labels.csv` 里放入“要验证的中心样本”（例如留一中心），即可完成“使用其他中心数据来训练”。

## 结果

训练过程中会输出 best `AUC`，并把最优模型保存到：

- `experiments/OCT_traige/checkpoints/best_model.pt`

