# OCT Triage 外部验证优化记录（2026-05-06）

## 结论摘要

本轮自动优化没有在严格、可发表的外部验证流程下把 AUC 正当提升到 0.80 以上。

当前最强的“未使用外部标签调参”的结果为多模型集成后按外部中心做无监督 score z-score 归一化：

- 外部验证集 AUC: 0.7751
- 外部验证集 PR-AUC: 0.6673
- 结果文件: `logs/final_external_ensemble_20260506/external_ensemble_all_summary.json`
- 预测文件: `logs/final_external_ensemble_20260506/external_predictions_combined_all.csv`

需要注意：按外部中心做无监督归一化没有使用外部标签，但使用了外部验证队列的中心分布信息。若用于投稿主结果，建议在方法中预先定义为部署时固定的数据标准化流程，或在完全独立的新增外部队列上再次锁定验证。

最强的严格非传导式简单集成结果为：

- 外部验证集 AUC: 0.7058
- 外部验证集 PR-AUC: 0.6269
- 结果文件: `logs/final_external_ensemble_20260506/external_ensemble_summary.json`

最佳单模型结果为 ResNet50 frozen feature + OCT + 阴道镜图像 + 临床变量的 Ridge 分类器：

- 外部验证集 AUC: 0.6982
- 外部验证集 PR-AUC: 0.5355
- 结果文件: `logs/frozen_feature_baseline_20260506/frozen_feature_results.json`

## 不能作为论文主结果的诊断性结果

我也做了一个仅用于诊断上限的外部标签子集选择实验。它在外部验证集标签上筛选模型组合，最高达到：

- 外部验证集 AUC: 0.7990
- 外部验证集 PR-AUC: 0.6833
- 组合: ViT boost + EfficientNet-B0 OCT/col/clinical + ResNet50 OCT/col/clinical + ConvNeXt OCT-only + ConvNeXt OCT/clinical

这个结果非常接近 0.80，但因为组合选择使用了外部验证集标签，不能作为 Lancet 级别投稿的有效外部验证主结果，也不应写成最终模型性能。

## 已做的代码优化

1. 修复 OCT 帧排序。
   - 文件: `data/dataset_oct_only.py`
   - 增加自然排序，避免 `C10` 被排在 `C2` 前面，减少帧采样顺序噪声。

2. 提升训练脚本可复现性和可调性。
   - 文件: `training/train_oct_traige.py`
   - 增加 `--seed`, `--batch_size`, `--num_workers`, `--oct_frames`, `--img_size` 参数。
   - 增加随机种子设置。

3. 评估脚本读取 checkpoint 配置。
   - 文件: `training/eval_external_oct_traige.py`
   - 文件: `training/eval_internal_oct_traige.py`
   - 外部/内部评估现在会从 checkpoint config 中读取 `oct_frames`, `img_size`, `embed_dim`, `dropout`, `alpha_cf`。

4. 修复训练 shell 的 Python 环境。
   - 文件: `run_train.sh`
   - 默认使用 `/data2/xh/xh/bin/python3.10`，该环境包含 torch/sklearn。

5. 新增 frozen feature baseline。
   - 文件: `scripts/frozen_feature_baseline.py`
   - 支持 EfficientNet-B0、ResNet50、ConvNeXt-Tiny 的 frozen ImageNet 特征。
   - 支持 OCT、阴道镜图像、临床变量组合。
   - 支持内部验证集选型，并输出外部验证预测 CSV 和 JSON 指标。

## 已运行的主要实验

### EfficientNet-B0 / ResNet50 frozen feature

输出目录: `logs/frozen_feature_baseline_20260506`

关键结果：

- EfficientNet-B0 OCT-only: AUC 0.6254
- EfficientNet-B0 OCT + colposcopy: AUC 0.6691
- EfficientNet-B0 OCT + colposcopy + clinical: AUC 0.6729
- ResNet50 OCT-only: AUC 0.6679
- ResNet50 OCT + colposcopy: AUC 0.6926
- ResNet50 OCT + colposcopy + clinical: AUC 0.6982
- Selected frozen ensemble: AUC 0.6972

### ConvNeXt-Tiny frozen feature

输出目录: `logs/frozen_feature_convnext_20260506`

关键结果：

- ConvNeXt-Tiny OCT-only: AUC 0.6759
- ConvNeXt-Tiny OCT + clinical: AUC 0.6764
- ConvNeXt-Tiny OCT + colposcopy + clinical: AUC 0.6473
- Selected frozen ensemble: AUC 0.6576

### 多模型外部集成

输出目录: `logs/final_external_ensemble_20260506`

已有深度模型和 frozen feature 模型集成后：

- Raw mean ensemble: AUC 0.7058, PR-AUC 0.6269
- Global z-score ensemble: AUC 0.7019, PR-AUC 0.6331
- Center-wise unsupervised z-score ensemble: AUC 0.7751, PR-AUC 0.6673

## 当前瓶颈

外部验证集总样本量为 148。已有模型在部分中心表现较好，但在 `22104` 中心明显下降，整体 AUC 主要受这个中心的域偏移影响。单纯调阈值不能提升 AUC；AUC 的提升需要改变排序质量，例如更强的跨中心表征、中心标准化、质量控制或新增训练数据。

## 建议的下一步

1. 将中心归一化作为预定义部署流程，在新的独立外部队列上锁定验证。
2. 对 `22104` 中心做图像质量、设备、扫描协议、标注一致性和病例构成分析。
3. 增加不使用外部验证标签的跨中心 domain generalization 训练，例如 leave-one-center-out 内部循环选型。
4. 若目标是 Lancet 级别投稿，建议把当前 0.7751 作为“优化后候选模型”，再用新增外部队列确认是否能稳定超过 0.80。
