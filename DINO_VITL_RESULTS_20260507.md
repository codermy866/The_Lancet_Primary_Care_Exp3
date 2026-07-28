# DINOv2 / ViT-L Frozen Feature 实验结果（2026-05-07）

## 结论

本轮 DINOv2/ViT-L frozen feature baseline 已自动完成。DINOv2 达到了本轮设定的目标：外部验证 AUC 优于 0.69。

最好的单独 DINOv2 frozen feature 结果为：

- DINOv2 ViT-L/14 + OCT + 阴道镜 + 临床变量: AUC 0.7023, PR-AUC 0.6390
- DINOv2 多 variant frozen mean: AUC 0.7034, PR-AUC 0.6527

ViT-L SWAG frozen feature 在本数据上表现不佳：

- ViT-L SWAG + OCT + 阴道镜 + 临床变量: AUC 0.6153
- ViT-L SWAG 多 variant frozen mean: AUC 0.6108

把 DINOv2 合入已有多模型集成后，整体结果进一步提升：

- small old ensemble raw mean: AUC 0.7058
- small old + DINOv2 frozen mean raw mean: AUC 0.7163
- old all-model raw mean: AUC 0.7166
- old all-model + all DINOv2 variants raw mean: AUC 0.7275
- old all-model center-z: AUC 0.7751
- old all-model + all DINOv2 variants center-z: AUC 0.7895

因此，DINOv2 比普通 ViT-L 更适合当前数据，并且确实提升了 raw ensemble 和 center-z ensemble。但目前仍未在严格独立外部验证意义上超过 0.80。

## 输出文件

- DINOv2 单模型结果: `logs/frozen_feature_dinov2_vitl14_20260507/frozen_feature_results.json`
- DINOv2 外部预测: `logs/frozen_feature_dinov2_vitl14_20260507/external_predictions_frozen_features.csv`
- ViT-L 单模型结果: `logs/frozen_feature_vitl_20260507/frozen_feature_results.json`
- ViT-L 外部预测: `logs/frozen_feature_vitl_20260507/external_predictions_frozen_features.csv`
- DINOv2 合并集成结果: `logs/final_external_ensemble_20260507_dinov2/dinov2_ensemble_summary.json`
- DINOv2 合并集成预测: `logs/final_external_ensemble_20260507_dinov2/external_predictions_dinov2_ensemble.csv`

## 已修改代码

文件: `scripts/frozen_feature_baseline.py`

新增 backbone 支持：

- `vit_l_16`
- `vit_l_16_swag_linear`
- `vit_l_16_swag_e2e`
- `dinov2_vitl14`
- `dinov2_vitl14_reg`

DINOv2 使用 `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")` 加载，输入使用 224x224 bicubic resize/crop 和 ImageNet normalization。输出使用 DINOv2 的 CLS embedding，再按病例聚合为 mean/std/max frozen feature。

## 方法学说明

本轮 frozen feature 的分类头仍只使用训练集训练，并通过内部验证集选择 Ridge 正则强度；外部验证集只用于最终评估。

集成中的 raw mean / z mean / center-z mean 没有使用外部标签拟合参数。center-z 使用外部样本的中心分布做无监督标准化，因此更适合作为预定义部署流程或候选模型结果；若用于高水平投稿主结果，建议在新的独立外部队列上锁定验证。
