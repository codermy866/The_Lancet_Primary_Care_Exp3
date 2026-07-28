# OCT_traige — 最终运行结果汇总（生成于 2026-04-25）

## 概览
- 训练集样本数: 788
- 内部验证集样本数: 197
- 外部测试集样本数: 902

## 关键总体指标
- **内部验证（internal_validation）**:
  - Accuracy: 0.7563
  - F1: 0.4000
  - AUROC: 0.86055 (bootstrap CI: 0.80287 — 0.91020)
  - PR-AUC: 0.81411
  - Sensitivity (召回): 0.25
  - Specificity: 1.00
  - Brier score: 0.16333
  - 校准（slope/intercept）: 6.4858 / 4.5425
  - 来源文件: logs/internal_overall_metrics_loc5out.json

- **外部测试（external_test）**:
  - Accuracy: 0.6386
  - F1: 0.3264
  - AUROC: 0.58336 (bootstrap CI: 0.54076 — 0.62212)
  - PR-AUC: 0.26313
  - Sensitivity: 0.37799
  - Specificity: 0.71717
  - Brier score: 0.22992
  - 校准（slope/intercept）: 0.34985 / -1.11975
  - 来源文件: logs/external_metrics_loc5out.json

## 每中心表现摘要（选取亮点）
- 内部（验证）表现较好的中心: 十堰市人民医院 AUROC=0.925, 荆州市第一人民医院 AUROC≈0.9286, 襄阳市中心医院 AUROC≈0.8331。来源: supplement_statistical_summary.json
- 外部按中心: Liaoning AUROC≈0.6086, ZhengDaSanFu AUROC≈0.5691；小中心（AnYang/Hua_Xi）样本量极小且 CI 极宽。来源: supplement_statistical_summary.json

## 主要观察（结论）
- 模型在内部验证上表现优秀（AUROC≈0.86），但在外部独立队列上显著下降（AUROC≈0.58），说明存在明显的域间差异或过拟合到训练中心分布。
- 内部结果显示高精度/低召回（precision≈1.0，但 recall≈0.25），可能是阈值偏高或类别不平衡导致的保守预测策略。
- 校准结果表明模型在不同队列上校准性能差异大（内部校准 slope/intercept 非常极端，外部偏低），需要重新校准以便概率可信度可用。
- 外部中心间差异显著，提示需要按中心做更细粒度的分析与/或中心特异性适配。

## 建议的后续工作（优先级排序）
1. 进行概率校准（Platt/Isotonic）并在外部集合上验证校准改善效果。
2. 优化决策阈值以平衡 Recall/Precision（针对临床需求决定倾向于高召回或高精度）。
3. 分析训练/测试集中影像分布差异（像素级统计、拍摄设备/协议、病人构成），找出导致域差异的因素。
4. 采用域适应或轻量微调：在少量外部标注样本上微调模型或使用对抗域适应方法。
5. 尝试模型集成（ensemble）与稳定性评估以提升外部鲁棒性。
6. 若可行，增加外部正例样本或采用更平衡的损失函数/采样策略以提升召回。

## 关键文件（方便复查）
- [logs/internal_overall_metrics_loc5out.json](experiments/OCT_traige/logs/internal_overall_metrics_loc5out.json#L1-L40)
- [logs/external_metrics_loc5out.json](experiments/OCT_traige/logs/external_metrics_loc5out.json#L1-L40)
- [logs/supplement_statistical_summary.json](experiments/OCT_traige/logs/supplement_statistical_summary.json#L1-L200)
- 最优模型： checkpoints/best_model.pt

---
（如需我把这个摘要转换成论文/报告用的 LaTeX 表格与图，或把 per-center 的完整 CSV/图形化展示出来，我可以继续生成。）

## 阈值优化（自动化后处理）

- 我基于已有的 `logs/internal_val_predictions.csv` 运行了阈值优化脚本，结果保存在 `logs/internal_threshold_optimization.json`，并生成了按最佳 F1 的预测文件 `logs/internal_val_predictions_bestf1.csv`。
- 关键结果（内部验证）：
  - ROC AUC: 0.86055
  - PR AUC: 0.81304
  - 最佳 F1 的阈值: 0.337
  - 在阈值 0.337 下：Accuracy=0.8477, Precision=0.9048, Recall=0.5938, F1=0.717, Specificity≈0.9699

- 说明：将阈值从原先（模型保存/默认阈值造成 Precision≈1.0, Recall≈0.25）调整到 0.337 后，内部召回显著提升（0.25 -> 0.5938），F1 从 0.40 提升到 ≈0.717（更适合兼顾精度和召回的临床应用）。

如果你同意，我可以：
- 把这个阈值应用到外部测试集以评估召回/精度的变化；
- 尝试概率校准（Platt/Isotonic）并比较校准前后的 Brier/校准斜率；
- 运行简单模型集成或微调提升泛化。

## 我已自动执行的后处理步骤（阈值 + 校准）

- 我已将内部最佳 F1 阈值 `0.337` 应用于外部预测（文件 `logs/external_predictions.csv`），并进行了 Platt（逻辑回归）与 Isotonic 校准，结果保存在 `logs/external_eval_calibration.json`，带校准概率的 CSV 保存在 `logs/external_predictions_with_calibrated.csv`。

- 关键比较（外部，共 n=148）:
  - 原始默认阈值 0.5: Accuracy=0.5541, Precision=0.3976, Recall=0.6735, F1=0.50, ROC AUC=0.6638, PR AUC=0.6237, Brier=0.3032
  - 应用内部最佳阈值 0.337: Accuracy=0.4730, Precision=0.3535, Recall=0.7143, F1=0.4730 (召回略增，但准确率/精度下降)
  - Platt 校准后（阈值 0.337）: Accuracy=0.4257, Precision=0.3333, Recall=0.7347, F1≈0.459 (Brier 上升，表明校准后概率分布有所变化)
  - Isotonic 校准后（阈值 0.337）: Accuracy=0.4730, Precision=0.3535, Recall=0.7143, F1=0.4730

- 校准斜率/截距（外部）:
  - 校准前: slope≈0.3672, intercept≈-0.9093
  - Platt 后: slope≈0.0566, intercept≈-1.1665
  - Isotonic 后: slope≈0.0031, intercept≈-0.7540

- 小结：把内部的最佳阈值应用到外部能稍微提高召回（内部提升后的阈值使外部召回到 ~0.71），但总体精度/F1 没有系统性提升，且校准后 Brier 分数并未改善（Platt/Isotonic 在本次评估中未能使外部分数更好，可能因为外部分布与内部差异较大，或内部样本不足以学习出泛化的校准映射）。

结果文件：
- `logs/external_eval_calibration.json`（包含各项评估指标）
- `logs/external_predictions_with_calibrated.csv`（带 `prob_platt` / `prob_isotonic` 与多种预测列）

我已把这些步骤写入 TODO 并保存。下一步建议：
- 若目标是提升外部泛化，优先进行少量外部微调或域适应；
- 若目标是改善校准，可用更多外部标注构建校准集或在外部子集上微调后再校准。
