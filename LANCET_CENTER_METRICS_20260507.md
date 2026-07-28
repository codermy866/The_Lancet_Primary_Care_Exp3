# Lancet-style External Centre Metrics（2026-05-07）

## Primary Candidate: all-model + DINOv2 centre-z ensemble

Score column: `old_all_plus_all_dino_variants_center_z_mean`

Operating point: fixed threshold `score >= 0.0`. This threshold was not optimised on external labels. It is the natural zero point after centre-wise unsupervised z-score normalisation.

| Centre | n | Positive | Negative | AUROC (95% CI) | PR-AUC | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | F1 | Accuracy (95% CI) | Balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 148 | 49 | 99 | 0.790 (0.705-0.866) | 0.699 | 0.755 (0.619-0.854) | 0.707 (0.611-0.788) | 0.561 (0.441-0.674) | 0.854 (0.761-0.914) | 0.643 | 0.723 (0.646-0.789) | 0.731 |
| 0008 | 70 | 23 | 47 | 0.870 (0.754-0.959) | 0.843 | 0.870 (0.679-0.955) | 0.702 (0.560-0.813) | 0.588 (0.422-0.736) | 0.917 (0.782-0.971) | 0.702 | 0.757 (0.645-0.842) | 0.786 |
| 22101 | 9 | 5 | 4 | 1.000 (1.000-1.000) | 1.000 | 1.000 (0.566-1.000) | 1.000 (0.510-1.000) | 1.000 (0.566-1.000) | 1.000 (0.510-1.000) | 1.000 | 1.000 (0.701-1.000) | 1.000 |
| 22104 | 69 | 21 | 48 | 0.688 (0.542-0.816) | 0.551 | 0.571 (0.365-0.755) | 0.688 (0.547-0.801) | 0.444 (0.276-0.627) | 0.786 (0.641-0.883) | 0.500 | 0.652 (0.534-0.754) | 0.629 |

## DINOv2-only frozen mean

Score column: `prob_frozen_mean`

Operating point: fixed threshold `score >= 0.0`, corresponding to the natural decision boundary of the Ridge decision scores averaged across DINOv2 variants.

| Centre | n | Positive | Negative | AUROC (95% CI) | PR-AUC | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | F1 | Accuracy (95% CI) | Balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 148 | 49 | 99 | 0.703 (0.608-0.793) | 0.653 | 0.469 (0.337-0.606) | 0.788 (0.697-0.857) | 0.523 (0.379-0.662) | 0.750 (0.659-0.823) | 0.495 | 0.682 (0.604-0.752) | 0.629 |
| 0008 | 70 | 23 | 47 | 0.759 (0.617-0.885) | 0.731 | 0.652 (0.449-0.812) | 0.660 (0.517-0.778) | 0.484 (0.320-0.652) | 0.795 (0.645-0.892) | 0.556 | 0.657 (0.540-0.758) | 0.656 |
| 22101 | 9 | 5 | 4 | 0.900 (0.571-1.000) | 0.927 | 0.600 (0.231-0.882) | 0.750 (0.301-0.954) | 0.750 (0.301-0.954) | 0.600 (0.231-0.882) | 0.667 | 0.667 (0.354-0.879) | 0.675 |
| 22104 | 69 | 21 | 48 | 0.644 (0.503-0.779) | 0.519 | 0.238 (0.106-0.451) | 0.917 (0.804-0.967) | 0.556 (0.267-0.811) | 0.733 (0.610-0.829) | 0.333 | 0.710 (0.594-0.804) | 0.577 |

## Files

- Primary CSV: `logs/lancet_center_metrics_20260507/primary_all_dino_center_z.csv`
- Primary Markdown: `logs/lancet_center_metrics_20260507/primary_all_dino_center_z.md`
- Primary LaTeX: `logs/lancet_center_metrics_20260507/primary_all_dino_center_z.tex`
- DINOv2-only CSV: `logs/lancet_center_metrics_20260507/dinov2_frozen_mean.csv`
- DINOv2-only LaTeX: `logs/lancet_center_metrics_20260507/dinov2_frozen_mean.tex`

## Statistical Notes

AUROC 95% CIs were computed by patient-level percentile bootstrap with 3000 resamples. Sensitivity, specificity, PPV, NPV, and accuracy 95% CIs were computed using Wilson intervals.

The `22101` external centre contains only 9 cases, so the apparent perfect performance in the primary candidate has very wide binomial uncertainty and should be interpreted cautiously.

The centre-z ensemble uses external-centre distribution information but not external labels. For a primary Lancet claim, it should be described as a pre-specified deployment normalisation strategy or locked and validated in an additional independent external cohort.
