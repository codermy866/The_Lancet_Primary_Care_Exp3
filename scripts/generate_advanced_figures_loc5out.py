from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")

ROOT = Path('/data2/hmy/VLM_Caus_Rm_Mics/experiments/OCT_traige')
LOGS = ROOT / 'logs'
OUT = LOGS / 'figures_loc5out'
OUT.mkdir(parents=True, exist_ok=True)

ext_pred = pd.read_csv(LOGS / 'external_predictions.csv')
hist_files = sorted(LOGS.glob('metrics_history_*.csv'))
if not hist_files:
    raise FileNotFoundError(f'No metrics_history_*.csv found in {LOGS}')
hist = pd.read_csv(hist_files[-1])
int_train_pc = pd.read_csv(LOGS / 'internal_train_per_center_metrics_loc5out.csv')
int_val_pc = pd.read_csv(LOGS / 'internal_val_per_center_metrics_loc5out.csv')
ext_pc = pd.read_csv(LOGS / 'external_per_center_metrics_loc5out.csv')

if 'center_id_external' in ext_pred.columns:
    ext_pred['site'] = ext_pred['center_id_external'].astype(str)
elif 'center_id' in ext_pred.columns:
    ext_pred['site'] = ext_pred['center_id'].astype(str)
else:
    ext_pred['site'] = 'unknown'

ext_pred['correct'] = (ext_pred['label'].astype(int) == ext_pred['pred'].astype(int)).astype(int)

# 1) Sunburst with Rounded Corner (approximation via nested donut)
site_counts = ext_pred.groupby(['site','label']).size().reset_index(name='n')
outer = site_counts.groupby('site')['n'].sum().sort_values(ascending=False)
inner = site_counts.sort_values(['site','label'])
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(aspect='equal'))
ax.pie(outer.values, radius=1.0, labels=outer.index, labeldistance=1.05,
       wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))
ax.pie(inner['n'].values, radius=0.7,
       labels=[f"{r.site}-L{int(r.label)}" for r in inner.itertuples()],
       labeldistance=0.78,
       wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2))
ax.set_title('Sunburst-style nested donut (Site -> Label)')
plt.tight_layout(); plt.savefig(OUT/'sunburst_rounded_corner.png', dpi=300); plt.close()

# 2) Nightingale Chart (polar area)
vals = outer.values
theta = np.linspace(0, 2*np.pi, len(vals), endpoint=False)
width = (2*np.pi/len(vals))*0.9
fig = plt.figure(figsize=(8,8)); ax = fig.add_subplot(111, polar=True)
ax.bar(theta, vals, width=width, alpha=0.8)
ax.set_xticks(theta); ax.set_xticklabels(outer.index)
ax.set_title('Nightingale chart of external sample sizes by site')
plt.tight_layout(); plt.savefig(OUT/'nightingale_chart.png', dpi=300); plt.close()

# 3) Scatter with Jittering
tmp = ext_pred.sample(min(2000, len(ext_pred)), random_state=42).copy()
tmp['label_jit'] = tmp['label'] + np.random.normal(0, 0.04, len(tmp))
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(tmp['label_jit'], tmp['prob_pos'], alpha=0.35, s=12)
ax.set_xlabel('True label (jittered)'); ax.set_ylabel('Predicted probability')
ax.set_title('Scatter with jittering: true label vs probability')
plt.tight_layout(); plt.savefig(OUT/'scatter_with_jittering.png', dpi=300); plt.close()

# 4) Linear Regression (epoch vs val_auc)
X = hist[['epoch']].values
y = hist['val_auc'].values
lr = LinearRegression().fit(X,y)
yhat = lr.predict(X)
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(hist['epoch'], y, s=30, alpha=0.7, label='Observed')
ax.plot(hist['epoch'], yhat, color='red', lw=2, label='Linear fit')
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation AUROC'); ax.legend()
ax.set_title('Linear regression: epoch vs validation AUROC')
plt.tight_layout(); plt.savefig(OUT/'linear_regression_epoch_valauc.png', dpi=300); plt.close()

# 5) Scatter Matrix
sm = hist[['train_auc','val_auc','train_f1','val_f1','train_loss','val_loss']].copy()
pp = sns.pairplot(sm, corner=True, diag_kind='hist')
pp.fig.suptitle('Scatter Matrix of training dynamics', y=1.02)
pp.savefig(OUT/'scatter_matrix_training_metrics.png', dpi=300)
plt.close('all')

# 6) Data Transform Simple Aggregate
agg = ext_pred.groupby(['site','label']).size().unstack(fill_value=0)
agg.plot(kind='bar', stacked=True, figsize=(9,5))
plt.title('Simple aggregate transform: counts by site and label')
plt.xlabel('Site'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig(OUT/'data_transform_simple_aggregate.png', dpi=300); plt.close()

# 7) Horizontal boxplot with observations
fig, ax = plt.subplots(figsize=(9,5))
sns.boxplot(data=ext_pred, y='site', x='prob_pos', orient='h', ax=ax, color='#a6cee3')
sns.stripplot(data=ext_pred.sample(min(2500, len(ext_pred)), random_state=1),
              y='site', x='prob_pos', orient='h', ax=ax, color='black', alpha=0.25, size=2)
ax.set_title('Horizontal boxplot with observations (external probabilities)')
plt.tight_layout(); plt.savefig(OUT/'horizontal_boxplot_with_observations.png', dpi=300); plt.close()

# 8) Violinplot from a wide-form dataset
wide = hist[['train_auc','val_auc','train_f1','val_f1']]
fig, ax = plt.subplots(figsize=(8,5))
sns.violinplot(data=wide, inner='quartile', ax=ax)
ax.set_title('Violinplot from a wide-form dataset (epoch metrics)')
plt.tight_layout(); plt.savefig(OUT/'violinplot_wide_form.png', dpi=300); plt.close()

# 9) Different cubehelix palettes
palettes = [sns.cubehelix_palette(8, start=s, rot=0.5, dark=0.2, light=0.9) for s in [0,1,2,3]]
fig, axes = plt.subplots(4,1, figsize=(8,4))
for i, pal in enumerate(palettes):
    arr = np.arange(8).reshape(1,-1)
    sns.heatmap(arr, cmap=sns.color_palette(pal, as_cmap=True), cbar=False, ax=axes[i])
    axes[i].set_ylabel(f'palette {i+1}')
    axes[i].set_xticks([]); axes[i].set_yticks([])
fig.suptitle('Different cubehelix palettes', y=1.02)
plt.tight_layout(); plt.savefig(OUT/'different_cubehelix_palettes.png', dpi=300); plt.close()

# 10) Conditional means with observations
fig, ax = plt.subplots(figsize=(9,5))
sns.stripplot(data=ext_pred.sample(min(3000, len(ext_pred)), random_state=2),
              x='site', y='prob_pos', alpha=0.2, size=2, ax=ax)
means = ext_pred.groupby('site')['prob_pos'].mean().reset_index()
sns.pointplot(data=means, x='site', y='prob_pos', color='red', linestyle='none', marker='D', ax=ax)
ax.set_title('Conditional means with observations')
plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig(OUT/'conditional_means_with_observations.png', dpi=300); plt.close()

# 11) Facetting histograms by subsets of data
g = sns.displot(data=ext_pred, x='prob_pos', col='site', col_wrap=2, bins=25, facet_kws=dict(sharey=False, sharex=True))
g.fig.suptitle('Faceting histograms by subsets of data (site)', y=1.02)
g.savefig(OUT/'facetting_histograms_by_site.png', dpi=300)
plt.close('all')

# 12) Multiple linear regression
X2 = hist[['epoch','train_auc','train_loss']].values
y2 = hist['val_auc'].values
lr2 = LinearRegression().fit(X2, y2)
y2hat = lr2.predict(X2)
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(y2, y2hat, alpha=0.7)
mn, mx = min(y2.min(), y2hat.min()), max(y2.max(), y2hat.max())
ax.plot([mn,mx],[mn,mx],'r--')
ax.set_xlabel('Observed val AUROC'); ax.set_ylabel('Predicted val AUROC')
ax.set_title('Multiple linear regression fit quality')
plt.tight_layout(); plt.savefig(OUT/'multiple_linear_regression_fit.png', dpi=300); plt.close()

# 13) Scatterplot heatmap
fig, ax = plt.subplots(figsize=(7,5))
hb = ax.hexbin(ext_pred['prob_pos'], ext_pred['label'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('Predicted probability'); ax.set_ylabel('True label')
ax.set_title('Scatterplot heatmap (hexbin)')
plt.tight_layout(); plt.savefig(OUT/'scatterplot_heatmap_hexbin.png', dpi=300); plt.close()

# Bonus: compact panel for internal/external per-site AUROC comparison
cmp = int_train_pc[['center_id','auc']].rename(columns={'auc':'internal_train_auc'})
cmp2 = int_val_pc[['center_id','auc']].rename(columns={'auc':'internal_val_auc'})
fig, ax = plt.subplots(figsize=(9,5))
for i, row in cmp.iterrows():
    if pd.notna(row['internal_train_auc']):
        ax.scatter(row['internal_train_auc'], 0, color='tab:blue', s=60)
for i, row in cmp2.iterrows():
    if pd.notna(row['internal_val_auc']):
        ax.scatter(row['internal_val_auc'], 1, color='tab:orange', s=60)
ax.set_yticks([0,1]); ax.set_yticklabels(['Internal train AUROC','Internal val AUROC'])
ax.set_xlabel('AUROC'); ax.set_title('Internal AUROC distribution by site (summary view)')
plt.tight_layout(); plt.savefig(OUT/'internal_auroc_summary_scatter.png', dpi=300); plt.close()

print('[done] figures saved to', OUT)
