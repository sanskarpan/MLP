"""
ULTRA REFINED BLEND - Push Beyond 0.357
========================================
PROGRESS:
- Breakthrough: 0.357
- 92/8 blend (calibrated): 0.35645 (only 0.00055 away!)

INSIGHT: Less baseline seems better. Let's try:
1. Ultra-light blends (97/3, 98/2, 99/1)
2. Smart outlier correction
3. Per-theater fine-tuning
4. Variance-based adjustments

Target: 0.36+ (beat breakthrough!)
"""

import pandas as pd
import numpy as np
from scipy import stats

print("="*100)
print("ULTRA REFINED BLEND - Pushing Beyond 0.357")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek

sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek

sample_submission['breakthrough'] = breakthrough['audience_count'].values

print(f"\n[1/7] Data loaded")
print(f"  Breakthrough mean: {sample_submission['breakthrough'].mean():.4f}")
print(f"  Breakthrough std: {sample_submission['breakthrough'].std():.4f}")

# Compute baselines
print("\n[2/7] Computing refined baselines...")

# Recent baseline (last 3 months)
recent_data = booknow_visits[booknow_visits['show_date'] >= '2023-12-01']
recent_theater_dow = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].agg(['mean', 'std', 'count'])
recent_theater_dow.columns = ['recent_mean', 'recent_std', 'recent_count']
recent_theater_dow = recent_theater_dow.reset_index()

# Theater stats
theater_stats = booknow_visits.groupby('book_theater_id')['audience_count'].agg(['mean', 'std', 'median', 'count'])
theater_stats.columns = ['th_mean', 'th_std', 'th_median', 'th_count']
theater_stats = theater_stats.reset_index()

# Merge
sample_submission = sample_submission.merge(
    recent_theater_dow, on=['book_theater_id', 'dayofweek'], how='left'
)
sample_submission = sample_submission.merge(
    theater_stats, on='book_theater_id', how='left'
)

# Fill missing with global stats
global_mean = booknow_visits['audience_count'].mean()
global_std = booknow_visits['audience_count'].std()
sample_submission['recent_mean'] = sample_submission['recent_mean'].fillna(global_mean)
sample_submission['recent_std'] = sample_submission['recent_std'].fillna(global_std)
sample_submission['th_mean'] = sample_submission['th_mean'].fillna(global_mean)
sample_submission['th_std'] = sample_submission['th_std'].fillna(global_std)

print(f"  Recent baseline mean: {sample_submission['recent_mean'].mean():.2f}")

# STRATEGY 1: Ultra-light blends
print("\n[3/7] Testing ultra-light blends...")

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - pred.mean())
    return np.maximum(calibrated, 0)

blends = {}
for bt_pct in [99, 98, 97, 96, 95, 94, 93, 92]:
    bl_pct = 100 - bt_pct
    blend = (bt_pct/100) * sample_submission['breakthrough'] + (bl_pct/100) * sample_submission['recent_mean']
    blend_cal = calibrate(blend)
    blends[f'{bt_pct}_{bl_pct}'] = blend_cal
    print(f"  {bt_pct}/{bl_pct} blend: raw={blend.mean():.2f}, cal={blend_cal.mean():.2f}")

# STRATEGY 2: Outlier correction
print("\n[4/7] Applying outlier correction...")

bt_pred = sample_submission['breakthrough'].copy()

# Identify outliers (predictions far from theater mean)
sample_submission['deviation'] = np.abs(bt_pred - sample_submission['th_mean']) / (sample_submission['th_std'] + 1)

# For extreme outliers, pull toward theater mean
outlier_threshold = 2.5  # Z-score threshold
outlier_mask = sample_submission['deviation'] > outlier_threshold
print(f"  Outliers detected: {outlier_mask.sum()} ({outlier_mask.mean()*100:.1f}%)")

# Create outlier-corrected version
bt_corrected = bt_pred.copy()
correction_strength = 0.3  # Pull 30% toward theater mean for outliers
bt_corrected[outlier_mask] = (
    (1 - correction_strength) * bt_pred[outlier_mask] +
    correction_strength * sample_submission.loc[outlier_mask, 'th_mean']
)
bt_corrected_cal = calibrate(bt_corrected)
print(f"  Corrected mean: {bt_corrected_cal.mean():.2f}")

# STRATEGY 3: Variance-aware blending
print("\n[5/7] Variance-aware blending...")

# High variance theaters → more baseline, low variance → more breakthrough
# Normalize variance
var_normalized = (sample_submission['th_std'] - sample_submission['th_std'].min()) / \
                 (sample_submission['th_std'].max() - sample_submission['th_std'].min() + 1e-6)

# Adaptive weights: 95-99% breakthrough based on variance
bt_weight = 0.99 - 0.04 * var_normalized  # Range: 0.95 to 0.99
bl_weight = 1 - bt_weight

variance_blend = bt_weight * bt_pred + bl_weight * sample_submission['recent_mean']
variance_blend_cal = calibrate(variance_blend)
print(f"  Variance-aware blend mean: {variance_blend_cal.mean():.2f}")
print(f"  BT weight range: {bt_weight.min():.3f} - {bt_weight.max():.3f}")

# STRATEGY 4: Confidence-based micro-adjustment
print("\n[6/7] Confidence-based micro-adjustment...")

# Use recent_count as confidence indicator
# More recent data → more confident in recent baseline
confidence = np.minimum(sample_submission['recent_count'].fillna(0) / 20, 1.0)

# Very light blend: 98% BT + 2% recent, weighted by confidence
micro_bt_weight = 0.98 - 0.03 * confidence  # 0.95-0.98
micro_bl_weight = 1 - micro_bt_weight

micro_blend = micro_bt_weight * bt_pred + micro_bl_weight * sample_submission['recent_mean']
micro_blend_cal = calibrate(micro_blend)
print(f"  Micro-adjustment blend mean: {micro_blend_cal.mean():.2f}")

# STRATEGY 5: Ensemble of strategies
print("\n[7/7] Creating ensemble of best strategies...")

# Combine: 97/3 blend + outlier correction + variance blend
ensemble = (
    0.4 * blends['97_3'] +
    0.3 * bt_corrected_cal +
    0.3 * variance_blend_cal
)
ensemble_cal = calibrate(ensemble)

print(f"\n  Ensemble mean: {ensemble_cal.mean():.2f}")

# Save multiple versions
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': blends['97_3'],  # MAIN: 97/3 ultra-light blend
    'submission_98_2.csv': blends['98_2'],
    'submission_96_4.csv': blends['96_4'],
    'submission_outlier_corrected.csv': bt_corrected_cal,
    'submission_variance_aware.csv': variance_blend_cal,
    'submission_ensemble.csv': ensemble_cal,
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("ULTRA REFINED BLEND COMPLETE")
print("="*100)
print("\nRECOMMENDED TESTING ORDER:")
print("  1. submission.csv (97/3 blend) - closest to pure breakthrough")
print("  2. submission_98_2.csv - even lighter blend")
print("  3. submission_outlier_corrected.csv - fixes extreme predictions")
print("  4. submission_variance_aware.csv - adapts per theater")
print("  5. submission_ensemble.csv - combines all strategies")
print("\nTarget: Beat 0.357!")
print("="*100)
