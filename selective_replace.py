"""
SELECTIVE REPLACEMENT - Not blending, REPLACING
================================================
Instead of blending ALL predictions with baseline,
identify predictions that are LIKELY WRONG and
REPLACE them entirely with baseline values.

This is fundamentally different from weighted averaging!
"""

import pandas as pd
import numpy as np

print("="*100)
print("SELECTIVE REPLACEMENT - Fix Wrong Predictions")
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

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

print(f"\n[1/5] Data loaded")

# Compute baselines and statistics
print("\n[2/5] Computing baselines and theater statistics...")

feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
all_theater_dow = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Theater statistics
theater_stats = booknow_visits.groupby('book_theater_id').agg({
    'audience_count': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std', 'th_min', 'th_max', 'th_count']

# Theater-DOW statistics
theater_dow_stats = booknow_visits.groupby(['book_theater_id', 'dayofweek']).agg({
    'audience_count': ['mean', 'std', 'min', 'max']
}).reset_index()
theater_dow_stats.columns = ['book_theater_id', 'dayofweek', 'td_mean', 'td_std', 'td_min', 'td_max']

global_mean = booknow_visits['audience_count'].mean()
global_std = booknow_visits['audience_count'].std()

def get_feb_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in all_theater_dow.index:
        return all_theater_dow[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_feb_baseline, axis=1)
sample_submission = sample_submission.merge(theater_stats, on='book_theater_id', how='left')
sample_submission = sample_submission.merge(theater_dow_stats, on=['book_theater_id', 'dayofweek'], how='left')

# Fill missing
for col in ['th_mean', 'th_std', 'th_min', 'th_max', 'th_count', 'td_mean', 'td_std', 'td_min', 'td_max']:
    if col in sample_submission.columns:
        if 'mean' in col:
            sample_submission[col] = sample_submission[col].fillna(global_mean)
        elif 'std' in col:
            sample_submission[col] = sample_submission[col].fillna(global_std)
        elif 'min' in col:
            sample_submission[col] = sample_submission[col].fillna(0)
        elif 'max' in col:
            sample_submission[col] = sample_submission[col].fillna(200)
        else:
            sample_submission[col] = sample_submission[col].fillna(1)

bt_pred = sample_submission['breakthrough'].values
feb_bl = sample_submission['feb_baseline'].values

print(f"  Breakthrough mean: {bt_pred.mean():.2f}")
print(f"  Feb baseline mean: {feb_bl.mean():.2f}")

# IDENTIFY "WRONG" PREDICTIONS
print("\n[3/5] Identifying potentially wrong predictions...")

# Criteria for "likely wrong":
# 1. Prediction is outside historical range for this theater-DOW
# 2. Prediction deviates significantly from theater mean
# 3. Prediction is an extreme outlier

# Check 1: Outside historical theater-DOW range
outside_range = (bt_pred < sample_submission['td_min'].values * 0.5) | \
                (bt_pred > sample_submission['td_max'].values * 1.5)
print(f"  Outside theater-DOW range (0.5x-1.5x): {outside_range.sum()} ({100*outside_range.mean():.1f}%)")

# Check 2: High z-score relative to theater mean
zscore = np.abs(bt_pred - sample_submission['th_mean'].values) / (sample_submission['th_std'].values + 1)
high_zscore = zscore > 2.5
print(f"  High z-score (>2.5): {high_zscore.sum()} ({100*high_zscore.mean():.1f}%)")

# Check 3: Global outliers
global_zscore = np.abs(bt_pred - global_mean) / global_std
global_outlier = global_zscore > 3
print(f"  Global outliers (z>3): {global_outlier.sum()} ({100*global_outlier.mean():.1f}%)")

# Check 4: Huge deviation from baseline
baseline_dev = np.abs(bt_pred - feb_bl) / (feb_bl + 1)
huge_baseline_dev = baseline_dev > 1.0  # More than 100% different
print(f"  Huge baseline deviation (>100%): {huge_baseline_dev.sum()} ({100*huge_baseline_dev.mean():.1f}%)")

# SELECTIVE REPLACEMENT STRATEGIES
print("\n[4/5] Creating selective replacement predictions...")

results = {}

# Reference: Standard 98/2 blend
blend_98_2 = 0.98 * bt_pred + 0.02 * feb_bl
results['blend_98_2'] = calibrate(blend_98_2)
print(f"  Reference (98/2 blend): {results['blend_98_2'].mean():.2f}")

# Strategy 1: Replace predictions outside theater-DOW range
pred_v1 = bt_pred.copy()
pred_v1[outside_range] = feb_bl[outside_range]
results['replace_outside_range'] = calibrate(pred_v1)
print(f"  Replace outside range: {results['replace_outside_range'].mean():.2f}")

# Strategy 2: Replace high z-score predictions
pred_v2 = bt_pred.copy()
pred_v2[high_zscore] = feb_bl[high_zscore]
results['replace_high_zscore'] = calibrate(pred_v2)
print(f"  Replace high z-score: {results['replace_high_zscore'].mean():.2f}")

# Strategy 3: Replace huge baseline deviations
pred_v3 = bt_pred.copy()
pred_v3[huge_baseline_dev] = feb_bl[huge_baseline_dev]
results['replace_huge_dev'] = calibrate(pred_v3)
print(f"  Replace huge deviation: {results['replace_huge_dev'].mean():.2f}")

# Strategy 4: Blend replaced predictions with 98/2 (hybrid)
pred_v4 = bt_pred.copy()
replace_mask = outside_range | high_zscore
pred_v4[replace_mask] = feb_bl[replace_mask]
# Then apply 98/2 blend to everything
pred_v4 = 0.98 * pred_v4 + 0.02 * feb_bl
results['hybrid_replace_blend'] = calibrate(pred_v4)
print(f"  Hybrid (replace then blend): {results['hybrid_replace_blend'].mean():.2f}")

# Strategy 5: Partial replacement (50% toward baseline for suspicious predictions)
pred_v5 = bt_pred.copy()
suspicious = high_zscore | huge_baseline_dev
pred_v5[suspicious] = 0.5 * bt_pred[suspicious] + 0.5 * feb_bl[suspicious]
pred_v5 = 0.98 * pred_v5 + 0.02 * feb_bl
results['partial_replace'] = calibrate(pred_v5)
print(f"  Partial replacement: {results['partial_replace'].mean():.2f}")

# Strategy 6: Clamp to historical range
pred_v6 = bt_pred.copy()
pred_v6 = np.maximum(pred_v6, sample_submission['td_min'].values * 0.7)
pred_v6 = np.minimum(pred_v6, sample_submission['td_max'].values * 1.3)
pred_v6 = 0.98 * pred_v6 + 0.02 * feb_bl
results['clamped'] = calibrate(pred_v6)
print(f"  Clamped to range: {results['clamped'].mean():.2f}")

# Strategy 7: Conservative - only replace EXTREME outliers
extreme = (zscore > 3.5) | (global_zscore > 4)
pred_v7 = bt_pred.copy()
pred_v7[extreme] = feb_bl[extreme]
pred_v7 = 0.98 * pred_v7 + 0.02 * feb_bl
results['replace_extreme'] = calibrate(pred_v7)
print(f"  Replace only extreme (z>3.5): {results['replace_extreme'].mean():.2f} ({extreme.sum()} replaced)")

# SAVE
print("\n[5/5] Saving submissions...")
print("="*100)

submissions = {
    'submission.csv': results['hybrid_replace_blend'],  # MAIN
    'submission_replace_range.csv': results['replace_outside_range'],
    'submission_replace_zscore.csv': results['replace_high_zscore'],
    'submission_partial.csv': results['partial_replace'],
    'submission_clamped.csv': results['clamped'],
    'submission_extreme_only.csv': results['replace_extreme'],
    'submission_98_2_ref.csv': results['blend_98_2'],
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("SELECTIVE REPLACEMENT COMPLETE")
print("="*100)
print("\nKEY INSIGHT:")
print("  Instead of blending EVERYTHING with baseline,")
print("  REPLACE only the predictions that are likely WRONG.")
print("  This preserves good predictions while fixing bad ones!")
print("\nCurrent best: 0.35749")
print("="*100)
