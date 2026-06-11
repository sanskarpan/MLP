"""
TARGETED CORRECTION - Fix Specific Weaknesses
==============================================
Instead of ML meta-learning, analyze WHERE breakthrough is weak
and apply targeted corrections to those segments.

The insight: 98/2 + Feb baseline works. What if we make
the baseline weight ADAPTIVE based on prediction confidence?

High-confidence predictions → stay close to breakthrough
Low-confidence predictions → lean more on baseline
"""

import pandas as pd
import numpy as np

print("="*100)
print("TARGETED CORRECTION - Adaptive Baseline Weighting")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month
booknow_visits['year'] = booknow_visits['show_date'].dt.year

print(f"\n[1/7] Data loaded")

# Parse test data
sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek
sample_submission['month'] = sample_submission['show_date'].dt.month
sample_submission['breakthrough'] = breakthrough['audience_count'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

# Compute baselines
print("\n[2/7] Computing baselines...")

# Feb 2024 baseline (proven best)
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# All-time theater-dow
all_theater_dow = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Theater statistics
theater_stats = booknow_visits.groupby('book_theater_id').agg({
    'audience_count': ['mean', 'std', 'count']
}).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std', 'th_count']

# Theater-DOW std (variability)
theater_dow_var = booknow_visits.groupby(['book_theater_id', 'dayofweek']).agg({
    'audience_count': ['std', 'count']
}).reset_index()
theater_dow_var.columns = ['book_theater_id', 'dayofweek', 'th_dow_std', 'th_dow_count']

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
sample_submission = sample_submission.merge(theater_dow_var, on=['book_theater_id', 'dayofweek'], how='left')

# Fill missing
sample_submission['th_mean'] = sample_submission['th_mean'].fillna(global_mean)
sample_submission['th_std'] = sample_submission['th_std'].fillna(global_std)
sample_submission['th_count'] = sample_submission['th_count'].fillna(1)
sample_submission['th_dow_std'] = sample_submission['th_dow_std'].fillna(global_std)
sample_submission['th_dow_count'] = sample_submission['th_dow_count'].fillna(1)

print(f"  Feb baseline mean: {sample_submission['feb_baseline'].mean():.2f}")

# ANALYZE: Where might breakthrough be uncertain?
print("\n[3/7] Analyzing prediction confidence...")

bt_pred = sample_submission['breakthrough'].values

# Confidence signals:
# 1. How far is breakthrough from theater mean? (z-score)
zscore = (bt_pred - sample_submission['th_mean'].values) / (sample_submission['th_std'].values + 1)

# 2. How variable is this theater-dow combination?
relative_var = sample_submission['th_dow_std'].values / (sample_submission['th_mean'].values + 1)

# 3. How much data do we have for this theater?
data_scarcity = 1.0 / (sample_submission['th_count'].values + 1)

# 4. Deviation from Feb baseline
feb_deviation = np.abs(bt_pred - sample_submission['feb_baseline'].values) / (sample_submission['th_std'].values + 1)

print(f"  Mean absolute z-score: {np.mean(np.abs(zscore)):.3f}")
print(f"  Mean relative variability: {np.mean(relative_var):.3f}")
print(f"  Mean Feb deviation: {np.mean(feb_deviation):.3f}")

# STRATEGY 1: Adaptive weights based on z-score
print("\n[4/7] Strategy 1: Z-score adaptive weighting...")

# Higher z-score → more uncertainty → lean more on baseline
# Base weight: 0.98 for breakthrough
# Adjustment: up to ±0.03 based on z-score

zscore_clip = np.clip(np.abs(zscore), 0, 3)  # Cap at 3 std
bt_weight_zscore = 0.98 - (zscore_clip / 3) * 0.03  # Range: 0.95 to 0.98
bl_weight_zscore = 1 - bt_weight_zscore

blend_zscore = bt_weight_zscore * bt_pred + bl_weight_zscore * sample_submission['feb_baseline'].values
blend_zscore_cal = calibrate(blend_zscore)
print(f"  Z-score adaptive: mean = {blend_zscore_cal.mean():.2f}")
print(f"  BT weight range: [{bt_weight_zscore.min():.3f}, {bt_weight_zscore.max():.3f}]")

# STRATEGY 2: Adaptive based on theater data availability
print("\n[5/7] Strategy 2: Data availability adaptive weighting...")

# Less data → more uncertainty → lean more on baseline
normalized_count = sample_submission['th_count'].values / sample_submission['th_count'].max()
bt_weight_data = 0.96 + 0.03 * normalized_count  # Range: 0.96 to 0.99 based on data
bl_weight_data = 1 - bt_weight_data

blend_data = bt_weight_data * bt_pred + bl_weight_data * sample_submission['feb_baseline'].values
blend_data_cal = calibrate(blend_data)
print(f"  Data adaptive: mean = {blend_data_cal.mean():.2f}")
print(f"  BT weight range: [{bt_weight_data.min():.3f}, {bt_weight_data.max():.3f}]")

# STRATEGY 3: Combined confidence score
print("\n[6/7] Strategy 3: Combined confidence adaptive...")

# Combine signals into confidence score
confidence = (
    0.5 * (1 - zscore_clip / 3) +  # Low z-score = high confidence
    0.3 * normalized_count +        # More data = higher confidence
    0.2 * (1 - np.clip(relative_var, 0, 1))  # Low variability = higher confidence
)

# Convert to weight: higher confidence → more breakthrough
bt_weight_combined = 0.95 + 0.04 * confidence  # Range: 0.95 to 0.99
bl_weight_combined = 1 - bt_weight_combined

blend_combined = bt_weight_combined * bt_pred + bl_weight_combined * sample_submission['feb_baseline'].values
blend_combined_cal = calibrate(blend_combined)
print(f"  Combined adaptive: mean = {blend_combined_cal.mean():.2f}")
print(f"  BT weight range: [{bt_weight_combined.min():.3f}, {bt_weight_combined.max():.3f}]")
print(f"  Mean BT weight: {bt_weight_combined.mean():.3f}")

# STRATEGY 4: Segment-based (aggressive for some, conservative for others)
print("\n[7/7] Strategy 4: Segment-based weighting...")

# High confidence segment: top 25% by confidence
high_conf_mask = confidence > np.percentile(confidence, 75)
low_conf_mask = confidence < np.percentile(confidence, 25)
mid_conf_mask = ~high_conf_mask & ~low_conf_mask

print(f"  High confidence: {high_conf_mask.sum()} predictions")
print(f"  Mid confidence: {mid_conf_mask.sum()} predictions")
print(f"  Low confidence: {low_conf_mask.sum()} predictions")

blend_segment = np.zeros_like(bt_pred)
# High confidence: 99/1 (trust breakthrough)
blend_segment[high_conf_mask] = 0.99 * bt_pred[high_conf_mask] + 0.01 * sample_submission.loc[high_conf_mask, 'feb_baseline'].values
# Mid confidence: 98/2 (standard)
blend_segment[mid_conf_mask] = 0.98 * bt_pred[mid_conf_mask] + 0.02 * sample_submission.loc[mid_conf_mask, 'feb_baseline'].values
# Low confidence: 96/4 (lean on baseline)
blend_segment[low_conf_mask] = 0.96 * bt_pred[low_conf_mask] + 0.04 * sample_submission.loc[low_conf_mask, 'feb_baseline'].values

blend_segment_cal = calibrate(blend_segment)
print(f"  Segment-based: mean = {blend_segment_cal.mean():.2f}")

# Also try different segment ratios
blend_segment2 = np.zeros_like(bt_pred)
blend_segment2[high_conf_mask] = 0.995 * bt_pred[high_conf_mask] + 0.005 * sample_submission.loc[high_conf_mask, 'feb_baseline'].values
blend_segment2[mid_conf_mask] = 0.98 * bt_pred[mid_conf_mask] + 0.02 * sample_submission.loc[mid_conf_mask, 'feb_baseline'].values
blend_segment2[low_conf_mask] = 0.95 * bt_pred[low_conf_mask] + 0.05 * sample_submission.loc[low_conf_mask, 'feb_baseline'].values
blend_segment2_cal = calibrate(blend_segment2)
print(f"  Segment v2 (99.5/98/95): mean = {blend_segment2_cal.mean():.2f}")

# Current best for reference
current_best = 0.98 * bt_pred + 0.02 * sample_submission['feb_baseline'].values
current_best_cal = calibrate(current_best)
print(f"\n  Reference (98/2 flat): mean = {current_best_cal.mean():.2f}")

# Save
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': blend_combined_cal,  # MAIN: Combined confidence adaptive
    'submission_zscore.csv': blend_zscore_cal,
    'submission_data.csv': blend_data_cal,
    'submission_segment.csv': blend_segment_cal,
    'submission_segment2.csv': blend_segment2_cal,
    'submission_flat_98_2.csv': current_best_cal,  # Reference
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("TARGETED CORRECTION COMPLETE")
print("="*100)
print("\nINSIGHT:")
print("  Instead of flat 98/2 for ALL predictions,")
print("  use ADAPTIVE weights based on prediction confidence:")
print("    - High confidence → trust breakthrough more (99%+)")
print("    - Low confidence → lean on baseline more (95-96%)")
print("\nCurrent best: 0.35749 (flat 98/2 + Feb)")
print("New approach: Confidence-adaptive weighting!")
print("="*100)
