"""
ENSEMBLE OF BLENDS - Meta-ensemble approach
=============================================
Instead of picking ONE blend ratio, ensemble MULTIPLE ratios.

If 98/2 scores 0.35749 and 97/3 scores 0.35700,
what if we average them? The errors might cancel out!

Also try: ensemble of different baselines.
"""

import pandas as pd
import numpy as np

print("="*100)
print("ENSEMBLE OF BLENDS - Meta-Ensemble Approach")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month

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

print(f"\n[1/5] Data loaded")

# Compute multiple baselines
print("\n[2/5] Computing multiple baselines...")

# Feb 2024 baseline (proven best)
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# All-time theater-dow
all_theater_dow = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Recent 3 months
recent_data = booknow_visits[booknow_visits['show_date'] >= '2023-12-01']
recent_theater_dow = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

global_mean = booknow_visits['audience_count'].mean()

def get_baseline(row, baseline_dict, fallback_dict=None):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in baseline_dict.index:
        return baseline_dict[key]
    if fallback_dict is not None and key in fallback_dict.index:
        return fallback_dict[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(
    lambda r: get_baseline(r, feb_theater_dow, all_theater_dow), axis=1)
sample_submission['recent_baseline'] = sample_submission.apply(
    lambda r: get_baseline(r, recent_theater_dow, all_theater_dow), axis=1)
sample_submission['all_baseline'] = sample_submission.apply(
    lambda r: get_baseline(r, all_theater_dow), axis=1)

print(f"  Feb baseline mean: {sample_submission['feb_baseline'].mean():.2f}")
print(f"  Recent baseline mean: {sample_submission['recent_baseline'].mean():.2f}")
print(f"  All-time baseline mean: {sample_submission['all_baseline'].mean():.2f}")

bt_pred = sample_submission['breakthrough'].values
feb_bl = sample_submission['feb_baseline'].values

# Create individual blends
print("\n[3/5] Creating individual blends...")

blends = {}

# Different ratios with Feb baseline
for ratio in [97.0, 97.5, 98.0, 98.5, 99.0]:
    blend = (ratio/100) * bt_pred + ((100-ratio)/100) * feb_bl
    blends[f'feb_{ratio}'] = calibrate(blend)
    print(f"  {ratio}/{100-ratio} Feb: mean = {blends[f'feb_{ratio}'].mean():.2f}")

# META-ENSEMBLE 1: Average of nearby ratios
print("\n[4/5] Creating meta-ensembles...")

# Average 97.5, 98.0, 98.5 (centered on 98/2)
meta_1 = (blends['feb_97.5'] + blends['feb_98.0'] + blends['feb_98.5']) / 3
meta_1_cal = calibrate(meta_1)
print(f"  Meta-1 (avg 97.5/98/98.5): {meta_1_cal.mean():.2f}")

# Average 97, 98, 99 (wider spread)
meta_2 = (blends['feb_97.0'] + blends['feb_98.0'] + blends['feb_99.0']) / 3
meta_2_cal = calibrate(meta_2)
print(f"  Meta-2 (avg 97/98/99): {meta_2_cal.mean():.2f}")

# Weighted: more weight on 98/2 (proven best)
meta_3 = 0.5 * blends['feb_98.0'] + 0.25 * blends['feb_97.5'] + 0.25 * blends['feb_98.5']
meta_3_cal = calibrate(meta_3)
print(f"  Meta-3 (weighted 98): {meta_3_cal.mean():.2f}")

# META-ENSEMBLE 2: Different baselines
print("\n  Blends with different baselines:")

blend_recent = 0.98 * bt_pred + 0.02 * sample_submission['recent_baseline'].values
blend_all = 0.98 * bt_pred + 0.02 * sample_submission['all_baseline'].values
blend_recent_cal = calibrate(blend_recent)
blend_all_cal = calibrate(blend_all)
print(f"  98/2 Recent: {blend_recent_cal.mean():.2f}")
print(f"  98/2 All-time: {blend_all_cal.mean():.2f}")

# Ensemble of baselines
meta_baselines = (blends['feb_98.0'] + blend_recent_cal + blend_all_cal) / 3
meta_baselines_cal = calibrate(meta_baselines)
print(f"  Meta-baselines (avg Feb/Recent/All): {meta_baselines_cal.mean():.2f}")

# Weighted baseline ensemble (favor Feb)
meta_baselines_weighted = 0.6 * blends['feb_98.0'] + 0.2 * blend_recent_cal + 0.2 * blend_all_cal
meta_baselines_weighted_cal = calibrate(meta_baselines_weighted)
print(f"  Meta-baselines weighted (0.6 Feb): {meta_baselines_weighted_cal.mean():.2f}")

# META-ENSEMBLE 3: Combine ratio and baseline diversity
meta_combined = (
    0.3 * blends['feb_98.0'] +
    0.2 * blends['feb_97.5'] +
    0.2 * blends['feb_98.5'] +
    0.15 * blend_recent_cal +
    0.15 * blend_all_cal
)
meta_combined_cal = calibrate(meta_combined)
print(f"  Meta-combined (ratio + baseline diversity): {meta_combined_cal.mean():.2f}")

# SAVE
print("\n[5/5] Saving submissions...")
print("="*100)

submissions = {
    'submission.csv': meta_3_cal,  # MAIN: Weighted around 98/2
    'submission_meta1.csv': meta_1_cal,
    'submission_meta2.csv': meta_2_cal,
    'submission_meta_baselines.csv': meta_baselines_weighted_cal,
    'submission_meta_combined.csv': meta_combined_cal,
    'submission_98_2_feb.csv': blends['feb_98.0'],  # Reference
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("ENSEMBLE OF BLENDS COMPLETE")
print("="*100)
print("\nINSIGHT:")
print("  Instead of picking ONE ratio, ensemble MULTIPLE.")
print("  Errors at different ratios might cancel out!")
print("\nCurrent best: 0.35749 (98/2 + Feb)")
print("New approach: Meta-ensemble of multiple blends!")
print("="*100)
