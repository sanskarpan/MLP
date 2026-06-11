"""
BOLD ENSEMBLE - Combine Multiple Winning Strategies
====================================================
Instead of picking ONE approach, ensemble MULTIPLE:
1. Standard 98/2 blend (0.35749 baseline)
2. Selective replacement (fix obvious errors)
3. Confidence-weighted (adaptive ratios)

The errors from different approaches might cancel out!
"""

import pandas as pd
import numpy as np

print("="*100)
print("BOLD ENSEMBLE - Multiple Strategies Combined")
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

bt_pred = breakthrough['audience_count'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

print(f"\n[1/6] Data loaded")

# Compute baselines
print("\n[2/6] Computing baselines...")

feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

recent_data = booknow_visits[booknow_visits['show_date'] >= '2023-12-01']
recent_theater_dow = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

theater_stats = booknow_visits.groupby('book_theater_id').agg({
    'audience_count': ['mean', 'std']
}).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std']

global_mean = booknow_visits['audience_count'].mean()

def get_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in recent_theater_dow.index:
        return recent_theater_dow[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_baseline, axis=1)
sample_submission = sample_submission.merge(theater_stats, on='book_theater_id', how='left')
sample_submission['th_mean'] = sample_submission['th_mean'].fillna(global_mean)
sample_submission['th_std'] = sample_submission['th_std'].fillna(booknow_visits['audience_count'].std())

feb_bl = sample_submission['feb_baseline'].values

# STRATEGY 1: Standard 98/2 (baseline approach)
print("\n[3/6] Strategy 1: Standard 98/2 blend...")
pred_1 = 0.98 * bt_pred + 0.02 * feb_bl
pred_1_cal = calibrate(pred_1)
print(f"  Standard 98/2: {pred_1_cal.mean():.4f}")

# STRATEGY 2: Selective replacement for outliers
print("\n[4/6] Strategy 2: Selective replacement...")
zscore = np.abs(bt_pred - sample_submission['th_mean'].values) / (sample_submission['th_std'].values + 1)
outlier_mask = zscore > 2.5  # Predictions far from theater mean

pred_2 = bt_pred.copy()
pred_2[outlier_mask] = 0.5 * bt_pred[outlier_mask] + 0.5 * feb_bl[outlier_mask]  # Blend outliers more
pred_2 = 0.98 * pred_2 + 0.02 * feb_bl
pred_2_cal = calibrate(pred_2)
print(f"  Selective replacement ({outlier_mask.sum()} outliers fixed): {pred_2_cal.mean():.4f}")

# STRATEGY 3: Confidence-adaptive weights
print("\n[5/6] Strategy 3: Confidence-adaptive...")
# Confidence based on how close prediction is to historical range
confidence = 1 - np.minimum(zscore / 3, 1)  # Higher z-score = lower confidence
bt_weight = 0.96 + 0.03 * confidence  # Range: 0.96 to 0.99
bl_weight = 1 - bt_weight

pred_3 = bt_weight * bt_pred + bl_weight * feb_bl
pred_3_cal = calibrate(pred_3)
print(f"  Confidence-adaptive: {pred_3_cal.mean():.4f}")

# BOLD ENSEMBLES
print("\n[6/6] Creating bold ensembles...")

# Ensemble A: Equal weight
ensemble_a = (pred_1_cal + pred_2_cal + pred_3_cal) / 3
ensemble_a_cal = calibrate(ensemble_a)
print(f"  Ensemble A (equal): {ensemble_a_cal.mean():.4f}")

# Ensemble B: Favor standard (proven)
ensemble_b = 0.6 * pred_1_cal + 0.2 * pred_2_cal + 0.2 * pred_3_cal
ensemble_b_cal = calibrate(ensemble_b)
print(f"  Ensemble B (60% standard): {ensemble_b_cal.mean():.4f}")

# Ensemble C: Favor replacement (might fix errors)
ensemble_c = 0.3 * pred_1_cal + 0.5 * pred_2_cal + 0.2 * pred_3_cal
ensemble_c_cal = calibrate(ensemble_c)
print(f"  Ensemble C (50% replacement): {ensemble_c_cal.mean():.4f}")

# Ensemble D: Just standard + replacement
ensemble_d = 0.5 * pred_1_cal + 0.5 * pred_2_cal
ensemble_d_cal = calibrate(ensemble_d)
print(f"  Ensemble D (50/50 std/repl): {ensemble_d_cal.mean():.4f}")

# SAVE
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': ensemble_b_cal,  # MAIN: Favor proven standard
    'submission_standard.csv': pred_1_cal,
    'submission_replacement.csv': pred_2_cal,
    'submission_adaptive.csv': pred_3_cal,
    'submission_equal_ensemble.csv': ensemble_a_cal,
    'submission_5050.csv': ensemble_d_cal,
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.4f}")

print("\n" + "="*100)
print("BOLD ENSEMBLE COMPLETE")
print("="*100)
print("\nKEY INSIGHT:")
print("  Ensemble multiple approaches - errors might cancel out!")
print("\nCurrent best: 0.35749")
print("="*100)
