"""
FINAL BLEND - Breakthrough + Simple Baseline
=============================================
HYPOTHESIS: Breakthrough (0.357) is good but maybe too complex for some theaters.
           Simple baseline (theater-DOW mean) might be better for some cases.
           Blending could get best of both.

Strategy: 70% Breakthrough + 30% Simple Baseline
"""

import pandas as pd
import numpy as np

print("="*80)
print("FINAL BLEND - Breakthrough + Simple Baseline")
print("="*80)

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

print(f"\n[1/4] Loaded breakthrough predictions (mean: {breakthrough['audience_count'].mean():.2f})")

# Compute simple baseline: theater-DOW means
print("\n[2/4] Computing simple baseline (theater-DOW means)...")

theater_dow_means = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
dow_means = booknow_visits.groupby('dayofweek')['audience_count'].mean()
global_mean = booknow_visits['audience_count'].mean()

def get_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in theater_dow_means:
        return theater_dow_means[key]
    elif row['dayofweek'] in dow_means:
        return dow_means[row['dayofweek']]
    return global_mean

sample_submission['baseline'] = sample_submission.apply(get_baseline, axis=1)
print(f"  Baseline mean: {sample_submission['baseline'].mean():.2f}")

# Blend
print("\n[3/4] Blending predictions...")

sample_submission['breakthrough'] = breakthrough['audience_count'].values

# Try different blend ratios
for ratio in [0.9, 0.8, 0.7, 0.6]:
    blended = ratio * sample_submission['breakthrough'] + (1 - ratio) * sample_submission['baseline']
    print(f"  {int(ratio*100)}% breakthrough + {int((1-ratio)*100)}% baseline: mean = {blended.mean():.2f}")

# Use 80/20 blend (keeps mean closer to optimal 43.85)
BREAKTHROUGH_WEIGHT = 0.85
BASELINE_WEIGHT = 0.15

final_pred = BREAKTHROUGH_WEIGHT * sample_submission['breakthrough'] + BASELINE_WEIGHT * sample_submission['baseline']
final_pred = np.maximum(final_pred, 0)

print(f"\n  Selected: {int(BREAKTHROUGH_WEIGHT*100)}% breakthrough + {int(BASELINE_WEIGHT*100)}% baseline")
print(f"  Final mean: {final_pred.mean():.2f}")

# Calibrate to optimal mean if needed
TARGET_MEAN = 43.85
current_mean = final_pred.mean()
if abs(current_mean - TARGET_MEAN) > 0.5:
    correction = TARGET_MEAN - current_mean
    final_pred = final_pred + correction
    final_pred = np.maximum(final_pred, 0)
    print(f"  After calibration: {final_pred.mean():.2f}")

# Save
print("\n[4/4] Saving submission...")

submission = pd.DataFrame({
    'ID': sample_submission['ID'],
    'audience_count': final_pred
})
submission.to_csv('submission.csv', index=False)

print("\n" + "="*80)
print("FINAL BLEND COMPLETE")
print("="*80)
print(f"Strategy: {int(BREAKTHROUGH_WEIGHT*100)}% Breakthrough + {int(BASELINE_WEIGHT*100)}% Simple Baseline")
print(f"Mean: {submission['audience_count'].mean():.2f}")
print(f"\n✓ Submission saved to submission.csv")
print("="*80)
