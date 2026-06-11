"""
EXACT 0.35749 REPLICATION
==========================
Replicate the EXACT approach from nbk_0.35749.ipynb:
- 98% breakthrough + 2% Feb baseline
- Fallback: Feb → Recent (Dec-Feb) → Global mean
- Calibrate to 43.85
"""

import pandas as pd
import numpy as np

print("="*100)
print("EXACT 0.35749 REPLICATION")
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

# Breakthrough predictions
breakthrough_predictions = breakthrough['audience_count'].values

print(f"\n[1/4] Data loaded")
print(f"  Breakthrough predictions: {len(breakthrough_predictions):,}")
print(f"  Breakthrough mean: {breakthrough_predictions.mean():.4f}")

# Compute baselines EXACTLY as in notebook
print("\n[2/4] Computing Feb-weighted baseline (exact notebook method)...")

# Feb-only baseline (most recent month)
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Recent baseline (Dec-Feb) as fallback - THIS IS THE KEY DIFFERENCE
recent_data = booknow_visits[booknow_visits['show_date'] >= '2023-12-01']
recent_theater_dow = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

global_mean = booknow_visits['audience_count'].mean()

print(f"  Feb theater-DOW entries: {len(feb_theater_dow)}")
print(f"  Recent theater-DOW entries: {len(recent_theater_dow)}")
print(f"  Global mean: {global_mean:.4f}")

def get_feb_weighted_baseline(row):
    """EXACT replication of notebook baseline logic"""
    key = (row['book_theater_id'], row['dayofweek'])

    # Try Feb baseline first (most recent = most relevant)
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    # Fall back to recent baseline (Dec-Feb)
    elif key in recent_theater_dow.index:
        return recent_theater_dow[key]
    # Fall back to global mean
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_feb_weighted_baseline, axis=1)
print(f"  Feb-weighted baseline mean: {sample_submission['feb_baseline'].mean():.4f}")

# Create blend EXACTLY as in notebook
print("\n[3/4] Creating 98/2 blend with calibration...")

BREAKTHROUGH_WEIGHT = 0.98
BASELINE_WEIGHT = 0.02
TARGET_MEAN = 43.85

# Blend
blend = BREAKTHROUGH_WEIGHT * breakthrough_predictions + BASELINE_WEIGHT * sample_submission['feb_baseline'].values

print(f"  Raw blend mean: {blend.mean():.4f}")

# Calibrate
calibration = TARGET_MEAN - blend.mean()
blend_calibrated = blend + calibration
blend_calibrated = np.maximum(blend_calibrated, 0)

print(f"  Calibration adjustment: {calibration:+.4f}")
print(f"  Final blend mean: {blend_calibrated.mean():.4f}")

# Save
print("\n[4/4] Saving submission...")

submission = pd.DataFrame({
    'ID': sample_submission['ID'],
    'audience_count': blend_calibrated
})
submission.to_csv('submission.csv', index=False)

print(f"\n  submission.csv saved")
print(f"  Mean: {blend_calibrated.mean():.4f}")
print(f"  Min: {blend_calibrated.min():.4f}")
print(f"  Max: {blend_calibrated.max():.4f}")

print("\n" + "="*100)
print("EXACT 0.35749 REPLICATION COMPLETE")
print("="*100)
print("\nThis should score exactly 0.35749")
print("="*100)
