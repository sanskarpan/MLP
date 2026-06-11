"""
SMART BLEND - New Strategies
============================
LEARNINGS:
  Pure 98/2 → 0.35703 ✓ BEST
  Smoothed 98/2 → 0.35668 (smoothing hurts!)
  99/1 → 0.35702 (too light)
  97/3 → 0.35700 (too heavy)

NEW IDEAS:
1. Weekend vs Weekday different blend ratios
2. High-volume vs Low-volume theaters different ratios
3. Weighted recent baseline (more weight on Feb)
4. Month-specific adjustments for March vs April

Target: Beat 0.35703!
"""

import pandas as pd
import numpy as np

print("="*100)
print("SMART BLEND - Context-Aware Strategies")
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
sample_submission['is_weekend'] = sample_submission['dayofweek'].isin([5, 6]).astype(int)

sample_submission['breakthrough'] = breakthrough['audience_count'].values
bt_pred = sample_submission['breakthrough'].copy()

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - pred.mean())
    return np.maximum(calibrated, 0)

print(f"\n[1/6] Data loaded")
print(f"  Weekend predictions: {sample_submission['is_weekend'].sum()}")
print(f"  Weekday predictions: {(~sample_submission['is_weekend'].astype(bool)).sum()}")

# Baselines
print("\n[2/6] Computing baselines...")

# Recent baseline (Dec-Feb)
recent_data = booknow_visits[booknow_visits['show_date'] >= '2023-12-01']
recent_theater_dow = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Feb-only baseline (most recent month)
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

global_mean = booknow_visits['audience_count'].mean()

def get_baseline(row, baseline_dict):
    key = (row['book_theater_id'], row['dayofweek'])
    return baseline_dict.get(key, global_mean)

sample_submission['recent_baseline'] = sample_submission.apply(
    lambda r: get_baseline(r, recent_theater_dow), axis=1
)
sample_submission['feb_baseline'] = sample_submission.apply(
    lambda r: get_baseline(r, feb_theater_dow), axis=1
)

# Theater volume stats
theater_volume = booknow_visits.groupby('book_theater_id')['audience_count'].mean()
median_volume = theater_volume.median()
sample_submission['theater_volume'] = sample_submission['book_theater_id'].map(theater_volume)
sample_submission['is_high_volume'] = (sample_submission['theater_volume'] > median_volume).astype(int)

print(f"  High-volume theaters: {sample_submission['is_high_volume'].sum()}")
print(f"  Low-volume theaters: {(~sample_submission['is_high_volume'].astype(bool)).sum()}")

# STRATEGY 1: Weekend vs Weekday different ratios
print("\n[3/6] Strategy 1: Weekend vs Weekday blend...")

# Hypothesis: Weekends might need more baseline (more variable)
# Weekdays might need less baseline (more predictable)
weekend_mask = sample_submission['is_weekend'] == 1
weekday_mask = ~weekend_mask

blend_weekend_weekday = bt_pred.copy()
# Weekend: 97/3 (slightly more baseline)
blend_weekend_weekday[weekend_mask] = (
    0.97 * bt_pred[weekend_mask] +
    0.03 * sample_submission.loc[weekend_mask, 'recent_baseline']
)
# Weekday: 99/1 (less baseline)
blend_weekend_weekday[weekday_mask] = (
    0.99 * bt_pred[weekday_mask] +
    0.01 * sample_submission.loc[weekday_mask, 'recent_baseline']
)
blend_weekend_weekday_cal = calibrate(blend_weekend_weekday)
print(f"  Weekend 97/3 + Weekday 99/1: mean = {blend_weekend_weekday_cal.mean():.2f}")

# STRATEGY 2: High vs Low volume theaters
print("\n[4/6] Strategy 2: Volume-based blend...")

high_vol_mask = sample_submission['is_high_volume'] == 1
low_vol_mask = ~high_vol_mask

blend_volume = bt_pred.copy()
# High volume: 99/1 (breakthrough is confident)
blend_volume[high_vol_mask] = (
    0.99 * bt_pred[high_vol_mask] +
    0.01 * sample_submission.loc[high_vol_mask, 'recent_baseline']
)
# Low volume: 97/3 (need more baseline help)
blend_volume[low_vol_mask] = (
    0.97 * bt_pred[low_vol_mask] +
    0.03 * sample_submission.loc[low_vol_mask, 'recent_baseline']
)
blend_volume_cal = calibrate(blend_volume)
print(f"  High-vol 99/1 + Low-vol 97/3: mean = {blend_volume_cal.mean():.2f}")

# STRATEGY 3: Feb-weighted baseline (most recent = most relevant)
print("\n[5/6] Strategy 3: Feb-weighted baseline...")

# Use Feb baseline where available, fall back to recent
sample_submission['weighted_baseline'] = sample_submission.apply(
    lambda r: r['feb_baseline'] if not pd.isna(r['feb_baseline']) and r['feb_baseline'] > 0
              else r['recent_baseline'], axis=1
)

blend_feb = 0.98 * bt_pred + 0.02 * sample_submission['weighted_baseline']
blend_feb_cal = calibrate(blend_feb)
print(f"  98/2 with Feb baseline: mean = {blend_feb_cal.mean():.2f}")

# STRATEGY 4: Month-specific (March vs April might differ)
print("\n[6/6] Strategy 4: Month-specific blend...")

march_mask = sample_submission['month'] == 3
april_mask = sample_submission['month'] == 4

blend_month = bt_pred.copy()
# March: 98/2 (proven best)
blend_month[march_mask] = (
    0.98 * bt_pred[march_mask] +
    0.02 * sample_submission.loc[march_mask, 'recent_baseline']
)
# April: 97.5/2.5 (slightly more baseline as we get further from training)
blend_month[april_mask] = (
    0.975 * bt_pred[april_mask] +
    0.025 * sample_submission.loc[april_mask, 'recent_baseline']
)
blend_month_cal = calibrate(blend_month)
print(f"  March 98/2 + April 97.5/2.5: mean = {blend_month_cal.mean():.2f}")

# STRATEGY 5: Combined smart blend
print("\n[BONUS] Combined smart blend...")

# Combine insights: weekend/weekday + volume
smart_blend = bt_pred.copy()

# 4 segments:
# High-vol weekend: 98/2
# High-vol weekday: 99/1
# Low-vol weekend: 96/4
# Low-vol weekday: 98/2

hv_we = high_vol_mask & weekend_mask
hv_wd = high_vol_mask & weekday_mask
lv_we = low_vol_mask & weekend_mask
lv_wd = low_vol_mask & weekday_mask

smart_blend[hv_we] = 0.98 * bt_pred[hv_we] + 0.02 * sample_submission.loc[hv_we, 'recent_baseline']
smart_blend[hv_wd] = 0.99 * bt_pred[hv_wd] + 0.01 * sample_submission.loc[hv_wd, 'recent_baseline']
smart_blend[lv_we] = 0.96 * bt_pred[lv_we] + 0.04 * sample_submission.loc[lv_we, 'recent_baseline']
smart_blend[lv_wd] = 0.98 * bt_pred[lv_wd] + 0.02 * sample_submission.loc[lv_wd, 'recent_baseline']

smart_blend_cal = calibrate(smart_blend)
print(f"  4-segment smart blend: mean = {smart_blend_cal.mean():.2f}")

# Save
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': blend_feb_cal,  # MAIN: 98/2 with Feb baseline
    'submission_weekend_weekday.csv': blend_weekend_weekday_cal,
    'submission_volume.csv': blend_volume_cal,
    'submission_month.csv': blend_month_cal,
    'submission_smart.csv': smart_blend_cal,
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("SMART BLEND COMPLETE")
print("="*100)
print("\nSTRATEGIES:")
print("  1. submission.csv - 98/2 with Feb-weighted baseline")
print("  2. submission_weekend_weekday.csv - Different ratios by day type")
print("  3. submission_volume.csv - Different ratios by theater volume")
print("  4. submission_month.csv - Different ratios by month")
print("  5. submission_smart.csv - Combined 4-segment approach")
print("\nTarget: Beat 0.35703!")
print("="*100)
