"""
MARCH-APRIL 2023 BASELINE - The Real Opportunity!
==================================================
DISCOVERY: We have March-April 2023 data in training!
  March 2023 mean: 45.73
  April 2023 mean: 45.07

This is the SAME seasonal period we're predicting (March-April 2024)!

STRATEGY:
1. Use March-April 2023 theater-DOW patterns as baseline
2. These capture the exact seasonal patterns we need
3. Blend with breakthrough predictions

This is fundamentally smarter than using Feb baseline!

Target: 0.36+
"""

import pandas as pd
import numpy as np

print("="*100)
print("MARCH-APRIL 2023 BASELINE - Seasonal Pattern Matching")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month
booknow_visits['year'] = booknow_visits['show_date'].dt.year

sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek
sample_submission['month'] = sample_submission['show_date'].dt.month

sample_submission['breakthrough'] = breakthrough['audience_count'].values
bt_pred = sample_submission['breakthrough'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

print(f"\n[1/5] Data loaded")

# BASELINE: March-April 2023 patterns
print("\n[2/5] Computing March-April 2023 baseline (seasonal match!)...")

# Filter to March-April 2023
mar_apr_2023 = booknow_visits[
    (booknow_visits['year'] == 2023) &
    (booknow_visits['month'].isin([3, 4]))
]
print(f"  March-April 2023 data: {len(mar_apr_2023):,} rows")
print(f"  Mean: {mar_apr_2023['audience_count'].mean():.2f}")

# Theater-DOW means from March-April 2023
mar_apr_theater_dow = mar_apr_2023.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Also compute month-specific (March vs April might differ)
march_2023 = mar_apr_2023[mar_apr_2023['month'] == 3]
april_2023 = mar_apr_2023[mar_apr_2023['month'] == 4]

march_theater_dow = march_2023.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
april_theater_dow = april_2023.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

# Fallback baselines
feb_2024 = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_2024.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

all_theater_dow = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
global_mean = booknow_visits['audience_count'].mean()

# BASELINE 1: Combined March-April 2023
def get_mar_apr_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in mar_apr_theater_dow.index:
        return mar_apr_theater_dow[key]
    elif key in all_theater_dow.index:
        return all_theater_dow[key]
    return global_mean

sample_submission['mar_apr_baseline'] = sample_submission.apply(get_mar_apr_baseline, axis=1)
print(f"  Mar-Apr baseline mean: {sample_submission['mar_apr_baseline'].mean():.2f}")

# BASELINE 2: Month-specific (March 2023 for March, April 2023 for April)
def get_month_specific_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])

    if row['month'] == 3:  # March prediction → use March 2023
        if key in march_theater_dow.index:
            return march_theater_dow[key]
    else:  # April prediction → use April 2023
        if key in april_theater_dow.index:
            return april_theater_dow[key]

    # Fallback to combined Mar-Apr
    if key in mar_apr_theater_dow.index:
        return mar_apr_theater_dow[key]
    elif key in all_theater_dow.index:
        return all_theater_dow[key]
    return global_mean

sample_submission['month_specific_baseline'] = sample_submission.apply(get_month_specific_baseline, axis=1)
print(f"  Month-specific baseline mean: {sample_submission['month_specific_baseline'].mean():.2f}")

# BASELINE 3: Weighted blend (70% Mar-Apr 2023 + 30% Feb 2024)
def get_feb_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in all_theater_dow.index:
        return all_theater_dow[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_feb_baseline, axis=1)
sample_submission['blended_seasonal'] = (
    0.7 * sample_submission['mar_apr_baseline'] +
    0.3 * sample_submission['feb_baseline']
)
print(f"  Blended seasonal (70% Mar-Apr + 30% Feb): {sample_submission['blended_seasonal'].mean():.2f}")

# BLENDING STRATEGIES
print("\n[3/5] Creating blended predictions...")

results = {}

# Strategy 1: 98/2 with March-April 2023 baseline
blend_98_mar_apr = 0.98 * bt_pred + 0.02 * sample_submission['mar_apr_baseline'].values
results['mar_apr_98_2'] = calibrate(blend_98_mar_apr)
print(f"  98/2 Mar-Apr: mean = {results['mar_apr_98_2'].mean():.2f}")

# Strategy 2: 97/3 with March-April 2023 baseline
blend_97_mar_apr = 0.97 * bt_pred + 0.03 * sample_submission['mar_apr_baseline'].values
results['mar_apr_97_3'] = calibrate(blend_97_mar_apr)
print(f"  97/3 Mar-Apr: mean = {results['mar_apr_97_3'].mean():.2f}")

# Strategy 3: 95/5 with March-April 2023 baseline (more seasonal influence)
blend_95_mar_apr = 0.95 * bt_pred + 0.05 * sample_submission['mar_apr_baseline'].values
results['mar_apr_95_5'] = calibrate(blend_95_mar_apr)
print(f"  95/5 Mar-Apr: mean = {results['mar_apr_95_5'].mean():.2f}")

# Strategy 4: 98/2 with month-specific baseline
blend_98_month = 0.98 * bt_pred + 0.02 * sample_submission['month_specific_baseline'].values
results['month_specific_98_2'] = calibrate(blend_98_month)
print(f"  98/2 Month-specific: mean = {results['month_specific_98_2'].mean():.2f}")

# Strategy 5: 98/2 with blended seasonal baseline
blend_98_blended = 0.98 * bt_pred + 0.02 * sample_submission['blended_seasonal'].values
results['blended_98_2'] = calibrate(blend_98_blended)
print(f"  98/2 Blended seasonal: mean = {results['blended_98_2'].mean():.2f}")

# Strategy 6: Stronger seasonal (96/4 with Mar-Apr)
blend_96_mar_apr = 0.96 * bt_pred + 0.04 * sample_submission['mar_apr_baseline'].values
results['mar_apr_96_4'] = calibrate(blend_96_mar_apr)
print(f"  96/4 Mar-Apr: mean = {results['mar_apr_96_4'].mean():.2f}")

# COMPARISON
print("\n[4/5] Baseline comparison...")
print(f"  Feb 2024 baseline mean:     {sample_submission['feb_baseline'].mean():.2f}")
print(f"  Mar-Apr 2023 baseline mean: {sample_submission['mar_apr_baseline'].mean():.2f}")
print(f"  Difference: {sample_submission['mar_apr_baseline'].mean() - sample_submission['feb_baseline'].mean():+.2f}")
print(f"\n  Mar-Apr 2023 captures the seasonal uplift naturally!")

# Save
print("\n[5/5] Saving submissions...")

submissions = {
    'submission.csv': results['mar_apr_98_2'],  # MAIN: 98/2 with Mar-Apr 2023
    'submission_mar_apr_97_3.csv': results['mar_apr_97_3'],
    'submission_mar_apr_95_5.csv': results['mar_apr_95_5'],
    'submission_mar_apr_96_4.csv': results['mar_apr_96_4'],
    'submission_month_specific.csv': results['month_specific_98_2'],
    'submission_blended_seasonal.csv': results['blended_98_2'],
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("MARCH-APRIL BASELINE COMPLETE")
print("="*100)
print("\nKEY INSIGHT:")
print("  March-April 2023 data captures the EXACT seasonal patterns")
print("  we need for March-April 2024 predictions!")
print("\nPrevious best: 0.35749 (98/2 + Feb baseline)")
print("New approach: Use March-April 2023 baseline instead!")
print("\nThis is fundamentally smarter - matching seasons, not just recency!")
print("="*100)
