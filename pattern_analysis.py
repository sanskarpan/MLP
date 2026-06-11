"""
PATTERN ANALYSIS - What Does Baseline Add?
==========================================
Analyze WHY Feb baseline helps and enhance those patterns.

Key question: What information does the 2% baseline contribute
that improves the score? Can we amplify that?
"""

import pandas as pd
import numpy as np

print("="*100)
print("PATTERN ANALYSIS - Understanding Baseline Contribution")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month
booknow_visits['week_of_month'] = (booknow_visits['show_date'].dt.day - 1) // 7 + 1

sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek
sample_submission['month'] = sample_submission['show_date'].dt.month
sample_submission['week_of_month'] = (sample_submission['show_date'].dt.day - 1) // 7 + 1
sample_submission['is_weekend'] = sample_submission['dayofweek'].isin([5, 6]).astype(int)
sample_submission['breakthrough'] = breakthrough['audience_count'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

print(f"\n[1/6] Data loaded")

# Feb baseline
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
all_theater_dow = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
global_mean = booknow_visits['audience_count'].mean()

def get_feb_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in all_theater_dow.index:
        return all_theater_dow[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_feb_baseline, axis=1)

bt_pred = sample_submission['breakthrough'].values
feb_bl = sample_submission['feb_baseline'].values

# ANALYSIS: What is the correction?
print("\n[2/6] Analyzing baseline correction patterns...")

correction = feb_bl - bt_pred
print(f"  Mean correction: {correction.mean():.3f}")
print(f"  Std correction: {correction.std():.3f}")
print(f"  Range: [{correction.min():.2f}, {correction.max():.2f}]")

# Where does baseline push UP vs DOWN?
push_up_mask = correction > 0
push_down_mask = correction < 0
print(f"  Push UP count: {push_up_mask.sum()} ({100*push_up_mask.mean():.1f}%)")
print(f"  Push DOWN count: {push_down_mask.sum()} ({100*push_down_mask.mean():.1f}%)")

# Correlation with features
print("\n  Correction by DOW:")
for dow in range(7):
    mask = sample_submission['dayofweek'] == dow
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"    {day_names[dow]}: mean correction = {correction[mask].mean():+.3f}")

print("\n  Correction by Month:")
for month in [3, 4]:
    mask = sample_submission['month'] == month
    print(f"    {'March' if month == 3 else 'April'}: mean correction = {correction[mask].mean():+.3f}")

# ENHANCED BASELINES
print("\n[3/6] Creating enhanced baselines...")

# Week-of-month aware baseline
theater_dow_wom = booknow_visits.groupby(['book_theater_id', 'dayofweek', 'week_of_month'])['audience_count'].mean()

def get_wom_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'], row['week_of_month'])
    if key in theater_dow_wom.index:
        return theater_dow_wom[key]
    key2 = (row['book_theater_id'], row['dayofweek'])
    if key2 in feb_theater_dow.index:
        return feb_theater_dow[key2]
    if key2 in all_theater_dow.index:
        return all_theater_dow[key2]
    return global_mean

sample_submission['wom_baseline'] = sample_submission.apply(get_wom_baseline, axis=1)
print(f"  Week-of-month baseline mean: {sample_submission['wom_baseline'].mean():.2f}")

# Trend-adjusted baseline (Feb is colder, Mar-Apr warmer - maybe more visitors)
# Compute monthly multipliers from historical data
monthly_means = booknow_visits.groupby('month')['audience_count'].mean()
feb_mean_hist = monthly_means.get(2, global_mean)
mar_mean_hist = monthly_means.get(3, global_mean)
apr_mean_hist = monthly_means.get(4, global_mean)

mar_multiplier = mar_mean_hist / feb_mean_hist
apr_multiplier = apr_mean_hist / feb_mean_hist

print(f"  March multiplier vs Feb: {mar_multiplier:.3f}")
print(f"  April multiplier vs Feb: {apr_multiplier:.3f}")

# Apply multiplier to Feb baseline
sample_submission['trend_adjusted_baseline'] = sample_submission.apply(
    lambda r: r['feb_baseline'] * (mar_multiplier if r['month'] == 3 else apr_multiplier), axis=1
)
print(f"  Trend-adjusted baseline mean: {sample_submission['trend_adjusted_baseline'].mean():.2f}")

# DOW-specific baseline (use all historical DOW data)
dow_means = booknow_visits.groupby('dayofweek')['audience_count'].mean()
sample_submission['dow_only_baseline'] = sample_submission['dayofweek'].map(dow_means)
print(f"  DOW-only baseline mean: {sample_submission['dow_only_baseline'].mean():.2f}")

# BLENDING WITH ENHANCED BASELINES
print("\n[4/6] Testing enhanced baseline blends...")

results = {}

# Standard 98/2 with Feb (current best)
blend_feb = 0.98 * bt_pred + 0.02 * feb_bl
results['feb_98_2'] = calibrate(blend_feb)
print(f"  98/2 Feb (reference): {results['feb_98_2'].mean():.2f}")

# 98/2 with week-of-month baseline
blend_wom = 0.98 * bt_pred + 0.02 * sample_submission['wom_baseline'].values
results['wom_98_2'] = calibrate(blend_wom)
print(f"  98/2 Week-of-month: {results['wom_98_2'].mean():.2f}")

# 98/2 with trend-adjusted baseline
blend_trend = 0.98 * bt_pred + 0.02 * sample_submission['trend_adjusted_baseline'].values
results['trend_98_2'] = calibrate(blend_trend)
print(f"  98/2 Trend-adjusted: {results['trend_98_2'].mean():.2f}")

# Ensemble of baselines: 50% Feb + 50% trend-adjusted
ensemble_baseline = 0.5 * feb_bl + 0.5 * sample_submission['trend_adjusted_baseline'].values
blend_ensemble = 0.98 * bt_pred + 0.02 * ensemble_baseline
results['ensemble_bl_98_2'] = calibrate(blend_ensemble)
print(f"  98/2 Ensemble baseline: {results['ensemble_bl_98_2'].mean():.2f}")

# STRATEGIC ADJUSTMENTS
print("\n[5/6] Strategic adjustments...")

# Idea: What if we AMPLIFY the correction where it's strong?
# Find where breakthrough and baseline AGREE vs DISAGREE

agreement = np.abs(bt_pred - feb_bl) / (feb_bl + 1)  # Relative disagreement
high_agree_mask = agreement < np.percentile(agreement, 25)  # Most agreement
high_disagree_mask = agreement > np.percentile(agreement, 75)  # Most disagreement

print(f"  High agreement predictions: {high_agree_mask.sum()}")
print(f"  High disagreement predictions: {high_disagree_mask.sum()}")

# Where they agree, trust breakthrough more (99.5%)
# Where they disagree, lean on baseline more (96%)
blend_agree = np.zeros_like(bt_pred)
blend_agree[high_agree_mask] = 0.995 * bt_pred[high_agree_mask] + 0.005 * feb_bl[high_agree_mask]
blend_agree[high_disagree_mask] = 0.96 * bt_pred[high_disagree_mask] + 0.04 * feb_bl[high_disagree_mask]
mid_mask = ~high_agree_mask & ~high_disagree_mask
blend_agree[mid_mask] = 0.98 * bt_pred[mid_mask] + 0.02 * feb_bl[mid_mask]
results['agreement_adaptive'] = calibrate(blend_agree)
print(f"  Agreement-adaptive: {results['agreement_adaptive'].mean():.2f}")

# DOW-specific blend ratios (weekends might need different treatment)
weekend_mask = sample_submission['is_weekend'] == 1
blend_dow = np.zeros_like(bt_pred)
# Weekends: 97/3 (more variable, need more baseline)
blend_dow[weekend_mask] = 0.97 * bt_pred[weekend_mask] + 0.03 * feb_bl[weekend_mask]
# Weekdays: 98.5/1.5 (more stable)
blend_dow[~weekend_mask] = 0.985 * bt_pred[~weekend_mask] + 0.015 * feb_bl[~weekend_mask]
results['dow_adaptive'] = calibrate(blend_dow)
print(f"  DOW-adaptive (WE 97/3, WD 98.5/1.5): {results['dow_adaptive'].mean():.2f}")

# Month-specific with trend (March vs April might need different treatment)
march_mask = sample_submission['month'] == 3
april_mask = sample_submission['month'] == 4
blend_month = np.zeros_like(bt_pred)
# March: 98/2 (closer to training, more confident)
blend_month[march_mask] = 0.98 * bt_pred[march_mask] + 0.02 * sample_submission.loc[march_mask, 'trend_adjusted_baseline'].values
# April: 97.5/2.5 (further from training, use more baseline)
blend_month[april_mask] = 0.975 * bt_pred[april_mask] + 0.025 * sample_submission.loc[april_mask, 'trend_adjusted_baseline'].values
results['month_trend'] = calibrate(blend_month)
print(f"  Month-trend adaptive: {results['month_trend'].mean():.2f}")

# SAVE
print("\n[6/6] Saving submissions...")
print("="*100)

submissions = {
    'submission.csv': results['agreement_adaptive'],  # MAIN: Agreement-adaptive
    'submission_wom.csv': results['wom_98_2'],
    'submission_trend.csv': results['trend_98_2'],
    'submission_ensemble_bl.csv': results['ensemble_bl_98_2'],
    'submission_dow_adaptive.csv': results['dow_adaptive'],
    'submission_month_trend.csv': results['month_trend'],
    'submission_feb_ref.csv': results['feb_98_2'],  # Reference
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("PATTERN ANALYSIS COMPLETE")
print("="*100)
print("\nKEY INSIGHTS:")
print(f"  - Baseline pushes predictions UP {100*push_up_mask.mean():.0f}% of the time")
print(f"  - March multiplier: {mar_multiplier:.3f}, April multiplier: {apr_multiplier:.3f}")
print(f"  - Agreement-adaptive: Trust BT where it agrees with baseline")
print("\nCurrent best: 0.35749 (98/2 + Feb)")
print("New approaches: Enhanced baselines & adaptive strategies!")
print("="*100)
