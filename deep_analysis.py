"""
DEEP ANALYSIS - Find Hidden Patterns
=====================================
Analyze breakthrough predictions deeply to find
systematic patterns that could be exploited.
"""

import pandas as pd
import numpy as np

print("="*100)
print("DEEP ANALYSIS - Finding Hidden Patterns")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month
booknow_visits['day'] = booknow_visits['show_date'].dt.day
booknow_visits['week_of_month'] = (booknow_visits['show_date'].dt.day - 1) // 7 + 1

sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek
sample_submission['month'] = sample_submission['show_date'].dt.month
sample_submission['day'] = sample_submission['show_date'].dt.day
sample_submission['week_of_month'] = (sample_submission['show_date'].dt.day - 1) // 7 + 1
sample_submission['breakthrough'] = breakthrough['audience_count'].values

bt_pred = sample_submission['breakthrough'].values

print(f"\n[1/5] Data loaded")

# Compute baselines
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
recent_theater_dow = booknow_visits[booknow_visits['show_date'] >= '2023-12-01'].groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
global_mean = booknow_visits['audience_count'].mean()

def get_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in recent_theater_dow.index:
        return recent_theater_dow[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_baseline, axis=1)

# ANALYSIS 1: Breakthrough vs Baseline differences by segment
print("\n[2/5] Analyzing breakthrough vs baseline by segment...")

diff = bt_pred - sample_submission['feb_baseline'].values
print(f"  Overall difference: BT mean={bt_pred.mean():.2f}, BL mean={sample_submission['feb_baseline'].mean():.2f}")
print(f"  Mean diff: {diff.mean():+.2f}, Std: {diff.std():.2f}")

print("\n  By Day of Week:")
for dow in range(7):
    mask = sample_submission['dayofweek'] == dow
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"    {day_names[dow]}: BT={bt_pred[mask].mean():.2f}, BL={sample_submission.loc[mask, 'feb_baseline'].mean():.2f}, diff={diff[mask].mean():+.2f}")

print("\n  By Month:")
for month in [3, 4]:
    mask = sample_submission['month'] == month
    name = 'March' if month == 3 else 'April'
    print(f"    {name}: BT={bt_pred[mask].mean():.2f}, BL={sample_submission.loc[mask, 'feb_baseline'].mean():.2f}, diff={diff[mask].mean():+.2f}")

print("\n  By Week of Month:")
for wom in range(1, 6):
    mask = sample_submission['week_of_month'] == wom
    if mask.sum() > 0:
        print(f"    Week {wom}: BT={bt_pred[mask].mean():.2f}, BL={sample_submission.loc[mask, 'feb_baseline'].mean():.2f}, diff={diff[mask].mean():+.2f}, count={mask.sum()}")

# ANALYSIS 2: Historical patterns by month
print("\n[3/5] Analyzing historical patterns...")

monthly_means = booknow_visits.groupby('month')['audience_count'].mean()
print("\n  Historical monthly means:")
for month in range(1, 13):
    if month in monthly_means.index:
        print(f"    Month {month:2d}: {monthly_means[month]:.2f}")

# March and April specifically
mar_mean = monthly_means.get(3, global_mean)
apr_mean = monthly_means.get(4, global_mean)
feb_mean = monthly_means.get(2, global_mean)
print(f"\n  Feb→Mar change: {mar_mean/feb_mean:.3f}x")
print(f"  Feb→Apr change: {apr_mean/feb_mean:.3f}x")

# ANALYSIS 3: Check if Feb baseline covers all test theater-DOW combinations
print("\n[4/5] Analyzing baseline coverage...")

test_keys = set(zip(sample_submission['book_theater_id'], sample_submission['dayofweek']))
feb_keys = set(feb_theater_dow.index)
recent_keys = set(recent_theater_dow.index)

covered_by_feb = len(test_keys & feb_keys)
covered_by_recent = len(test_keys & recent_keys)
not_covered = len(test_keys - recent_keys)

print(f"  Test theater-DOW combinations: {len(test_keys)}")
print(f"  Covered by Feb baseline: {covered_by_feb} ({100*covered_by_feb/len(test_keys):.1f}%)")
print(f"  Covered by Recent baseline: {covered_by_recent} ({100*covered_by_recent/len(test_keys):.1f}%)")
print(f"  Not covered (using global mean): {not_covered} ({100*not_covered/len(test_keys):.1f}%)")

# ANALYSIS 4: Distribution analysis
print("\n[5/5] Distribution analysis...")

p5, p25, p50, p75, p95 = np.percentile(bt_pred, [5, 25, 50, 75, 95])
print(f"  Breakthrough percentiles: 5%={p5:.1f}, 25%={p25:.1f}, 50%={p50:.1f}, 75%={p75:.1f}, 95%={p95:.1f}")

p5, p25, p50, p75, p95 = np.percentile(sample_submission['feb_baseline'], [5, 25, 50, 75, 95])
print(f"  Baseline percentiles:     5%={p5:.1f}, 25%={p25:.1f}, 50%={p50:.1f}, 75%={p75:.1f}, 95%={p95:.1f}")

# KEY INSIGHT: Apply monthly multiplier
print("\n" + "="*100)
print("KEY INSIGHT: Monthly Adjustment")
print("="*100)

# Historical pattern shows March-April is higher than February
# The Feb baseline might be systematically LOW for the test period

TARGET_MEAN = 43.85

def calibrate(pred):
    return np.maximum(pred + (TARGET_MEAN - np.mean(pred)), 0)

# Strategy: Scale Feb baseline by monthly multiplier before blending
mar_mult = mar_mean / feb_mean
apr_mult = apr_mean / feb_mean

feb_bl = sample_submission['feb_baseline'].values
scaled_bl = np.where(
    sample_submission['month'] == 3,
    feb_bl * mar_mult,
    feb_bl * apr_mult
)

print(f"\n  March multiplier: {mar_mult:.4f}")
print(f"  April multiplier: {apr_mult:.4f}")
print(f"  Original baseline mean: {feb_bl.mean():.2f}")
print(f"  Scaled baseline mean: {scaled_bl.mean():.2f}")

# Create blend with scaled baseline
blend_scaled = 0.98 * bt_pred + 0.02 * scaled_bl
blend_scaled_cal = calibrate(blend_scaled)

# Standard for comparison
blend_standard = 0.98 * bt_pred + 0.02 * feb_bl
blend_standard_cal = calibrate(blend_standard)

print(f"\n  Standard blend mean: {blend_standard_cal.mean():.2f}")
print(f"  Scaled blend mean: {blend_scaled_cal.mean():.2f}")

# Save
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': blend_scaled_cal,  # MAIN: Scaled baseline
    'submission_standard.csv': blend_standard_cal,
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.4f}")

print("\n" + "="*100)
print("DEEP ANALYSIS COMPLETE")
print("="*100)
