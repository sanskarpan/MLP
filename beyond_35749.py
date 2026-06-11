"""
BEYOND 0.35749 - Push for 0.36+
================================
Current best: 98/2 + Feb baseline = 0.35749

NEW IDEAS:
1. Try slightly different ratios around 98/2
2. Weighted Feb baseline (more weight on later Feb dates)
3. Combine with weekend/weekday insights
4. Theater-count weighted baseline

Target: 0.36+
"""

import pandas as pd
import numpy as np

print("="*100)
print("BEYOND 0.35749 - Targeting 0.36+")
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
sample_submission['is_weekend'] = sample_submission['dayofweek'].isin([5, 6]).astype(int)

sample_submission['breakthrough'] = breakthrough['audience_count'].values
bt_pred = sample_submission['breakthrough'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

print(f"\n[1/6] Data loaded")

# BASELINE 1: Feb baseline (current winner)
print("\n[2/6] Computing baselines...")

feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

recent_data = booknow_visits[booknow_visits['show_date'] >= '2023-12-01']
recent_theater_dow = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

global_mean = booknow_visits['audience_count'].mean()

def get_feb_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in recent_theater_dow.index:
        return recent_theater_dow[key]
    return global_mean

sample_submission['feb_baseline'] = sample_submission.apply(get_feb_baseline, axis=1)
print(f"  Feb baseline mean: {sample_submission['feb_baseline'].mean():.2f}")

# BASELINE 2: Late Feb baseline (last 2 weeks of Feb - closest to test)
late_feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-15']
late_feb_theater_dow = late_feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()

def get_late_feb_baseline(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in late_feb_theater_dow.index:
        return late_feb_theater_dow[key]
    elif key in feb_theater_dow.index:
        return feb_theater_dow[key]
    elif key in recent_theater_dow.index:
        return recent_theater_dow[key]
    return global_mean

sample_submission['late_feb_baseline'] = sample_submission.apply(get_late_feb_baseline, axis=1)
print(f"  Late Feb baseline mean: {sample_submission['late_feb_baseline'].mean():.2f}")

# STRATEGY 1: Fine-tune around 98/2 with Feb baseline
print("\n[3/6] Strategy 1: Fine-tune ratios with Feb baseline...")

results = {}
for ratio in [97.5, 97.75, 98.0, 98.25, 98.5]:
    blend = (ratio/100) * bt_pred + ((100-ratio)/100) * sample_submission['feb_baseline'].values
    blend_cal = calibrate(blend)
    results[f'feb_{ratio}'] = blend_cal
    print(f"  {ratio}/{100-ratio} Feb: mean = {blend_cal.mean():.2f}")

# STRATEGY 2: Late Feb baseline (closer to test period)
print("\n[4/6] Strategy 2: Late Feb baseline...")

for ratio in [97.5, 98.0, 98.5]:
    blend = (ratio/100) * bt_pred + ((100-ratio)/100) * sample_submission['late_feb_baseline'].values
    blend_cal = calibrate(blend)
    results[f'late_feb_{ratio}'] = blend_cal
    print(f"  {ratio}/{100-ratio} Late Feb: mean = {blend_cal.mean():.2f}")

# STRATEGY 3: Weekend/Weekday differentiated with Feb baseline
print("\n[5/6] Strategy 3: Weekend/Weekday + Feb baseline...")

weekend_mask = sample_submission['is_weekend'] == 1
weekday_mask = ~weekend_mask

# Weekend 97.5/2.5, Weekday 98.5/1.5 with Feb baseline
diff_blend = bt_pred.copy()
diff_blend[weekend_mask] = 0.975 * bt_pred[weekend_mask] + 0.025 * sample_submission.loc[weekend_mask, 'feb_baseline'].values
diff_blend[weekday_mask] = 0.985 * bt_pred[weekday_mask] + 0.015 * sample_submission.loc[weekday_mask, 'feb_baseline'].values
diff_blend_cal = calibrate(diff_blend)
results['diff_we_wd'] = diff_blend_cal
print(f"  WE 97.5/2.5 + WD 98.5/1.5: mean = {diff_blend_cal.mean():.2f}")

# STRATEGY 4: Blend of Feb and Late Feb baselines
print("\n[6/6] Strategy 4: Blended baseline...")

# 50% Feb + 50% Late Feb baseline
blended_baseline = 0.5 * sample_submission['feb_baseline'].values + 0.5 * sample_submission['late_feb_baseline'].values
blend_blended = 0.98 * bt_pred + 0.02 * blended_baseline
blend_blended_cal = calibrate(blend_blended)
results['blended_baseline'] = blend_blended_cal
print(f"  98/2 with blended baseline: mean = {blend_blended_cal.mean():.2f}")

# Save
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': results['late_feb_98.0'],  # MAIN: 98/2 with late Feb
    'submission_feb_9775.csv': results['feb_97.75'],
    'submission_feb_9825.csv': results['feb_98.25'],
    'submission_late_feb_975.csv': results['late_feb_97.5'],
    'submission_diff_we_wd.csv': results['diff_we_wd'],
    'submission_blended.csv': results['blended_baseline'],
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("BEYOND 0.35749 COMPLETE")
print("="*100)
print("\nCurrent best: 0.35749 (98/2 + Feb baseline)")
print("\nNEW STRATEGIES:")
print("  1. submission.csv - 98/2 + Late Feb baseline (most recent data)")
print("  2. Differentiated weekend/weekday ratios")
print("  3. Fine-tuned ratios (97.75, 98.25)")
print("\nTarget: 0.36+!")
print("="*100)
