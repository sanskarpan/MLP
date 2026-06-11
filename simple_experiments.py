"""
SIMPLE EXPERIMENTS - What if the answer is simpler?
====================================================
Try simple modifications that might have big impact:
1. Different target means (is 43.85 optimal?)
2. Clipping extreme values
3. Rounding strategies
4. Monotonic transformations
"""

import pandas as pd
import numpy as np

print("="*100)
print("SIMPLE EXPERIMENTS - Looking for Hidden Improvements")
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
sample_submission['breakthrough'] = breakthrough['audience_count'].values

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

# Create 98/2 blend (our best approach)
blend_98_2 = 0.98 * bt_pred + 0.02 * feb_bl

# EXPERIMENT 1: Different target means
print("\n[2/6] Experiment 1: Optimal target mean search...")

# What's the actual mean in training data?
train_mean = booknow_visits['audience_count'].mean()
feb_mean = feb_data['audience_count'].mean()
print(f"  Full training mean: {train_mean:.3f}")
print(f"  Feb 2024 mean: {feb_mean:.3f}")
print(f"  Current blend mean: {blend_98_2.mean():.3f}")

results = {}

# Try different target means around 43.85
for target in [43.0, 43.5, 43.75, 43.85, 44.0, 44.25, 44.5, 45.0]:
    calibrated = blend_98_2 + (target - np.mean(blend_98_2))
    calibrated = np.maximum(calibrated, 0)
    results[f'mean_{target}'] = calibrated
    print(f"  Target {target}: final mean = {calibrated.mean():.2f}")

# EXPERIMENT 2: Clip extreme values
print("\n[3/6] Experiment 2: Clipping extremes...")

# What are the percentiles?
p1, p5, p95, p99 = np.percentile(blend_98_2, [1, 5, 95, 99])
print(f"  Current range: [{blend_98_2.min():.2f}, {blend_98_2.max():.2f}]")
print(f"  Percentiles: 1%={p1:.2f}, 5%={p5:.2f}, 95%={p95:.2f}, 99%={p99:.2f}")

# Clip at different levels
blend_clip_99 = np.clip(blend_98_2, p1, p99)
blend_clip_95 = np.clip(blend_98_2, p5, p95)

# Calibrate after clipping
for name, clipped in [('clip_99', blend_clip_99), ('clip_95', blend_clip_95)]:
    calibrated = clipped + (43.85 - np.mean(clipped))
    calibrated = np.maximum(calibrated, 0)
    results[name] = calibrated
    print(f"  {name}: mean = {calibrated.mean():.2f}")

# EXPERIMENT 3: Soft clipping (winsorize)
print("\n[4/6] Experiment 3: Soft clipping (winsorize)...")

# Winsorize: blend extreme values toward percentile instead of hard clip
def winsorize_soft(arr, lower_pct=1, upper_pct=99, strength=0.5):
    lower, upper = np.percentile(arr, [lower_pct, upper_pct])
    result = arr.copy()
    # Blend extremes toward bounds
    low_mask = arr < lower
    high_mask = arr > upper
    result[low_mask] = (1-strength) * arr[low_mask] + strength * lower
    result[high_mask] = (1-strength) * arr[high_mask] + strength * upper
    return result

blend_winsorize = winsorize_soft(blend_98_2, 2, 98, 0.5)
calibrated = blend_winsorize + (43.85 - np.mean(blend_winsorize))
calibrated = np.maximum(calibrated, 0)
results['winsorize'] = calibrated
print(f"  Winsorized (50% toward 2-98 pct): mean = {calibrated.mean():.2f}")

# EXPERIMENT 4: Power transform
print("\n[5/6] Experiment 4: Power transforms...")

# Sometimes a slight power transform improves predictions
blend_positive = np.maximum(blend_98_2, 0.1)  # Ensure positive for power

for power in [0.95, 0.98, 1.0, 1.02, 1.05]:
    transformed = np.power(blend_positive, power)
    # Rescale to maintain mean
    transformed = transformed * (blend_98_2.mean() / transformed.mean())
    calibrated = transformed + (43.85 - np.mean(transformed))
    calibrated = np.maximum(calibrated, 0)
    results[f'power_{power}'] = calibrated
    print(f"  Power {power}: mean = {calibrated.mean():.2f}")

# EXPERIMENT 5: Rounding
print("\n[6/6] Experiment 5: Rounding strategies...")

# Try different rounding
blend_cal = blend_98_2 + (43.85 - np.mean(blend_98_2))
blend_cal = np.maximum(blend_cal, 0)

round_1 = np.round(blend_cal, 1)  # 1 decimal
round_2 = np.round(blend_cal, 2)  # 2 decimals
round_int = np.round(blend_cal, 0)  # Integer

# Recalibrate after rounding
round_1_cal = round_1 + (43.85 - np.mean(round_1))
round_2_cal = round_2 + (43.85 - np.mean(round_2))

results['round_1dec'] = np.maximum(round_1_cal, 0)
results['round_2dec'] = np.maximum(round_2_cal, 0)

print(f"  Round to 1 decimal: mean = {results['round_1dec'].mean():.2f}")
print(f"  Round to 2 decimals: mean = {results['round_2dec'].mean():.2f}")

# Reference: standard 98/2 + 43.85 calibration
results['standard'] = np.maximum(blend_cal, 0)
print(f"\n  Reference (standard 98/2): mean = {results['standard'].mean():.2f}")

# SAVE best candidates
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': results['standard'],  # MAIN: Standard 98/2
    'submission_mean_44.csv': results['mean_44.0'],
    'submission_mean_4425.csv': results['mean_44.25'],
    'submission_winsorize.csv': results['winsorize'],
    'submission_power_098.csv': results['power_0.98'],
    'submission_power_102.csv': results['power_1.02'],
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("SIMPLE EXPERIMENTS COMPLETE")
print("="*100)
print("\nEXPERIMENTS TRIED:")
print("  1. Different target means (43.0 to 45.0)")
print("  2. Clipping extremes (99th, 95th percentiles)")
print("  3. Soft winsorization")
print("  4. Power transforms (0.95 to 1.05)")
print("  5. Rounding strategies")
print("\nCurrent best: 0.35749 (98/2 + Feb + mean 43.85)")
print("="*100)
