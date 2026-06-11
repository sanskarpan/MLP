"""
DOW-SPECIFIC CORRECTION
========================
The analysis shows breakthrough overestimates differently by day:
- Tuesday: +6.11 (most overestimated)
- Sunday: +0.53 (most accurate)

Use DOW-specific blend ratios to correct this!
"""

import pandas as pd
import numpy as np

print("="*100)
print("DOW-SPECIFIC CORRECTION")
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
sample_submission['week_of_month'] = (sample_submission['show_date'].dt.day - 1) // 7 + 1

bt_pred = breakthrough['audience_count'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    return np.maximum(pred + (TARGET_MEAN - np.mean(pred)), 0)

print(f"\n[1/4] Data loaded")

# Compute baseline
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
feb_bl = sample_submission['feb_baseline'].values

# DOW-specific blend ratios
# Higher diff → model might be overestimating → use more baseline
# Diffs from analysis:
# Mon: +4.22, Tue: +6.11, Wed: +2.54, Thu: +3.32, Fri: +4.49, Sat: +3.17, Sun: +0.53

print("\n[2/4] DOW-specific blend ratios...")

# Strategy: Scale baseline weight by deviation
# Sun (0.53) → 1% baseline (trust BT)
# Tue (6.11) → 3% baseline (correct BT)
# Linear interpolation between

diffs = {0: 4.22, 1: 6.11, 2: 2.54, 3: 3.32, 4: 4.49, 5: 3.17, 6: 0.53}
max_diff = max(diffs.values())  # 6.11
min_diff = min(diffs.values())  # 0.53

# Map diff to baseline weight: 1% (low diff) to 3% (high diff)
dow_bl_weights = {}
for dow, diff in diffs.items():
    normalized = (diff - min_diff) / (max_diff - min_diff)  # 0 to 1
    bl_weight = 0.01 + 0.02 * normalized  # 1% to 3%
    dow_bl_weights[dow] = bl_weight
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"  {day_names[dow]}: diff={diff:+.2f} → baseline weight = {bl_weight*100:.1f}%")

# Apply DOW-specific weights
dow_array = sample_submission['dayofweek'].values
bl_weights = np.array([dow_bl_weights[dow] for dow in dow_array])
bt_weights = 1 - bl_weights

print("\n[3/4] Creating predictions...")

pred_dow = bt_weights * bt_pred + bl_weights * feb_bl
pred_dow_cal = calibrate(pred_dow)
print(f"  DOW-specific blend: {pred_dow_cal.mean():.4f}")

# Also try week-of-month specific
# Diffs: Week1: +1.40, Week2: +2.64, Week3: +4.20, Week4: +5.73, Week5: +5.70
wom_diffs = {1: 1.40, 2: 2.64, 3: 4.20, 4: 5.73, 5: 5.70}
max_wom_diff = max(wom_diffs.values())
min_wom_diff = min(wom_diffs.values())

wom_bl_weights = {}
for wom, diff in wom_diffs.items():
    normalized = (diff - min_wom_diff) / (max_wom_diff - min_wom_diff)
    bl_weight = 0.01 + 0.02 * normalized
    wom_bl_weights[wom] = bl_weight
    print(f"  Week {wom}: diff={diff:+.2f} → baseline weight = {bl_weight*100:.1f}%")

wom_array = sample_submission['week_of_month'].values
wom_bl_weights_arr = np.array([wom_bl_weights.get(wom, 0.02) for wom in wom_array])
wom_bt_weights = 1 - wom_bl_weights_arr

pred_wom = wom_bt_weights * bt_pred + wom_bl_weights_arr * feb_bl
pred_wom_cal = calibrate(pred_wom)
print(f"  Week-of-month specific: {pred_wom_cal.mean():.4f}")

# Combined: DOW + WOM
combined_bl_weight = (bl_weights + wom_bl_weights_arr) / 2  # Average
combined_bt_weight = 1 - combined_bl_weight

pred_combined = combined_bt_weight * bt_pred + combined_bl_weight * feb_bl
pred_combined_cal = calibrate(pred_combined)
print(f"  Combined DOW+WOM: {pred_combined_cal.mean():.4f}")

# Standard for reference
pred_standard = 0.98 * bt_pred + 0.02 * feb_bl
pred_standard_cal = calibrate(pred_standard)
print(f"  Standard 98/2: {pred_standard_cal.mean():.4f}")

# Save
print("\n[4/4] Saving submissions...")
print("="*100)

submissions = {
    'submission.csv': pred_dow_cal,  # MAIN: DOW-specific
    'submission_wom.csv': pred_wom_cal,
    'submission_combined.csv': pred_combined_cal,
    'submission_standard.csv': pred_standard_cal,
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.4f}")

print("\n" + "="*100)
print("DOW-SPECIFIC CORRECTION COMPLETE")
print("="*100)
print("\nKey insight: Correct more where breakthrough overestimates most!")
print("  Tuesday: 3% baseline (high overestimate)")
print("  Sunday: 1% baseline (accurate)")
print("="*100)
