"""
BOOKING-ENHANCED BASELINE
==========================
Use the booking data to create smarter baselines.
The booknow_booking.csv has ticket booking patterns that might
correlate with final audience counts.
"""

import pandas as pd
import numpy as np

print("="*100)
print("BOOKING-ENHANCED BASELINE")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
booknow_booking = pd.read_csv('booknow_booking.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

print(f"\n[1/5] Data loaded")
print(f"  Visits: {len(booknow_visits):,}")
print(f"  Bookings: {len(booknow_booking):,}")

# Parse dates
booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek

booknow_booking['show_datetime'] = pd.to_datetime(booknow_booking['show_datetime'])
booknow_booking['show_date'] = booknow_booking['show_datetime'].dt.date
booknow_booking['show_date'] = pd.to_datetime(booknow_booking['show_date'])
booknow_booking['dayofweek'] = booknow_booking['show_date'].dt.dayofweek

sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek
sample_submission['breakthrough'] = breakthrough['audience_count'].values

TARGET_MEAN = 43.85

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

# ANALYZE: How do bookings relate to visits?
print("\n[2/5] Analyzing booking patterns...")

# Aggregate bookings by theater and date
booking_agg = booknow_booking.groupby(['book_theater_id', 'show_date']).agg({
    'tickets_booked': 'sum'
}).reset_index()

# Merge with visits to see correlation
merged = booknow_visits.merge(booking_agg, on=['book_theater_id', 'show_date'], how='left')
merged['tickets_booked'] = merged['tickets_booked'].fillna(0)

# Correlation between bookings and audience
correlation = merged[['audience_count', 'tickets_booked']].corr().iloc[0, 1]
print(f"  Correlation (bookings vs audience): {correlation:.4f}")

# Compute booking-based theater-DOW means
booking_theater_dow = booknow_booking.groupby(['book_theater_id', 'dayofweek'])['tickets_booked'].mean()
print(f"  Booking theater-DOW patterns computed")

# BASELINES
print("\n[3/5] Creating baselines...")

# Standard Feb baseline
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
print(f"  Feb baseline mean: {sample_submission['feb_baseline'].mean():.2f}")

# Booking-enhanced baseline: scale Feb baseline by booking ratio
# Idea: if bookings suggest higher demand, adjust baseline up
booking_global_mean = booknow_booking['tickets_booked'].mean()

def get_booking_ratio(row):
    key = (row['book_theater_id'], row['dayofweek'])
    if key in booking_theater_dow.index:
        booking_val = booking_theater_dow[key]
        return booking_val / booking_global_mean if booking_global_mean > 0 else 1.0
    return 1.0

sample_submission['booking_ratio'] = sample_submission.apply(get_booking_ratio, axis=1)
sample_submission['booking_enhanced_baseline'] = sample_submission['feb_baseline'] * sample_submission['booking_ratio']

# Normalize to have same mean as feb baseline
booking_enhanced_mean = sample_submission['booking_enhanced_baseline'].mean()
sample_submission['booking_enhanced_baseline'] *= sample_submission['feb_baseline'].mean() / booking_enhanced_mean

print(f"  Booking-enhanced baseline mean: {sample_submission['booking_enhanced_baseline'].mean():.2f}")

# BLENDING
print("\n[4/5] Creating blends...")

bt_pred = sample_submission['breakthrough'].values
feb_bl = sample_submission['feb_baseline'].values
booking_bl = sample_submission['booking_enhanced_baseline'].values

results = {}

# Standard 98/2 Feb (reference)
blend_feb = 0.98 * bt_pred + 0.02 * feb_bl
results['feb_98_2'] = calibrate(blend_feb)
print(f"  98/2 Feb (reference): {results['feb_98_2'].mean():.2f}")

# 98/2 Booking-enhanced
blend_booking = 0.98 * bt_pred + 0.02 * booking_bl
results['booking_98_2'] = calibrate(blend_booking)
print(f"  98/2 Booking-enhanced: {results['booking_98_2'].mean():.2f}")

# Combined: 50% Feb + 50% Booking-enhanced baseline
combined_bl = 0.5 * feb_bl + 0.5 * booking_bl
blend_combined = 0.98 * bt_pred + 0.02 * combined_bl
results['combined_98_2'] = calibrate(blend_combined)
print(f"  98/2 Combined baseline: {results['combined_98_2'].mean():.2f}")

# Try different ratios with booking-enhanced
for ratio in [97.5, 98.5]:
    blend = (ratio/100) * bt_pred + ((100-ratio)/100) * booking_bl
    results[f'booking_{ratio}'] = calibrate(blend)
    print(f"  {ratio}/{100-ratio} Booking-enhanced: {results[f'booking_{ratio}'].mean():.2f}")

# SAVE
print("\n[5/5] Saving submissions...")
print("="*100)

submissions = {
    'submission.csv': results['booking_98_2'],  # MAIN: Booking-enhanced
    'submission_feb_ref.csv': results['feb_98_2'],
    'submission_booking_975.csv': results['booking_97.5'],
    'submission_combined_bl.csv': results['combined_98_2'],
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("BOOKING-ENHANCED BASELINE COMPLETE")
print("="*100)
print(f"\nBooking-visits correlation: {correlation:.4f}")
print("Current best: 0.35749 (98/2 + Feb)")
print("New approach: Booking-enhanced baseline!")
print("="*100)
