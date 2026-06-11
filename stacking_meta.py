"""
STACKING META-LEARNER
=====================
Instead of just blending breakthrough with baseline,
train a SECOND-STAGE MODEL that learns to refine predictions.

The breakthrough predictions become a FEATURE, not the final answer.
The meta-learner learns the optimal correction patterns.

This is fundamentally different from simple blending!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb

print("="*100)
print("STACKING META-LEARNER - Learn to Refine Predictions")
print("="*100)

# Load data
booknow_visits = pd.read_csv('booknow_visits.csv')
sample_submission = pd.read_csv('sample_submission.csv')
breakthrough = pd.read_csv('submission_breakthrough_357.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month
booknow_visits['year'] = booknow_visits['show_date'].dt.year
booknow_visits['day'] = booknow_visits['show_date'].dt.day
booknow_visits['week'] = booknow_visits['show_date'].dt.isocalendar().week.astype(int)

print(f"\n[1/6] Data loaded")
print(f"  Training rows: {len(booknow_visits):,}")

# Parse test data
sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])
sample_submission['dayofweek'] = sample_submission['show_date'].dt.dayofweek
sample_submission['month'] = sample_submission['show_date'].dt.month
sample_submission['day'] = sample_submission['show_date'].dt.day
sample_submission['week'] = sample_submission['show_date'].dt.isocalendar().week.astype(int)
sample_submission['breakthrough'] = breakthrough['audience_count'].values

TARGET_MEAN = 43.85

# Compute rich features from training data
print("\n[2/6] Computing rich features...")

# Theater statistics
theater_stats = booknow_visits.groupby('book_theater_id').agg({
    'audience_count': ['mean', 'std', 'median', 'min', 'max', 'count']
}).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std', 'th_median', 'th_min', 'th_max', 'th_count']

# Theater-DOW statistics
theater_dow_stats = booknow_visits.groupby(['book_theater_id', 'dayofweek']).agg({
    'audience_count': ['mean', 'std', 'count']
}).reset_index()
theater_dow_stats.columns = ['book_theater_id', 'dayofweek', 'th_dow_mean', 'th_dow_std', 'th_dow_count']

# Theater-Month statistics
theater_month_stats = booknow_visits.groupby(['book_theater_id', 'month']).agg({
    'audience_count': ['mean', 'std']
}).reset_index()
theater_month_stats.columns = ['book_theater_id', 'month', 'th_month_mean', 'th_month_std']

# DOW global statistics
dow_stats = booknow_visits.groupby('dayofweek')['audience_count'].agg(['mean', 'std']).reset_index()
dow_stats.columns = ['dayofweek', 'dow_mean', 'dow_std']

# Month global statistics
month_stats = booknow_visits.groupby('month')['audience_count'].agg(['mean', 'std']).reset_index()
month_stats.columns = ['month', 'month_mean', 'month_std']

# Recent Feb 2024 patterns
feb_data = booknow_visits[booknow_visits['show_date'] >= '2024-02-01']
feb_theater_dow = feb_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean().reset_index()
feb_theater_dow.columns = ['book_theater_id', 'dayofweek', 'feb_baseline']

# All-time theater-dow
all_theater_dow = booknow_visits.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean().reset_index()
all_theater_dow.columns = ['book_theater_id', 'dayofweek', 'all_baseline']

print(f"  Theater stats computed for {len(theater_stats)} theaters")

# Build test features
print("\n[3/6] Building test features...")

test_df = sample_submission.copy()

# Merge all features
test_df = test_df.merge(theater_stats, on='book_theater_id', how='left')
test_df = test_df.merge(theater_dow_stats, on=['book_theater_id', 'dayofweek'], how='left')
test_df = test_df.merge(theater_month_stats, on=['book_theater_id', 'month'], how='left')
test_df = test_df.merge(dow_stats, on='dayofweek', how='left')
test_df = test_df.merge(month_stats, on='month', how='left')
test_df = test_df.merge(feb_theater_dow, on=['book_theater_id', 'dayofweek'], how='left')
test_df = test_df.merge(all_theater_dow, on=['book_theater_id', 'dayofweek'], how='left')

# Fill missing
global_mean = booknow_visits['audience_count'].mean()
for col in ['th_mean', 'th_std', 'th_median', 'th_min', 'th_max', 'th_count',
            'th_dow_mean', 'th_dow_std', 'th_dow_count',
            'th_month_mean', 'th_month_std',
            'feb_baseline', 'all_baseline']:
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(global_mean if 'mean' in col or 'baseline' in col else 0)

# Derived features
test_df['is_weekend'] = test_df['dayofweek'].isin([5, 6]).astype(int)
test_df['bt_vs_th_mean'] = test_df['breakthrough'] - test_df['th_mean']
test_df['bt_vs_dow_mean'] = test_df['breakthrough'] - test_df['dow_mean']
test_df['bt_vs_feb'] = test_df['breakthrough'] - test_df['feb_baseline']
test_df['bt_ratio_th'] = test_df['breakthrough'] / (test_df['th_mean'] + 1)
test_df['bt_zscore'] = (test_df['breakthrough'] - test_df['th_mean']) / (test_df['th_std'] + 1)

# How different is breakthrough from baselines?
test_df['bt_deviation'] = np.abs(test_df['breakthrough'] - test_df['th_dow_mean']) / (test_df['th_dow_std'] + 1)

print(f"  Test features built: {len(test_df)} rows, {len(test_df.columns)} columns")

# TRAIN META-LEARNER using training data
print("\n[4/6] Training meta-learner on historical data...")

# For training, we need to simulate what "breakthrough" predictions would be
# We use the actual values with some noise, or use leave-one-out predictions
# For simplicity, use theater-DOW means as "base predictions" to train the meta-learner

train_df = booknow_visits.copy()
train_df = train_df.merge(theater_stats, on='book_theater_id', how='left')
train_df = train_df.merge(theater_dow_stats, on=['book_theater_id', 'dayofweek'], how='left')
train_df = train_df.merge(theater_month_stats, on=['book_theater_id', 'month'], how='left')
train_df = train_df.merge(dow_stats, on='dayofweek', how='left')
train_df = train_df.merge(month_stats, on='month', how='left')
train_df = train_df.merge(feb_theater_dow, on=['book_theater_id', 'dayofweek'], how='left')
train_df = train_df.merge(all_theater_dow, on=['book_theater_id', 'dayofweek'], how='left')

for col in ['th_mean', 'th_std', 'th_median', 'th_min', 'th_max', 'th_count',
            'th_dow_mean', 'th_dow_std', 'th_dow_count',
            'th_month_mean', 'th_month_std',
            'feb_baseline', 'all_baseline']:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(global_mean if 'mean' in col or 'baseline' in col else 0)

# Use all_baseline as "simulated breakthrough" for training
# The meta-learner learns to correct this baseline toward actual values
train_df['sim_breakthrough'] = train_df['all_baseline']
train_df['is_weekend'] = train_df['dayofweek'].isin([5, 6]).astype(int)
train_df['bt_vs_th_mean'] = train_df['sim_breakthrough'] - train_df['th_mean']
train_df['bt_vs_dow_mean'] = train_df['sim_breakthrough'] - train_df['dow_mean']
train_df['bt_vs_feb'] = train_df['sim_breakthrough'] - train_df['feb_baseline']
train_df['bt_ratio_th'] = train_df['sim_breakthrough'] / (train_df['th_mean'] + 1)
train_df['bt_zscore'] = (train_df['sim_breakthrough'] - train_df['th_mean']) / (train_df['th_std'] + 1)
train_df['bt_deviation'] = np.abs(train_df['sim_breakthrough'] - train_df['th_dow_mean']) / (train_df['th_dow_std'] + 1)

# Target: the RESIDUAL (actual - simulated breakthrough)
train_df['residual'] = train_df['audience_count'] - train_df['sim_breakthrough']

# Features for meta-learner
feature_cols = ['sim_breakthrough', 'dayofweek', 'month', 'day', 'is_weekend',
                'th_mean', 'th_std', 'th_median', 'th_count',
                'th_dow_mean', 'th_dow_std', 'th_dow_count',
                'th_month_mean', 'dow_mean', 'month_mean',
                'feb_baseline', 'all_baseline',
                'bt_vs_th_mean', 'bt_vs_dow_mean', 'bt_vs_feb',
                'bt_ratio_th', 'bt_zscore', 'bt_deviation']

# Filter to valid features
feature_cols = [c for c in feature_cols if c in train_df.columns]

X_train = train_df[feature_cols].values
y_train = train_df['residual'].values

print(f"  Training meta-learner on {len(X_train):,} samples with {len(feature_cols)} features")

# Train with K-Fold to avoid overfitting
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
meta_preds_train = np.zeros(len(X_train))

xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

lgb_params = {
    'objective': 'regression',
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbose': -1
}

# Train ensemble of meta-learners
xgb_models = []
lgb_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=200)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_models.append(xgb_model)

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=200)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    lgb_models.append(lgb_model)

    # OOF predictions
    meta_preds_train[val_idx] = 0.5 * xgb_model.predict(X_val) + 0.5 * lgb_model.predict(X_val)

train_corrected = train_df['sim_breakthrough'].values + meta_preds_train
train_rmse = np.sqrt(np.mean((train_corrected - train_df['audience_count'].values)**2))
baseline_rmse = np.sqrt(np.mean((train_df['sim_breakthrough'].values - train_df['audience_count'].values)**2))

print(f"  Baseline RMSE: {baseline_rmse:.4f}")
print(f"  Meta-corrected RMSE: {train_rmse:.4f}")
print(f"  Improvement: {baseline_rmse - train_rmse:.4f}")

# Apply to test
print("\n[5/6] Applying meta-learner to breakthrough predictions...")

# Prepare test features (use breakthrough as the base prediction)
test_features = test_df.copy()
test_features['sim_breakthrough'] = test_features['breakthrough']

X_test = test_features[feature_cols].values

# Predict residuals
meta_residuals = np.zeros(len(X_test))
for xgb_model, lgb_model in zip(xgb_models, lgb_models):
    meta_residuals += 0.5 * xgb_model.predict(X_test) + 0.5 * lgb_model.predict(X_test)
meta_residuals /= len(xgb_models)

# Apply correction
corrected = test_df['breakthrough'].values + meta_residuals

# Also try scaled correction (less aggressive)
corrected_mild = test_df['breakthrough'].values + 0.5 * meta_residuals
corrected_gentle = test_df['breakthrough'].values + 0.25 * meta_residuals

def calibrate(pred):
    calibrated = pred + (TARGET_MEAN - np.mean(pred))
    return np.maximum(calibrated, 0)

corrected_cal = calibrate(corrected)
corrected_mild_cal = calibrate(corrected_mild)
corrected_gentle_cal = calibrate(corrected_gentle)

print(f"  Full correction mean: {corrected_cal.mean():.2f}")
print(f"  Mild correction (50%) mean: {corrected_mild_cal.mean():.2f}")
print(f"  Gentle correction (25%) mean: {corrected_gentle_cal.mean():.2f}")

# Also blend with Feb baseline (combine best approaches)
print("\n[6/6] Combining with best approaches...")

# Meta-corrected + 2% Feb baseline (like the winning approach)
meta_blend = 0.98 * corrected_gentle + 0.02 * test_df['feb_baseline'].values
meta_blend_cal = calibrate(meta_blend)
print(f"  Meta + Feb baseline blend: {meta_blend_cal.mean():.2f}")

# 98/2 breakthrough + Feb (current best for comparison)
current_best = 0.98 * test_df['breakthrough'].values + 0.02 * test_df['feb_baseline'].values
current_best_cal = calibrate(current_best)
print(f"  Current best (98/2 + Feb): {current_best_cal.mean():.2f}")

# Save
print("\n" + "="*100)
print("SAVING SUBMISSIONS")
print("="*100)

submissions = {
    'submission.csv': corrected_gentle_cal,  # MAIN: Gentle meta-correction
    'submission_meta_full.csv': corrected_cal,
    'submission_meta_mild.csv': corrected_mild_cal,
    'submission_meta_blend.csv': meta_blend_cal,
    'submission_current_best.csv': current_best_cal,  # Reference
}

for filename, pred in submissions.items():
    pd.DataFrame({
        'ID': sample_submission['ID'],
        'audience_count': pred
    }).to_csv(filename, index=False)
    print(f"  {filename}: mean = {pred.mean():.2f}")

print("\n" + "="*100)
print("STACKING META-LEARNER COMPLETE")
print("="*100)
print("\nAPPROACH:")
print("  Train a meta-learner to predict RESIDUALS (correction factors)")
print("  Apply learned corrections to breakthrough predictions")
print("  This is fundamentally different from simple blending!")
print("\nCurrent best: 0.35749 (98/2 + Feb baseline)")
print("New approach: Meta-learned corrections!")
print("="*100)
