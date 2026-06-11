"""
RECENT DATA ONLY - Ultra-Radical Approach
==========================================
HYPOTHESIS: Training on ALL data (14 months) may include outdated patterns.
           What if patterns from early 2023 are HURTING March 2024 predictions?

STRATEGY:
1. Train ONLY on recent 2-3 months (Dec 2023 - Feb 2024)
2. Less data but MORE RELEVANT to test period
3. This directly addresses temporal distribution shift

Expected: 0.38-0.42 (if old data was hurting)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("RECENT DATA ONLY - Training on Last 3 Months Only")
print("="*100)

# Load
booknow_theaters = pd.read_csv('booknow_theaters.csv')
booknow_visits = pd.read_csv('booknow_visits.csv')
date_info = pd.read_csv('date_info.csv')
sample_submission = pd.read_csv('sample_submission.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
date_info['show_date'] = pd.to_datetime(date_info['show_date'])
sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])

print(f"\n[1/8] Data loaded")
print(f"  Full data: {len(booknow_visits):,} rows")
print(f"  Date range: {booknow_visits['show_date'].min().date()} to {booknow_visits['show_date'].max().date()}")

# FILTER TO RECENT DATA ONLY (Dec 2023 - Feb 2024)
recent_start = pd.Timestamp('2023-12-01')
recent_data = booknow_visits[booknow_visits['show_date'] >= recent_start].copy()

print(f"\n[2/8] Filtering to RECENT data only")
print(f"  Recent data: {len(recent_data):,} rows ({len(recent_data)/len(booknow_visits)*100:.1f}%)")
print(f"  Date range: {recent_data['show_date'].min().date()} to {recent_data['show_date'].max().date()}")

# But we still need historical stats for all theaters
# Compute them from FULL data
print("\n[3/8] Computing theater stats from full historical data...")

booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month

theater_stats = booknow_visits.groupby('book_theater_id')['audience_count'].agg([
    'mean', 'std', 'median', 'count'
]).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std', 'th_median', 'th_count']
theater_stats['th_std'] = theater_stats['th_std'].fillna(0)

dayofweek_stats = booknow_visits.groupby('dayofweek')['audience_count'].mean().reset_index()
dayofweek_stats.columns = ['dayofweek', 'dow_mean']

# Target encoding on RECENT data only
recent_data['dayofweek'] = recent_data['show_date'].dt.dayofweek
recent_data['month'] = recent_data['show_date'].dt.month

def target_encode_cv(df, cols, target, n_splits=5):
    encoded = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        train_means = df.iloc[train_idx].groupby(cols)[target].mean()
        global_mean = df.iloc[train_idx][target].mean()
        for idx in val_idx:
            key = tuple(df.iloc[idx][cols]) if isinstance(cols, list) else df.iloc[idx][cols]
            encoded[idx] = train_means.get(key, global_mean)
    return encoded

recent_data['th_dow_enc'] = target_encode_cv(recent_data, ['book_theater_id', 'dayofweek'], 'audience_count')
recent_data['th_month_enc'] = target_encode_cv(recent_data, ['book_theater_id', 'month'], 'audience_count')

def create_features(df):
    df = df.copy()
    df['month'] = df['show_date'].dt.month
    df['day'] = df['show_date'].dt.day
    df['dayofweek'] = df['show_date'].dt.dayofweek
    df['dayofyear'] = df['show_date'].dt.dayofyear
    df['week'] = df['show_date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['week_of_month'] = ((df['day'] - 1) // 7) + 1

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    df = df.merge(date_info, on='show_date', how='left')
    df = df.merge(booknow_theaters, on='book_theater_id', how='left')
    df = df.merge(theater_stats, on='book_theater_id', how='left')
    df = df.merge(dayofweek_stats, on='dayofweek', how='left')

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and df[col].isna().any():
            df[col] = df[col].fillna(0)

    for col in ['day_of_week', 'theater_type', 'theater_area']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    return df

print("\n[4/8] Feature engineering on recent data...")
train_df = create_features(recent_data.copy())
train_df['th_dow_enc'] = recent_data['th_dow_enc'].values
train_df['th_month_enc'] = recent_data['th_month_enc'].values
train_df = train_df.sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

# Lag features
print("\n[5/8] Creating lag features...")
for lag in [1, 2, 3, 7, 14, 21, 28]:
    train_df[f'lag{lag}'] = train_df.groupby('book_theater_id')['audience_count'].shift(lag)

for window in [3, 7, 14, 28]:
    train_df[f'roll{window}'] = train_df.groupby('book_theater_id')['audience_count'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

train_df = train_df.fillna(0)

exclude = ['audience_count', 'show_date', 'book_theater_id']
feature_cols = [c for c in train_df.columns if c not in exclude]

X = train_df[feature_cols].copy()
y = train_df['audience_count'].copy()

# Validation: last 2 weeks of February
split_date = pd.Timestamp('2024-02-15')
train_mask = train_df['show_date'] < split_date
val_mask = train_df['show_date'] >= split_date
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"  Features: {len(feature_cols)}")
print(f"  Train (Dec-mid Feb): {X_train.shape}")
print(f"  Val (mid Feb-end Feb): {X_val.shape}")

# Train with MORE aggressive settings (less data needs simpler model)
print("\n[6/8] Training models (simpler for less data)...")

xgb_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.06, max_depth=5,  # Simpler
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=1.0, reg_lambda=3.0,  # More regularization
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.06, max_depth=6, num_leaves=35,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=1.0, reg_lambda=3.0,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

cat_model = CatBoostRegressor(
    iterations=500, learning_rate=0.06, depth=5,
    l2_leaf_reg=4.0, subsample=0.85,
    random_seed=42, verbose=False
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

pred_xgb = xgb_model.predict(X_val)
pred_lgb = lgb_model.predict(X_val)
pred_cat = cat_model.predict(X_val)
pred = 0.33 * pred_xgb + 0.33 * pred_lgb + 0.34 * pred_cat

r2 = r2_score(y_val, pred)
print(f"  Validation R²: {r2:.4f}")
print(f"  Validation mean: {pred.mean():.2f}")

# Retrain on ALL recent data
print("\n[7/8] Retraining on ALL recent data...")

xgb_full = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.06, max_depth=5,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=1.0, reg_lambda=3.0,
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_full.fit(X, y, verbose=False)

lgb_full = lgb.LGBMRegressor(
    n_estimators=400, learning_rate=0.06, max_depth=6, num_leaves=35,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=1.0, reg_lambda=3.0,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_full.fit(X, y)

cat_full = CatBoostRegressor(
    iterations=350, learning_rate=0.06, depth=5,
    l2_leaf_reg=4.0, subsample=0.85,
    random_seed=42, verbose=False
)
cat_full.fit(X, y)

# Iterative prediction
print("\n[8/8] Iterative prediction...")

# Use FULL historical data for iterative prediction (need lag history)
combined = pd.concat([
    booknow_visits[['book_theater_id', 'show_date', 'audience_count', 'dayofweek', 'month']].assign(
        th_dow_enc=np.nan, th_month_enc=np.nan
    ),
    sample_submission[['book_theater_id', 'show_date']].assign(
        audience_count=np.nan, dayofweek=np.nan, month=np.nan, th_dow_enc=np.nan, th_month_enc=np.nan
    )
], ignore_index=True).sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

# Fill target encoding for training data
train_th_dow_means = recent_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
train_th_month_means = recent_data.groupby(['book_theater_id', 'month'])['audience_count'].mean()
global_mean = recent_data['audience_count'].mean()

test_dates = sorted(sample_submission['show_date'].unique())

for i, date in enumerate(test_dates):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(test_dates)}", end='\r')

    date_mask = combined['show_date'] == date
    date_indices = combined[date_mask].index

    combined_feats = create_features(combined.copy())

    # Target encoding from recent data
    available_data = combined[combined['show_date'] < date].dropna(subset=['audience_count'])
    # Only use recent portion for encoding
    recent_cutoff = date - pd.Timedelta(days=90)
    recent_available = available_data[available_data['show_date'] >= recent_cutoff]

    if len(recent_available) > 0:
        recent_available['dayofweek'] = recent_available['show_date'].dt.dayofweek
        recent_available['month'] = recent_available['show_date'].dt.month
        combined_feats['dayofweek'] = combined_feats['show_date'].dt.dayofweek
        combined_feats['month'] = combined_feats['show_date'].dt.month

        th_dow_means = recent_available.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
        th_month_means = recent_available.groupby(['book_theater_id', 'month'])['audience_count'].mean()
        recent_global_mean = recent_available['audience_count'].mean()

        combined_feats['th_dow_enc'] = combined_feats.apply(
            lambda row: th_dow_means.get((row['book_theater_id'], row['dayofweek']), recent_global_mean), axis=1
        )
        combined_feats['th_month_enc'] = combined_feats.apply(
            lambda row: th_month_means.get((row['book_theater_id'], row['month']), recent_global_mean), axis=1
        )
    else:
        combined_feats['th_dow_enc'] = global_mean
        combined_feats['th_month_enc'] = global_mean

    # Lags from full history
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        combined_feats[f'lag{lag}'] = combined_feats.groupby('book_theater_id')['audience_count'].shift(lag)
    for window in [3, 7, 14, 28]:
        combined_feats[f'roll{window}'] = combined_feats.groupby('book_theater_id')['audience_count'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    combined_feats = combined_feats.fillna(0)
    X_date = combined_feats.loc[date_indices, feature_cols].copy()

    pred_xgb = xgb_full.predict(X_date)
    pred_lgb = lgb_full.predict(X_date)
    pred_cat = cat_full.predict(X_date)
    pred = 0.33 * pred_xgb + 0.33 * pred_lgb + 0.34 * pred_cat
    pred = np.maximum(pred, 0)

    combined.loc[date_indices, 'audience_count'] = pred

print(f"\n  ✓ Complete")

# Save
test_pred = combined[combined['show_date'] >= pd.Timestamp('2024-03-01')].copy()
test_pred = test_pred.merge(sample_submission[['book_theater_id', 'show_date', 'ID']],
                              on=['book_theater_id', 'show_date'], how='right')

submission = pd.DataFrame({
    'ID': test_pred['ID'],
    'audience_count': test_pred['audience_count'].fillna(0)
})
submission.to_csv('submission.csv', index=False)

print("\n" + "="*100)
print("RECENT DATA ONLY - COMPLETE")
print("="*100)
print(f"Training data: ONLY Dec 2023 - Feb 2024 (3 months)")
print(f"Validation R²: {r2:.4f}")
print(f"\nPredictions: {len(submission):,}")
print(f"Mean: {submission['audience_count'].mean():.2f}")
print(f"\n✓ Submission saved to submission.csv")
print(f"\nHypothesis: Recent patterns are more relevant to March-April 2024")
print("="*100)
