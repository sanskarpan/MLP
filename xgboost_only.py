"""
XGBOOST ONLY - No Ensemble
==========================
HYPOTHESIS: Maybe the ensemble averaging (XGB+LGB+CAT) is hurting.
           What if a single well-tuned XGBoost performs better?

This is the simplest possible "breakthrough" - just XGBoost.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("XGBOOST ONLY - Single Model, No Ensemble")
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

print(f"\n[1/7] Data loaded: {len(booknow_visits):,} rows")

# Features (same as breakthrough)
booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month

theater_stats = booknow_visits.groupby('book_theater_id')['audience_count'].agg([
    'mean', 'std', 'median', 'count'
]).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std', 'th_median', 'th_count']
theater_stats['th_std'] = theater_stats['th_std'].fillna(0)

dayofweek_stats = booknow_visits.groupby('dayofweek')['audience_count'].mean().reset_index()
dayofweek_stats.columns = ['dayofweek', 'dow_mean']

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

booknow_visits['th_dow_enc'] = target_encode_cv(booknow_visits, ['book_theater_id', 'dayofweek'], 'audience_count')
booknow_visits['th_month_enc'] = target_encode_cv(booknow_visits, ['book_theater_id', 'month'], 'audience_count')

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

print("\n[2/7] Feature engineering...")
train_df = create_features(booknow_visits.copy())
train_df['th_dow_enc'] = booknow_visits['th_dow_enc']
train_df['th_month_enc'] = booknow_visits['th_month_enc']
train_df = train_df.sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

print("\n[3/7] Creating lag features...")
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

split_date = pd.Timestamp('2024-02-01')
train_mask = train_df['show_date'] < split_date
val_mask = train_df['show_date'] >= split_date
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"  Features: {len(feature_cols)}")
print(f"  Train: {X_train.shape} | Val: {X_val.shape}")

# Train ONLY XGBoost
print("\n[4/7] Training XGBoost ONLY...")

xgb_model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.045, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.8, reg_lambda=2.5,
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

pred_val = xgb_model.predict(X_val)
r2 = r2_score(y_val, pred_val)
print(f"  Validation R² (XGBoost only): {r2:.4f}")
print(f"  Validation mean: {pred_val.mean():.2f}")

# Retrain on full data
print("\n[5/7] Retraining on full data...")

xgb_full = xgb.XGBRegressor(
    n_estimators=850, learning_rate=0.045, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.8, reg_lambda=2.5,
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_full.fit(X, y, verbose=False)

# Iterative prediction
print("\n[6/7] Iterative prediction (XGBoost only)...")

combined = pd.concat([
    booknow_visits[['book_theater_id', 'show_date', 'audience_count', 'dayofweek', 'month', 'th_dow_enc', 'th_month_enc']],
    sample_submission[['book_theater_id', 'show_date']].assign(
        audience_count=np.nan, dayofweek=np.nan, month=np.nan, th_dow_enc=np.nan, th_month_enc=np.nan
    )
], ignore_index=True).sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

test_dates = sorted(sample_submission['show_date'].unique())

for i, date in enumerate(test_dates):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(test_dates)}", end='\r')

    date_mask = combined['show_date'] == date
    date_indices = combined[date_mask].index

    combined_feats = create_features(combined.copy())

    available_data = combined[combined['show_date'] < date].dropna(subset=['audience_count'])
    if len(available_data) > 0:
        available_data['dayofweek'] = available_data['show_date'].dt.dayofweek
        available_data['month'] = available_data['show_date'].dt.month
        combined_feats['dayofweek'] = combined_feats['show_date'].dt.dayofweek
        combined_feats['month'] = combined_feats['show_date'].dt.month

        th_dow_means = available_data.groupby(['book_theater_id', 'dayofweek'])['audience_count'].mean()
        th_month_means = available_data.groupby(['book_theater_id', 'month'])['audience_count'].mean()
        global_mean = available_data['audience_count'].mean()

        combined_feats['th_dow_enc'] = combined_feats.apply(
            lambda row: th_dow_means.get((row['book_theater_id'], row['dayofweek']), global_mean), axis=1
        )
        combined_feats['th_month_enc'] = combined_feats.apply(
            lambda row: th_month_means.get((row['book_theater_id'], row['month']), global_mean), axis=1
        )
    else:
        combined_feats['th_dow_enc'] = 0
        combined_feats['th_month_enc'] = 0

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        combined_feats[f'lag{lag}'] = combined_feats.groupby('book_theater_id')['audience_count'].shift(lag)
    for window in [3, 7, 14, 28]:
        combined_feats[f'roll{window}'] = combined_feats.groupby('book_theater_id')['audience_count'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    combined_feats = combined_feats.fillna(0)
    X_date = combined_feats.loc[date_indices, feature_cols].copy()

    pred = xgb_full.predict(X_date)
    pred = np.maximum(pred, 0)

    combined.loc[date_indices, 'audience_count'] = pred

print(f"\n  ✓ Complete")

# Save
print("\n[7/7] Creating submission...")
test_pred = combined[combined['show_date'] >= pd.Timestamp('2024-03-01')].copy()
test_pred = test_pred.merge(sample_submission[['book_theater_id', 'show_date', 'ID']],
                              on=['book_theater_id', 'show_date'], how='right')

submission = pd.DataFrame({
    'ID': test_pred['ID'],
    'audience_count': test_pred['audience_count'].fillna(0)
})
submission.to_csv('submission.csv', index=False)

print("\n" + "="*100)
print("XGBOOST ONLY COMPLETE")
print("="*100)
print(f"Validation R²: {r2:.4f}")
print(f"\nPredictions: {len(submission):,}")
print(f"Mean: {submission['audience_count'].mean():.2f}")
print(f"\n✓ Submission saved to submission.csv")
print("="*100)
