"""
NEURAL NETWORK APPROACH - Completely Different Paradigm
========================================================
After 11 failed gradient boosting attempts, trying neural networks.

Key differences:
- Non-linear learned representations
- Different optimization (Adam vs tree-based)
- Can learn complex feature interactions automatically
- Different inductive bias

Target: 0.4+
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Check if PyTorch is available, otherwise use sklearn MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    USE_PYTORCH = True
    print("Using PyTorch")
except ImportError:
    from sklearn.neural_network import MLPRegressor
    USE_PYTORCH = False
    print("Using sklearn MLPRegressor")

print("="*100)
print("NEURAL NETWORK APPROACH - Breaking Free from Gradient Boosting")
print("="*100)

# Load data
booknow_theaters = pd.read_csv('booknow_theaters.csv')
booknow_visits = pd.read_csv('booknow_visits.csv')
date_info = pd.read_csv('date_info.csv')
sample_submission = pd.read_csv('sample_submission.csv')

booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])
date_info['show_date'] = pd.to_datetime(date_info['show_date'])
sample_submission['book_theater_id'] = (sample_submission['ID'].str.split('_').str[0] + '_' +
                                         sample_submission['ID'].str.split('_').str[1])
sample_submission['show_date'] = pd.to_datetime(sample_submission['ID'].str.split('_').str[2])

print(f"\n[1/8] Data loaded: {len(booknow_visits):,} rows")

# Feature engineering (same as breakthrough)
print("\n[2/8] Feature engineering...")

booknow_visits['dayofweek'] = booknow_visits['show_date'].dt.dayofweek
booknow_visits['month'] = booknow_visits['show_date'].dt.month

theater_stats = booknow_visits.groupby('book_theater_id')['audience_count'].agg([
    'mean', 'std', 'median', 'count'
]).reset_index()
theater_stats.columns = ['book_theater_id', 'th_mean', 'th_std', 'th_median', 'th_count']
theater_stats['th_std'] = theater_stats['th_std'].fillna(0)

dayofweek_stats = booknow_visits.groupby('dayofweek')['audience_count'].mean().reset_index()
dayofweek_stats.columns = ['dayofweek', 'dow_mean']

# Theater encoding
theater_encoder = LabelEncoder()
booknow_visits['theater_code'] = theater_encoder.fit_transform(booknow_visits['book_theater_id'])

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

print("\n[3/8] Creating features...")
train_df = create_features(booknow_visits.copy())
train_df['th_dow_enc'] = booknow_visits['th_dow_enc']
train_df['th_month_enc'] = booknow_visits['th_month_enc']
train_df['theater_code'] = booknow_visits['theater_code']
train_df = train_df.sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

# Lag features
print("\n[4/8] Creating lag features...")
for lag in [1, 2, 3, 7, 14, 21, 28]:
    train_df[f'lag{lag}'] = train_df.groupby('book_theater_id')['audience_count'].shift(lag)

for window in [3, 7, 14, 28]:
    train_df[f'roll{window}'] = train_df.groupby('book_theater_id')['audience_count'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

train_df = train_df.fillna(0)

exclude = ['audience_count', 'show_date', 'book_theater_id']
feature_cols = [c for c in train_df.columns if c not in exclude]

X = train_df[feature_cols].values
y = train_df['audience_count'].values

# Scale features for neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_date = pd.Timestamp('2024-02-01')
train_mask = train_df['show_date'] < split_date
val_mask = train_df['show_date'] >= split_date
X_train, y_train = X_scaled[train_mask], y[train_mask]
X_val, y_val = X_scaled[val_mask], y[val_mask]

print(f"  Features: {len(feature_cols)}")
print(f"  Train: {X_train.shape} | Val: {X_val.shape}")

# Train Neural Network
print("\n[5/8] Training Neural Network...")

if USE_PYTORCH:
    # PyTorch implementation
    class AudienceNet(nn.Module):
        def __init__(self, input_dim):
            super(AudienceNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudienceNet(X_train.shape[1]).to(device)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device)).squeeze().cpu().numpy()
            val_loss = np.mean((val_pred - y_val) ** 2)
            val_r2 = r2_score(y_val, val_pred)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Val R² = {val_r2:.4f}")

        if patience_counter >= 15:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        val_pred = model(torch.FloatTensor(X_val).to(device)).squeeze().cpu().numpy()

else:
    # Sklearn implementation
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=512,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

val_r2 = r2_score(y_val, val_pred)
print(f"\n  Final Validation R²: {val_r2:.4f}")
print(f"  Validation mean: {val_pred.mean():.2f}")

# Retrain on full data
print("\n[6/8] Retraining on full data...")

if USE_PYTORCH:
    model_full = AudienceNet(X_scaled.shape[1]).to(device)
    full_dataset = TensorDataset(
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y)
    )
    full_loader = DataLoader(full_dataset, batch_size=512, shuffle=True)

    optimizer = optim.Adam(model_full.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(50):  # Fewer epochs for full data
        model_full.train()
        for batch_X, batch_y in full_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_full(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50")

    model_full.eval()
else:
    model_full = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=512,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        verbose=False
    )
    model_full.fit(X_scaled, y)

# Iterative prediction
print("\n[7/8] Iterative prediction...")

combined = pd.concat([
    booknow_visits[['book_theater_id', 'show_date', 'audience_count', 'dayofweek', 'month', 'th_dow_enc', 'th_month_enc', 'theater_code']],
    sample_submission[['book_theater_id', 'show_date']].assign(
        audience_count=np.nan, dayofweek=np.nan, month=np.nan, th_dow_enc=np.nan, th_month_enc=np.nan, theater_code=np.nan
    )
], ignore_index=True).sort_values(['book_theater_id', 'show_date']).reset_index(drop=True)

# Fill theater codes for test data
theater_code_map = dict(zip(booknow_visits['book_theater_id'], booknow_visits['theater_code']))
combined['theater_code'] = combined['book_theater_id'].map(theater_code_map)

test_dates = sorted(sample_submission['show_date'].unique())

for i, date in enumerate(test_dates):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(test_dates)}", end='\r')

    date_mask = combined['show_date'] == date
    date_indices = combined[date_mask].index

    combined_feats = create_features(combined.copy())
    combined_feats['theater_code'] = combined['theater_code']

    # Target encoding
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

    # Lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        combined_feats[f'lag{lag}'] = combined_feats.groupby('book_theater_id')['audience_count'].shift(lag)
    for window in [3, 7, 14, 28]:
        combined_feats[f'roll{window}'] = combined_feats.groupby('book_theater_id')['audience_count'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    combined_feats = combined_feats.fillna(0)
    X_date = combined_feats.loc[date_indices, feature_cols].values
    X_date_scaled = scaler.transform(X_date)

    if USE_PYTORCH:
        with torch.no_grad():
            pred = model_full(torch.FloatTensor(X_date_scaled).to(device)).squeeze().cpu().numpy()
    else:
        pred = model_full.predict(X_date_scaled)

    pred = np.maximum(pred, 0)
    combined.loc[date_indices, 'audience_count'] = pred

print(f"\n  ✓ Complete")

# Save
print("\n[8/8] Creating submission...")
test_pred = combined[combined['show_date'] >= pd.Timestamp('2024-03-01')].copy()
test_pred = test_pred.merge(sample_submission[['book_theater_id', 'show_date', 'ID']],
                              on=['book_theater_id', 'show_date'], how='right')

submission = pd.DataFrame({
    'ID': test_pred['ID'],
    'audience_count': test_pred['audience_count'].fillna(0)
})

# Calibrate mean if needed
raw_mean = submission['audience_count'].mean()
TARGET_MEAN = 43.85
if abs(raw_mean - TARGET_MEAN) > 1:
    correction = TARGET_MEAN - raw_mean
    print(f"  Applying mean correction: {raw_mean:.2f} → {TARGET_MEAN}")
    submission['audience_count'] = submission['audience_count'] + correction
    submission['audience_count'] = np.maximum(submission['audience_count'], 0)

submission.to_csv('submission.csv', index=False)

print("\n" + "="*100)
print("NEURAL NETWORK APPROACH COMPLETE")
print("="*100)
print(f"Validation R²: {val_r2:.4f}")
print(f"\nPredictions: {len(submission):,}")
print(f"Mean: {submission['audience_count'].mean():.2f}")
print(f"\n✓ Submission saved to submission.csv")
print(f"\nThis is a COMPLETELY DIFFERENT approach from gradient boosting!")
print("="*100)
