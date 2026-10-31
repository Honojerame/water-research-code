"""
WATER RESEARCH EDA - PREPROCESSING CODE CHEAT SHEET
====================================================

Copy-paste ready code for moving from EDA to modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Load the aligned dataset
aligned = pd.read_csv(r'c:\Water_research\modeling\aligned_modeling_data.csv')
aligned['date'] = pd.to_datetime(aligned['date'])

# =====================================================================
# STEP 1: HANDLE MISSING DATA
# =====================================================================

# Precipitation: interpolate small gaps, then forward-fill
aligned['prcp'] = (aligned['prcp']
                   .interpolate(method='linear', limit=7)  # interpolate up to 7 days
                   .fillna(method='ffill'))                # forward-fill remaining

# Discharge: simple forward-fill (only 0.43% missing)
aligned['discharge_mean'] = aligned['discharge_mean'].fillna(method='ffill')
aligned['gage_mean'] = aligned['gage_mean'].fillna(method='ffill')

# Drop any remaining NaNs (should be minimal)
aligned = aligned.dropna()

print(f"✓ After cleaning: {aligned.shape[0]:,} rows (dropped {62750 - aligned.shape[0]} rows)")


# =====================================================================
# STEP 2: TRANSFORM SKEWED VARIABLES
# =====================================================================

# Log-transform discharge (skewness=6.16 → needs transform)
aligned['discharge_log'] = np.log(aligned['discharge_mean'] + 0.01)

# Optional: use discharge_log for modeling, discharge_mean for interpretation
# Keep gage_mean as-is (skewness=1.79 is manageable)

print(f"✓ Log-transformed discharge (skewness: {6.156:.2f} → {aligned['discharge_log'].skew():.2f})")


# =====================================================================
# STEP 3: EXTRACT TEMPORAL FEATURES
# =====================================================================

aligned['year'] = aligned['date'].dt.year
aligned['month'] = aligned['date'].dt.month
aligned['dayofyear'] = aligned['date'].dt.dayofyear
aligned['dayofweek'] = aligned['date'].dt.dayofweek  # 0=Mon, 6=Sun

# Season indicator (useful for seasonal models)
aligned['season'] = aligned['month'].map({
    12: 'winter', 1: 'winter', 2: 'winter',
    3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer',
    9: 'fall', 10: 'fall', 11: 'fall'
})

# Wet vs dry season binary (based on EDA findings)
aligned['is_wet_season'] = aligned['month'].isin([5, 6, 7, 8]).astype(int)

print(f"✓ Created temporal features: year, month, dayofyear, season, is_wet_season")


# =====================================================================
# STEP 4: CREATE LAGGED FEATURES
# =====================================================================

# Lagged discharge (critical for capturing temporal dependency)
aligned['discharge_lag1'] = aligned['discharge_mean'].shift(1)
aligned['discharge_lag2'] = aligned['discharge_mean'].shift(2)
aligned['discharge_lag7'] = aligned['discharge_mean'].shift(7)      # Weekly pattern
aligned['discharge_lag365'] = aligned['discharge_mean'].shift(365)  # Yearly pattern

# Lagged gage height
aligned['gage_lag1'] = aligned['gage_mean'].shift(1)

# Lagged precipitation
aligned['prcp_lag1'] = aligned['prcp'].shift(1)
aligned['prcp_lag7'] = aligned['prcp'].shift(7)  # Accumulated recent rain

print(f"✓ Created lagged features: discharge/gage/prcp with lags=[1, 2, 7, 365]")


# =====================================================================
# STEP 5: CREATE ROLLING STATISTICS
# =====================================================================

# Rolling means (capture trend)
aligned['discharge_roll7_mean'] = aligned['discharge_mean'].rolling(7).mean()
aligned['discharge_roll30_mean'] = aligned['discharge_mean'].rolling(30).mean()
aligned['gage_roll7_mean'] = aligned['gage_mean'].rolling(7).mean()

# Rolling standard deviations (capture variability)
aligned['discharge_roll7_std'] = aligned['discharge_mean'].rolling(7).std()
aligned['discharge_roll30_std'] = aligned['discharge_mean'].rolling(30).std()

# Rolling precipitation sum (accumulated rain)
aligned['prcp_roll7_sum'] = aligned['prcp'].rolling(7).sum()
aligned['prcp_roll30_sum'] = aligned['prcp'].rolling(30).sum()

# Temperature rolling mean
aligned['tmax_roll7_mean'] = aligned['tmax'].rolling(7).mean()

print(f"✓ Created rolling statistics: 7-day and 30-day means/stds/sums")


# =====================================================================
# STEP 6: DROP ROWS WITH LAGGED NaNs
# =====================================================================

# Lagged features create NaNs at the beginning
aligned_clean = aligned.dropna()

print(f"✓ After removing NaN lags: {aligned_clean.shape[0]:,} rows")
print(f"  Lost {aligned.shape[0] - aligned_clean.shape[0]} rows to lagging")


# =====================================================================
# STEP 7: DEFINE FEATURES & TARGET
# =====================================================================

# Define feature columns (all except target and metadata)
feature_cols = [
    # Meteorological
    'tmax', 'tmin', 'srad', 'vp', 'swe',
    'prcp', 'prcp_lag1', 'prcp_lag7', 'prcp_roll7_sum',
    # Hydrological
    'gage_mean', 'gage_lag1', 'gage_roll7_mean',
    # Lagged discharge (critical for LSTM/RF)
    'discharge_lag1', 'discharge_lag2', 'discharge_lag7', 'discharge_lag365',
    # Rolling stats
    'discharge_roll7_mean', 'discharge_roll7_std',
    'discharge_roll30_mean', 'discharge_roll30_std',
    'tmax_roll7_mean',
    # Temporal
    'month', 'dayofyear', 'is_wet_season'
]

# Target options:
# - Use 'discharge_log' for neural networks (better for skewed data)
# - Use 'discharge_mean' for tree models (handles skew natively)
target_col = 'discharge_log'  # or 'discharge_mean'

X = aligned_clean[feature_cols].copy()
y = aligned_clean[target_col].copy()

print(f"✓ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"✓ Target variable: {target_col}")


# =====================================================================
# STEP 8: NORMALIZE FOR NEURAL NETWORKS
# =====================================================================

# For LSTM/Dense networks: StandardScaler (zero mean, unit variance)
scaler_X = StandardScaler()
X_scaled = pd.DataFrame(
    scaler_X.fit_transform(X),
    columns=X.columns,
    index=X.index
)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print(f"✓ Normalized features: μ≈0, σ≈1")
print(f"✓ Normalized target: μ≈0, σ≈1")


# =====================================================================
# STEP 9: TIME-AWARE TRAIN/TEST SPLIT
# =====================================================================

# CRITICAL: Don't use random_state or shuffle=True for time series!

# Find 80% cutoff date
cutoff_idx = int(len(aligned_clean) * 0.8)
cutoff_date = aligned_clean.iloc[cutoff_idx]['date']

# Split
train_mask = aligned_clean['date'] <= cutoff_date
test_mask = aligned_clean['date'] > cutoff_date

X_train = X_scaled[train_mask]
X_test = X_scaled[test_mask]
y_train = y_scaled[train_mask]
y_test = y_scaled[test_mask]

print(f"✓ Train: {len(X_train):,} rows ({cutoff_date.date()})")
print(f"✓ Test:  {len(X_test):,} rows ({aligned_clean['date'].max().date()})")


# =====================================================================
# STEP 10: SAVE PROCESSED DATA
# =====================================================================

# Save for later
processed_data = aligned_clean.copy()
processed_data['discharge_log'] = np.log(processed_data['discharge_mean'] + 0.01)
processed_data.to_csv(r'c:\Water_research\processed_modeling_data.csv', index=False)

# Save features and targets separately
np.save(r'c:\Water_research\X_train_scaled.npy', X_train.values)
np.save(r'c:\Water_research\X_test_scaled.npy', X_test.values)
np.save(r'c:\Water_research\y_train_scaled.npy', y_train)
np.save(r'c:\Water_research\y_test_scaled.npy', y_test)

print(f"✓ Saved processed data to c:\\Water_research\\processed_modeling_data.csv")
print(f"✓ Saved scaled arrays for quick model training")


# =====================================================================
# OPTIONAL: EXPLORATORY PLOTS
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Plot 1: Original vs log-transformed discharge
axes[0,0].hist(aligned_clean['discharge_mean'], bins=50, alpha=0.7, label='Original')
axes[0,0].set_xlabel('Discharge (m³/s)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Original Discharge Distribution')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

axes[0,1].hist(aligned_clean['discharge_log'], bins=50, alpha=0.7, color='orange', label='Log-transformed')
axes[0,1].set_xlabel('Log(Discharge)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Log-Transformed Discharge')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# Plot 2: Time series with train/test split
axes[1,0].plot(aligned_clean[train_mask]['date'], y[train_mask], 'b-', alpha=0.5, label='Train')
axes[1,0].plot(aligned_clean[test_mask]['date'], y[test_mask], 'r-', alpha=0.5, label='Test')
axes[1,0].axvline(cutoff_date, color='black', linestyle='--', linewidth=2, label='Split')
axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Discharge (log scale)')
axes[1,0].set_title('Train/Test Split')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# Plot 3: Lagged discharge correlation
axes[1,1].scatter(aligned_clean['discharge_lag1'], aligned_clean['discharge_mean'], alpha=0.1, s=1)
axes[1,1].set_xlabel('Discharge(t-1)')
axes[1,1].set_ylabel('Discharge(t)')
axes[1,1].set_title(f'Temporal Autocorrelation (r={aligned_clean[[\"discharge_lag1\", \"discharge_mean\"]].corr().iloc[0,1]:.3f})')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Water_research\preprocessing_validation.png', dpi=150)
plt.show()

print(f"✓ Saved validation plot to preprocessing_validation.png")


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "="*70)
print("PREPROCESSING COMPLETE - Ready for Modeling!")
print("="*70)
print(f"\nDataset Summary:")
print(f"  Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
print(f"  Target: {target_col} (scaled: μ={y_train.mean():.3f}, σ={y_train.std():.3f})")
print(f"\nRecommended Models:")
print(f"  1. LSTM (best): Sequential temporal modeling")
print(f"  2. XGBoost (fast): Tree ensemble baseline")
print(f"  3. Prophet (simple): Seasonal decomposition")
print("\nNext Step: Copy this data into your model training script!")
