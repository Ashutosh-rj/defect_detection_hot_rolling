"""
================================================================================
 PHASE 2 — ADVANCED FEATURE ENGINEERING
================================================================================
 Industrial-grade feature engineering for rare defect detection.
 Statistical aggregates, interactions, anomaly features, PCA, clustering.
================================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, QuantileTransformer, PowerTransformer
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import os, sys, pickle

sys.path.insert(0, r"d:\Tata_steel")
from config import *

print("=" * 80)
print("  PHASE 2 — ADVANCED FEATURE ENGINEERING")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# 2.0  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y_train = train[TARGET_COL].values
train_ids = train[ID_COL].values
test_ids  = test[ID_COL].values

X_train = train[FEATURE_COLS].copy()
X_test  = test[FEATURE_COLS].copy()

print(f"[INFO] Original features: {X_train.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.1  MISSING VALUE IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Missing Value Imputation ────────────────────────────────")
# Create missing indicator features first
for col in FEATURE_COLS:
    if X_train[col].isnull().any() or X_test[col].isnull().any():
        X_train[f"{col}_missing"] = X_train[col].isnull().astype(int)
        X_test[f"{col}_missing"]  = X_test[col].isnull().astype(int)
        print(f"  Created missing indicator for {col}")

# Impute with median (robust to outliers)
imputer = SimpleImputer(strategy="median")
X_train[FEATURE_COLS] = imputer.fit_transform(X_train[FEATURE_COLS])
X_test[FEATURE_COLS]  = imputer.transform(X_test[FEATURE_COLS])

# Save imputer
with open(os.path.join(MODEL_DIR, "imputer.pkl"), "wb") as f:
    pickle.dump(imputer, f)

# ══════════════════════════════════════════════════════════════════════════════
# 2.2  STATISTICAL AGGREGATE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Statistical Aggregate Features ─────────────────────────")

def add_stat_features(df, feature_cols):
    """Row-wise statistical aggregates across process parameters."""
    vals = df[feature_cols].values
    df["feat_mean"]     = np.nanmean(vals, axis=1)
    df["feat_std"]      = np.nanstd(vals, axis=1)
    df["feat_median"]   = np.nanmedian(vals, axis=1)
    df["feat_min"]      = np.nanmin(vals, axis=1)
    df["feat_max"]      = np.nanmax(vals, axis=1)
    df["feat_range"]    = df["feat_max"] - df["feat_min"]
    df["feat_skew"]     = pd.DataFrame(vals).apply(lambda x: x.skew(), axis=1).values
    df["feat_kurtosis"] = pd.DataFrame(vals).apply(lambda x: x.kurtosis(), axis=1).values
    df["feat_q25"]      = np.nanpercentile(vals, 25, axis=1)
    df["feat_q75"]      = np.nanpercentile(vals, 75, axis=1)
    df["feat_iqr"]      = df["feat_q75"] - df["feat_q25"]
    df["feat_cv"]       = df["feat_std"] / (df["feat_mean"] + 1e-8)  # coefficient of variation
    df["feat_mad"]      = np.nanmedian(np.abs(vals - np.nanmedian(vals, axis=1, keepdims=True)), axis=1)
    df["feat_sum"]      = np.nansum(vals, axis=1)
    df["feat_energy"]   = np.nansum(vals ** 2, axis=1)
    # Count of features above/below global median
    global_medians = np.nanmedian(vals, axis=0)
    df["feat_above_median_count"] = np.sum(vals > global_medians, axis=1)
    df["feat_below_median_count"] = np.sum(vals < global_medians, axis=1)
    return df

X_train = add_stat_features(X_train, FEATURE_COLS)
X_test  = add_stat_features(X_test, FEATURE_COLS)
print(f"  Added statistical aggregates. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.3  PROCESS STAGE GROUPING FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Process Stage Features ─────────────────────────────────")
# Group features by assumed process stages (hypothetical industrial groupings)
# Stage 1: Pre-rolling (X1-X10), Stage 2: Rolling (X11-X20),
# Stage 3: Cooling (X21-X30), Stage 4: Post-processing (X31-X40),
# Stage 5: Quality metrics (X41-X49)
stages = {
    "stage1": [f"X{i}" for i in range(1, 11)],
    "stage2": [f"X{i}" for i in range(11, 21)],
    "stage3": [f"X{i}" for i in range(21, 31)],
    "stage4": [f"X{i}" for i in range(31, 41)],
    "stage5": [f"X{i}" for i in range(41, 50)],
}

for stage_name, stage_cols in stages.items():
    vals = X_train[stage_cols].values
    X_train[f"{stage_name}_mean"]  = np.mean(vals, axis=1)
    X_train[f"{stage_name}_std"]   = np.std(vals, axis=1)
    X_train[f"{stage_name}_range"] = np.ptp(vals, axis=1)
    X_train[f"{stage_name}_max"]   = np.max(vals, axis=1)
    X_train[f"{stage_name}_min"]   = np.min(vals, axis=1)

    vals_t = X_test[stage_cols].values
    X_test[f"{stage_name}_mean"]  = np.mean(vals_t, axis=1)
    X_test[f"{stage_name}_std"]   = np.std(vals_t, axis=1)
    X_test[f"{stage_name}_range"] = np.ptp(vals_t, axis=1)
    X_test[f"{stage_name}_max"]   = np.max(vals_t, axis=1)
    X_test[f"{stage_name}_min"]   = np.min(vals_t, axis=1)

# Cross-stage interactions
for i, (s1_name, s1_cols) in enumerate(stages.items()):
    for j, (s2_name, s2_cols) in enumerate(stages.items()):
        if j > i:
            X_train[f"{s1_name}_{s2_name}_mean_diff"] = X_train[f"{s1_name}_mean"] - X_train[f"{s2_name}_mean"]
            X_train[f"{s1_name}_{s2_name}_mean_ratio"] = X_train[f"{s1_name}_mean"] / (X_train[f"{s2_name}_mean"] + 1e-8)
            X_test[f"{s1_name}_{s2_name}_mean_diff"]  = X_test[f"{s1_name}_mean"] - X_test[f"{s2_name}_mean"]
            X_test[f"{s1_name}_{s2_name}_mean_ratio"]  = X_test[f"{s1_name}_mean"] / (X_test[f"{s2_name}_mean"] + 1e-8)

print(f"  Added process stage features. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.4  KEY INTERACTION FEATURES (Top correlated pairs)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Interaction Features ───────────────────────────────────")
# Identify top features by target correlation for interaction creation
train_filled_temp = X_train[FEATURE_COLS].copy()
target_corr = train_filled_temp.corrwith(pd.Series(y_train)).abs().sort_values(ascending=False)
top_features = target_corr.head(15).index.tolist()

interaction_count = 0
for i in range(len(top_features)):
    for j in range(i + 1, min(i + 5, len(top_features))):
        f1, f2 = top_features[i], top_features[j]
        # Product
        X_train[f"{f1}_{f2}_product"] = X_train[f1] * X_train[f2]
        X_test[f"{f1}_{f2}_product"]  = X_test[f1]  * X_test[f2]
        # Ratio
        X_train[f"{f1}_{f2}_ratio"] = X_train[f1] / (X_train[f2] + 1e-8)
        X_test[f"{f1}_{f2}_ratio"]  = X_test[f1]  / (X_test[f2] + 1e-8)
        # Difference
        X_train[f"{f1}_{f2}_diff"] = X_train[f1] - X_train[f2]
        X_test[f"{f1}_{f2}_diff"]  = X_test[f1]  - X_test[f2]
        interaction_count += 3

print(f"  Created {interaction_count} interaction features. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.5  NONLINEAR TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Nonlinear Transformations ──────────────────────────────")
for col in top_features[:8]:
    # Log transform (shifted)
    min_val = min(X_train[col].min(), X_test[col].min())
    shift = abs(min_val) + 1 if min_val <= 0 else 0
    X_train[f"{col}_log"]  = np.log1p(X_train[col] + shift)
    X_test[f"{col}_log"]   = np.log1p(X_test[col] + shift)
    # Square root
    X_train[f"{col}_sqrt"] = np.sqrt(np.abs(X_train[col])) * np.sign(X_train[col])
    X_test[f"{col}_sqrt"]  = np.sqrt(np.abs(X_test[col])) * np.sign(X_test[col])
    # Squared
    X_train[f"{col}_sq"]   = X_train[col] ** 2
    X_test[f"{col}_sq"]    = X_test[col] ** 2

print(f"  Added nonlinear transforms. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.6  ANOMALY-ORIENTED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Anomaly-Oriented Features ──────────────────────────────")

# Z-scores for all original features
scaler_z = StandardScaler()
z_scores = scaler_z.fit_transform(X_train[FEATURE_COLS])
z_scores_test = scaler_z.transform(X_test[FEATURE_COLS])

X_train["z_score_max"]      = np.max(np.abs(z_scores), axis=1)
X_train["z_score_mean"]     = np.mean(np.abs(z_scores), axis=1)
X_train["z_score_above3"]   = np.sum(np.abs(z_scores) > 3, axis=1)
X_train["z_score_above2"]   = np.sum(np.abs(z_scores) > 2, axis=1)

X_test["z_score_max"]       = np.max(np.abs(z_scores_test), axis=1)
X_test["z_score_mean"]      = np.mean(np.abs(z_scores_test), axis=1)
X_test["z_score_above3"]    = np.sum(np.abs(z_scores_test) > 3, axis=1)
X_test["z_score_above2"]    = np.sum(np.abs(z_scores_test) > 2, axis=1)

# Mahalanobis-inspired: distance from class 0 centroid
class0_data = X_train.loc[y_train == 0, FEATURE_COLS]
class0_mean = class0_data.mean().values
class0_std  = class0_data.std().values + 1e-8

X_train["dist_from_normal_centroid"] = np.sqrt(
    np.sum(((X_train[FEATURE_COLS].values - class0_mean) / class0_std) ** 2, axis=1)
)
X_test["dist_from_normal_centroid"] = np.sqrt(
    np.sum(((X_test[FEATURE_COLS].values - class0_mean) / class0_std) ** 2, axis=1)
)

# Isolation Forest anomaly score
iso_forest = IsolationForest(
    n_estimators=200, contamination=0.05,
    random_state=RANDOM_SEED, n_jobs=-1
)
iso_forest.fit(X_train[FEATURE_COLS])
X_train["isolation_score"] = -iso_forest.score_samples(X_train[FEATURE_COLS])
X_test["isolation_score"]  = -iso_forest.score_samples(X_test[FEATURE_COLS])

with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
    pickle.dump(iso_forest, f)

print(f"  Added anomaly features. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.7  PCA FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── PCA Features ───────────────────────────────────────────")
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X_train[FEATURE_COLS])
X_scaled_test = scaler_pca.transform(X_test[FEATURE_COLS])

n_components = 10
pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
pca_train = pca.fit_transform(X_scaled)
pca_test  = pca.transform(X_scaled_test)

for i in range(n_components):
    X_train[f"PCA_{i+1}"] = pca_train[:, i]
    X_test[f"PCA_{i+1}"]  = pca_test[:, i]

print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  Added {n_components} PCA components. Shape: {X_train.shape}")

with open(os.path.join(MODEL_DIR, "scaler_pca.pkl"), "wb") as f:
    pickle.dump(scaler_pca, f)
with open(os.path.join(MODEL_DIR, "pca.pkl"), "wb") as f:
    pickle.dump(pca, f)

# ══════════════════════════════════════════════════════════════════════════════
# 2.8  CLUSTERING FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Clustering Features ────────────────────────────────────")

# KMeans
for k in [3, 5, 8]:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    X_train[f"kmeans_{k}_label"] = km.fit_predict(X_scaled)
    X_test[f"kmeans_{k}_label"]  = km.predict(X_scaled_test)
    # Distance to nearest cluster center
    X_train[f"kmeans_{k}_dist"]  = km.transform(X_scaled).min(axis=1)
    X_test[f"kmeans_{k}_dist"]   = km.transform(X_scaled_test).min(axis=1)
    with open(os.path.join(MODEL_DIR, f"kmeans_{k}.pkl"), "wb") as f:
        pickle.dump(km, f)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=RANDOM_SEED, covariance_type="full")
gmm.fit(X_scaled)
X_train["gmm_label"] = gmm.predict(X_scaled)
X_test["gmm_label"]  = gmm.predict(X_scaled_test)
gmm_probs_train = gmm.predict_proba(X_scaled)
gmm_probs_test  = gmm.predict_proba(X_scaled_test)
for i in range(5):
    X_train[f"gmm_prob_{i}"] = gmm_probs_train[:, i]
    X_test[f"gmm_prob_{i}"]  = gmm_probs_test[:, i]
X_train["gmm_score"] = -gmm.score_samples(X_scaled)
X_test["gmm_score"]  = -gmm.score_samples(X_scaled_test)

with open(os.path.join(MODEL_DIR, "gmm.pkl"), "wb") as f:
    pickle.dump(gmm, f)

print(f"  Added clustering features. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.9  QUANTILE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Quantile Rank Features ─────────────────────────────────")
for col in top_features[:6]:
    # Rank percentile within the column
    X_train[f"{col}_rank"] = X_train[col].rank(pct=True)
    X_test[f"{col}_rank"]  = X_test[col].rank(pct=True)

print(f"  Added quantile rank features. Shape: {X_train.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.10  CLEAN UP — Replace inf/nan
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Final Cleanup ──────────────────────────────────────────")
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

# Final imputation for engineered features
final_imputer = SimpleImputer(strategy="median")
feature_names = X_train.columns.tolist()
X_train = pd.DataFrame(final_imputer.fit_transform(X_train), columns=feature_names)
X_test  = pd.DataFrame(final_imputer.transform(X_test), columns=feature_names)

with open(os.path.join(MODEL_DIR, "final_imputer.pkl"), "wb") as f:
    pickle.dump(final_imputer, f)

print(f"\n  Final feature count: {X_train.shape[1]}")
print(f"  Any NaN in train  : {X_train.isnull().any().any()}")
print(f"  Any NaN in test   : {X_test.isnull().any().any()}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE ENGINEERED DATA
# ══════════════════════════════════════════════════════════════════════════════
X_train.to_pickle(os.path.join(OUTPUT_DIR, "X_train_engineered.pkl"))
X_test.to_pickle(os.path.join(OUTPUT_DIR, "X_test_engineered.pkl"))
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "train_ids.npy"), train_ids)
np.save(os.path.join(OUTPUT_DIR, "test_ids.npy"), test_ids)

with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

print("\n" + "=" * 80)
print(f"  ✅ PHASE 2 COMPLETE — {X_train.shape[1]} features engineered")
print("=" * 80)
