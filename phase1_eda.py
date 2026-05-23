"""
================================================================================
 PHASE 1 — DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS
================================================================================
 Alpha Defect Detection in Hot Rolling Mills
 Exhaustive EDA: shape, types, missing values, class balance, distributions,
 correlations, outliers, mutual information, and feature-target analysis.
================================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import os, sys

sys.path.insert(0, r"d:\Tata_steel")
from config import *

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
sns.set_palette("magma")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1.1  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  PHASE 1 — DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS")
print("=" * 80)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"\n[INFO] Train shape : {train.shape}")
print(f"[INFO] Test  shape : {test.shape}")
print(f"[INFO] Train cols  : {list(train.columns[:5])} ... {list(train.columns[-5:])}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.2  DATA TYPES & BASIC STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Data Types ─────────────────────────────────────────────")
print(train.dtypes.value_counts())

print("\n─── Descriptive Statistics (Train) ─────────────────────────")
desc = train[FEATURE_COLS].describe().T
desc["skew"]     = train[FEATURE_COLS].skew()
desc["kurtosis"] = train[FEATURE_COLS].kurtosis()
print(desc.round(4).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 1.3  MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Missing Values ─────────────────────────────────────────")
train_missing = train.isnull().sum()
test_missing  = test.isnull().sum()
missing_df = pd.DataFrame({
    "Train_Missing": train_missing,
    "Train_Pct":     (train_missing / len(train) * 100).round(2),
    "Test_Missing":  test_missing,
    "Test_Pct":      (test_missing / len(test) * 100).round(2),
})
missing_df = missing_df[(missing_df["Train_Missing"] > 0) | (missing_df["Test_Missing"] > 0)]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("  No missing values found.")

# Visualize missing values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
train.isnull().sum().plot(kind="bar", ax=axes[0], color="#E94560", alpha=0.8)
axes[0].set_title("Train — Missing Values per Feature")
axes[0].set_ylabel("Count")
test.isnull().sum().plot(kind="bar", ax=axes[1], color="#00D2FF", alpha=0.8)
axes[1].set_title("Test — Missing Values per Feature")
axes[1].set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_missing_values.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.4  DUPLICATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Duplicate Analysis ─────────────────────────────────────")
dup_count = train.duplicated(subset=FEATURE_COLS).sum()
print(f"  Duplicate feature rows in train: {dup_count}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.5  CLASS DISTRIBUTION & IMBALANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Class Distribution ─────────────────────────────────────")
class_counts = train[TARGET_COL].value_counts()
imbalance_ratio = class_counts[0] / class_counts[1]
print(f"  Class 0 (No Defect) : {class_counts[0]} ({class_counts[0]/len(train)*100:.1f}%)")
print(f"  Class 1 (Defect)    : {class_counts[1]} ({class_counts[1]/len(train)*100:.1f}%)")
print(f"  Imbalance Ratio     : {imbalance_ratio:.1f}:1")
print(f"  Severity            : {'EXTREME' if imbalance_ratio > 10 else 'MODERATE' if imbalance_ratio > 3 else 'MILD'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
colors = ["#00D2FF", "#E94560"]
class_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Target Class Distribution", fontweight="bold")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["No Defect (0)", "Defect (1)"], rotation=0)
for i, v in enumerate(class_counts):
    axes[0].text(i, v + 10, f"{v}\n({v/len(train)*100:.1f}%)", ha="center", fontweight="bold")

# Pie chart
axes[1].pie(class_counts, labels=["No Defect", "Defect"], colors=colors,
            autopct="%1.1f%%", startangle=90, explode=(0, 0.1),
            shadow=True, textprops={"fontsize": 12, "fontweight": "bold"})
axes[1].set_title(f"Imbalance Ratio: {imbalance_ratio:.1f}:1", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_class_distribution.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.6  FEATURE DISTRIBUTIONS — SKEWNESS & KURTOSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Skewness & Kurtosis Summary ────────────────────────────")
skew_vals = train[FEATURE_COLS].skew()
kurt_vals = train[FEATURE_COLS].kurtosis()
print(f"  Highly skewed features (|skew| > 2)   : {(skew_vals.abs() > 2).sum()}")
print(f"  Heavy-tailed features  (kurtosis > 7)  : {(kurt_vals > 7).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
skew_vals.sort_values().plot(kind="barh", ax=axes[0], color="#FF6B6B", alpha=0.8)
axes[0].set_title("Feature Skewness", fontweight="bold")
axes[0].axvline(x=0, color="white", linestyle="--", alpha=0.5)

kurt_vals.sort_values().plot(kind="barh", ax=axes[1], color="#4ECDC4", alpha=0.8)
axes[1].set_title("Feature Kurtosis", fontweight="bold")
axes[1].axvline(x=3, color="white", linestyle="--", alpha=0.5, label="Normal=3")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_skewness_kurtosis.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.7  FEATURE DISTRIBUTIONS BY CLASS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Feature Distribution Plots (by class) ──────────────────")
n_feat = len(FEATURE_COLS)
n_cols_plot = 5
n_rows_plot = (n_feat + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(25, 4 * n_rows_plot))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    ax = axes[i]
    for cls, color, label in [(0, "#00D2FF", "No Defect"), (1, "#E94560", "Defect")]:
        subset = train[train[TARGET_COL] == cls][col].dropna()
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Feature Distributions by Class", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_feature_distributions_by_class.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.8  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Correlation Heatmap ────────────────────────────────────")
# Fill missing for correlation computation
train_filled = train[FEATURE_COLS].fillna(train[FEATURE_COLS].median())
corr_matrix = train_filled.corr()

fig, ax = plt.subplots(figsize=(20, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
ax.set_title("Feature Correlation Heatmap (Lower Triangle)", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_correlation_heatmap.png"), bbox_inches="tight")
plt.close()

# Highly correlated pairs
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j],
                              round(corr_matrix.iloc[i, j], 3)))
print(f"  Highly correlated feature pairs (|r| > 0.85): {len(high_corr)}")
for f1, f2, r in sorted(high_corr, key=lambda x: -abs(x[2]))[:10]:
    print(f"    {f1} ↔ {f2} : r = {r}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.9  CORRELATION WITH TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Feature-Target Correlation ─────────────────────────────")
target_corr = train_filled.corrwith(train[TARGET_COL]).sort_values(ascending=False)
print(target_corr.to_string())

fig, ax = plt.subplots(figsize=(14, 8))
colors_bar = ["#E94560" if v > 0 else "#00D2FF" for v in target_corr]
target_corr.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white", linewidth=0.5)
ax.set_title("Feature Correlation with Target (Y)", fontweight="bold")
ax.set_xlabel("Pearson Correlation")
ax.axvline(x=0, color="white", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "06_target_correlation.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.10  MUTUAL INFORMATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Mutual Information Scores ──────────────────────────────")
X_mi = train_filled.copy()
y_mi = train[TARGET_COL].values
mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=False, random_state=RANDOM_SEED)
mi_series = pd.Series(mi_scores, index=FEATURE_COLS).sort_values(ascending=False)
print(mi_series.round(4).to_string())

fig, ax = plt.subplots(figsize=(14, 8))
mi_series.plot(kind="barh", ax=ax, color="#FFD700", edgecolor="white", linewidth=0.5, alpha=0.8)
ax.set_title("Mutual Information Scores (Feature → Target)", fontweight="bold")
ax.set_xlabel("MI Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "07_mutual_information.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.11  OUTLIER DETECTION (IQR-based)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── Outlier Analysis (IQR Method) ──────────────────────────")
outlier_counts = {}
for col in FEATURE_COLS:
    Q1, Q3 = train[col].quantile(0.25), train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = ((train[col] < lower) | (train[col] > upper)).sum()
    outlier_counts[col] = n_outliers

outlier_df = pd.Series(outlier_counts).sort_values(ascending=False)
print(f"  Total features with outliers: {(outlier_df > 0).sum()} / {len(FEATURE_COLS)}")
print(f"  Top 10 features by outlier count:")
print(outlier_df.head(10).to_string())

fig, ax = plt.subplots(figsize=(14, 8))
outlier_df.plot(kind="barh", ax=ax, color="#FF6B6B", edgecolor="white", alpha=0.8)
ax.set_title("Outlier Counts per Feature (IQR Method)", fontweight="bold")
ax.set_xlabel("Number of Outliers")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "08_outlier_counts.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.12  BOX PLOTS FOR TOP DISCRIMINATING FEATURES
# ══════════════════════════════════════════════════════════════════════════════
top_mi_features = mi_series.head(12).index.tolist()

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(top_mi_features):
    ax = axes[i]
    data = [train[train[TARGET_COL] == 0][col].dropna(),
            train[train[TARGET_COL] == 1][col].dropna()]
    bp = ax.boxplot(data, labels=["No Defect", "Defect"], patch_artist=True,
                    boxprops=dict(facecolor="#00D2FF", alpha=0.6),
                    medianprops=dict(color="#E94560", linewidth=2))
    bp["boxes"][1].set_facecolor("#E94560")
    ax.set_title(col, fontweight="bold")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Box Plots — Top Discriminating Features by MI Score",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "09_boxplots_top_features.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 1.13  STATISTICAL TESTS (Kolmogorov-Smirnov)
# ══════════════════════════════════════════════════════════════════════════════
print("\n─── KS Test: Feature Distribution Difference (Class 0 vs 1) ─")
ks_results = {}
for col in FEATURE_COLS:
    d0 = train[train[TARGET_COL] == 0][col].dropna()
    d1 = train[train[TARGET_COL] == 1][col].dropna()
    if len(d0) > 0 and len(d1) > 0:
        stat, pval = stats.ks_2samp(d0, d1)
        ks_results[col] = {"KS_stat": round(stat, 4), "p_value": pval}
ks_df = pd.DataFrame(ks_results).T.sort_values("KS_stat", ascending=False)
print(ks_df.head(15).to_string())
print(f"\n  Features with significantly different distributions (p < 0.05): "
      f"{(ks_df['p_value'] < 0.05).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE PROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════
train.to_pickle(os.path.join(OUTPUT_DIR, "train_raw.pkl"))
test.to_pickle(os.path.join(OUTPUT_DIR, "test_raw.pkl"))

print("\n" + "=" * 80)
print("  ✅ PHASE 1 COMPLETE — All EDA plots saved to:", PLOT_DIR)
print("=" * 80)
