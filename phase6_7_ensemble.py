"""
================================================================================
 PHASE 6 — THRESHOLD ENGINEERING & PHASE 7 — ENSEMBLE SYSTEM
================================================================================
 Aggressive threshold optimization, ensemble construction with stacking,
 weighted averaging, and final submission generation.
================================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, precision_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import os, sys, pickle

sys.path.insert(0, r"d:\Tata_steel")
from config import *

plt.style.use("dark_background")
plt.rcParams["figure.dpi"] = 120

print("=" * 80)
print("  PHASE 6 & 7 — THRESHOLD ENGINEERING + ENSEMBLE SYSTEM")
print("=" * 80)

# ── Load predictions ─────────────────────────────────────────────────────────
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
test_ids = np.load(os.path.join(OUTPUT_DIR, "test_ids.npy"))

with open(os.path.join(OUTPUT_DIR, "oof_predictions.pkl"), "rb") as f:
    oof_dict = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, "test_predictions.pkl"), "rb") as f:
    test_dict = pickle.load(f)

# Load tuned predictions if available
try:
    with open(os.path.join(OUTPUT_DIR, "tuned_oof_predictions.pkl"), "rb") as f:
        tuned_oof = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "tuned_test_predictions.pkl"), "rb") as f:
        tuned_test = pickle.load(f)
    oof_dict.update(tuned_oof)
    test_dict.update(tuned_test)
    print("[INFO] Loaded tuned model predictions")
except FileNotFoundError:
    print("[INFO] No tuned predictions found, using base models only")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — THRESHOLD ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("  PHASE 6 — THRESHOLD ENGINEERING")
print("─" * 80)

def comprehensive_threshold_analysis(y_true, y_prob, model_name):
    """Full threshold analysis with visualization."""
    thresholds = np.linspace(0.01, 0.99, THRESHOLD_SEARCH_STEPS)
    recalls = []
    precisions = []
    f1s = []
    f2s = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_true, preds, zero_division=0))
        precisions.append(precision_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))
        f2s.append(fbeta_score(y_true, preds, beta=2, zero_division=0))

    # Find optimal threshold: recall=1.0, max precision
    best_thresh = 0.01
    best_prec = 0
    for t, rec, prec in zip(thresholds, recalls, precisions):
        if rec >= 1.0 and prec > best_prec:
            best_prec = prec
            best_thresh = t

    return {
        "thresholds": thresholds,
        "recalls": np.array(recalls),
        "precisions": np.array(precisions),
        "f1s": np.array(f1s),
        "f2s": np.array(f2s),
        "optimal_threshold": best_thresh,
        "optimal_precision": best_prec,
    }

# Run threshold analysis for all models
threshold_results = {}
for name, oof in oof_dict.items():
    result = comprehensive_threshold_analysis(y_train, oof, name)
    threshold_results[name] = result
    print(f"  {name:25s} → Optimal Threshold={result['optimal_threshold']:.4f}, "
          f"Precision@Recall=1.0: {result['optimal_precision']:.4f}")

# ── Threshold visualization (top models) ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Threshold vs Recall
ax = axes[0, 0]
for name, res in threshold_results.items():
    ax.plot(res["thresholds"], res["recalls"], label=name, linewidth=1.5)
ax.set_xlabel("Threshold")
ax.set_ylabel("Recall")
ax.set_title("Threshold vs Recall", fontweight="bold")
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Recall=1.0 target")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Threshold vs Precision
ax = axes[0, 1]
for name, res in threshold_results.items():
    ax.plot(res["thresholds"], res["precisions"], label=name, linewidth=1.5)
ax.set_xlabel("Threshold")
ax.set_ylabel("Precision")
ax.set_title("Threshold vs Precision", fontweight="bold")
ax.axhline(y=0.9, color="gold", linestyle="--", alpha=0.7, label="Precision=0.9 target")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: Threshold vs F2
ax = axes[1, 0]
for name, res in threshold_results.items():
    ax.plot(res["thresholds"], res["f2s"], label=name, linewidth=1.5)
ax.set_xlabel("Threshold")
ax.set_ylabel("F2 Score")
ax.set_title("Threshold vs F2 Score", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: PR Curves
ax = axes[1, 1]
for name, oof in oof_dict.items():
    prec_arr, rec_arr, _ = precision_recall_curve(y_train, oof)
    ap = average_precision_score(y_train, oof)
    ax.plot(rec_arr, prec_arr, label=f"{name} (AP={ap:.4f})", linewidth=1.5)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("PHASE 6 — Threshold Engineering Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "11_threshold_engineering.png"), bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — ENSEMBLE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("  PHASE 7 — ENSEMBLE SYSTEM")
print("─" * 80)

# ── 7.1 Simple Average Ensemble ─────────────────────────────────────────────
print("\n  ▸ Ensemble 1: Simple Average")
all_oof = np.column_stack([oof_dict[k] for k in oof_dict])
all_test = np.column_stack([test_dict[k] for k in test_dict])

oof_simple_avg = all_oof.mean(axis=1)
test_simple_avg = all_test.mean(axis=1)

# ── 7.2 Weighted Average Ensemble (optimized) ───────────────────────────────
print("  ▸ Ensemble 2: Optimized Weighted Average")

def optimize_weights(oof_matrix, y_true, n_iter=10000):
    """Find optimal ensemble weights via random search."""
    n_models = oof_matrix.shape[1]
    best_f2 = 0
    best_weights = np.ones(n_models) / n_models
    best_thresh = 0.1

    for _ in range(n_iter):
        # Random weights (Dirichlet distribution)
        weights = np.random.dirichlet(np.ones(n_models))
        blended = oof_matrix @ weights

        # Find best threshold for this weight combo
        thresholds = np.linspace(0.01, 0.5, 200)
        for t in thresholds:
            preds = (blended >= t).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)
            if rec >= 1.0:
                f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
                prec = precision_score(y_true, preds, zero_division=0)
                if f2 > best_f2:
                    best_f2 = f2
                    best_weights = weights.copy()
                    best_thresh = t
                break  # highest threshold with recall=1.0

    return best_weights, best_thresh, best_f2

weights, ens_thresh, ens_f2 = optimize_weights(all_oof, y_train)
model_names = list(oof_dict.keys())
print(f"  Optimized Weights:")
for name, w in zip(model_names, weights):
    print(f"    {name:25s} : {w:.4f}")
print(f"  Ensemble Threshold: {ens_thresh:.4f}")
print(f"  Ensemble F2 Score : {ens_f2:.4f}")

oof_weighted = all_oof @ weights
test_weighted = all_test @ weights

# ── 7.3 Top-3 Boosting Ensemble ─────────────────────────────────────────────
print("\n  ▸ Ensemble 3: Top-3 Gradient Boosting Models Only")
# Use only LightGBM, XGBoost, CatBoost (and tuned variants)
boost_models = [k for k in oof_dict.keys() if any(s in k for s in ["LightGBM", "XGBoost", "CatBoost"])]
print(f"    Models: {boost_models}")

boost_oof = np.column_stack([oof_dict[k] for k in boost_models])
boost_test = np.column_stack([test_dict[k] for k in boost_models])

boost_weights, boost_thresh, boost_f2 = optimize_weights(boost_oof, y_train)
print(f"    Weights: {dict(zip(boost_models, boost_weights.round(4)))}")
print(f"    F2={boost_f2:.4f}, Threshold={boost_thresh:.4f}")

oof_boost_ens = boost_oof @ boost_weights
test_boost_ens = boost_test @ boost_weights

# ── 7.4 Stacking Ensemble (Meta-Learner) ────────────────────────────────────
print("\n  ▸ Ensemble 4: Stacking with Logistic Regression Meta-Learner")

skf_stack = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_stack_pred = np.zeros(len(y_train))
test_stack_pred = np.zeros(len(all_test))
meta_models = []

for fold_idx, (train_idx, val_idx) in enumerate(skf_stack.split(all_oof, y_train)):
    X_meta_tr = all_oof[train_idx]
    X_meta_val = all_oof[val_idx]
    y_meta_tr = y_train[train_idx]

    meta = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs",
        max_iter=5000, random_state=RANDOM_SEED
    )
    meta.fit(X_meta_tr, y_meta_tr)
    oof_stack_pred[val_idx] = meta.predict_proba(X_meta_val)[:, 1]
    test_stack_pred += meta.predict_proba(all_test)[:, 1] / N_FOLDS
    meta_models.append(meta)

# ── 7.5 Rank-Based Ensemble ─────────────────────────────────────────────────
print("  ▸ Ensemble 5: Rank-Based Ensemble")
# Convert probabilities to ranks, then average
oof_ranks = np.column_stack([
    pd.Series(oof_dict[k]).rank(pct=True).values for k in oof_dict
])
test_ranks = np.column_stack([
    pd.Series(test_dict[k]).rank(pct=True).values for k in test_dict
])
oof_rank_avg = oof_ranks.mean(axis=1)
test_rank_avg = test_ranks.mean(axis=1)

# ── 7.6  Conservative "OR" Ensemble ─────────────────────────────────────────
print("  ▸ Ensemble 6: Conservative OR Ensemble (any model flags defect → defect)")
# If ANY model (at its optimal threshold) flags a sample as defect, mark it

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ALL ENSEMBLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("  ENSEMBLE COMPARISON")
print("─" * 80)

def find_optimal_threshold(y_true, y_prob, min_recall=1.0):
    thresholds = np.linspace(0.01, 0.99, THRESHOLD_SEARCH_STEPS)
    best_thresh = 0.01
    best_precision = 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        if rec >= min_recall and prec > best_precision:
            best_precision = prec
            best_thresh = t
    if best_precision == 0:
        best_thresh = 0.01
    return best_thresh, best_precision

ensemble_dict = {
    "Simple_Average": (oof_simple_avg, test_simple_avg),
    "Weighted_Average": (oof_weighted, test_weighted),
    "Top3_Boost": (oof_boost_ens, test_boost_ens),
    "Stacking_LR": (oof_stack_pred, test_stack_pred),
    "Rank_Average": (oof_rank_avg, test_rank_avg),
}

ensemble_results = []
for ens_name, (oof_ens, test_ens) in ensemble_dict.items():
    opt_t, opt_p = find_optimal_threshold(y_train, oof_ens, min_recall=1.0)
    y_pred = (oof_ens >= opt_t).astype(int)
    rec  = recall_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred, zero_division=0)
    f1   = f1_score(y_train, y_pred, zero_division=0)
    f2   = fbeta_score(y_train, y_pred, beta=2, zero_division=0)
    auc  = roc_auc_score(y_train, oof_ens)
    ap   = average_precision_score(y_train, oof_ens)
    cm   = confusion_matrix(y_train, y_pred)

    ensemble_results.append({
        "Ensemble": ens_name, "Threshold": round(opt_t, 4),
        "Recall": round(rec, 4), "Precision": round(prec, 4),
        "F1": round(f1, 4), "F2": round(f2, 4),
        "ROC_AUC": round(auc, 4), "Avg_Precision": round(ap, 4),
    })
    print(f"\n  {ens_name}:")
    print(f"    Threshold={opt_t:.4f}, Recall={rec:.4f}, Precision={prec:.4f}, "
          f"F2={f2:.4f}, AUC={auc:.4f}")
    print(f"    Confusion Matrix:\n{cm}")

ens_df = pd.DataFrame(ensemble_results).sort_values("F2", ascending=False)
print("\n" + ens_df.to_string(index=False))
ens_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_comparison.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# SELECT BEST ENSEMBLE & GENERATE SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 80)
print("  GENERATING FINAL SUBMISSIONS")
print("─" * 80)

# Strategy: Pick ensemble with highest F2 at recall=1.0
best_ens = ens_df.iloc[0]
best_ens_name = best_ens["Ensemble"]
print(f"\n  Selected Best Ensemble: {best_ens_name}")
print(f"    F2={best_ens['F2']}, Recall={best_ens['Recall']}, Precision={best_ens['Precision']}")

best_oof, best_test = ensemble_dict[best_ens_name]
best_threshold, _ = find_optimal_threshold(y_train, best_oof, min_recall=1.0)

# Submission 1: Best ensemble
sub1 = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: (best_test >= best_threshold).astype(int)
})
sub1.to_csv(os.path.join(SUBMISSION_DIR, "submission_best_ensemble.csv"), index=False)
print(f"\n  Submission 1 (Best Ensemble): {sub1[TARGET_COL].sum()} defects predicted")

# Submission 2: Ultra-conservative (lower threshold)
ultra_thresh = best_threshold * 0.5
sub2 = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: (best_test >= ultra_thresh).astype(int)
})
sub2.to_csv(os.path.join(SUBMISSION_DIR, "submission_ultra_conservative.csv"), index=False)
print(f"  Submission 2 (Ultra-Conservative): {sub2[TARGET_COL].sum()} defects predicted")

# Submission 3: Weighted average ensemble
wt_thresh, _ = find_optimal_threshold(y_train, oof_weighted, min_recall=1.0)
sub3 = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: (test_weighted >= wt_thresh).astype(int)
})
sub3.to_csv(os.path.join(SUBMISSION_DIR, "submission_weighted_avg.csv"), index=False)
print(f"  Submission 3 (Weighted Avg): {sub3[TARGET_COL].sum()} defects predicted")

# Final expected_submission.csv → Best ensemble
sub1.to_csv(os.path.join(BASE_DIR, "expected_submission.csv"), index=False)
print(f"\n  ★ expected_submission.csv → saved to {BASE_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
y_pred_final = (best_oof >= best_threshold).astype(int)
cm = confusion_matrix(y_train, y_pred_final)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r", ax=ax,
            xticklabels=["No Defect", "Defect"],
            yticklabels=["No Defect", "Defect"],
            annot_kws={"size": 18, "fontweight": "bold"},
            linewidths=2, linecolor="white")
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("Actual", fontsize=13)
ax.set_title(f"Confusion Matrix — {best_ens_name}\n"
             f"(Threshold={best_threshold:.4f}, Recall={best_ens['Recall']}, "
             f"Precision={best_ens['Precision']})",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "12_confusion_matrix_final.png"), bbox_inches="tight")
plt.close()

# Save ensemble data
with open(os.path.join(OUTPUT_DIR, "ensemble_dict.pkl"), "wb") as f:
    pickle.dump({k: {"oof": v[0], "test": v[1]} for k, v in ensemble_dict.items()}, f)
with open(os.path.join(OUTPUT_DIR, "best_ensemble_info.pkl"), "wb") as f:
    pickle.dump({
        "name": best_ens_name,
        "threshold": best_threshold,
        "weights": weights if best_ens_name == "Weighted_Average" else None,
        "model_names": model_names,
    }, f)

print("\n" + "=" * 80)
print("  ✅ PHASE 6 & 7 COMPLETE — Ensemble built, submissions generated")
print("=" * 80)
