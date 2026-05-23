"""
================================================================================
 PHASE 3 & 4 — MODEL TRAINING, IMBALANCE HANDLING, & HYPERPARAMETER TUNING
================================================================================
 Multi-model training with Optuna hyperparameter optimization.
 Stratified KFold CV, class weight tuning, SMOTE variants.
 Models: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGBM, LR.
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
    precision_recall_curve, classification_report
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, BaggingClassifier
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os, sys, pickle, time, json
from collections import defaultdict

sys.path.insert(0, r"d:\Tata_steel")
from config import *

plt.style.use("dark_background")
sns.set_palette("magma")
plt.rcParams["figure.dpi"] = 120

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ENGINEERED DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  PHASE 3 & 4 — MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

X_train = pd.read_pickle(os.path.join(OUTPUT_DIR, "X_train_engineered.pkl"))
X_test  = pd.read_pickle(os.path.join(OUTPUT_DIR, "X_test_engineered.pkl"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))

with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"[INFO] Class distribution: {np.bincount(y_train.astype(int))}")

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)
print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_prob, min_recall=1.0):
    """
    Find the highest threshold that achieves the required recall,
    then maximize precision among those thresholds.
    """
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

    # If no threshold gives perfect recall, use the lowest threshold
    if best_precision == 0:
        for t in sorted(thresholds):
            preds = (y_prob >= t).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)
            if rec >= min_recall:
                best_thresh = t
                best_precision = precision_score(y_true, preds, zero_division=0)
                break
        else:
            best_thresh = 0.01  # Ultra-conservative fallback

    return best_thresh, best_precision


def evaluate_model(y_true, y_prob, threshold=0.5, model_name="Model"):
    """Comprehensive evaluation at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    f2   = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    ap   = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0
    cm   = confusion_matrix(y_true, y_pred)

    return {
        "model": model_name, "threshold": threshold,
        "recall": rec, "precision": prec, "f1": f1, "f2": f2,
        "roc_auc": auc, "avg_precision": ap, "confusion_matrix": cm
    }


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Storage for OOF predictions and test predictions
oof_predictions = {}
test_predictions = {}
model_results = {}
trained_models = {}

def train_and_evaluate(model_name, create_model_fn, X, y, X_test_df,
                       needs_scaling=False, optuna_trials=0):
    """
    Full CV training pipeline with optional Optuna tuning.
    Returns OOF predictions and averaged test predictions.
    """
    print(f"\n{'─' * 70}")
    print(f"  Training: {model_name}")
    print(f"{'─' * 70}")

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test_df))
    fold_models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)
        else:
            X_test_scaled = X_test_df

        model = create_model_fn()

        # CatBoost needs special handling
        if "CatBoost" in model_name:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        elif "LGBM" in model_name or "LightGBM" in model_name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
        elif "XGBoost" in model_name or "XGB" in model_name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        else:
            model.fit(X_tr, y_tr)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            val_prob = model.decision_function(X_val)
            test_prob = model.decision_function(X_test_scaled)
            # Normalize to [0, 1]
            val_prob = (val_prob - val_prob.min()) / (val_prob.max() - val_prob.min() + 1e-8)
            test_prob = (test_prob - test_prob.min()) / (test_prob.max() - test_prob.min() + 1e-8)

        oof_preds[val_idx] = val_prob
        test_preds += test_prob / N_FOLDS
        fold_models.append(model)

        # Fold evaluation at optimal threshold
        thresh, prec = find_optimal_threshold(y_val, val_prob, min_recall=1.0)
        metrics = evaluate_model(y_val, val_prob, threshold=thresh, model_name=model_name)
        fold_metrics.append(metrics)

        print(f"  Fold {fold_idx+1}: Recall={metrics['recall']:.4f}, "
              f"Precision={metrics['precision']:.4f}, F2={metrics['f2']:.4f}, "
              f"AUC={metrics['roc_auc']:.4f}, Thresh={thresh:.4f}")

    # Overall OOF evaluation
    opt_thresh, opt_prec = find_optimal_threshold(y, oof_preds, min_recall=1.0)
    overall = evaluate_model(y, oof_preds, threshold=opt_thresh, model_name=model_name)

    print(f"\n  ▸ OOF Results ({model_name}):")
    print(f"    Optimal Threshold : {opt_thresh:.4f}")
    print(f"    Recall            : {overall['recall']:.4f}")
    print(f"    Precision         : {overall['precision']:.4f}")
    print(f"    F1 Score          : {overall['f1']:.4f}")
    print(f"    F2 Score          : {overall['f2']:.4f}")
    print(f"    ROC-AUC           : {overall['roc_auc']:.4f}")
    print(f"    Avg Precision     : {overall['avg_precision']:.4f}")
    print(f"    Confusion Matrix  :\n{overall['confusion_matrix']}")

    return oof_preds, test_preds, overall, fold_models, opt_thresh


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: LightGBM (Optimized)
# ══════════════════════════════════════════════════════════════════════════════
def create_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=7,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        is_unbalance=False,
    )

oof_lgbm, test_lgbm, res_lgbm, models_lgbm, thresh_lgbm = \
    train_and_evaluate("LightGBM", create_lgbm, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: XGBoost (Optimized)
# ══════════════════════════════════════════════════════════════════════════════
def create_xgb():
    return xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        verbosity=0,
        use_label_encoder=False,
    )

oof_xgb, test_xgb, res_xgb, models_xgb, thresh_xgb = \
    train_and_evaluate("XGBoost", create_xgb, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: CatBoost (Optimized)
# ══════════════════════════════════════════════════════════════════════════════
def create_catboost():
    return cb.CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric="AUC",
        task_type="CPU",
    )

oof_cat, test_cat, res_cat, models_cat, thresh_cat = \
    train_and_evaluate("CatBoost", create_catboost, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: RandomForest (with class weight)
# ══════════════════════════════════════════════════════════════════════════════
def create_rf():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

oof_rf, test_rf, res_rf, models_rf, thresh_rf = \
    train_and_evaluate("RandomForest", create_rf, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5: ExtraTrees (with class weight)
# ══════════════════════════════════════════════════════════════════════════════
def create_et():
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

oof_et, test_et, res_et, models_et, thresh_et = \
    train_and_evaluate("ExtraTrees", create_et, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 6: HistGradientBoosting
# ══════════════════════════════════════════════════════════════════════════════
def create_histgbm():
    return HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.05,
        max_depth=7,
        min_samples_leaf=10,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
    )

oof_hist, test_hist, res_hist, models_hist, thresh_hist = \
    train_and_evaluate("HistGBM", create_histgbm, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 7: Logistic Regression (scaled)
# ══════════════════════════════════════════════════════════════════════════════
def create_lr():
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
        random_state=RANDOM_SEED,
        penalty="l2",
    )

oof_lr, test_lr, res_lr, models_lr, thresh_lr = \
    train_and_evaluate("LogisticRegression", create_lr, X_train, y_train, X_test,
                       needs_scaling=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 8: LightGBM with DART (diversity)
# ══════════════════════════════════════════════════════════════════════════════
def create_lgbm_dart():
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=50,
        min_child_samples=15,
        subsample=0.75,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        boosting_type="dart",
        drop_rate=0.1,
        random_state=RANDOM_SEED + 1,
        n_jobs=-1,
        verbose=-1,
    )

oof_lgbm_dart, test_lgbm_dart, res_lgbm_dart, models_lgbm_dart, thresh_lgbm_dart = \
    train_and_evaluate("LightGBM_DART", create_lgbm_dart, X_train, y_train, X_test)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  MODEL COMPARISON — CROSS-VALIDATION RESULTS")
print("=" * 80)

all_results = [res_lgbm, res_xgb, res_cat, res_rf, res_et, res_hist, res_lr, res_lgbm_dart]
results_df = pd.DataFrame([{
    "Model": r["model"], "Threshold": r["threshold"],
    "Recall": r["recall"], "Precision": r["precision"],
    "F1": r["f1"], "F2": r["f2"],
    "ROC-AUC": r["roc_auc"], "Avg-Precision": r["avg_precision"]
} for r in all_results])
results_df = results_df.sort_values("F2", ascending=False)
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE ALL OOF & TEST PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
oof_dict = {
    "LightGBM": oof_lgbm, "XGBoost": oof_xgb, "CatBoost": oof_cat,
    "RandomForest": oof_rf, "ExtraTrees": oof_et, "HistGBM": oof_hist,
    "LogisticRegression": oof_lr, "LightGBM_DART": oof_lgbm_dart,
}
test_dict = {
    "LightGBM": test_lgbm, "XGBoost": test_xgb, "CatBoost": test_cat,
    "RandomForest": test_rf, "ExtraTrees": test_et, "HistGBM": test_hist,
    "LogisticRegression": test_lr, "LightGBM_DART": test_lgbm_dart,
}
thresh_dict = {
    "LightGBM": thresh_lgbm, "XGBoost": thresh_xgb, "CatBoost": thresh_cat,
    "RandomForest": thresh_rf, "ExtraTrees": thresh_et, "HistGBM": thresh_hist,
    "LogisticRegression": thresh_lr, "LightGBM_DART": thresh_lgbm_dart,
}

with open(os.path.join(OUTPUT_DIR, "oof_predictions.pkl"), "wb") as f:
    pickle.dump(oof_dict, f)
with open(os.path.join(OUTPUT_DIR, "test_predictions.pkl"), "wb") as f:
    pickle.dump(test_dict, f)
with open(os.path.join(OUTPUT_DIR, "thresholds.pkl"), "wb") as f:
    pickle.dump(thresh_dict, f)

# Save models
for name, models in [("LightGBM", models_lgbm), ("XGBoost", models_xgb),
                      ("CatBoost", models_cat), ("RandomForest", models_rf),
                      ("ExtraTrees", models_et), ("HistGBM", models_hist),
                      ("LogisticRegression", models_lr), ("LightGBM_DART", models_lgbm_dart)]:
    with open(os.path.join(MODEL_DIR, f"{name}_models.pkl"), "wb") as f:
        pickle.dump(models, f)

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — PR CURVES FOR ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(oof_dict)))
for (name, oof), color in zip(oof_dict.items(), colors):
    precision_arr, recall_arr, _ = precision_recall_curve(y_train, oof)
    ap = average_precision_score(y_train, oof)
    ax.plot(recall_arr, precision_arr, label=f"{name} (AP={ap:.4f})", color=color, linewidth=2)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — All Models (OOF)", fontweight="bold", fontsize=14)
ax.legend(loc="best", fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "10_pr_curves_all_models.png"), bbox_inches="tight")
plt.close()

print("\n" + "=" * 80)
print("  ✅ PHASE 3 & 4 COMPLETE — All models trained and evaluated")
print("=" * 80)
