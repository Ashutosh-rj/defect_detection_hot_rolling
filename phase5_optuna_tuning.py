"""
================================================================================
 PHASE 5 — OPTUNA HYPERPARAMETER OPTIMIZATION
================================================================================
 Bayesian optimization using Optuna for top 3 models.
 Optimizes F2 score while maintaining recall >= 1.0.
================================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, precision_score, fbeta_score,
    roc_auc_score, average_precision_score
)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import os, sys, pickle

sys.path.insert(0, r"d:\Tata_steel")
from config import *

print("=" * 80)
print("  PHASE 5 — OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

# ── Load data ────────────────────────────────────────────────────────────────
X_train = pd.read_pickle(os.path.join(OUTPUT_DIR, "X_train_engineered.pkl"))
X_test  = pd.read_pickle(os.path.join(OUTPUT_DIR, "X_test_engineered.pkl"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

N_OPTUNA_TRIALS = 40  # Per model

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def objective_lgbm(trial):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight",
                                                 scale_pos_weight * 0.5,
                                                 scale_pos_weight * 2.0),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    oof_preds = np.zeros(len(X_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Find threshold for recall=1.0
    thresholds = np.linspace(0.01, 0.99, 500)
    best_f2 = 0
    for t in thresholds:
        preds = (oof_preds >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        if rec >= 1.0:
            f2 = fbeta_score(y_train, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2

    return best_f2 if best_f2 > 0 else -1


def objective_xgb(trial):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight",
                                                 scale_pos_weight * 0.5,
                                                 scale_pos_weight * 2.0),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "verbosity": 0,
        "use_label_encoder": False,
    }

    oof_preds = np.zeros(len(X_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 500)
    best_f2 = 0
    for t in thresholds:
        preds = (oof_preds >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        if rec >= 1.0:
            f2 = fbeta_score(y_train, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2

    return best_f2 if best_f2 > 0 else -1


def objective_catboost(trial):
    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "auto_class_weights": "Balanced",
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "early_stopping_rounds": 50,
        "eval_metric": "AUC",
        "task_type": "CPU",
    }

    oof_preds = np.zeros(len(X_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = cb.CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 500)
    best_f2 = 0
    for t in thresholds:
        preds = (oof_preds >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        if rec >= 1.0:
            f2 = fbeta_score(y_train, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2

    return best_f2 if best_f2 > 0 else -1


# ══════════════════════════════════════════════════════════════════════════════
# RUN OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
studies = {}

for name, objective in [("LightGBM_Tuned", objective_lgbm),
                         ("XGBoost_Tuned", objective_xgb),
                         ("CatBoost_Tuned", objective_catboost)]:
    print(f"\n{'─' * 60}")
    print(f"  Optimizing {name} ({N_OPTUNA_TRIALS} trials)")
    print(f"{'─' * 60}")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    studies[name] = study
    print(f"  Best F2 Score: {study.best_value:.4f}")
    print(f"  Best Params : {study.best_params}")

# ══════════════════════════════════════════════════════════════════════════════
# RETRAIN WITH BEST PARAMS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  RETRAINING WITH OPTIMIZED HYPERPARAMETERS")
print("=" * 80)

# -- LightGBM Tuned --
best_lgbm_params = studies["LightGBM_Tuned"].best_params
best_lgbm_params.update({
    "n_estimators": 2000, "random_state": RANDOM_SEED,
    "n_jobs": -1, "verbose": -1,
})

oof_lgbm_tuned = np.zeros(len(X_train))
test_lgbm_tuned = np.zeros(len(X_test))
models_lgbm_tuned = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model = lgb.LGBMClassifier(**best_lgbm_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_lgbm_tuned[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgbm_tuned += model.predict_proba(X_test)[:, 1] / N_FOLDS
    models_lgbm_tuned.append(model)

# -- XGBoost Tuned --
best_xgb_params = studies["XGBoost_Tuned"].best_params
best_xgb_params.update({
    "n_estimators": 2000, "random_state": RANDOM_SEED, "n_jobs": -1,
    "eval_metric": "aucpr", "early_stopping_rounds": 50,
    "verbosity": 0, "use_label_encoder": False,
})

oof_xgb_tuned = np.zeros(len(X_train))
test_xgb_tuned = np.zeros(len(X_test))
models_xgb_tuned = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model = xgb.XGBClassifier(**best_xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    oof_xgb_tuned[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb_tuned += model.predict_proba(X_test)[:, 1] / N_FOLDS
    models_xgb_tuned.append(model)

# -- CatBoost Tuned --
best_cat_params = studies["CatBoost_Tuned"].best_params
best_cat_params.update({
    "iterations": 2000, "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED, "verbose": 0,
    "early_stopping_rounds": 50, "eval_metric": "AUC", "task_type": "CPU",
})

oof_cat_tuned = np.zeros(len(X_train))
test_cat_tuned = np.zeros(len(X_test))
models_cat_tuned = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model = cb.CatBoostClassifier(**best_cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    oof_cat_tuned[val_idx] = model.predict_proba(X_val)[:, 1]
    test_cat_tuned += model.predict_proba(X_test)[:, 1] / N_FOLDS
    models_cat_tuned.append(model)

# Evaluate tuned models
from phase3_4_model_training import find_optimal_threshold, evaluate_model

for name, oof in [("LightGBM_Tuned", oof_lgbm_tuned),
                   ("XGBoost_Tuned", oof_xgb_tuned),
                   ("CatBoost_Tuned", oof_cat_tuned)]:
    opt_t, _ = find_optimal_threshold(y_train, oof, min_recall=1.0)
    res = evaluate_model(y_train, oof, threshold=opt_t, model_name=name)
    print(f"\n  {name}:")
    print(f"    Threshold={opt_t:.4f}, Recall={res['recall']:.4f}, "
          f"Precision={res['precision']:.4f}, F2={res['f2']:.4f}, AUC={res['roc_auc']:.4f}")

# Save tuned predictions
tuned_oof = {
    "LightGBM_Tuned": oof_lgbm_tuned,
    "XGBoost_Tuned": oof_xgb_tuned,
    "CatBoost_Tuned": oof_cat_tuned,
}
tuned_test = {
    "LightGBM_Tuned": test_lgbm_tuned,
    "XGBoost_Tuned": test_xgb_tuned,
    "CatBoost_Tuned": test_cat_tuned,
}

with open(os.path.join(OUTPUT_DIR, "tuned_oof_predictions.pkl"), "wb") as f:
    pickle.dump(tuned_oof, f)
with open(os.path.join(OUTPUT_DIR, "tuned_test_predictions.pkl"), "wb") as f:
    pickle.dump(tuned_test, f)

for name, models in [("LightGBM_Tuned", models_lgbm_tuned),
                      ("XGBoost_Tuned", models_xgb_tuned),
                      ("CatBoost_Tuned", models_cat_tuned)]:
    with open(os.path.join(MODEL_DIR, f"{name}_models.pkl"), "wb") as f:
        pickle.dump(models, f)

# Save best params
best_params_all = {
    "LightGBM": best_lgbm_params,
    "XGBoost": best_xgb_params,
    "CatBoost": best_cat_params,
}
with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.pkl"), "wb") as f:
    pickle.dump(best_params_all, f)

print("\n" + "=" * 80)
print("  ✅ PHASE 5 COMPLETE — Optuna optimization done")
print("=" * 80)
