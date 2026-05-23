import numpy as np
import pandas as pd
import os
import sys
import pickle
import warnings
import argparse
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Force utf-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

sys.path.insert(0, r"d:\Tata_steel")
from config import *

def add_stat_features(df, feature_cols):
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
    df["feat_cv"]       = df["feat_std"] / (df["feat_mean"] + 1e-8)
    df["feat_mad"]      = np.nanmedian(np.abs(vals - np.nanmedian(vals, axis=1, keepdims=True)), axis=1)
    df["feat_sum"]      = np.nansum(vals, axis=1)
    df["feat_energy"]   = np.nansum(vals ** 2, axis=1)
    
    global_medians = np.nanmedian(vals, axis=0)
    df["feat_above_median_count"] = np.sum(vals > global_medians, axis=1)
    df["feat_below_median_count"] = np.sum(vals < global_medians, axis=1)
    return df

class InferencePipeline:
    def __init__(self, mode="accuracy"):
        self.mode = mode
        self.load_transformers()
        self.load_models()

    def load_transformers(self):
        print("[INFO] Loading preprocessing transformers...")
        with open(os.path.join(MODEL_DIR, "imputer.pkl"), "rb") as f:
            self.imputer = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "rb") as f:
            self.iso_forest_base = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler_pca.pkl"), "rb") as f:
            self.scaler_pca = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "pca.pkl"), "rb") as f:
            self.pca = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "kmeans_3.pkl"), "rb") as f:
            self.km3 = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "kmeans_5.pkl"), "rb") as f:
            self.km5 = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "kmeans_8.pkl"), "rb") as f:
            self.km8 = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "gmm.pkl"), "rb") as f:
            self.gmm = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "final_imputer.pkl"), "rb") as f:
            self.final_imputer = pickle.load(f)
        with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), "rb") as f:
            self.feature_names = pickle.load(f)

        if self.mode == "accuracy":
            with open(os.path.join(MODEL_DIR, "iso_phase9.pkl"), "rb") as f:
                self.iso_phase9 = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "autoencoder_scaler.pkl"), "rb") as f:
                self.ae_scaler = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "autoencoder_phase9.pkl"), "rb") as f:
                self.autoencoder = pickle.load(f)

    def load_models(self):
        print(f"[INFO] Loading models for mode: {self.mode}...")
        if self.mode == "accuracy":
            with open(os.path.join(MODEL_DIR, "lgb_phase9.pkl"), "rb") as f:
                self.lgb_models = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "cat_phase9.pkl"), "rb") as f:
                self.cat_models = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "xgb_phase9.pkl"), "rb") as f:
                self.xgb_models = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "threshold_phase9.pkl"), "rb") as f:
                self.threshold = pickle.load(f)

    def preprocess(self, df_raw):
        print("[INFO] Preprocessing features...")
        X = df_raw[FEATURE_COLS].copy()
        
        # 1. Missing imputation
        for col in FEATURE_COLS:
            X[f"{col}_missing"] = X[col].isnull().astype(int)
        X[FEATURE_COLS] = self.imputer.transform(X[FEATURE_COLS])

        # 2. Statistical
        X = add_stat_features(X, FEATURE_COLS)

        # 3. Process Stages
        stages = {
            "stage1": [f"X{i}" for i in range(1, 11)],
            "stage2": [f"X{i}" for i in range(11, 21)],
            "stage3": [f"X{i}" for i in range(21, 31)],
            "stage4": [f"X{i}" for i in range(31, 41)],
            "stage5": [f"X{i}" for i in range(41, 50)],
        }
        for stage_name, stage_cols in stages.items():
            vals = X[stage_cols].values
            X[f"{stage_name}_mean"]  = np.mean(vals, axis=1)
            X[f"{stage_name}_std"]   = np.std(vals, axis=1)
            X[f"{stage_name}_range"] = np.ptp(vals, axis=1)
            X[f"{stage_name}_max"]   = np.max(vals, axis=1)
            X[f"{stage_name}_min"]   = np.min(vals, axis=1)

        for i, (s1_name, s1_cols) in enumerate(stages.items()):
            for j, (s2_name, s2_cols) in enumerate(stages.items()):
                if j > i:
                    X[f"{s1_name}_{s2_name}_mean_diff"] = X[f"{s1_name}_mean"] - X[f"{s2_name}_mean"]
                    X[f"{s1_name}_{s2_name}_mean_ratio"] = X[f"{s1_name}_mean"] / (X[f"{s2_name}_mean"] + 1e-8)

        # 4. Interactions (top 15 features from phase 2)
        top_features = ['X35', 'X13', 'X36', 'X34', 'X10', 'X30', 'X31', 'X32', 'X29', 'X37', 'X33', 'X41', 'X7', 'X4', 'X15']
        for i in range(len(top_features)):
            for j in range(i + 1, min(i + 5, len(top_features))):
                f1, f2 = top_features[i], top_features[j]
                X[f"{f1}_{f2}_product"] = X[f1] * X[f2]
                X[f"{f1}_{f2}_ratio"]   = X[f1] / (X[f2] + 1e-8)
                X[f"{f1}_{f2}_diff"]    = X[f1] - X[f2]

        # 5. Non-linear transforms
        for col in top_features[:8]:
            min_val = X[col].min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            X[f"{col}_log"]  = np.log1p(X[col] + shift)
            X[f"{col}_sqrt"] = np.sqrt(np.abs(X[col])) * np.sign(X[col])
            X[f"{col}_sq"]   = X[col] ** 2

        # 6. Anomaly features
        scaler_z = StandardScaler()
        z_scores = scaler_z.fit_transform(X[FEATURE_COLS])
        X["z_score_max"]      = np.max(np.abs(z_scores), axis=1)
        X["z_score_mean"]     = np.mean(np.abs(z_scores), axis=1)
        X["z_score_above3"]   = np.sum(np.abs(z_scores) > 3, axis=1)
        X["z_score_above2"]   = np.sum(np.abs(z_scores) > 2, axis=1)

        # Base Isolation Forest
        X["isolation_score"]  = -self.iso_forest_base.score_samples(X[FEATURE_COLS])

        # Dist from centroid skipped (was hardcoded on training set)
        X["dist_from_normal_centroid"] = 0  # Replaced to avoid needing the exact train array here

        # 7. PCA
        X_scaled = self.scaler_pca.transform(X[FEATURE_COLS])
        pca_feats = self.pca.transform(X_scaled)
        for i in range(10):
            X[f"PCA_{i+1}"] = pca_feats[:, i]

        # 8. Clustering
        for k, km in zip([3, 5, 8], [self.km3, self.km5, self.km8]):
            X[f"kmeans_{k}_label"] = km.predict(X_scaled)
            X[f"kmeans_{k}_dist"]   = km.transform(X_scaled).min(axis=1)

        X["gmm_label"]  = self.gmm.predict(X_scaled)
        gmm_probs = self.gmm.predict_proba(X_scaled)
        for i in range(5):
            X[f"gmm_prob_{i}"]  = gmm_probs[:, i]
        X["gmm_score"]  = -self.gmm.score_samples(X_scaled)

        # 9. Quantile features
        for col in top_features[:6]:
            X[f"{col}_rank"]  = X[col].rank(pct=True)

        # Cleanup
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Ensure exact column order before imputation
        X = X[self.feature_names]
        X = pd.DataFrame(self.final_imputer.transform(X), columns=self.feature_names)

        # 10. Phase 9 Specific Anomaly Scoring
        if self.mode == "accuracy":
            X['anomaly_score'] = self.iso_phase9.decision_function(X)
            X_scaled_ae = self.ae_scaler.transform(X.fillna(0))
            reconstruction = self.autoencoder.predict(X_scaled_ae)
            X['reconstruction_error'] = np.mean(np.square(X_scaled_ae - reconstruction), axis=1)

        return X

    def predict(self, df_raw):
        X_processed = self.preprocess(df_raw)
        
        print("[INFO] Running ensemble inference...")
        preds_lgb = np.zeros(len(X_processed))
        preds_cat = np.zeros(len(X_processed))
        preds_xgb = np.zeros(len(X_processed))

        for i in range(5):
            preds_lgb += self.lgb_models[i].predict_proba(X_processed)[:, 1] / 5.0
            preds_cat += self.cat_models[i].predict_proba(X_processed)[:, 1] / 5.0
            preds_xgb += self.xgb_models[i].predict_proba(X_processed)[:, 1] / 5.0

        ensemble_probs = (preds_lgb + preds_cat + preds_xgb) / 3.0
        
        print(f"[INFO] Applying threshold: {self.threshold:.4f}")
        final_preds = (ensemble_probs >= self.threshold).astype(int)
        
        return final_preds, ensemble_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Inference Pipeline")
    parser.add_argument("--input", type=str, default=TEST_PATH, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=os.path.join(SUBMISSION_DIR, "predictions.csv"), help="Path to output CSV")
    parser.add_argument("--mode", type=str, default="accuracy", choices=["accuracy"], help="Optimization mode")
    
    args = parser.parse_args()

    print("=" * 80)
    print(f"  INDUSTRIAL ML INFERENCE PIPELINE — Mode: {args.mode.upper()}")
    print("=" * 80)

    try:
        df = pd.read_csv(args.input)
        print(f"[INFO] Loaded input data: {df.shape}")
        
        pipeline = InferencePipeline(mode=args.mode)
        preds, probs = pipeline.predict(df)
        
        output_df = pd.DataFrame({
            ID_COL: df[ID_COL] if ID_COL in df.columns else np.arange(len(df)),
            TARGET_COL: preds,
            "Probability": probs
        })
        
        output_df.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Predictions saved to {args.output}")
        print(f"[SUMMARY] Flagged {preds.sum()} defects out of {len(preds)} coils.")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {str(e)}")
        sys.exit(1)
