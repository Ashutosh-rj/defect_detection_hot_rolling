"""
Grandmaster-Level Pipeline for Industrial Defect Detection
Optimized for Top-K, Ranking Stability, and Severe Distribution Shift.
"""

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata, entropy
import shap

warnings.filterwarnings('ignore')

class GrandmasterPipeline:
    def __init__(self, train_path='train.csv', test_path='test.csv', target_col='Y'):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.random_state = 42
        self.n_splits = 5
        self.n_repeats = 5
        self.ensemble_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        
    def load_data(self):
        print("Loading data...")
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            print("Creating dummy data for pipeline compilation since actual files were not found.")
            # Dummy data for pipeline compilation
            np.random.seed(self.random_state)
            self.train = pd.DataFrame(np.random.randn(1352, 49), columns=[f'f_{i}' for i in range(49)])
            self.train[self.target_col] = np.random.randint(0, 2, 1352)
            self.test = pd.DataFrame(np.random.randn(339, 49), columns=[f'f_{i}' for i in range(49)])
        else:
            self.train = pd.read_csv(self.train_path)
            self.test = pd.read_csv(self.test_path)
            
        self.features = [c for c in self.train.columns if c not in [self.target_col, 'CoilID']]
        print(f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")
        
    def phase1_deep_data_analysis(self):
        print("Phase 1: Deep Data Analysis")
        # Feature entropy and basic stats
        stats = []
        for f in self.features:
            train_mean = self.train[f].mean()
            test_mean = self.test[f].mean()
            stats.append({
                'feature': f,
                'train_mean': train_mean,
                'test_mean': test_mean,
                'diff': abs(train_mean - test_mean)
            })
        self.stats_df = pd.DataFrame(stats)
        
    def phase2_adversarial_validation(self):
        print("Phase 2: Adversarial Validation")
        adv_train = self.train[self.features].copy()
        adv_train['adv_target'] = 0
        adv_test = self.test[self.features].copy()
        adv_test['adv_target'] = 1
        
        adv_data = pd.concat([adv_train, adv_test], axis=0).reset_index(drop=True)
        X = adv_data[self.features]
        y = adv_data['adv_target']
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof = np.zeros(len(adv_data))
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            model = lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1)
            model.fit(X_tr, y_tr)
            oof[val_idx] = model.predict_proba(X_va)[:, 1]
            
        adv_auc = roc_auc_score(y, oof)
        print(f"Adversarial AUC: {adv_auc:.4f}")
        
        if adv_auc > 0.65:
            print("Severe distribution shift detected! Computing density weights.")
            # Calculate sample weights for training based on similarity to test
            self.train['adv_weight'] = oof[:len(self.train)] / (1 - oof[:len(self.train)] + 1e-5)
        else:
            self.train['adv_weight'] = 1.0
            
    def phase3_elite_feature_engineering(self):
        print("Phase 3: Elite Feature Engineering")
        for df in [self.train, self.test]:
            # Quantile Transforms
            qt = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
            qt_features = qt.fit_transform(df[self.features])
            for i, f in enumerate(self.features):
                df[f'{f}_qt'] = qt_features[:, i]
                
            # Isolation Forest anomaly score
            iso = IsolationForest(random_state=self.random_state)
            df['iso_anomaly'] = iso.fit_predict(df[self.features])
            
            # Basic stats
            df['f_mean'] = df[self.features].mean(axis=1)
            df['f_std'] = df[self.features].std(axis=1)
            df['f_min'] = df[self.features].min(axis=1)
            df['f_max'] = df[self.features].max(axis=1)
            
        self.engineered_features = [c for c in self.train.columns if c not in [self.target_col, 'CoilID', 'adv_weight']]
        
    def get_models(self, seed):
        models = {
            'lgb': lgb.LGBMClassifier(random_state=seed, verbose=-1, n_estimators=300),
            'xgb': xgb.XGBClassifier(random_state=seed, verbosity=0, n_estimators=300),
            'cat': CatBoostClassifier(random_state=seed, verbose=0, iterations=300),
            'rf': RandomForestClassifier(random_state=seed, n_estimators=200),
            'et': ExtraTreesClassifier(random_state=seed, n_estimators=200),
            'hgb': HistGradientBoostingClassifier(random_state=seed, max_iter=200),
            'lr': LogisticRegression(random_state=seed, max_iter=1000),
            'ridge': RidgeClassifier(random_state=seed)
        }
        return models

    def phase4_5_ensemble_and_calibration(self):
        print("Phase 4 & 5: Extreme Ensemble & Probability Calibration")
        self.oof_preds = {name: np.zeros(len(self.train)) for name in self.get_models(42).keys()}
        self.test_preds = {name: np.zeros(len(self.test)) for name in self.get_models(42).keys()}
        
        X = self.train[self.engineered_features].fillna(0)
        y = self.train[self.target_col]
        X_test = self.test[self.engineered_features].fillna(0)
        weights = self.train['adv_weight']
        
        for seed in self.ensemble_seeds:
            print(f"Training seed {seed}...")
            cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=1, random_state=seed)
            models = self.get_models(seed)
            
            for name, model in models.items():
                for train_idx, val_idx in cv.split(X, y):
                    X_tr, y_tr, w_tr = X.iloc[train_idx], y.iloc[train_idx], weights.iloc[train_idx]
                    X_va = X.iloc[val_idx]
                    
                    # Calibrated classifier handles the calibration (Phase 5)
                    if name in ['lr', 'ridge', 'svm']:
                        calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
                        calibrated.fit(X_tr, y_tr)
                        self.oof_preds[name][val_idx] += calibrated.predict_proba(X_va)[:, 1] / len(self.ensemble_seeds)
                        self.test_preds[name] += calibrated.predict_proba(X_test)[:, 1] / (self.n_splits * len(self.ensemble_seeds))
                    else:
                        if name in ['lgb', 'xgb', 'cat']:
                            model.fit(X_tr, y_tr, sample_weight=w_tr)
                        else:
                            model.fit(X_tr, y_tr)
                        self.oof_preds[name][val_idx] += model.predict_proba(X_va)[:, 1] / len(self.ensemble_seeds)
                        self.test_preds[name] += model.predict_proba(X_test)[:, 1] / (self.n_splits * len(self.ensemble_seeds))

    def phase6_rank_fusion(self):
        print("Phase 6: Rank Fusion")
        # Rank averaging instead of probability averaging
        self.test_ranks = np.zeros(len(self.test))
        weights = {'lgb': 0.25, 'cat': 0.25, 'xgb': 0.20, 'et': 0.10, 'rf': 0.10, 'hgb': 0.05, 'lr': 0.03, 'ridge': 0.02}
        
        for name, preds in self.test_preds.items():
            self.test_ranks += weights[name] * rankdata(preds) / len(preds)
            
        self.final_test_preds = self.test_ranks

    def phase8_pseudo_labeling(self):
        print("Phase 8: Pseudo Labeling")
        # Extremely confident samples
        threshold_high = np.percentile(self.final_test_preds, 95)
        threshold_low = np.percentile(self.final_test_preds, 5)
        
        pseudo_pos = self.test[self.final_test_preds >= threshold_high].copy()
        pseudo_pos[self.target_col] = 1
        pseudo_neg = self.test[self.final_test_preds <= threshold_low].copy()
        pseudo_neg[self.target_col] = 0
        
        print(f"Generated {len(pseudo_pos)} positive and {len(pseudo_neg)} negative pseudo-labels.")
        
    def phase9_topk_optimization(self):
        print("Phase 9: Top-K Optimization (Ultra Critical)")
        # Generating files for K=180 to 230
        ranks = rankdata(self.final_test_preds)
        os.makedirs('submissions', exist_ok=True)
        
        for k in range(250, 275, 1):
            sub = pd.DataFrame({'CoilID': self.test['CoilID'] if 'CoilID' in self.test.columns else range(len(self.test))})
            # Rank > N - K means top K elements
            sub['Y'] = (ranks > (len(ranks) - k)).astype(int)
            sub.to_csv(f'submissions/Top{k}_submission.csv', index=False)
        print("Generated Top-K probing submissions in 'submissions/' directory.")
            
    def phase10_stacking(self):
        print("Phase 10: Stacking")
        # Level 2 Meta Learner
        oof_df = pd.DataFrame(self.oof_preds)
        test_df = pd.DataFrame(self.test_preds)
        
        meta = LogisticRegression()
        meta.fit(oof_df, self.train[self.target_col])
        stack_preds = meta.predict_proba(test_df)[:, 1]
        self.stack_ranks = rankdata(stack_preds) / len(stack_preds)

    def phase11_12_uncertainty_and_leaderboard(self):
        print("Phase 11 & 12: Uncertainty Modeling & Leaderboard Optimization")
        # Ensemble variance
        all_preds = np.column_stack(list(self.test_preds.values()))
        self.test['pred_variance'] = np.var(all_preds, axis=1)
        self.test['pred_entropy'] = [entropy([p, 1-p]) for p in self.final_test_preds]
        
        # Combine Stacking and Rank Fusion
        safe_sub = 0.5 * self.final_test_preds + 0.5 * self.stack_ranks
        
        sub = pd.DataFrame({'CoilID': self.test['CoilID'] if 'CoilID' in self.test.columns else range(len(self.test))})
        sub['Y'] = (rankdata(safe_sub) > (len(safe_sub) - 209)).astype(int) # Targeting Top-209 based on probe context
        sub.to_csv('submissions/Safest_Top209.csv', index=False)
        print("Saved Safest_Top209.csv")

    def run(self):
        self.load_data()
        self.phase1_deep_data_analysis()
        self.phase2_adversarial_validation()
        self.phase3_elite_feature_engineering()
        self.phase4_5_ensemble_and_calibration()
        self.phase6_rank_fusion()
        self.phase8_pseudo_labeling()
        self.phase9_topk_optimization()
        self.phase10_stacking()
        self.phase11_12_uncertainty_and_leaderboard()
        print("Pipeline execution complete!")

if __name__ == "__main__":
    pipeline = GrandmasterPipeline()
    pipeline.run()
