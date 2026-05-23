import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import os
import pickle
import sys

# Add config path
sys.path.insert(0, r"d:\Tata_steel")
from config import *

def check_accuracy():
    print("=" * 60)
    print("  MODEL ACCURACY & EVALUATION CHECKER")
    print("=" * 60)
    
    # Load actual labels
    y_train_path = os.path.join(OUTPUT_DIR, "y_train.npy")
    if not os.path.exists(y_train_path):
        print("[ERROR] Training labels not found. Have you run Phase 1-2?")
        return
        
    y_true = np.load(y_train_path)
    
    # Load predictions (Out-Of-Fold predictions from our models)
    oof_path = os.path.join(OUTPUT_DIR, "oof_predictions.pkl")
    if not os.path.exists(oof_path):
        print("[ERROR] Model predictions not found. Have you run Phase 3-7?")
        return
        
    with open(oof_path, "rb") as f:
        oof_dict = pickle.load(f)
        
    # Let's use the Rank_Average ensemble which performed best
    print("\n[INFO] Evaluating final 'Rank_Average' Ensemble Strategy...")
    
    # Rank averaging logic (same as Phase 7)
    df_ranks = pd.DataFrame()
    for name, oof in oof_dict.items():
        df_ranks[name] = pd.Series(oof).rank(pct=True)
    
    ensemble_prob = df_ranks.mean(axis=1).values
    
    # Best threshold for Rank_Average to maintain 100% recall
    # (Pre-calculated during Phase 7)
    threshold = 0.3200 
    
    y_pred = (ensemble_prob >= threshold).astype(int)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n─── 📊 FINAL ENSEMBLE METRICS ────────────────────")
    print(f"  • Overall Accuracy : {acc * 100:.2f}%")
    print(f"  • Recall           : {rec * 100:.2f}%  (Goal: 100% - No defects missed!)")
    print(f"  • Precision        : {prec * 100:.2f}%")
    
    print("\n─── 🔢 CONFUSION MATRIX ──────────────────────────")
    print(f"  True Negatives (Correctly identified as healthy) : {cm[0, 0]}")
    print(f"  False Positives (Healthy flagged as defective)   : {cm[0, 1]}")
    print(f"  False Negatives (Defective flagged as healthy)   : {cm[1, 0]} <-- MUST BE 0")
    print(f"  True Positives (Defect correctly caught)         : {cm[1, 1]}")
    
    print("\n─── 💡 INTERPRETATION ────────────────────────────")
    print("Why is Accuracy ~24%?")
    print("Because the dataset is extremely imbalanced (19.5 to 1 ratio) and our strict")
    print("business rule requires ZERO false negatives (missing an actual defect).")
    print("To achieve 100% Recall, the model must be very conservative and flag many")
    print("healthy items as potential defects (False Positives), which lowers the")
    print("overall Accuracy score.")
    print("=" * 60)

if __name__ == "__main__":
    check_accuracy()
