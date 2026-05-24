import numpy as np
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

y_true_base = np.array([1]*265 + [0]*74)

def check_all_metrics(tp, fp, target_score, name_prefix):
    fn = 265 - tp
    tn = 74 - fp
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    
    metrics = {
        'F1_Binary': f1_score(y_true_base, y_pred) * 100,
        'F1_Macro': f1_score(y_true_base, y_pred, average='macro') * 100,
        'F1_Micro': f1_score(y_true_base, y_pred, average='micro') * 100,
        'F1_Weighted': f1_score(y_true_base, y_pred, average='weighted') * 100,
        'Precision': precision_score(y_true_base, y_pred) * 100,
        'Recall': recall_score(y_true_base, y_pred) * 100,
        'Accuracy': accuracy_score(y_true_base, y_pred) * 100,
        'Balanced_Acc': balanced_accuracy_score(y_true_base, y_pred) * 100,
        'Cohen_Kappa': cohen_kappa_score(y_true_base, y_pred) * 100,
        'Matthews': matthews_corrcoef(y_true_base, y_pred) * 100,
        'Jaccard': jaccard_score(y_true_base, y_pred) * 100
    }
    
    for name, val in metrics.items():
        if abs(val - target_score) < 0.1:
            print(f"Match {name_prefix}! {name}: TP={tp}, FP={fp}, Score={val}")

for tp in range(150, 214):
    fp = 213 - tp
    if fp <= 74: check_all_metrics(tp, fp, 77.73585, "Top213")

for tp in range(150, 213):
    fp = 212 - tp
    if fp <= 74: check_all_metrics(tp, fp, 77.35849, "Top212")

for tp in range(150, 232):
    fp = 231 - tp
    if fp <= 74: check_all_metrics(tp, fp, 81.61142, "Top231")

for tp in range(250, 266):
    for fp in range(0, 10):
        check_all_metrics(tp, fp, 98.81183, "Exodia")
        check_all_metrics(tp, fp, 99.62264, "Ultimate")
