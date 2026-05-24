import numpy as np
from sklearn.metrics import *

P = 265
N = 74

def test_metrics(TP, FP):
    FN = P - TP
    TN = N - FP
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*TP + [0]*FN + [1]*FP + [0]*TN)
    
    print(f"TP={TP}, FP={FP}")
    print("Acc:", accuracy_score(y_true, y_pred) * 100)
    print("Bal Acc:", balanced_accuracy_score(y_true, y_pred) * 100)
    print("F1 Mac:", f1_score(y_true, y_pred, average='macro') * 100)
    print("F1 Mic:", f1_score(y_true, y_pred, average='micro') * 100)
    print("F1 Wgt:", f1_score(y_true, y_pred, average='weighted') * 100)
    print("F1 Bin:", f1_score(y_true, y_pred) * 100)
    print("Prec:", precision_score(y_true, y_pred) * 100)
    print("Rec:", recall_score(y_true, y_pred) * 100)
    print("Jaccard:", jaccard_score(y_true, y_pred) * 100)
    print("Matthew:", matthews_corrcoef(y_true, y_pred) * 100)
    print("-" * 20)

test_metrics(264, 7)
test_metrics(265, 6)
test_metrics(263, 8)
test_metrics(262, 9)
