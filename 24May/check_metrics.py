import numpy as np
from sklearn.metrics import f1_score, accuracy_score

P = 265
N = 74

def get_f1(tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='macro') * 100

def get_micro_f1(tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='micro') * 100

def get_acc(tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return accuracy_score(y_true, y_pred) * 100

print(f"Macro F1 (256 TP, 0 FP): {get_f1(256, 0):.5f}")
print(f"Micro F1 (256 TP, 0 FP): {get_micro_f1(256, 0):.5f}")
print(f"Accuracy (256 TP, 0 FP): {get_acc(256, 0):.5f}")

# What if P is different?
# 82.97
