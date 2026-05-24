import numpy as np
from sklearn.metrics import f1_score

P = 265
N = 74

def get_f1(tp, fp):
    fn = P - tp
    tn = N - fp
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    return f1_score(y_true, y_pred, average='macro') * 100

print(f"265 TP, 0 FP: {get_f1(265, 0):.5f}")
print(f"264 TP, 0 FP: {get_f1(264, 0):.5f}")
print(f"265 TP, 1 FP: {get_f1(265, 1):.5f}")
print(f"264 TP, 1 FP: {get_f1(264, 1):.5f}")
