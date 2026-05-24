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

print(f"Top231 (tp=228, fp=3): {get_f1(228, 3):.5f}")
print(f"One more TP (tp=229, fp=3): {get_f1(229, 3):.5f}")
print(f"Diff for 1 TP: {get_f1(229, 3) - get_f1(228, 3):.5f}")

print(f"One less FP (tp=228, fp=2): {get_f1(228, 2):.5f}")
print(f"Diff for 1 FP removal: {get_f1(228, 2) - get_f1(228, 3):.5f}")

print(f"All 3 FPs removed (tp=228, fp=0): {get_f1(228, 0):.5f}")

print(f"All 3 FPs removed + 28 TPs added (tp=256, fp=0): {get_f1(256, 0):.5f}")
