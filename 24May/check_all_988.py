import numpy as np
from sklearn.metrics import *

def check(tp, fp, P=265, N=74):
    fn = P - tp
    tn = N - fp
    if fn < 0 or tn < 0: return
    y_true = np.array([1]*P + [0]*N)
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    
    score1 = accuracy_score(y_true, y_pred) * 100
    score2 = balanced_accuracy_score(y_true, y_pred) * 100
    score3 = f1_score(y_true, y_pred, average='macro') * 100
    score4 = f1_score(y_true, y_pred, average='micro') * 100
    score5 = f1_score(y_true, y_pred, average='weighted') * 100
    score6 = f1_score(y_true, y_pred) * 100
    score7 = precision_score(y_true, y_pred) * 100
    score8 = recall_score(y_true, y_pred) * 100
    
    scores = [score1, score2, score3, score4, score5, score6, score7, score8]
    for s in scores:
        if abs(s - 98.81183) < 0.001:
            print(f"MATCH! TP={tp}, FP={fp}, Score={s:.5f}")

for tp in range(200, 266):
    for fp in range(0, 75):
        check(tp, fp)
