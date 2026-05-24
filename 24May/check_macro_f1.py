import numpy as np
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

y_true_base = np.array([1]*265 + [0]*74)

for tp in range(0, 214):
    fp = 213 - tp
    if fp > 74: continue
    fn = 265 - tp
    tn = 74 - fp
    
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    
    val = f1_score(y_true_base, y_pred, average='macro') * 100
    if abs(val - 77.73585) < 0.1:
        print(f"Match Macro F1 Top213! TP={tp}, FP={fp}, Score={val}")
        
for tp in range(0, 232):
    fp = 231 - tp
    if fp > 74: continue
    fn = 265 - tp
    tn = 74 - fp
    
    y_pred = np.array([1]*tp + [0]*fn + [1]*fp + [0]*tn)
    
    val = f1_score(y_true_base, y_pred, average='macro') * 100
    if abs(val - 81.61142) < 0.1:
        print(f"Match Macro F1 Top231! TP={tp}, FP={fp}, Score={val}")
