import numpy as np
from sklearn.metrics import f1_score

for tp in range(0, 232):
    fp = 231 - tp
    # F1 Score
    f1 = 2 * tp / (2 * tp + fp + (265 - tp)) * 100
    if abs(f1 - 81.61142) < 0.1:
        print(f"Match F1! TP={tp}, FP={fp}, Score={f1}")
        
    # Macro F1
    fn = 265 - tp
    tn = 74 - fp
    f1_1 = 2 * tp / (2 * tp + fp + fn) * 100
    if tn + fp + fn > 0:
        f1_0 = 2 * tn / (2 * tn + fn + fp) * 100
        macro_f1 = (f1_1 + f1_0) / 2
        if abs(macro_f1 - 81.61142) < 0.1:
            print(f"Match Macro F1! TP={tp}, FP={fp}, Score={macro_f1}")
            
    # Accuracy
    acc = (tp + tn) / 339 * 100
    if abs(acc - 81.61142) < 0.1:
        print(f"Match Acc! TP={tp}, FP={fp}, Score={acc}")
