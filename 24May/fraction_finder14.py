for tp in range(192, 266):
    tn = tp - 192
    fp = 266 - tp
    fn = 265 - tp
    f1_1 = 2 * tp / (266 + 265)
    f1_0 = 2 * tn / (tn + fp + tn + fn) # wait, F1_0 = 2*TN / (2*TN + FP + FN)
    f1_0 = 2 * tn / (2 * tn + fp + fn)
    macro = (f1_1 + f1_0) / 2 * 100
    if abs(macro - 58.51606) < 0.5:
        print(f"Macro F1 match! TP={tp}, Macro={macro}")
