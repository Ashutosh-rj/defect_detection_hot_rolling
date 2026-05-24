for tp in range(0, 232):
    fp = 231 - tp
    fn = 265 - tp
    f1 = 2 * tp / (2 * tp + fp + fn) * 100
    if abs(f1 - 81.61142) < 0.1:
        print(f"Match F1 for GodTier! TP={tp}, FP={fp}, Score={f1}")
