for tp in range(150, 266):
    for fp in range(0, 50):
        # FN = 265 - tp
        f1 = 2 * tp / (2 * tp + fp + (265 - tp)) * 100
        if abs(f1 - 81.61142) < 0.0001:
            print(f"TP={tp}, FP={fp}, F1={f1}")
