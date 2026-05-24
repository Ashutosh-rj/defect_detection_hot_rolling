for P_pub in range(1, 339):
    for tp in range(0, P_pub+1):
        for fp in range(0, 339-P_pub+1):
            fn = P_pub - tp
            if 2*tp + fp + fn == 0: continue
            f1 = 2 * tp / (2 * tp + fp + fn) * 100
            if abs(f1 - 81.61142) < 0.001:
                print(f"Match Base! P={P_pub}, TP={tp}, FP={fp}, FN={fn}, F1={f1:.5f}")
