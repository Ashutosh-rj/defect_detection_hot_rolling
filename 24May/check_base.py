for P in range(200, 339):
    for tp in range(150, 232):
        fp = 231 - tp
        fn = P - tp
        if fn < 0: continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if abs(f1 - 0.8161142) < 0.00001:
            print(f"Match base! P={P}, TP={tp}, FP={fp}")
            
        # Also check 82.86
        fp_new = fp - 1
        f1_new = 2 * tp / (2 * tp + fp_new + fn)
        if abs(f1 - 0.8161142) < 0.00001 and abs(f1_new - 0.8286) < 0.01:
            print(f"  -> And removing 1 FP gives {f1_new}")
