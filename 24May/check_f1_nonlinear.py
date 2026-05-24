for TP in range(150, 266):
    for FP in range(0, 75):
        for P in range(200, 300):
            fn = P - TP
            if fn < 0: continue
            
            f1_base = 2 * TP / (2 * TP + FP + fn) * 100
            
            if abs(f1_base - 81.61142) < 0.05:
                # check TP+1
                f1_tp1 = 2 * (TP+1) / (2 * (TP+1) + FP + (fn-1)) * 100
                if abs(f1_tp1 - 81.98878) < 0.05:
                    print(f"Match! P={P}, TP={TP}, FP={FP}, Base={f1_base:.5f}, TP+1={f1_tp1:.5f}")
                    # check FP+1
                    f1_fp1 = 2 * TP / (2 * TP + FP + 1 + fn) * 100
                    print(f"  FP+1={f1_fp1:.5f}")
                    # check FP-1
                    f1_fpm1 = 2 * TP / (2 * TP + FP - 1 + fn) * 100
                    print(f"  FP-1={f1_fpm1:.5f}")
