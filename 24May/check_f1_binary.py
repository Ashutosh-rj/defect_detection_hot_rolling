for P in range(200, 339):
    for tp in range(150, P+1):
        for fp in range(0, 100):
            fn = P - tp
            f1 = 2 * tp / (2 * tp + fp + fn) * 100
            if abs(f1 - 99.62264) < 0.0001:
                print(f"Match F1 Binary! P={P}, TP={tp}, FP={fp}, FN={fn}")
