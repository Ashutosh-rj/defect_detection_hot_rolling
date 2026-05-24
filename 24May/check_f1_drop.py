for k in range(1, 1000):
    for tp in range(0, k+1):
        drop = 2 * tp / (k * (k + 1))
        if abs(drop - 0.0135135135) < 0.00001:
            f1_old = 2 * tp / k
            print(f"Match Drop! TP={tp}, K={k}, F1_old={f1_old*100}, Drop={drop*100}")
