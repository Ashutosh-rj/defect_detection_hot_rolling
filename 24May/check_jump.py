for p_pub in range(1, 340):
    for tp in range(0, 232):
        num_pred = 231
        f1_old = 2 * tp / (num_pred + p_pub) * 100
        f1_tp = 2 * (tp + 1) / (num_pred + 1 + p_pub) * 100
        f1_fp = 2 * tp / (num_pred + 1 + p_pub) * 100
        
        if abs(f1_old - 81.61142) < 0.1 and abs(f1_tp - f1_old - 0.37736) < 0.1 and abs(f1_old - f1_fp - 1.35135) < 0.1:
            print(f"Match! P_pub={p_pub}, TP={tp}, F1={f1_old}, TP_jump={f1_tp - f1_old}, FP_drop={f1_old - f1_fp}")
