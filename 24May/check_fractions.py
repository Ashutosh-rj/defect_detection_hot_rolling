for D in range(1, 1000):
    for num in range(1, D+1):
        if num % 2 != 0: continue
        f1 = num / D
        if abs(f1 - 0.8161142) < 0.00001:
            print(f"Match base! 2*TP={num}, D={D}, F1={f1}")
        if abs(f1 - 0.9962264) < 0.00001:
            print(f"Match final! 2*TP={num}, D={D}, F1={f1}")
