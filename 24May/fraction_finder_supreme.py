scores = [59.47986, 60.29577, 39.87761]

print("Finding fractions for Supreme scores...")
for val_str in scores:
    val = val_str / 100
    print(f"\nScore: {val_str}")
    for den in range(1, 10000):
        num = round(val * den)
        if abs(num / den - val) < 1e-6:
            print(f"Match: {num} / {den} = {num/den}")
