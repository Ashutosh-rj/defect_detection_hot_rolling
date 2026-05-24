val = 0.5851606
print("Finding all matches for 58.51606...")
for den in range(1, 1000):
    num = round(val * den)
    if abs(num / den - val) < 1e-5:
        print(f"Match: {num} / {den} = {num/den}")
