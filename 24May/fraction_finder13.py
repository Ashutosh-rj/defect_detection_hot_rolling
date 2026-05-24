val1 = 0.7358491
val2 = 0.762264
val3 = 0.5851606

for den in range(1, 100000):
    num1 = round(val1 * den)
    num2 = round(val2 * den)
    
    if abs(num1/den - val1) < 1e-6 and abs(num2/den - val2) < 1e-6:
        num3 = round(val3 * den)
        if abs(num3/den - val3) < 1e-6:
            print(f"BINGO! Den={den}, num1={num1}, num2={num2}, num3={num3}")
