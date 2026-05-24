val1 = 0.7358491
val2 = 0.762264
val3 = 0.5851606

for den in range(1, 5000):
    num1 = round(val1 * den)
    num2 = round(val2 * den)
    
    if abs(num1/den - val1) < 1e-5 and abs(num2/den - val2) < 1e-5:
        print(f"Match for Top200 and Top202: Den={den}, num1={num1}, num2={num2}")
        
        # Now check if this den can make 58.51606 an integer!
        num3 = round(val3 * den)
        if abs(num3/den - val3) < 1e-5:
            print(f"  AND it makes Top266 an integer! num3={num3}")
        else:
            print(f"  But it does NOT make Top266 an integer. (num3/den = {num3/den})")
