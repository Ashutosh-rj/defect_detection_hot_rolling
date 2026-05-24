for y in range(1, 1000):
    for x in range(1, y+1):
        if abs(x/y - 0.8161142) < 0.00001:
            print(f"{x}/{y} = {x/y:.7f}")
