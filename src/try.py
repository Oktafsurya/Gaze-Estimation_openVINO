import numpy as np 

fl = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
print(fl)
p = len(fl)
j = 0
points = []

for i in range(int(p/2)):
    point = (fl[j], fl[j+1])
    points.append(point)

    j += 2

print(points)

for point in points:
    print(point)

