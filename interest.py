import matplotlib.pyplot as plt
import numpy as np

i = np.arange(0, 50, 1)
i_n = []
for int in i:
    temp = 100*int/(100 - int)
    i_n.append(temp)

print(i)
print("\n" + "--------------" + "\n")
print(i_n)

fig, ax = plt.subplots()
ax.scatter(list(i), i_n)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

plt.show()