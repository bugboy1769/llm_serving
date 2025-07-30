import matplotlib.pyplot as plt
import numpy as np
import torch

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

batch_size = 5
token_remaining = [1, 7, 7, 0, 2]
rem_ind = []
for i, token_rem in enumerate(token_remaining):
    if token_rem <= 0:
        rem_ind.append(i)

mask = torch.ones(batch_size, dtype = torch.bool)
mask[rem_ind] = False

print(f"mask: {mask}")