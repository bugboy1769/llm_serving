import numpy as np
import torch

arr = [1, 2, 3, 4, 5]
arr = np.array(arr)

arr2d = [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]

arr3d = [
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ],
    [
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
]

next_token_id = torch.tensor([1750])

tensor = torch.from_numpy(arr)
print(tensor.long().cumsum(-1) -1)

batch = [('Moby Dick is a whale', 100), ('Moby Dick is a whale', 10), ('Moby Dick is a whale', 10), ('Moby Dick is a whale', 10), ('Moby Dick is a whale', 10), ('Moby Dick is a whale', 10), ('Moby Dick is a whale', 10), ('Moby Dick is a whale', 10)]
batch = np.array(batch)

print(f"State of Batch: {batch}")
print("-----------------")
print(f"Batch Shape: {batch.shape}")
print(f"Compare_to_first {next_token_id}")
print(f"Compare_to: {next_token_id.reshape((-1, 1))}")
