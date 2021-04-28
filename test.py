import numpy as np
import torch

c = [
    np.array([
        [[1, 2], [2, 1], [3, 2]],
        [[1, 3], [3, 3], [2, 4]],
        [[1, 3], [3, 3], [2, 4]]
    ]),
    np.array([
        [[1, 2], [2, 1], [3, 2]],
        [[1, 3], [3, 3], [2, 4]],
        [[1, 3], [3, 3], [2, 4]]
    ]),
    np.array([
        [[1, 2], [2, 1], [3, 2]],
        [[1, 3], [3, 3], [2, 4]],
        [[1, 3], [3, 3], [2, 4]]
    ])
]

b = np.array(c).reshape(3, 3, 3, 2).transpose(0, 3, 1, 2).astype(np.float32)
b = torch.from_numpy(b)

a = np.array([[[0, 0, 1], [1, 0, 1], [0, 1, 1]], [[0, 0, 1], [1, 0, 1], [0, 1, 1]], [[0, 0, 1], [1, 0, 1], [0, 1, 1]]]).reshape(3, 3, 3).astype(np.bool)
a = torch.from_numpy(a)

res = torch.masked_select(b.transpose(0, 1), a).reshape(2, 3, -1).transpose(0, 1)

print(b.shape)
print(a.shape)
print(res)
print(res.shape)