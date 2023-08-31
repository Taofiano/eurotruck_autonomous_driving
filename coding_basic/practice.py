import numpy as np

lines = [[1, 2, 3, 4],
         [1, 2, 1, 7],
         [1, 2, -2, 5],
         [1, 2, -5, 2]]

lines = np.array(lines)

print(lines[:, 0])
