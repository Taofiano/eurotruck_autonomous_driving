import numpy as np

lines = [[1, 2, 3, 4],
         [1, 2, 1, 7],
         [1, 2, -2, 5],
         [1, 2, -5, 2]]

lines = np.array(lines)

deg = np.abs(np.rad2deg(np.arctan2(lines[:, 1] - lines[:, 3], lines[:, 0] - lines[:, 2])))
print(deg[deg < 130])
print(deg)
