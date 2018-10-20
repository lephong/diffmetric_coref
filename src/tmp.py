# import tensorflow as tf
import numpy as np

for _ in range(500):
    iwi = np.zeros([350, 350], dtype=np.int32)
    for i in range(350):
        for j in range(350):
            if i == j + 2:
                iwi[i, j] = 1

