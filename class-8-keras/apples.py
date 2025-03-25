import numpy as np

def get(counts):
    xs = np.sort(np.random.rand(counts))
    ys = np.ones(counts)
    for i in range(counts):
        if 0 <= i < 15 or 35 < i < 50:
           ys[i] = 0
    return xs, ys