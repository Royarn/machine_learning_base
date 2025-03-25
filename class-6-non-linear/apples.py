import numpy as np

def get(counts):
    ws = np.sort(np.random.rand(counts))
    ys = []
    for i in range(counts):
        if i < counts / 2:
           ys.append(0)
        else:
           ys.append(1)
    return ws, ys

