import numpy as np

def get(counts):
    xs = np.sort(np.random.rand(counts))
    #
    xs = xs
    ys = []

    for x in xs:
        y = 1.5 * x + np.random.rand() / 10
        ys.append(y)
    #
    return xs, ys