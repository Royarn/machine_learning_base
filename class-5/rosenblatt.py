import matplotlib.pyplot as plot
import numpy

import apples

# generate data
# 输出一个误差代价函数--抛物线形式 因为 误差e = (y - (w * xs + b)) ** 2


# 制作一x轴个随机数
ws = numpy.arange(-50, 100, 0.1)

xs, ys = apples.get(100)
b = 0.1

# 选取某个特定坐标值，并绘制出误差函数
xi = xs[10]
yi = ys[10]

es = []
for w in ws:
    e = (yi - (w * xi + b)) ** 2
    es.append(e)

plot.plot(ws, es)

plot.show()
