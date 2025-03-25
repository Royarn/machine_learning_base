import matplotlib.pyplot as plot
import numpy

import apples

# generate data
# 输出一个误差代价函数--抛物线形式 因为 误差e = (y - (w * xs + b)) ** 2


# 制作一x轴个随机数
ws = numpy.arange(-50, 100, 0.1)

# 现在b 的值也作为一个自变量
bs = numpy.arange(0, 1, 0.01)

xs, ys = apples.get(100)

# 曲面库
ax = plot.axes(projection='3d')

# 选取某个特定坐标值，并绘制出误差函数
xi = xs[10]
yi = ys[10]


for b in bs:
    es = []
    for w in ws:
        e = (yi - (w * xi + b)) ** 2
        es.append(e)
    plot.plot(ws, es, zdir = 'y')

plot.show()
