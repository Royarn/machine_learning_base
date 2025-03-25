import matplotlib.pyplot as plot
import apples
import numpy

xs, ys = apples.get(100)

plot.title("size-sweetness")
plot.xlabel("size")
plot.ylabel("sweetness")

plot.scatter(xs, ys)


# 激活函数 -- y = 1 / (1 + numpy.exp(-x))

# 从原始的激活函数，无法满足需求，所以，我们将x 替换为x = wx + b。 通过调整w, b 来优化激活函数  y = 1 / (1 + numpy.exp(-wx - b))

w = 0.1
b = 0.1

for j in range(200):
    for i in range(100):
        # 样本
        xi = xs[i]
        yi = ys[i]

        # 求导
        t = w * xi + b
        s = 1 / (1 + numpy.exp(-t))
        e = (yi - s) ** 2
        dtdw = xi
        dtdb = 1

        dsdt = s * (1 - s)
        deds = -2 * (yi - s)

        dedw = deds * dsdt * dtdw
        dedb = deds * dsdt * dtdb

        alpha = 0.1
        w = w - alpha * dedw
        b = b - alpha * dedb

    if (j % 100 == 0):
        # 预测值
        y = 1 / (1 + numpy.exp(-w * xs - b))

        plot.clf()
        plot.ylim(0, 1.6)
        plot.xlim(0, 1)
        plot.scatter(xs, ys)
        plot.plot(xs, y)
        plot.pause(0.01)
plot.show()