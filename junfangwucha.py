import numpy
from matplotlib import pyplot as plot
import apples


# 批量梯度下降，是求和所有的平方和的误差 -- 所有样本的平均误差（均方误差）  所有误差的平方的和的平均值
xs, ys = apples.get(100)

# plot data
plot.scatter(xs, ys)

# set x, y and title
plot.title("Apples")
plot.xlabel("size")
plot.ylabel("sweet")

w = 0.1
a = numpy.sum((xs ** 2)) / 100
b = -2 * numpy.sum(xs * ys) / 100

for i in range(400):
    alpha = 0.02
    k = 2 * a * w + b
    w = w - alpha * k
    # predict y
    y = xs * w
    # show dynamically
    plot.clf()
    plot.ylim(0, 1.6)
    plot.xlim(0, 1)
    plot.scatter(xs, ys)
    plot.plot(xs, y)
    plot.pause(0.01)


plot.show()