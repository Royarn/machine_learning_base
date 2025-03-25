import matplotlib.pyplot as plot
import apples
import numpy

# generate data
xs, ys = apples.get(100)

# plot data
plot.scatter(xs, ys)

# set x, y and title
plot.title("Apples")
plot.xlabel("size")
plot.ylabel("sweet")


# initial slope
w = 0.1


# mini 梯度下降 --仍然用的是平均方差 只不过是逐步截取部分样本
for i in range(0, 400, 10):
    xsi = xs[i:i + 10]
    ysi = ys[i:i + 10]
    a = numpy.sum((xsi ** 2)) / 10
    b = -2 * numpy.sum(xsi * ysi) / 10

    k = 2 * a * w + b
    alpha = 0.01
    # alpha 为超参数
    w = w - alpha * k
    # predict y
    y = xsi * w
    # show dynamically
    plot.clf()
    plot.ylim(0, 1.6)
    plot.xlim(0, 1)
    plot.scatter(xs, ys)
    plot.plot(xs, y)
    plot.pause(0.01)

plot.show()
