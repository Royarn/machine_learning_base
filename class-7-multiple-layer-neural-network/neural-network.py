import matplotlib.pyplot as plot
import apples
import numpy

xs, ys = apples.get(50)

plot.title("size-sweetness")
plot.ylabel("sweetness")
plot.xlabel("size")
plot.scatter(xs, ys)

# initial param
w1 = numpy.random.randn()
w2 = numpy.random.randn()
w3 = numpy.random.randn()
w4 = numpy.random.randn()
b1 = numpy.random.randn()
b2 = numpy.random.randn()
b3 = numpy.random.randn()

def forward_propagation(xs):

    # 第一层第一个神经元的输出结果
    t1 = w1 * xs + b1
    s1 = sigmoid(t1)

    # 第一层第二个神经元的输出结果
    t2 = w2 * xs + b2
    s2 = sigmoid(t2)

    # 第二层的第一个神经元的输出结果
    t3 = w3 * s1 + w4 * s2 + b3
    s3 = sigmoid(t3)

    return s3, t3, s2, t2, s1, t1

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


for j in range(200):
    for i in range(100):
        # 样本
        xi = xs[i]
        yi = ys[i]

        # 预测值的各个神经元的输出 s1, s2, s3
        s3, t3, s2, t2, s1, t1 = forward_propagation(xi)

        # 算误差函数
        e = (yi - s3) ** 2
        deds3 = -2 * (yi - s3)
        de3dt3 = s3 * (1 - s3)
        dt3dw3 = s1
        de3dw4 = s2
        dt3db3 = 1


        # 求导
        t = w * xi + b
        s = 1 / (1 + numpy.exp(-t))
        e = (yi - s) ** 2
        dtdw1 =
        dtdw2 =
        dtdw3 =
        dtdw4 =
        dtdb1 =
        dtdb2 =
        dtdb3 =

        dsdt = s * (1 - s)
        deds = -2 * (yi - s)

        dedw = deds * dsdt * dtdw
        dedb = deds * dsdt * dtdb

        alpha = 0.1
        w1 = w1 - alpha * dedw1
        w2 = w2 - alpha * dedw2
        w3 = w3 - alpha * dedw3
        w4 = w4 - alpha * dedw4
        b1 = b1 - alpha * dedb1
        b2 = b2 - alpha * dedb2
        b3 = b3 - alpha * dedb3

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

