import matplotlib.pyplot as plot
import apples

# generate data
xs, ys = apples.get(100)

# set x, y and title
plot.title("Apples")
plot.xlabel("size")
plot.ylabel("sweet")


#

# initial slope
w = 0.1
b = 0.1
for _ in range(100):
    for i in range(100):
        # 样本
        xi = xs[i]
        yi = ys[i]

        # e 现在有两个自变量， w , b 均为调整值，所以要对w, b 各自求导算斜率
        dw = 2 * (xi ** 2) * w - 2 * xi * (yi - b)
        db = 2 * b - 2 * (yi - w * xi)

        alpha = 0.1
        w = w - alpha * dw
        b = b - alpha * db

        # 预测值
        y = w * xs + b

        plot.clf()
        plot.ylim(0, 1.6)
        plot.xlim(0, 1)
        plot.scatter(xs, ys)
        plot.plot(xs, y)
        plot.pause(0.01)

plot.show()
