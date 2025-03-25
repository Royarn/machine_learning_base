import matplotlib.pyplot as plot
import apples

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

for i in range(100):
    # calculate error
    xi = xs[i]
    yi = ys[i]

    e = yi - w * xi
    a = 0.1
    w = w + a * e *xi
    y = w * xs
    plot.clf()
    plot.ylim(-1.6, 1.6)
    plot.xlim(-1, 1)
    plot.scatter(xs, ys)
    plot.plot(xs, y)
    plot.pause(0.01)

plot.show()
