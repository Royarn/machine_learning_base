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

w = 0.1

sum_xy = 0
sum_xx = 0

for i in range(len(xs)):

    x = xs[i]
    y = ys[i]
    sum_xy += x * y
    sum_xx += x * x

w = sum_xy / sum_xx

y = w * xs

plot.plot(xs, y)
plot.show()