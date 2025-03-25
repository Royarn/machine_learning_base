import matplotlib.pyplot as plot
from keras import Sequential
from keras.src.layers import Dense

import apples

xs, ys = apples.get(50)

plot.title("size-sweetness")
plot.xlabel("size")
plot.ylabel("sweetness")

plot.scatter(xs, ys)

plot.show()

model = Sequential()

model.add(Dense(units=2, activation='sigmoid', input_dim=1))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='SGD(lr=1)', metrics=['accuracy'])


model.fit(xs, ys, epochs=1000, batch_size=10)

y = model.predict(xs)

plot.plot(xs, y)

plot.show()