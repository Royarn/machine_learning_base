from keras.src.datasets import mnist

(x_train, y_train) = mnist.load_data()
(x_test, y_test) = mnist.load_data()


print(x_train[0])