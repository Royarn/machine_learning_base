import tensorflow as tf
from keras import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from torch.ao.nn.qat import Conv2d

minst = tf.keras.datasets.mnist
to_categorical = tf.keras.utils.to_categorical
Sequential = tf.keras.models.Sequential
Conv2d = tf._tf_uses_legacy_keras.layers.Conv2d
Flatten = tf.keras.layers.Flatten
AveragePooling2D = tf.keras.layers.AveragePooling2D
Dense = tf.keras.layers.Dense
SGD = tf.keras.optimizers.SGD
