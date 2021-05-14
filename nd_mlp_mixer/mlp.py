import tensorflow as tf
from tensorflow.keras import layers

from nd_mlp_mixer.dense_on_axis import DenseOnAxis


class MLP(layers.Layer):
    "Multilayer perceptron that is applied to a particular axis of a tensor."

    def __init__(self, outsize, hiddensize=None, axis=-1, activation=tf.nn.gelu):
        super().__init__()
        hiddensize = hiddensize if hiddensize else outsize
        self.layer_1 = DenseOnAxis(hiddensize, axis, activation)
        self.layer_2 = DenseOnAxis(outsize, axis)

    def call(self, inputs):
        h = self.layer_1(inputs)
        return self.layer_2(h)
