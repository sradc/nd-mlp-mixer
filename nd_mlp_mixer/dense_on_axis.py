import tensorflow as tf
from tensorflow.keras import layers


class DenseOnAxis(layers.Layer):
    "Dense layer that is applied to a particular axis of a tensor."

    def __init__(self, units, axis=-1, activation=lambda x: x):
        super().__init__()
        self.units = units
        self.axis = axis
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[input_shape[self.axis], self.units],
            initializer="he_uniform",
            trainable=True,
        )

        bias_shape = [1 for _ in input_shape]
        bias_shape[self.axis] = self.units
        self.b = self.add_weight(shape=bias_shape, initializer="zeros", trainable=True)

    def call(self, inputs):
        return self.activation(linear_on_axis(inputs, self.w, self.b, self.axis))


def linear_on_axis(X, weights, bias, axis):
    A = "abcdefghijklmnopqrstuvwxyz"

    ndim = len(X.shape)
    assert ndim <= len(A), "Too many dimensions."

    axis = axis if axis >= 0 else ndim + axis
    assert axis >= 0 and axis < ndim, f"Invalid axis: {axis}, for ndim: {ndim}"

    s1 = "".join(A[i] for i in range(ndim))
    s2 = "".join(s1[axis] + A[ndim])
    s3 = "".join(A[i] if i != axis else A[ndim] for i in range(ndim))

    return tf.einsum(f"{s1},{s2}->{s3}", X, weights, optimize="auto") + bias


# In my informal tests I found einsum faster than transposing, but...
# TODO: test einsum vs transpose method
def __old_linear_on_axis(X, weights, bias, axis):
    perm = list(range(len(X.shape)))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    h = tf.transpose(X, perm)
    h = tf.matmul(h, weights)
    return tf.transpose(h, perm) + bias
