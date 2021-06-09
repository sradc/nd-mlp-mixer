import tensorflow as tf

import nd_mlp_mixer.dense_on_axis as dense_on_axis


def test_DenseOnAxis():
    "Tests that the shapes are as expected."
    X = tf.random.uniform([10, 10, 10, 2])
    layer = dense_on_axis.DenseOnAxis(5, axis=1)
    result = layer(X)
    assert result.shape == (10, 5, 10, 2), f"Unexpected output size, {result.shape}."


def test_linear_on_axis():
    "Tests that the shapes are as expected."
    X = tf.random.uniform([10, 10, 10, 2])
    weights = tf.random.uniform([10, 5])
    bias = tf.random.uniform([1, 5, 1, 1])
    axis = 1
    result = dense_on_axis.linear_on_axis(X, weights, bias, axis)
    assert result.shape == (10, 5, 10, 2), f"Unexpected output size, {result.shape}."
