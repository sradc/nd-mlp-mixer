import tensorflow as tf
from tensorflow.keras import layers

from nd_mlp_mixer.dense_on_axis import DenseOnAxis
from nd_mlp_mixer.mlp import MLP
from nd_mlp_mixer.scalar_gate import ScalarGate
from nd_mlp_mixer.layers import ResidualLayers


class NdMixer(tf.keras.layers.Layer):
    "N-dimensional mixer block, without batchnorm or skip connections."

    def __init__(self, outshape=None, Net=MLP, gate=True):
        """
        Args:
            outshape (tuple/list):
                The output shape, not including the samples dimension.
            Net (layers.Layer):
                A dense-like layer that operates on a particular axis.
            gate (bool):
                Whether to gate layers' (with a learnable scalar).
        """
        super().__init__()
        self.outshape = outshape
        self.Net = MLP
        self.gate = gate

    def build(self, input_shape):
        outshape = self.outshape if self.outshape else input_shape[1:]
        self.nets = [self.Net(size, axis=i + 1) for i, size in enumerate(outshape)]
        self.gates = [ScalarGate() if self.gate else lambda x: x for _ in outshape]
        self.norms = [layers.BatchNormalization() for _ in outshape]

    def call(self, inputs):
        h = inputs
        for norm, net, gate in zip(self.norms, self.nets, self.gates):
            h = norm(h)
            h = h + gate(net(h))
        return h


def NdClassifier(
    in_shape, repr_shape, out_shape, num_mix_layers, num_classes, hidden_size=None
):
    """A classifier based on NDMixer."""
    Net = lambda outsize, axis: MLP(outsize, axis, hidden_size)
    make_mixer = lambda: NdMixer(repr_shape, Net, gate=True)
    inputs = layers.Input(in_shape)
    repr_init = NdMixer(repr_shape, Net=DenseOnAxis, gate=False)(inputs)
    mixed = ResidualLayers(num_mix_layers, make_mixer)(repr_init)
    repr_final = NdMixer(out_shape, Net=DenseOnAxis, gate=False)(mixed)
    h = layers.Flatten()(repr_final)
    h = layers.Dense(num_classes)(h)
    return tf.keras.Model(inputs=inputs, outputs=h)


def NdAutoencoder(in_shape, repr_shape, num_mix_layers, hidden_size=None):
    """An autoencoder based on NDMixer."""
    Net = lambda outsize, axis: MLP(outsize, axis, hidden_size)
    make_mixer = lambda: NdMixer(repr_shape, Net, gate=True)
    inputs = layers.Input(in_shape)
    repr_init = NdMixer(repr_shape, Net=Net, gate=False)(inputs)
    mixed = ResidualLayers(num_mix_layers, make_mixer)(repr_init)
    repr_final = NdMixer(in_shape, Net=Net, gate=False)(mixed)
    return tf.keras.Model(inputs=inputs, outputs=repr_final)


class _old_NdMixer(tf.keras.layers.Layer):
    "N-dimensional mixer block, without batchnorm or skip connections."

    def __init__(self, outshape=None, Net=MLP, gate=True):
        """
        Args:
            outshape (tuple/list):
                The output shape, not including the samples dimension.
            Net (layers.Layer):
                A dense-like layer that operates on a particular axis.
            gate (bool):
                Whether to gate the output (with a learnable scalar),
                in order to initialise as the identify function.
        """
        super().__init__()
        self.outshape = outshape
        self.Net = MLP
        self.gate = ScalarGate() if gate else lambda x: x

    def build(self, input_shape):
        outshape = self.outshape if self.outshape else input_shape[1:]
        self.nets = [self.Net(size, axis=i + 1) for i, size in enumerate(outshape)]

    def call(self, inputs):
        h = inputs
        for net in self.nets:
            h = net(h)
        return self.gate(h)
