# N-dimensional MLP-Mixer TensorFlow

Based on [MLP-Mixer](https://arxiv.org/abs/2105.01601) [1], but NdMixerBlock is generalized to n-dimensions.

## Original MLP-Mixer

To use the MLP-Mixer as described in the paper:

```python
from nd_mlp_mixer import MLPMixer

# S/32, from table 1
mlp_mixer = MLPMixer(num_classes=1000, 
                     num_blocks=8,
                     patch_size=32, 
                     hidden_dim=512,
                     tokens_mlp_dim=256,
                     channels_mlp_dim=2048)
```

Or a more reasonable size model, on MNIST:

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers
from nd_mlp_mixer import MLPMixer

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images, test_images = train_images.astype("float32"), test_images.astype("float32")
height, width = train_images.shape[-2:]
num_classes = 10

# Prepare the model (add channel dimension to images)
inputs = layers.Input(shape=(height, width))
h = layers.Reshape([28, 28, 1])(inputs)
mlp_mixer = MLPMixer(num_classes=10, 
                     num_blocks=2, 
                     patch_size=4, 
                     hidden_dim=28, 
                     tokens_mlp_dim=28,
                     channels_mlp_dim=28)(h)
model = tf.keras.Model(inputs=inputs, outputs=mlp_mixer)
print(model.summary())

# Train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=64, epochs=10,
                    validation_data=(test_images, test_labels), verbose=2)
```

### [1] MLP-Mixer paper:

https://arxiv.org/abs/2105.01601

```
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision}, 
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
