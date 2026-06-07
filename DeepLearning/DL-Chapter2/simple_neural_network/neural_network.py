import keras
import math
from keras import ops   # tensor operations


class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size), initializer="uniform"
        )
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")

    def __call__(self, inputs):  # applies forward pass
        # ŷ = Wx + b
        x = ops.matmul(inputs, self.W)  # Wx
        x = x + self.b  # + b
        # Checking for activation f(n) is assigned
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def weights(self):
        return [self.W, self.b]


class SequentialDense:
    def __init__(self, layers):
        self.layers = layers    # multiple naive layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        # combining all weights into a single array
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images / batch_size))

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels