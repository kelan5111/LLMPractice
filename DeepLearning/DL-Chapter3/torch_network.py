import torch


class SequentialModel:
    def __init__(self, layers):
        self.layers = layers

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


class DenseLayer:
    def __init__(self, in_features, out_features, activation=None):
        # Defining weights and biases - random weight and bias when initialized
        self.W = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.b = torch.nn.Parameter(torch.randn(out_features))

        self.activation = activation

    def __call__(self, inputs):
        y = torch.matmul(inputs, self.W.T) + self.b  # y = xW^T + b
        if self.activation is not None:
            if self.activation == "relu":
                y = torch.relu(y)

        return y

    @property
    def weights(self):
        return [self.W, self.b]


learning_rate = 0.1


def training_step(model, inputs, targets):
    predictions = model(inputs)
    loss_value = torch.nn.MSELoss(predictions, targets)
    loss_value.backward()

    with torch.no_grad():
        for param in model.weights:
            param -= param.grad * learning_rate
            
    for param in model.weights:
        param.grad = None

    return loss_value
