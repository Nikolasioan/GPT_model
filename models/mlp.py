import torch.nn as nn

class MLP(nn.Module): # Inherits from nn.Module, pytorch's base class for learnable parameters.
    "Multilayer perceptron model."

    def __init__(self, layer_dims, dropout_probability = 0.0, hidden_activation=nn.ReLU, output_activation=None):
        super().__init__()
        number_of_layers = len(layer_dims)

        layers = []
        for i in range(number_of_layers - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if hidden_activation is not None:
                layers.append(hidden_activation())
            if dropout_probability:
                layers.append(nn.Dropout(dropout_probability))
        
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers) # Sequential is a container module that combines all the layers modules. * decomposes the list and returns back different arguments.

    # When I subclass nn.Module this method is required (Standard method).
    def forward(self, x):
        return self.net(x)
    

