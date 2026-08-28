
import numpy as np
from ..parameter import Parameter
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights with small random values
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * 0.1
        )
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        # Fixed typo: self.weights -> self.weight
        return [self.weight, self.bias]
