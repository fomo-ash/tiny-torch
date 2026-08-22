
import numpy as np

from ..parameter import Parameter
from .module import Module


class Linear(Module):
    """
    Fully connected (dense) neural network layer.

    Performs the affine transformation:

        y = xW + b

    where:

        x = input
        W = learnable weight matrix
        b = learnable bias vector

    Input shape:
        (batch_size, in_features)

    Weight shape:
        (in_features, out_features)

    Bias shape:
        (out_features,)

    Output shape:
        (batch_size, out_features)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Each input feature connects to every output neuron.
        #
        # Therefore:
        #
        #       weight.shape = (in_features, out_features)
        #
        # Example:
        #
        #       Linear(3, 2)
        #
        #       weight.shape = (3, 2)
        #
        #              output neurons
        #                ↓   ↓
        #             [ w  w ]
        # input 1 --> [ w  w ]
        # input 2 --> [ w  w ]
        # input 3 --> [ w  w ]
        #
        # We use a small random initialization so that different
        # neurons start with different values.
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * 0.01
        )

        # One bias value for each output neuron.
        #
        # Shape:
        #
        #       (out_features,)
        #
        # During xW + b, NumPy broadcasting adds the same bias
        # vector to every sample in the batch.
        self.bias = Parameter(
            np.zeros(out_features)
        )

    def forward(self, x):
        """
        Forward pass.

        x has shape:

            (batch_size, in_features)

        weight has shape:

            (in_features, out_features)

        Therefore:

            x @ weight

        gives:

            (batch_size, out_features)

        Then bias is broadcast across the batch:

            (batch_size, out_features)
                    +
            (out_features,)

        resulting in:

            (batch_size, out_features)
        """

        return x @ self.weight + self.bias
