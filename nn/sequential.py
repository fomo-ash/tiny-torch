
from .module import Module


class Sequential(Module):

    def __init__(self, *layers):
        """
        Store all layers in the order they were provided.

        Example:

            Sequential(
                Linear(3, 4),
                ReLU(),
                Linear(4, 2)
            )

        stores:

            layer 0 → Linear
            layer 1 → ReLU
            layer 2 → Linear
        """

        self.layers = list(layers)

    def forward(self, x):
        """
        Pass x through every layer sequentially.
        """

        for layer in self.layers:
            x = layer(x)

        return x
