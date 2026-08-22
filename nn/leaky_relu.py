
import numpy as np
from tinytorch import Tensor


class LeakyReLU: # if the nn is always receiving negative inputs, it will help it learn

    def __init__(self, negative_slope=0.01):
        # Controls how much gradient passes
        # through the negative side.
        self.negative_slope = negative_slope

    def __call__(self, x):

        # -------------------------
        # Forward pass
        # -------------------------
        #
        # positive x -> x
        # negative x -> alpha * x

        out_data = np.where(
            x.data > 0,
            x.data,
            self.negative_slope * x.data
        )

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad
        )

        # x is the parent of out
        out._prev = {x}

        # -------------------------
        # Backward pass
        # -------------------------

        def _backward():

            if x.requires_grad:

                if x.grad is None:
                    x.grad = np.zeros_like(x.data)

                # Derivative:
                #
                # x > 0 -> 1
                # x <= 0 -> negative_slope

                leaky_grad = np.where(
                    x.data > 0,
                    1.0,
                    self.negative_slope
                )

                # Chain rule
                x.grad += leaky_grad * out.grad

        out._backward = _backward

        return out
