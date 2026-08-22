
import numpy as np
from tinytorch import Tensor


class Sigmoid:

    def __call__(self, x):

        # -------------------------
        # Forward pass
        # -------------------------
        #
        # sigmoid(x) = 1 / (1 + e^(-x))
        #
        # We calculate sigmoid using NumPy.
        #
        # We store the result in `out` because
        # the backward pass can reuse the sigmoid
        # output instead of calculating it again.

        out_data = 1 / (1 + np.exp(-x.data))

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad
        )

        # -------------------------
        # Computation graph
        # -------------------------

        # `out` was created from `x`,
        # so x is a parent of out.

        out._prev = {x}

        # -------------------------
        # Backward pass
        # -------------------------

        def _backward():

            if x.requires_grad:

                if x.grad is None:
                    x.grad = np.zeros_like(x.data)

                # Derivative of sigmoid:
                #
                # sigmoid'(x)
                #     = sigmoid(x) * (1 - sigmoid(x))
                #
                # `out.data` already contains sigmoid(x),
                # so we can directly use it.

                sigmoid_grad = out.data * (1 - out.data)

                # Chain rule:
                #
                # dL/dx =
                # dL/dout * dout/dx

                x.grad += sigmoid_grad * out.grad

        out._backward = _backward

        return out
