
import numpy as np
from tinytorch import Tensor


class ReLU:

    def __call__(self, x):
        # Forward pass
        # ReLU(x) = max(0, x)
        out = Tensor(
            np.maximum(0, x.data),
            requires_grad=x.requires_grad
        )

        # ReLU is connected to x in the computation graph
        out._prev = {x}

        def _backward():

            if x.requires_grad:

                if x.grad is None:
                    x.grad = np.zeros_like(x.data)

                # Derivative of ReLU:
                #
                # x > 0  -> 1
                # x <= 0 -> 0
                #
                # So the incoming gradient only passes
                # through where x was positive.

                relu_grad = (x.data > 0).astype(np.float32)

                x.grad += relu_grad * out.grad

        out._backward = _backward

        return out
