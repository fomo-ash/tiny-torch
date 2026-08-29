
import numpy as np
from .module import Module
from ..tensor import Tensor

class Flatten(Module):

  def forward(self, x):
     # Save the original shape so backward()
        # can restore it later.
        original_shape=x.data.shape;
        batch_size=x.data.shape[0];

        out=Tensor(
            x.data.reshape(batch_size, -1),
            requires_grad=x.requires_grad
        )

        out._prev={x};

        def _backward(): # Renamed from 'backward' to '_backward'
          if x.grad is None:
           x.grad=np.zeros_like(x.data)

          # Gradient of reshape is simply
          # reshape the gradient back to the original shape.
          x.grad += out.grad.reshape(original_shape)

        out._backward = _backward

        return out
