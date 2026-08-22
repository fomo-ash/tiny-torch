#parameter will eventually help us get properties of tensor

from .tensor import Tensor

class Parameter(Tensor): #inherit all properties from tensorclass

  def __init__(self, data):
    super().__init__(
        data,
        requires_grad=True
    )
