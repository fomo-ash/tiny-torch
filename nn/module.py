
from ..parameter import Parameter

class Module: #common functionality to all neural network layers
  def parameters(self):
    params=[] #gives list of all parameters

    for value in self.__dict__.values(): #we use dict to store the values
      if isinstance(value, Parameter):
        params.append(value)

      elif isinstance(value, Module): # amodule can have multiple modules inside it
        params.extend(value.parameters()) #The outer model needs to find parameters inside the inner modules.
      
    return params

  def zero_grad(self): #old gradients should be overwritten, reset gradiesnt befpre training

        for param in self.parameters():
            param.grad = None 

  def __call__(self, *args, **kwargs):

        return self.forward(*args, **kwargs) #Module to accept flexible arguments.

  def forward(self, *args, **kwargs):

        raise NotImplementedError(
            "Every Module must implement forward()"
        )
