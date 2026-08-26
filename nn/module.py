
from ..parameter import Parameter

class Module:
    def parameters(self):
        params = []
        for value in self.__dict__.values():
            # Robust check: if it has requires_grad, it's a Parameter (even if reloaded)
            if hasattr(value, 'requires_grad'):
                params.append(value)
            # If it has a parameters method, it's a sub-module
            elif hasattr(value, 'parameters') and callable(value.parameters) and value is not self:
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, 'parameters') and callable(item.parameters):
                        params.extend(item.parameters())
                    elif hasattr(item, 'requires_grad'):
                        params.append(item)
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Every Module must implement forward()")
