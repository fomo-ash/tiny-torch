import numpy as np


class SGD:

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):

        for parameter in self.parameters:

            if parameter.grad is not None:

                parameter.data -= self.lr * parameter.grad

    def zero_grad(self):

        for parameter in self.parameters:

            if parameter.grad is not None:

                parameter.grad = np.zeros_like(parameter.data)
