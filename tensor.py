
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set()
        self._backward = lambda: None

    @staticmethod
    def _unbroadcast(grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                grad_to_add = out.grad
                if axis is not None and not keepdims:
                    grad_to_add = np.expand_dims(grad_to_add, axis=axis)
                self.grad += np.ones_like(self.data) * grad_to_add
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                n = self.data.size if axis is None else self.data.shape[axis]
                grad_to_add = out.grad
                if axis is not None and not keepdims:
                    grad_to_add = np.expand_dims(grad_to_add, axis=axis)
                self.grad += (grad_to_add / n)
        out._backward = _backward
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
                other.grad -= Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += Tensor._unbroadcast((1 / other.data) * out.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
                other.grad += Tensor._unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad += (out.grad @ other.data.T)
            if other.requires_grad:
                if other.grad is None: other.grad = np.zeros_like(other.data)
                other.grad += (self.data.T @ out.grad)
        out._backward = _backward
        return out

    def matmul(self, other):
        return self @ other

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = np.zeros_like(self.data)
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.grad is None: self.grad = np.zeros_like(self.data)
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __getitem__(self, index):
        out = Tensor(self.data[index], requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad[index] += out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.grad = np.zeros_like(self.data)
                self.grad -= out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
