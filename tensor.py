
import numpy as np


class Tensor:

    def __init__(self, data, requires_grad=False):

        # NumPy is currently the numerical engine underneath TinyTorch.
        # We store everything as float32 because this is what is commonly
        # used for neural-network computations.
        self.data = np.array(data, dtype=np.float32)

        # If True, this Tensor participates in gradient computation.
        #
        # Example:
        # x = Tensor(5, requires_grad=True)
        #
        # We eventually want:
        # y = x * x
        # y.backward()
        # print(x.grad)
        self.requires_grad = requires_grad

        # Gradient will be calculated during backward().
        #
        # Initially there is no gradient.
        self.grad = None

        # Every operation creates a computation graph.
        #
        # Example:
        #
        # x ----\
        #        * ----> z
        # y ----/
        #
        # z._prev = {x, y}
        #
        # This allows backward() to travel through the graph.
        self._prev = set()

        # Each operation will replace this with a function describing
        # how to propagate gradients backward through that operation.
        self._backward = lambda: None

    # ============================================================
    # BROADCASTING HELPER
    # ============================================================

    @staticmethod
    def _unbroadcast(grad, shape):
        """
        Convert a broadcasted gradient back to the original Tensor shape.

        Why do we need this?
        --------------------

        NumPy allows operations like:

            X = shape (3, 2)
            b = shape (2,)

            X + b

        NumPy automatically broadcasts b:

            [a, b]
            [a, b]
            [a, b]

        So the output has shape:

            (3, 2)

        During backward(), however, b still has shape:

            (2,)

        Therefore, its gradient must also have shape (2,).

        Suppose:

            X = [[1, 2],
                 [3, 4],
                 [5, 6]]

            b = [10, 20]

        Then:

            X + b

        uses each element of b three times.

        Therefore:

            db = [3, 3]

        instead of a (3, 2) gradient.

        This function performs that reduction.
        """

        # If the gradient has more dimensions than the original Tensor,
        # those extra dimensions were introduced by broadcasting.
        #
        # Example:
        #
        # grad shape = (3, 2)
        # original shape = (2,)
        #
        # We need to reduce (3, 2) -> (2,).
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)

        # If an original dimension was 1, NumPy may have broadcast it.
        #
        # Example:
        #
        # original shape = (3, 1)
        # broadcast shape = (3, 4)
        #
        # We need to sum across that broadcasted dimension.
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    # ============================================================
    # BASIC PROPERTIES
    # ============================================================

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data})"

    # ============================================================
    # ADDITION
    # ============================================================

    def __add__(self, other):

        # Allow:
        #
        # Tensor + Tensor
        # Tensor + number
        #
        # Example:
        # x + 5
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Forward pass
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        # Remember which Tensors created this Tensor.
        out._prev = {self, other}

        def _backward():

            # For:
            #
            # z = x + y
            #
            # dz/dx = 1
            #
            # Therefore:
            #
            # dx = dz
            if self.requires_grad:

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                # IMPORTANT:
                #
                # out.grad may have a larger shape than self.
                #
                # This happens when NumPy broadcasting was used.
                #
                # Example:
                #
                # X (3,2) + b (2,)
                #
                # out.grad -> (3,2)
                # b.grad   -> (2,)
                #
                # _unbroadcast() reduces the gradient back to
                # the original Tensor shape.
                self.grad = self.grad + Tensor._unbroadcast(
                    out.grad,
                    self.data.shape
                )

            if other.requires_grad:

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                other.grad = other.grad + Tensor._unbroadcast(
                    out.grad,
                    other.data.shape
                )

        out._backward = _backward

        return out

    # ============================================================
    # SUBTRACTION
    # ============================================================

    def __sub__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Forward:
        #
        # z = x - y
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():

            # dz/dx = 1
            if self.requires_grad:

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad = self.grad + Tensor._unbroadcast(
                    out.grad,
                    self.data.shape
                )

            # dz/dy = -1
            if other.requires_grad:

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                other.grad = other.grad - Tensor._unbroadcast(
                    out.grad,
                    other.data.shape
                )

        out._backward = _backward

        return out

    # ============================================================
    # MULTIPLICATION
    # ============================================================

    def __mul__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Forward:
        #
        # z = x * y
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():

            # For:
            #
            # z = x * y
            #
            # dz/dx = y
            #
            # Therefore:
            #
            # dx = y * dz
            if self.requires_grad:

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                grad_self = other.data * out.grad

                # Broadcasting may have happened during:
                #
                # self * other
                #
                # So reduce the gradient back to self's shape.
                self.grad = self.grad + Tensor._unbroadcast(
                    grad_self,
                    self.data.shape
                )

            # dz/dy = x
            if other.requires_grad:

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                grad_other = self.data * out.grad

                # Again, if other was broadcasted, its gradient
                # must be reduced back to its original shape.
                other.grad = other.grad + Tensor._unbroadcast(
                    grad_other,
                    other.data.shape
                )

        out._backward = _backward

        return out

    # ============================================================
    # DIVISION
    # ============================================================

    def __truediv__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Forward:
        #
        # z = x / y
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():

            # dz/dx = 1/y
            if self.requires_grad:

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                grad_self = (1 / other.data) * out.grad

                self.grad = self.grad + Tensor._unbroadcast(
                    grad_self,
                    self.data.shape
                )

            # dz/dy = -x/y²
            if other.requires_grad:

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                grad_other = (
                    -self.data / (other.data ** 2)
                ) * out.grad

                # IMPORTANT:
                #
                # The gradient belongs to 'other',
                # not 'self'.
                other.grad = other.grad + Tensor._unbroadcast(
                    grad_other,
                    other.data.shape
                )

        out._backward = _backward

        return out

    # ============================================================
    # POWER
    # ============================================================

    def __pow__(self, power):

        # Forward:
        #
        # z = x^n
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad
        )

        out._prev = {self}

        def _backward():

            # d(x^n)/dx = n*x^(n-1)
            if self.requires_grad:

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad = self.grad + (
                    power * self.data ** (power - 1)
                ) * out.grad

        out._backward = _backward

        return out

    # ============================================================
    # MATRIX MULTIPLICATION
    # ============================================================

    def __matmul__(self, other):

        # Forward:
        #
        # Z = X @ W
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad
        )

        out._prev = {self, other}

        def _backward():

            # For:
            #
            # Z = X @ W
            #
            # dZ/dX = dZ @ W^T
            if self.requires_grad:

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)

                self.grad = self.grad + (out.grad @ other.data.T)

            # dZ/dW = X^T @ dZ
            if other.requires_grad:

                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                other.grad = other.grad + (self.data.T @ out.grad)

        out._backward = _backward

        return out

    # Keep the old API working:
    #
    # x.matmul(y)
    #
    # while also supporting:
    #
    # x @ y
    def matmul(self, other):
        return self @ other

    # ============================================================
    # BACKWARD
    # ============================================================

    def backward(self):

        """
        Start reverse-mode automatic differentiation.

        Example:

            x -> y -> z

        Calling:

            z.backward()

        calculates:

            dz/dx
            dz/dy
        """

        topo = []
        visited = set()

        # --------------------------------------------------------
        # Build topological ordering of computation graph
        # --------------------------------------------------------

        def build_topo(v):

            if v not in visited:

                visited.add(v)

                # Visit everything that created v first.
                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        # --------------------------------------------------------
        # Derivative of output with respect to itself
        # --------------------------------------------------------
        #
        # If:
        #
        # z = final output
        #
        # then:
        #
        # dz/dz = 1
        #
        # This is where backpropagation starts.

        self.grad = np.ones_like(self.data)

        # --------------------------------------------------------
        # Traverse graph backwards
        # --------------------------------------------------------
        #
        # Each Tensor has a _backward() function that knows the
        # derivative rule for the operation that created it.

        for v in reversed(topo):
            v._backward()
