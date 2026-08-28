
import numpy as np


class Adam:

    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
        # Store the parameters that Adam will update
        self.parameters = list(parameters)

        # Hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Number of optimizer steps
        self.t = 0

        # First moment: moving average of gradients
        self.m = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

        # Second moment: moving average of squared gradients
        self.v = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

    def zero_grad(self):
        """
        Clear gradients before the next backward pass.
        """

        for p in self.parameters:

            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def step(self):
        """
        Perform one Adam parameter update.
        """

        # Move to the next optimization step
        self.t += 1

        for i, p in enumerate(self.parameters):

            # Skip parameters without gradients
            if p.grad is None:
                continue

            g = p.grad

            # --------------------------------
            # 1. First moment
            # --------------------------------
            self.m[i] = (
                self.beta1 * self.m[i]
                + (1 - self.beta1) * g
            )

            # --------------------------------
            # 2. Second moment
            # --------------------------------
            self.v[i] = (
                self.beta2 * self.v[i]
                + (1 - self.beta2) * (g ** 2)
            )

            # --------------------------------
            # 3. Bias correction
            # --------------------------------
            m_hat = self.m[i] / (
                1 - self.beta1 ** self.t
            )

            v_hat = self.v[i] / (
                1 - self.beta2 ** self.t
            )

            # --------------------------------
            # 4. Adam parameter update
            # --------------------------------
            p.data -= self.lr * (
                m_hat /
                (np.sqrt(v_hat) + self.eps)
            )
