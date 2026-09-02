
import numpy as np

from .module import Module
from ..tensor import Tensor
from ..parameter import Parameter


class Conv2D(Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Xavier-style initialization
        scale = np.sqrt(
            2.0 / (in_channels * kernel_size * kernel_size)
        )

        self.weight = Parameter(
            np.random.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ) * scale
        )

        self.bias = Parameter(
            np.zeros(out_channels)
        )

    def forward(self, x):

        # x shape:
        # (batch, channels, height, width)

        batch_size, _, H, W = x.data.shape

        K = self.kernel_size
        S = self.stride
        P = self.padding

        # --------------------------------
        # Padding
        # --------------------------------

        if P > 0:

            x_padded = np.pad(
                x.data,
                (
                    (0, 0),
                    (0, 0),
                    (P, P),
                    (P, P)
                ),
                mode="constant"
            )

        else:

            x_padded = x.data

        H_p, W_p = x_padded.shape[2:]

        # Output dimensions

        H_out = (H_p - K) // S + 1
        W_out = (W_p - K) // S + 1

        # Output tensor

        out_data = np.zeros(
            (
                batch_size,
                self.out_channels,
                H_out,
                W_out
            )
        )

        # --------------------------------
        # Forward convolution
        # --------------------------------

        for b in range(batch_size):

            for f in range(self.out_channels):

                for i in range(H_out):

                    for j in range(W_out):

                        h_start = i * S
                        h_end = h_start + K

                        w_start = j * S
                        w_end = w_start + K

                        region = x_padded[
                            b,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        out_data[b, f, i, j] = (
                            np.sum(
                                region * self.weight.data[f]
                            )
                            + self.bias.data[f]
                        )

        out = Tensor(
            out_data,
            requires_grad=(
                x.requires_grad
                or self.weight.requires_grad
                or self.bias.requires_grad
            )
        )

        out._prev = {
            x,
            self.weight,
            self.bias
        }

        # --------------------------------
        # Backward
        # --------------------------------

        def _backward():

            if x.grad is None:

                x.grad = np.zeros_like(
                    x.data
                )

            if self.weight.grad is None:

                self.weight.grad = np.zeros_like(
                    self.weight.data
                )

            if self.bias.grad is None:

                self.bias.grad = np.zeros_like(
                    self.bias.data
                )

            # Gradient with respect to
            # padded input

            dx_padded = np.zeros_like(
                x_padded
            )

            for b in range(batch_size):

                for f in range(self.out_channels):

                    for i in range(H_out):

                        for j in range(W_out):

                            h_start = i * S
                            h_end = h_start + K

                            w_start = j * S
                            w_end = w_start + K

                            region = x_padded[
                                b,
                                :,
                                h_start:h_end,
                                w_start:w_end
                            ]

                            # Gradient of output
                            # with respect to this
                            # convolution result

                            grad_out = out.grad[
                                b, f, i, j
                            ]

                            # --------------------------------
                            # Weight gradient
                            # --------------------------------

                            self.weight.grad[f] += (
                                grad_out * region
                            )

                            # --------------------------------
                            # Bias gradient
                            # --------------------------------

                            self.bias.grad[f] += (
                                grad_out
                            )

                            # --------------------------------
                            # Input gradient
                            # --------------------------------

                            dx_padded[
                                b,
                                :,
                                h_start:h_end,
                                w_start:w_end
                            ] += (
                                grad_out
                                * self.weight.data[f]
                            )

            # Remove padding from input gradient

            if P > 0:

                x.grad += dx_padded[
                    :,
                    :,
                    P:-P,
                    P:-P
                ]

            else:

                x.grad += dx_padded

        out._backward = _backward

        return out
