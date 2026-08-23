
import numpy as np
from tinytorch import Tensor # Added missing import for Tensor


def softmax(x):
    """
    Convert logits into probabilities.

    softmax(x_i) = exp(x_i) / sum(exp(x))
    """

    exponentials = x.exp()

    denominator = exponentials.sum()

    return exponentials / denominator
