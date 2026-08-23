
from .module import Module
import tinytorch.nn.functional as functional # Import the functional module


class MSELoss(Module):

    def forward(self, prediction, target):
        """
        Mean Squared Error:

            MSE = mean((prediction - target)^2)

        We use existing Tensor operations so that
        TinyTorch's autograd system automatically
        builds the computation graph.
        """

        difference = prediction - target
        squared = difference ** 2

        return squared.mean()

class CrossEntropyLoss(Module):

    def forward(self, logits, target):
        """
        Cross Entropy for a single example.

        Steps:

            logits
               ↓
            softmax
               ↓
          probability
               ↓
            -log()
               ↓
             loss
        """

        # Use the fully qualified name for softmax
        probabilities = functional.softmax(logits)

        correct_probability = probabilities[target]

        loss = -correct_probability.log()

        return loss
