import numpy as np
from .module import Module
import tinytorch.nn.functional as functional # Import the functional module
from tinytorch import Tensor # Import Tensor for creating new Tensor objects


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

        # Get batch size from the target tensor
        batch_size = target.data.shape[0]

        # Create an array of row indices (0, 1, 2, ...) for advanced indexing
        row_indices = np.arange(batch_size)

        # Use advanced indexing to select the probability of the correct class for each example
        # probabilities.data is a NumPy array of shape (batch_size, num_classes)
        # target.data is a NumPy array of shape (batch_size,) containing class indices
        selected_probabilities_data = probabilities.data[row_indices, target.data.astype(int)]

        # Wrap the result back into a TinyTorch Tensor
        correct_probability = Tensor(selected_probabilities_data, requires_grad=probabilities.requires_grad)

        # Calculate negative log likelihood for each example in the batch
        per_sample_loss = -correct_probability.log()

        # Return the mean of the per-sample losses to get a scalar loss for the batch
        return per_sample_loss.mean()
