import numpy as np
from .module import Module
from tinytorch import Tensor

class MSELoss(Module):
    def forward(self, prediction, target):
        difference = prediction - target
        squared = difference ** 2
        return squared.mean()

class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        # 1. Apply softmax functional logic directly or via functional
        # To keep it simple and ensure graph connectivity:
        exps = logits.exp()
        probs = exps / exps.sum(axis=-1, keepdims=True)
        
        # 2. Use TinyTorch indexing to keep the graph alive
        # target contains class indices. We need to pick the prob of the correct class.
        batch_size = logits.data.shape[0]
        losses = []
        
        for i in range(batch_size):
            # Indexing a Tensor returns a Tensor connected to the graph
            class_idx = int(target.data[i])
            prob = probs[i][class_idx]
            losses.append(-prob.log())
            
        # 3. Average the losses
        total_loss = losses[0]
        for i in range(1, len(losses)):
            total_loss = total_loss + losses[i]
            
        return total_loss / batch_size
