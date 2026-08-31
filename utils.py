
import numpy as np

def accuracy(logits, targets):

  #highest logit= prediction
  predictions = np.argmax(logits.data, axis=1)

  #numver of correct predictions
  correct= np.sum(predictions==targets.data)

  return correct/len(targets.data)
