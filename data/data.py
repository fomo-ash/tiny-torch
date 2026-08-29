
import numpy as np


class DataLoader:
  def __init__(
      self,
      X,
      Y,
      batch_size=32,
      shuffle=True
  ):
    self.X=X
    self.Y=Y

    self.batch_size=batch_size

    self.shuffle=shuffle

    self.n_samples= X.data.shape[0]

  def __iter__(self):
    #indices for dataset
    indices=np.arange(self.n_samples);

    #shuffling
    if self.shuffle:
      np.random.shuffle(indices);

    for start in range(
        0, self.n_samples, self.batch_size
    ): 
      end=start+self.batch_size


      batch_indices = indices[start:end]

      X_batch = self.X.data[batch_indices]
      Y_batch = self.Y.data[batch_indices]

      yield X_batch, Y_batch

  def __len__(self):

        return int(
            np.ceil(
                self.n_samples / self.batch_size
            )
        )
