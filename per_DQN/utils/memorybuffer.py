"""Priority Experience Replay (PER) Memory buffer script

This manages the prioritized memory buffer. 
"""

import random

import numpy as np

from .sumtree import SumTree

class PriorityBuffer:
    """
    Class for the PER Buffer creation
    """

    e = 0.001
    a = 0.6
    beta = 0.4
    beta_inc = 0.001

    def __init__(self, capacity):
        """Instantiate the sumtree and the capacity

        Args:
            capacity (int): maxsize of the buffer

        Returns:
            None
        """

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def clear(self):
        """Clear the buffer 
        Args:
            None

        Returns:
            None
        """

        self.tree = SumTree(self.capacity)

    def _get_priority(self, error):
        """Calculate the priority from the td error

        Args:
            error (float): td error of the sample

        Returns:
            None
        """

        return (error + self.e) ** self.a

    def store(self, error, sample):
        """Compute the sample priority and store in the sumtree

        Args:
            error (float): td error of the sample
            sample (list): state, action, reward, obs_state, done values

        Returns:
            None
        """
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        """Get the samples from the buffer

        Args:
            batch_size (int): size of the batch to sample

        Returns:
           batch (list): sampled experiences
           idxs (list): tree indices of the experiences
           is_weight (list): weights of the experiences
        """
        
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_inc])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        """Update the sample priority, given the new td error

        Args:
            idx (int): sample index in the three
            error (float): new td error of the sample

        Returns:
            None
        """

        p = self._get_priority(error)
        self.tree.update(idx, p)

    @property
    def size(self):
        """Return the size of the buffer
        """
        
        return self.tree.n_entries
