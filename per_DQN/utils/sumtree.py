"""SumTree structure for the PER buffer

This manages the sumtree.
"""

import random
from collections import deque

import numpy as np

class SumTree:
    """
    Class for the sumtree creation
    """

    write = 0

    def __init__(self, capacity):
        """Instantiate the tree and the data arrays

        Args:
            capacity (int): maxsize of the buffer

        Returns:
            None
        """

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate the priority change up into the tree

        Args:
            idx (int): index of the sample that changed prio
            change (float): updated priority 

        Returns:
            None
        """

        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Retrieve ...

        Args:
            idx (int): index of the sample
            s ...

        Returns:
            None
        """

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Total tree priority (i.e., the ones in the root)

        Args:
            None

        Returns:
            self.tree[0] (float): the total priority
        """

        return self.tree[0]

    def add(self, p, data):
        """Add data to the tree

        Args:
            p (float): priority of the new data (typically the td_error)
            data (list): sample to add

        Returns:
            None
        """

        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update the priority of the idx sample

        Args:
            idx (int): index of the sample to update
            p (float): priority of the data

        Returns:
            None
        """

        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get the sample, its index and priority

        Args:
            s ...

        Returns:
            idx (int): index of the sample
            self.tree[idx] (float): priority of the sample
            self.data[dataIdx] (list): sample

        """

        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])