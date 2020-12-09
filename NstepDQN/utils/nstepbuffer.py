"""NStep Buffer script

This manages the nstep buffer. 
"""

import random
from collections import deque

import numpy as np

class NStepBuffer:
    """
    Class for the NStep Buffer creation
    """

    def __init__(self, nsteps):
        """Instantiate the nstep buffer as empty lists

        Args:
            nsteps (int): steps to consider

        Returns:
            None
        """

        self.nstep = nsteps
        self.states, self.actions, self.rewards, self.obs_states, self.dones = \
            [deque(maxlen=nsteps) for _ in range(5)]

    def store(self, state, action, reward, obs_state, done):
        """Append the sample in the nstep buffer

        Args:
            state (list): state of the agent
            action (list): performed action
            reward (float): received reward
            obs_state (list): observed state after the action
            done (int): 1 if terminal states in the last episode

        Returns:
            None
        """

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.obs_states.append(obs_state)
        self.dones.append(done)

    def sample(self):
        """Get the samples from the nstep buffer

        Args:
            None

        Returns:
            states (list): states of the last episode
            actions (list): performed action in the last episode
            rewards (float): received reward in the last episode
            obs_states (list): observed state after the action in the last episode
            dones (int): 1 if terminal states in the last episode
        """
        
        if len(self.states) == self.nstep:
            dones = [self.dones[i] == 0 for i in range(self.nstep - 1)]

            if True in dones:
                return None

            return {'states': np.array(self.states),
                    'actions': np.array(self.actions),
                    'rewards': np.array(self.rewards),
                    'obs_states': np.array(self.obs_states),
                    'dones': np.array(self.dones)
                    }
        else:
            return None