"""DQN agent script 

This manages the training phase of the off-policy DQN.
"""

import random
from collections import deque

import yaml
import numpy as np

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()
    
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(seed)

from utils.deepnetwork import DeepNetwork
from utils.memorybuffer import Buffer

class DQN:
    """
    Class for the DQN agent
    """

    def __init__(self, env, params):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g.,dnn structure)

        Returns:
            None
        """

        self.env = env

        self.model = DeepNetwork.build(env, params['dnn'])
        self.model_opt = Adam()

        self.buffer = Buffer(params['buffer']['size'])
        
    def get_action(self, state):
        """Get the action to perform

        Args:
            state (list): agent current state

        Returns:
            action (float): sampled actions to perform
        """

        q_values = self.model(np.array([state])).numpy()[0]
        return np.argmax(q_values)


    def update(self, gamma, batch_size):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            batch_size (int): batch size for the off-policy A2C

        Returns:
            None
        """

        batch_size = min(self.buffer.size, batch_size)
        states, actions, rewards, obs_states, dones = self.buffer.sample(batch_size)

        # The updates require shape (n° samples, len(metric))
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        self.fit(gamma, states, actions, rewards, obs_states, dones)

    def fit(self, gamma, states, actions, rewards, obs_states, dones):
        """We want to minimizing mse of the temporal difference error given by Q(s,a|θ) and the target y = r + γ max_a' Q(s', a'|θ). The first problem is that target values for Q depends on Q itself, hence we are chasing a non-stationary target.

        Args:
            gamma (float): discount factor
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update

        Returns:
            None
        """
        
        with tf.GradientTape() as tape:
            # Compute the target y = r + γ max_a' Q(s', a'|θ)
            obs_qvalues = self.model(obs_states)
            obs_action = tf.math.argmax(obs_qvalues, axis=-1).numpy()
            idxs = np.array([[int(i), int(action)] for i, action in enumerate(obs_action)])

            max_obs_qvalues = tf.expand_dims(tf.gather_nd(obs_qvalues, idxs), axis=-1)
            y = rewards + gamma * max_obs_qvalues * dones

            # Compute values Q(s,a|θ)
            qvalues = self.model(states)
            idxs = np.array([[int(i), int(action)] for i, action in enumerate(actions)])
            qvalues = tf.expand_dims(tf.gather_nd(qvalues, idxs), axis=-1)

            # Compute the loss as mse of Q(s, a) - y
            td_errors = tf.math.subtract(qvalues, y)
            td_errors = 0.5 * tf.math.square(td_errors)
            loss = tf.math.reduce_mean(td_errors)

            # Compute the model gradient and update the network
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.model_opt.apply_gradients(zip(grad, self.model.trainable_variables))
          
    def train(self, tracker, n_episodes, verbose, params, hyperp):
        """Main loop for the agent's training phase

        Args:
            tracker (object): used to store and save the training stats
            n_episodes (int): n° of episodes to perform
            verbose (int): how frequent we save the training stats
            params (dict): agent parameters (e.g., the critic's gamma)
            hyperp (dict): algorithmic specific values (e.g., tau)

        Returns:
            None
        """

        mean_reward = deque(maxlen=100)

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  
        
            state = self.env.reset()

            while True:
                action = self.get_action(state)
                obs_state, obs_reward, done, _ = self.env.step(action)

                self.buffer.store(state, 
                    action, 
                    obs_reward, 
                    obs_state, 
                    1 - int(done)
                )

                ep_reward += obs_reward
                steps += 1

                state = obs_state

                if e > params['update_start']: 
                    self.update(
                        params['gamma'], 
                        params['buffer']['batch']
                    )                

                if done: break  
            
            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        


   