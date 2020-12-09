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
from utils.memorybuffer import PriorityBuffer

class PERDQN:
    """
    Class for the PERDQN agent
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

        self.buffer = PriorityBuffer(params['buffer']['size'])

    def get_action(self, state, eps):
        """Get the action to perform

        Args:
            state (list): agent current state
            eps (float): random action probability

        Returns:
            action (float): sampled actions to perform
        """

        if np.random.uniform() <= eps:
            return np.random.randint(0, self.env.action_space.n)
        
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

        minibatch, mb_idxs, mb_weights = self.buffer.sample(batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        obs_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # The updates require shape (n° samples, len(metric))
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        self.fit(gamma, states, actions, rewards, obs_states, dones, mb_idxs, mb_weights)

    def fit(self, gamma, states, actions, rewards, obs_states, dones, mb_idxs, mb_weights):
        """We want to minimizing mse of the temporal difference error given by Q(s,a|θ) and the target y = r + γ max_a' Q(s', a'|θ). The first problem is that target values for Q depends on Q itself, hence we are chasing a non-stationary target. Finally, update the priorities

        Args:
            gamma (float): discount factor
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update
            samples_idxs (list): indices of sampled experiences
            is_weights (list): priority of sampled experiences

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

            # Compute the loss as mse of Q(s, a) - y, adjusting the errors with their is_weight
            td_errors = tf.math.subtract(qvalues, y)
            td_errors = 0.5 * tf.math.square(td_errors)
            mb_weights = tf.convert_to_tensor(mb_weights, dtype=tf.float32)
            td_errors = tf.math.multiply(td_errors, mb_weights)

            loss = tf.math.reduce_mean(td_errors)

            # Compute the model gradient and update the network
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.model_opt.apply_gradients(zip(grad, self.model.trainable_variables))

        qvalues = self.model(states).numpy()

        qvalues = np.array([qvalue[action] for action, qvalue in zip(actions, qvalues)])
        qvalues = np.expand_dims(qvalues, axis=-1)

        td_error = np.abs(qvalues - y).squeeze().squeeze()

        for i in range(len(states)):
            idx = mb_idxs[i]
            self.buffer.update(idx, td_error[i])  

    def store(self, gamma, state, action, reward, obs_state, done):
        """Compute the td error of the sample and store both in the sumtree

        Args:
            gamma (float): discount factor
            state (list): state of the agent
            action (list): performed action
            reward (float): received reward
            obs_state (list): observed state after the action
            done (int): 1 if terminal states in the last episode

        Returns:
            None
        """
        # Compute the td error of the sample

        # Compute the target y = r + γ max_a' Q(s', a'|θ)
        obs_qvalues = self.model(np.array([state])).numpy()[0]
        obs_action = tf.math.argmax(obs_qvalues, axis=-1)

        max_obs_qvalue = obs_qvalues[obs_action]
        y = reward + gamma * max_obs_qvalue * done

        qvalues = self.model(np.array([state])).numpy()[0]
        qvalue = qvalues[action]

        td_error = np.abs(qvalue - y)

        self.buffer.store(td_error, [state, action, reward, obs_state, done])

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

        eps, eps_min = params['eps'], params['eps_min']
        eps_decay = hyperp['eps_d']

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  
        
            state = self.env.reset()

            while True:
                action = self.get_action(state, eps)
                obs_state, obs_reward, done, _ = self.env.step(action)

                self.store(params['gamma'], 
                    state, 
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
            
            eps = max(eps_min, eps * eps_decay)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        


   