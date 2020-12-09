"""NStepDQN agent script 

This manages the training phase of the off-policy NStepDQN.
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
from utils.nstepbuffer import NStepBuffer

class NStepDQN:
    """
    Class for the NStepDQN agent
    """

    def __init__(self, env, params, nsteps):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g.,dnn structure)
            nsteps (int): nsteps to consider

        Returns:
            None
        """

        self.env = env

        self.model = DeepNetwork.build(env, params['dnn'])
        self.model_opt = Adam()

        self.buffer = Buffer(params['buffer']['size'])
        self.nstep_buffer = NStepBuffer(nsteps)

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
        states, actions, rewards, obs_states, dones = self.buffer.sample(batch_size)

        # The updates require shape (n° samples, len(metric))
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        self.fit(gamma, states, actions, rewards, obs_states, dones)

    def fit(self, gamma, states, actions, rewards, obs_states, dones):
        """We want to minimizing mse of the temporal difference error given by Q(s,a|θ) and the nstep target y = r_nstep + γ max_a' Q(s_nstep, a'|θ), the nstep values are already in the buffer. It shares the non-stationarity issue of DQN. The second problem is that nstep returns considers trajectories, and we can’t assume that these trajectories would have been taken if the agent was using the current policy. So some sort of correction (e.g., importance sampling) should be used to adjust the target. However, according to "Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target": it is possible to ignore off-policy correction without seeing an adverse effect in the overall performance of Q-learning and Sarsa. This is problem specific, but it suggests that off-policy correction is not always necessary for learning from samples from the experience replay buffer. Furthermore, the Rainbow algorithm also does not perform any correction

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
            # Compute the target y = r_nstep + γ max_a' Q(s_nstep, a'|θ)
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
    
    def store(self, gamma, state, action, reward, obs_state, done):
        """Store in the nstepbuffer, compute and store the sample in the buffer

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

        self.nstep_buffer.store(state, action, reward, obs_state, done)
        nstep_sample = self.nstep_buffer.sample()

        if nstep_sample is not None:
            state = nstep_sample['states'][0]
            action = nstep_sample['actions'][0]
            rewards = nstep_sample['rewards']
            obs_state = nstep_sample['obs_states'][-1]
            done = nstep_sample['dones'][-1]

            reward = np.sum([np.power(gamma, i) * r for i, r in enumerate(rewards)])

            self.buffer.store(state, action, reward, obs_state, done)  

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
        


   