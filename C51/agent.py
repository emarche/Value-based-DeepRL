"""C51 agent script 

This manages the training phase of the off-policy C51.
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

class C51:
    """
    Class for the C51 agent
    """

    def __init__(self, env, params, n_atoms):
        """Initialize the agent, its network, optimizer and buffer

        Args:
            env (gym): gym environment
            params (dict): agent parameters (e.g.,dnn structure)
            n_atoms (int): n° atoms used to model the discrete probability of values

        Returns:
            None
        """

        self.env = env

        self.model = DeepNetwork.build(env, params['dnn'], n_atoms)
        self.model_tg = DeepNetwork.build(env, params['dnn'], n_atoms)
        self.model_tg.set_weights(self.model.get_weights())

        self.model_opt = Adam()

        self.buffer = Buffer(params['buffer']['size'])

    def get_action(self, state, eps, z):
        """Get the action to perform based on the Z distribution

        Args:
            state (list): agent current state
            eps (float): random action probability
            z (list): the equally distributed values

        Returns:
            action (float): sampled actions to perform
        """

        if np.random.uniform() <= eps:
            return np.random.randint(0, self.env.action_space.n)
        
        # Get list of tensors of shape (1, natoms) with size = n_actions
        # Cast the tensors to np and stack it, returning shape (n_actions, n_atoms)
        p = np.vstack(self.model(np.array([state])))

        # Compute Q(s, a) according C51: Σ z * p(s, a)
        qvalues = np.sum(np.multiply(p, z), axis=-1)

        # Return the best action
        return np.argmax(qvalues)   


    def update(self, gamma, batch_size, delta_z, z, v_min, v_max):
        """Prepare the samples to update the network

        Args:
            gamma (float): discount factor
            batch_size (int): batch size for the off-policy A2C
            delta_z (float): size of each "step" between the z values
            z (list): the equally distributed values
            v_min (int): min value 
            v_max (int): max value

        Returns:
            None
        """

        batch_size = min(self.buffer.size, batch_size)
        states, actions, rewards, obs_states, dones = self.buffer.sample(batch_size)

        #rewards = rewards.reshape(-1, 1)
        #dones = dones.reshape(-1, 1)

        self.fit(gamma, delta_z, z, v_min, v_max, states, actions, rewards, obs_states, dones)

    def fit(self, gamma, delta_z, z, v_min, v_max, states, actions, rewards, obs_states, dones):
        """TODO: fix C51 DQN loss documentation

        Args:
            gamma (float): discount factor
            delta_z (float): size of each "step" between the z values
            z (list): the equally distributed values
            v_min (int): min value 
            v_max (int): max value
            states (list): episode's states for the update
            actions (list): episode's actions for the update
            rewards (list): episode's rewards for the update
            obs_states (list): episode's obs_states for the update
            dones (list): episode's dones for the update

        Returns:
            None
        """

        obs_p_tg = np.array(self.model_tg(obs_states))

        # Get obs_actions
        obs_p = self.model.predict(obs_states)
        obs_qvalues = tf.math.multiply(obs_p, z)
        obs_qvalues = tf.math.reduce_sum(obs_qvalues, axis=-1)
        obs_actions = tf.math.argmax(obs_qvalues)
        
        m_prob = np.zeros((self.env.action_space.n, len(states), z.shape[1]))

        for i in range(len(states)):
            if dones[i] == 0:   # The distribution collapses to a point
                Tz = min(v_max, max(v_min, rewards[i]))
                bj = (Tz - v_min) / delta_z
                l, u = np.floor(bj), np.ceil(bj)
                m_prob[actions[i]][i][int(l)] += (u - bj)
                m_prob[actions[i]][i][int(u)] += (bj - l)
            else:
                for j in range(z.shape[1]):
                    Tz = min(v_max, max(v_min, rewards[i] + gamma * z.squeeze()[j]))
                    bj = (Tz - v_min) / delta_z
                    l, u = np.floor(bj), np.ceil(bj)
                    m_prob[actions[i]][i][int(l)] += (u - bj) * obs_p_tg[obs_actions[i]][i][j] 
                    m_prob[actions[i]][i][int(u)] += (bj - l) * obs_p_tg[obs_actions[i]][i][j]
                    
        with tf.GradientTape() as tape:
            p = self.model(states)
            log_p = tf.math.log(p)
            loss = tf.math.multiply(log_p, m_prob)
            loss = tf.math.reduce_sum(loss, axis=-1)
            loss = -tf.math.reduce_mean(loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))


    @tf.function
    def polyak_update(self, weights, target_weights, tau):
        """Polyak update for the target networks

        Args:
            weights (list): network weights
            target_weights (list): target network weights
            tau (float): controls the update rate

        Returns:
            None
        """

        for (w, tw) in zip(weights, target_weights):
            tw.assign(w * tau + tw * (1 - tau))

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
        tau, use_polyak, tg_update = hyperp['tau'], hyperp['use_polyak'], hyperp['tg_update']

        n_atoms, v_min, v_max = hyperp['atoms'], hyperp['v_min'], hyperp['v_max']
        # Z is the discrete probability distribution of the values
        # Values are then equally distributed between v_min and v_max in n_atoms values
        # delta_z is the size of each "step" between the values
        delta_z = (v_max - v_min) / float(n_atoms - 1)
        z = np.linspace([v_min], [v_max], num=n_atoms, axis=1)

        for e in range(n_episodes):
            ep_reward, steps = 0, 0  
        
            state = self.env.reset()

            while True:
                action = self.get_action(state, eps, z)
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
                        params['buffer']['batch'],
                        delta_z,
                        z,
                        v_min,
                        v_max
                    )     
                    if use_polyak:
                        # DDPG Polyak update improve stability over the periodical full copy
                        self.polyak_update(self.model.variables, self.model_tg.variables, tau)
                    if steps % tg_update == 0:
                        self.model_tg.set_weights(self.model.get_weights())
                        
                if done: break           

            eps = max(eps_min, eps * eps_decay)

            mean_reward.append(ep_reward)
            tracker.update([e, ep_reward])

            if e % verbose == 0: tracker.save_metrics()

            print(f'Ep: {e}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')
        


   