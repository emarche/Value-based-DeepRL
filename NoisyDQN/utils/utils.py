"""Noisy dense layer utility script

This manages the noisy dense definition
"""

import yaml
import numpy as np

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

import tensorflow as tf
tf.random.set_seed(seed)

def noisy_dense(x, size, name, bias=True, activation_fn=tf.identity):

    # the function used in eq.7,8
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

    # Initializer of \mu and \sigma 
    mu_init = tf.random_uniform_initializer( \
        minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5), \
        maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5)
    )
    sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))

    # Sample noise from gaussian
    p = tf.random.normal([x.get_shape().as_list()[1], 1])
    q = tf.random.normal([1, size])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

    # w = w_mu + w_sigma * w_epsilon
    w_mu = tf.compat.v1.get_variable(name+"/w_mu", [x.get_shape()[1], size], initializer=mu_init)
    w_sigma = tf.compat.v1.get_variable(name+"/w_sigma", [x.get_shape()[1], size],initializer=sigma_init)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(x, w)

    if bias:
        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.compat.v1.get_variable(name + "/b_mu", [size], initializer=mu_init)
        b_sigma = tf.compat.v1.get_variable(name + "/b_sigma", [size], initializer=sigma_init)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation_fn(ret + b)
    else:
        return activation_fn(ret)