
import numpy as np
import tensorflow as tf

class policy (object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, clipping_range=None):
        
        
        self.epochs = 20
        self.beta = 1.0
        self.eta = 50
        self.hid1_mult = hid1_mult
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = None
        self.learning_rate_mult = 1.0 # d0ynamically adjust lr when D_KL out of control
    
        self.clipping_range = clipping_range
        self._build_graph()
        self._init_session()


    def _init_session():
        # Launch Tensorflow session 
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph():


