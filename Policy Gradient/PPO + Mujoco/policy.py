
import numpy as np
import tensorflow as tf

class Policy (object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, clipping_range):
        
        self.epochs = 20
        self.beta = 1.0
        self.eta = 50
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.kl_targ = kl_targ
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = None
        self.learning_rate_mult = 1.0 # Dynamically adjust lr when D_KL out of control
    
        self.clipping_range = clipping_range
        self._build_graph()
        self._init_session()


    def _build_graph(self):
        # Build, initialize Tensorflow graph
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        # Input placeholders
        self.state = tf.placeholder(tf.float32,  (None, self.obs_dim), 'obs')
        self.action = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages = tf.placeholder(tf.float32, (None,), 'advantages')

        # Strength of D_KL loss terms
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta') 
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')

        #Learning rate
        self.learning_rate_ph = tf.placeholder(tf.float32, (),'eta')

        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        
        """
            hid1 size = observation dimensions x 10
            hid2 size = geometric mean of hid1 and hid3 sizes
            hid3 size = action dimensions x 10
        """
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.learning_rate = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined

        # Place tanh activation function for 3 hidden layers
        out = tf.layers.dense(self.state, hid1_size, tf.tanh, kernel_initializer= tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim)), name="h1")

        out = tf.layers.dense(out, hid2_size, tf.tanh,
        kernel_initializer= tf.random_normal_initializer(stddev=np.sqrt(1 / hid1_size)), name="h2")

        out = tf.layers.dense(out, hid3_size, tf.tanh,
        kernel_initializer= tf.random_normal_initializer(stddev=np.sqrt(1 / hid2_size)), name="h3")

        self.means = tf.layers.dense(out, self.act_dim, kernel_initializer=tf.random_normal_initializer
        (stddev=np.sqrt(1 / hid3_size)), name="means")

        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        print('Policy Params -- h1: {}, h2: {}, h3: {}, learning_rate: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.learning_rate, logvar_speed))

        
    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.means) /
                                        tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_means_ph) /
                                            tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
         Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        # Sample from distribution
        self.sampled_act = (self.means +  tf.exp(self.log_vars / 2.0) *
                        tf.random_normal(shape=(self.act_dim,)))

    def _loss_train_op(self):  
        print('setting up loss with clipping objective')
        pg_ratio = tf.exp(self.logp - self.logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range[0], 1 + self.clipping_range[1])
        surrogate_loss = tf.minimum(self.advantages* pg_ratio,
                                    self.advantages * clipped_pg_ratio)
        self.loss = -tf.reduce_mean(surrogate_loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_op = optimizer.minimize(self.loss)
    
    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self,observation):
        # Draw sample from policy dist
        feed_dict = {self.state : observation}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
        """
        feed_dict = {self.state: observes,
                     self.action: actions,
                     self.advantages: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.learning_rate_ph : self.learning_rate * self.learning_rate_mult}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.learning_rate_mult > 0.1:     
                self.learning_rate_mult /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.learning_rate_mult < 10:
                self.learning_rate_mult *= 1.5

    def close_sess(self):
        # Close TensorFlow session """
        self.sess.close()