import tensorflow as tf
import numpy as np
import threading

tf.compat.v1.disable_v2_behavior()

class ACNet:
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        self.scope = scope
        self.globalAC = globalAC
        self.lock = threading.Lock()

        self.s_size = 8
        self.a_size = 5

        with tf.compat.v1.variable_scope(self.scope):
            self.s = tf.compat.v1.placeholder(tf.float32, [None, self.s_size], 'state')
            self.a_his = tf.compat.v1.placeholder(tf.int32, [None, ], 'action')
            self.v_target = tf.compat.v1.placeholder(tf.float32, [None, 1], 'v_target')

            # Actor network
            with tf.compat.v1.variable_scope('actor'):
                l1 = tf.keras.layers.Dense(256, activation='relu', name='l1')(self.s)
                l1 = tf.keras.layers.Dropout(0.2)(l1)
                l2 = tf.keras.layers.Dense(128, activation='relu', name='l2')(l1)
                self.a_prob = tf.keras.layers.Dense(self.a_size, activation='softmax', name='a_prob')(l2)

            # Critic network
            with tf.compat.v1.variable_scope('critic'):
                l1_c = tf.keras.layers.Dense(256, activation='relu', name='l1_c')(self.s)
                l1_c = tf.keras.layers.Dropout(0.2)(l1_c)
                l2_c = tf.keras.layers.Dense(128, activation='relu', name='l2_c')(l1_c)
                self.v = tf.keras.layers.Dense(1, name='v')(l2_c)

            # Loss and optimization
            with tf.compat.v1.variable_scope('loss'):
                self.td = self.v_target - self.v
                self.critic_loss = tf.reduce_mean(tf.square(self.td))

                log_prob = tf.math.log(tf.reduce_sum(tf.one_hot(self.a_his, self.a_size) * self.a_prob, axis=1))
                self.actor_loss = -tf.reduce_mean(log_prob * tf.stop_gradient(self.td))

                self.entropy = -tf.reduce_mean(tf.reduce_sum(self.a_prob * tf.math.log(self.a_prob + 1e-10), axis=1))
                self.total_loss = self.actor_loss + 0.5 * self.critic_loss - 0.5 * self.entropy

            with tf.compat.v1.variable_scope('train'):
                self.global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'Global_Net')
                self.local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)

                self.grads = tf.gradients(self.total_loss, self.local_vars)
                self.grads, _ = tf.clip_by_global_norm(self.grads, 40)

                optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)
                self.apply_grads = optimizer.apply_gradients(zip(self.grads, self.global_vars))
                self.sync = [tf.compat.v1.assign(l_v, g_v) for l_v, g_v in zip(self.local_vars, self.global_vars)]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.a_prob, {self.s: s})
        return np.random.choice(self.a_size, p=probs[0])

    def update_global(self, feed_dict):
        with self.lock:
            self.sess.run(self.apply_grads, feed_dict)

    def pull_global(self):
        self.sess.run(self.sync)