from collections import deque
import numpy as np
import tensorflow as tf
import random


class A2C:

    def __init__(self, action_size, state_size, sess, memory_size=2000):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)

        # reward discount
        self.gamma = 0.95

        # build model and optimizer
        self._build_actor()
        self._build_critic()

        # Session and initialize variable
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        # Control the frequency that updates the target network
        self.counter = 0
        self.replace_duration = 50

    def _build_actor(self):
        """
        define the actor structure
        :return: actor
        """
        with tf.variable_scope("actor"):
            self.actor_x = tf.placeholder(tf.float32, (None, self.state_size))
            actor_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(self.actor_x)
            actor_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(actor_out)
            actor_out = tf.layers.Dense(self.action_size, activation=tf.nn.softmax).apply(actor_out)
            self.actor_net = actor_out

        self.actor_y = tf.placeholder(tf.float32, (None, self.action_size))

        # policy gradient loss, similar to cross entropy loss
        self.actor_loss = -tf.reduce_mean(self.actor_y * tf.log(tf.clip_by_value(self.actor_net, 1e-10, 1.0)))
        self.actor_train_step = tf.train.AdamOptimizer().minimize(self.actor_loss)

        with tf.variable_scope("actor_backup"):
            self.actor_backup_x = tf.placeholder(tf.float32, (None, self.state_size))
            actor_backup_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(self.actor_backup_x)
            actor_backup_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(actor_backup_out)
            actor_backup_out = tf.layers.Dense(self.action_size, activation=tf.nn.softmax).apply(actor_backup_out)
            self.actor_backup_net = actor_backup_out

        a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        b_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_backup')
        self.actor_replace_op = [tf.assign(b, a) for b, a in zip(b_params, a_params)]

    def _build_critic(self):
        """
        define the critic structure
        :return: critic
        """
        with tf.variable_scope("critic"):
            self.critic_x = tf.placeholder(tf.float32, (None, self.state_size))
            critic_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(self.critic_x)
            critic_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(critic_out)
            critic_out = tf.layers.Dense(1).apply(critic_out)
            self.critic_net = critic_out

        self.critic_y = tf.placeholder(tf.float32, (None, 1))
        self.critic_loss = tf.losses.mean_squared_error(self.critic_y, self.critic_net)
        self.critic_train_step = tf.train.AdamOptimizer().minimize(self.critic_loss)

        with tf.variable_scope("critic_backup"):
            self.critic_backup_x = tf.placeholder(tf.float32, (None, self.state_size))
            critic_backup_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(self.critic_backup_x)
            critic_backup_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(critic_backup_out)
            critic_backup_out = tf.layers.Dense(1).apply(critic_backup_out)
            self.critic_backup_net = critic_backup_out

        a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        b_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_backup')
        self.critic_replace_op = [tf.assign(b, a) for b, a in zip(b_params, a_params)]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_act(self, state):
        # select actions with backup network, which is more stable when training
        w = self.sess.run(self.actor_backup_net, feed_dict={self.actor_backup_x: state.reshape((1, -1))})
        return np.random.choice(self.action_size, p=w[0])

    def act(self, state):
        return self.train_act(state)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.counter += 1
        if self.counter > self.replace_duration:
            self.sess.run(self.actor_replace_op)
            self.sess.run(self.critic_replace_op)
            self.counter = 0

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)
        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        done = np.array(done)

        # train the critic network
        critic_target = rewards.reshape((-1, 1))
        critic_target[~done] += self.gamma * self.sess.run(self.critic_backup_net, feed_dict={self.critic_backup_x: next_states[~done]})
        self.sess.run(self.critic_train_step, feed_dict={self.critic_x: states, self.critic_y: critic_target})

        # train the actor network
        td_error = critic_target - self.sess.run(self.critic_net, feed_dict={self.critic_x: states})
        one_hot_mat = np.zeros((batch_size, self.action_size), dtype=np.int)
        one_hot_mat[range(batch_size), actions] = 1
        one_hot_mat = td_error * one_hot_mat
        self.sess.run(self.actor_train_step, feed_dict={self.actor_x: states, self.actor_y: one_hot_mat})

    def save(self):
        """
        save the model
        :return:
        """
        saver = tf.train.Saver()
        saver.save(self.sess, "ac/model.ckpt")

    def restore(self):
        """
        restore the model
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, "ac/model.ckpt")

