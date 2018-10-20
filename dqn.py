from collections import deque
import numpy as np
import random
import tensorflow as tf


class DQN:
    def __init__(self, action_size, state_size, sess, memory_size=2000):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)

        # The reward discount
        self.gamma = 0.95

        # These controls the trade-off between exploration and exploitation
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = 1.0

        # Build the act_net and target_net, and define the loss and optimizer
        self._build_model()

        # Session and initialize variable
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        # Control the frequency that updates the target network
        self.counter = 0
        self.replace_duration = 50

    def _build_model(self):
        """
        define the model structure
        :return: model
        """
        # Define the act_net
        with tf.variable_scope("act_net"):
            self.act_x = tf.placeholder(tf.float32, (None, self.state_size))
            self.target_y = tf.placeholder(tf.float32, (None, self.action_size))
            act_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(self.act_x)
            act_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(act_out)
            act_out = tf.layers.Dense(self.action_size).apply(act_out)
            self.act_net = act_out

        # loss and optimizer
        self.loss = tf.losses.mean_squared_error(self.target_y, self.act_net)
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        # Define the target network
        with tf.variable_scope("target_net"):
            self.target_x = tf.placeholder(tf.float32, (None, self.state_size))
            target_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(self.target_x)
            target_out = tf.layers.Dense(24, activation=tf.nn.relu).apply(target_out)
            target_out = tf.layers.Dense(self.action_size).apply(target_out)
            self.target_net = target_out

        # Define the update operation
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='act_net')
        self.target_replace_op = [tf.assign(t, a) for t, a in zip(t_params, a_params)]

    def remember(self, state, action, reward, next_state, done):
        """
        Utilize memory replay
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_act(self, state):
        """
        choose action when training
        :param state:
        :return:
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.sess.run(self.act_net, feed_dict={self.act_x: state.reshape(1, -1)})
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """
        replay experience with batch_size
        :param batch_size:
        :return:
        """
        if len(self.memory) < batch_size:
            return

        self.counter += 1
        if self.counter > self.replace_duration:
            self.sess.run(self.target_replace_op)
            self.counter = 0
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)
        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        done = np.array(done)

        target = rewards

        # Compute target with target network
        target[~done] += self.gamma * self.sess.run(tf.reduce_max(self.target_net, reduction_indices=[1]), feed_dict= {self.target_x: next_states[~done]})
        present = self.sess.run(self.act_net, feed_dict={self.act_x: states})
        present[range(batch_size), actions] = target

        # update the action network
        self.sess.run(self.train_step, feed_dict={self.act_x: states, self.target_y: present})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        """
        choose action when testing, without randomness
        :param state:
        :return:
        """
        act_values = self.sess.run(self.target_net, feed_dict={self.target_x: state.reshape(1, -1)})
        return np.argmax(act_values[0])

    def save(self):
        """
        save the model
        :return:
        """
        saver = tf.train.Saver()
        saver.save(self.sess, "dqn/model.ckpt")

    def restore(self):
        """
        restore the model
        :return:
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, "dqn/model.ckpt")
