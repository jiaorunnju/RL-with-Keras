from collections import deque
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import random
from tensorflow.keras.optimizers import Adam
from keras.layers import Lambda
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy


class A2C:

    def __init__(self, action_size, state_size, memory_size=2000):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.actor = self._build_actor()
        self.critic = self._build_critic()


    def _one_hot_action(self, x):
        t = np.array(x, dtype=np.int)
        result = np.zeros((len(t), self.action_size), dtype=np.int)
        result[range(len(t)), t] = 1
        return result


    def _build_actor(self):
        """
        define the actor structure
        :return: actor
        """
        x = keras.Input(shape=(self.state_size,))
        #r = keras.Input(shape=(1,))
        out = keras.layers.Dense(24, activation="relu")(x)
        out = keras.layers.Dense(24, activation='relu')(out)
        out = keras.layers.Dense(self.action_size, activation='softmax')(out)
        model = keras.Model(inputs =x, outputs = out)

        #def pg_loss(r, x):
            #return -K.mean(tf.reshape(r[:,0], (-1,1)) * K.log(K.batch_dot(x, K.one_hot(tf.to_int32(r[:,1]), self.action_size), axes=1)))

        def pg_loss(r, x):
            return tf.reshape(r[:,0],(-1, 1)) * sparse_categorical_crossentropy(tf.to_int32(r[:, 1]), x)

        model.compile(loss=pg_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_critic(self):
        """
        define the critic structure
        :return: critic
        """
        x = keras.Input(shape=(self.state_size,))
        out = keras.layers.Dense(24, activation="relu")(x)
        out = keras.layers.Dense(24, activation="relu")(out)
        out = keras.layers.Dense(1, activation="linear")(out)
        model = keras.Model(inputs=x, outputs=out)
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        w = self.actor.predict(state.reshape((1, -1)))
        return np.random.choice(self.action_size, p=w[0])

    def fit(self, state, action, reward, next_state, done):
        self.critic_replay(32)
        target = np.array([[reward]])
        if not done:
            target = reward + self.gamma * self.critic.predict(next_state.reshape((1, -1)))
        val_curr = self.critic.predict(state.reshape((1, -1)))
        self.actor.fit(state.reshape((1, -1)), np.array([[(target - val_curr)[0], action]]), epochs=1, verbose=0)


    def critic_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = np.array([[reward]])
            if not done:
                target = reward + self.gamma * self.critic.predict(next_state.reshape((1,-1)))
            self.critic.fit(state.reshape((1,-1)), target, epochs=1, verbose=0)

