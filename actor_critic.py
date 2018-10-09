from collections import deque
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import random
from tensorflow.keras.optimizers import Adam
import logging


class A2C:

    def __init__(self, action_size, state_size, memory_size=2000):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def _build_actor(self):
        """
        define the actor structure
        :return: actor
        """
        x = keras.Input(shape=(self.state_size,))
        out = keras.layers.Dense(24, activation="relu")(x)
        out = keras.layers.Dense(24, activation='relu')(out)
        out = keras.layers.Dense(self.action_size, activation='softmax')(out)
        model = keras.Model(inputs = x, outputs = out)

        def pg_loss(x, r):
            return -r[:,0]*K.log(K.batch_dot(x, K.one_hot(K.cast(r[:,1], 'int32'), self.action_size), axes=1)[0])

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
        #return np.random.choice(self.action_size, p=np.exp(w[0])/np.sum(np.exp(w[0])))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, done = zip(*minibatch)
        states = np.array(states)
        actions = np.array(actions, dtype=np.int)
        rewards = np.array(rewards)
        done = np.array(done)
        next_states = np.array(next_states)

        val_next = self.critic.predict(next_states)
        val_present = self.critic.predict(states)
        rewards[~done] = rewards[~done] + self.gamma * val_next.squeeze()[~done]
        self.critic.fit(states, rewards, verbose=0, epochs=1)
        tmp = np.hstack(((rewards - val_present.squeeze()).reshape((-1,1)), actions.reshape((-1,1))))
        self.actor.fit(states, tmp, verbose=0, epochs=1)
