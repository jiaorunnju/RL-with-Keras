from collections import deque
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

import gym
EPISODES = 60

class DQN:
    def __init__(self, action_size, state_size, memory_size=2000):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = 1.0
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        define the model structure
        :return: model
        """
        '''
        x = keras.Input(shape=(self.state_size,))
        out = keras.layers.Dense(16, activation="relu")(x)
        out = keras.layers.Dense(16, activation="relu")(out)
        out = keras.layers.Dense(self.action_size, activation="linear")(out)
        model = keras.Model(inputs=x, outputs=out)
        model.compile(loss='mse', optimizer=Adam())
        return model
        '''
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape((1,-1)))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1,-1)))[0])
            target_f = self.model.predict(state.reshape((1,-1)))
            target_f[0][action] = target
            self.model.fit(state.reshape((1,-1)), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay