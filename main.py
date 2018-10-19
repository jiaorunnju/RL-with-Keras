import numpy as np
import gym
#from actor_critic import A2C
from dqn import DQN
import tensorflow as tf

EPISODES = 300


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    with tf.Session() as sess:
        agent = DQN(action_size, state_size, sess)
        #agent = A2C(action_size, state_size)
        for e in range(EPISODES):
            state = env.reset()
            for t in range(500):
                #env.render()
                action = agent.train_act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print("episode: {}/{}, score: {}".format(e, EPISODES, t))
                    break
                agent.replay(32)