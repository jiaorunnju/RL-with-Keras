import numpy as np
import gym
from actor_critic import A2C
from dqn import DQN
from demo import DQNAgent

EPISODES = 50


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    #agent = DQN(action_size, state_size)
    agent = A2C(action_size, state_size)

    for e in range(EPISODES):
        state = env.reset()
        #state = state.reshape((1,-1))
        for t in range(500):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            #next_state = next_state.reshape((1,-1))
            agent.remember(state, action,reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}".format(e, EPISODES, t))
                break
            if len(agent.memory)>32:
                agent.replay(32)