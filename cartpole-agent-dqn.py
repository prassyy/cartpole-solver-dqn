import sys
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.reshape(1,4), action, reward, next_state.reshape(1,4), done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1,4))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        for state, action, reward, next_state, done in random.sample(self.memory, batch_size):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

EPISODES = 100

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    # agent.load("save/cartpole-dqn.h5")

    for e in range(1, EPISODES):
        state = env.reset()

        for time_t in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if e%10 == 0:
            	env.render()
            	agent.save("save/cartpole-dqn.h5")

            if len(agent.memory)>batch_size:
            	agent.replay(batch_size)

            if done:
                print("Episode: {}/{}, Score: {}".format(e, EPISODES, time_t))
                break