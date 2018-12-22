# This is the implication of DQN(Mnih et al 2015)
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# requires keras with tensorflow as backend.

"""
Created on Dec 12  2018
@author: Wenping Cui
"""
# ==============================================================================
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K
K.tensorflow_backend._get_available_gpus()
class AstraiAgent:
    def __init__(self, INPUT_SHAPE, action_size):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.WINDOW_LENGTH=4
        self.input_shape= (self.WINDOW_LENGTH,) + self.INPUT_SHAPE
        # parameters are from Extended Data Table 1 at Mnih et al 2015
        self.gamma = 0.95    	# discount rate
        self.epsilon = 1.0  	# exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0025
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=input_shape))
        model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        print (model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('Assault-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        print (state.shape)
        state = np.reshape(state, [1, state_size])
        print (state)
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                        .format(e, EPISODES, time, loss))  
        if e % 10 == 0:
             agent.save("save/Assault-dqn.h5")
                     Array_E.append(e)