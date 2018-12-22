
# This is the implication of DQN(Mnih et al 2015)
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# requires keras with tensorflow as backend.
"""
Created on Dec 13 2018
@author: Wenping Cui
"""
# ==============================================================================
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
import keras.optimizers as opt
import keras.backend as K

class AstraiAgent:
    def __init__(self, INPUT_SHAPE, action_size,WINDOW_LENGTH):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.WINDOW_LENGTH=WINDOW_LENGTH
        self.input_shape= (self.WINDOW_LENGTH,) + self.INPUT_SHAPE
        # parameters are from Extended Data Table 1 at Mnih et al 2015
        self.gamma = 0.99  	# discount rate
        self.epsilon = 1.0  	# exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0025
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # CNN for Deep-Q learning Model
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=self.input_shape))
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
        opt.RMSprop(lr=self.learning_rate, epsilon=0.01)
        model.compile(optimizer='rmsprop',loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def _update_model(self):
        self.target_model =self.model
        return 0
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *np.amax(self.target_model.predict(next_state)[0]))
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


