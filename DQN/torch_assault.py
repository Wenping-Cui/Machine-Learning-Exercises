
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
import gym
import numpy as np
from collections import deque
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
import keras.optimizers as opt
import keras.backend as K
import setGPU

class AstraiAgent:
    def __init__(self, INPUT_SHAPE, action_size,WINDOW_LENGTH):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.WINDOW_LENGTH=WINDOW_LENGTH
        self.input_shape= (self.WINDOW_LENGTH,) + self.INPUT_SHAPE
        # parameters are from Extended Data Table 1 at Mnih et al 2015
        self.gamma = 0.95    	# discount rate
        self.epsilon = 1.0  	# exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
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
        print (model.summary())
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
            target_f = self.target_model.predict(state)
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
def state_process(state, Input_shape):
        assert state.ndim == 3  # (height, width, channel)
        Input_shape=(88,88)
        img = Image.fromarray(state)
        img = img.resize(Input_shape).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == Input_shape
        return processed_observation.astype('uint8')  # saves storage in experience memory


if __name__ == "__main__":
    env = gym.make('Assault-v0')
    EPISODES=1000000
    action_size = env.action_space.n
    Input_shape=(88,88)
    WINDOW_LENGTH=4
    agent = AstraiAgent(Input_shape, action_size,WINDOW_LENGTH)
    done = False
    batch_size = 32
    G=0
    Input_States=np.zeros([1,WINDOW_LENGTH, 88,88])
    Next_States=np.zeros([1,WINDOW_LENGTH, 88,88])
    time =0
    for e in range(EPISODES):
        for i in range(WINDOW_LENGTH):
            state = env.reset()
            Input_States[0,i,:,:]= state_process(state, Input_shape)
        assert Input_States.shape == (1,4, 88, 88)
        for time in range(10000):
            action = agent.act(Input_States)
            for i in range(WINDOW_LENGTH):
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                G=G+reward
                Next_States[0,i,:,:]= state_process(next_state,Input_shape)
            Input_States = Next_States
            assert Next_States.shape == (1,4, 88, 88)
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            else:
                agent.remember(Input_States, action, reward, Next_States, done)
                time =time+1;
                if time%10000==0: agent._update_model

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 100 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}".format(e, EPISODES, time, loss))  
        if e % 100 == 0:
             file_name="Assault-dqn_e"+str(e)+'.h5'
             agent.save(file_name)