
# This is the implication of DQN(Mnih et al 2015)
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# requires keras with tensorflow as backend.

"""
Created on Dec 12  2018
@author: Wenping Cui
"""
# ==============================================================================

# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from cliff_env import CliffEnv
import pickle

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=1000)
		self.gamma = 0.9    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.99
		self.learning_rate = 0.01
		self.model = self._build_model()

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(6, input_dim=self.state_size, activation='relu'))
		model.add(Dense(6, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
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
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
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
	env=CliffEnv()
	Array_E=[];
	Array_G=[];
	Array_t=[];
	EPISODES = 500
	state_size = 48
	action_size = 4
	agent = DQNAgent(state_size, action_size)
	done = False
	batch_size = 16
	G=0
	for e in range(EPISODES):
		state = env.reset()
		array=np.zeros(48)
		array[state[0]*12+state[1]]=1;
		array = np.reshape(array, [1, state_size])
		G=0;
		for time in range(100):
			# env.render()
			action = agent.act(array)
			next_state, reward, done, _ = env.step(action)
			#print (next_state)
			if done:
				reward=100
			else:
				reward = reward
			next_array=np.zeros(48)
			next_array[next_state[0]*12+next_state[1]]=1;
			next_array= np.reshape(next_array, [1, state_size])
			agent.remember(array, action, reward, next_array, done)
			state = next_state
			array=next_array
			G=G+reward
			if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
					  .format(e, EPISODES, G, agent.epsilon))
				break

			if len(agent.memory) > batch_size:
				loss = agent.replay(batch_size)
		if e % 10 == 0:
			 agent.save("save/CliffWalk-dqn_1.h5")
		Array_E.append(e)
		Array_G.append(G)
		Array_t.append(time)
	Varible={}
	Varible['E']=Array_E
	Varible['G']=Array_G
	Varible['t']=Array_t
	with open('CliffWalk.pkl', 'wb') as fp:
		pickle.dump(Varible,fp, protocol=pickle.HIGHEST_PROTOCOL)