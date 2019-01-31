# -*- coding: utf-8 -*-
# This DQN Agent Software is from the following  Jaromir Janisch  source: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
class _model_NN(nn.Module):
	def __init__(self, state_size, action_size):
		super(_model_NN, self).__init__()
		self.linear1 = nn.Linear(state_size, 32)
		self.linear2 = nn.Linear(32, 24)
		self.head = nn.Linear(24, action_size)
	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		#return self.head(x.view(x.size(0), -1))
		return self.head(x)

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=8000)
		self.gamma = 0.99    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.01
		self.model = _model_NN(state_size, action_size)
		self.model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model(torch.from_numpy(state).float())
		act_values = act_values.detach().numpy()
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		#optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		#optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
		optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		optimizer.zero_grad()
		criterion= nn.MSELoss()
		#criterion= nn.SmoothL1Loss()
		#criterion= nn.L1Loss()
		states, targets_f = [], []
		for state, action, reward, next_state, done in minibatch:
			state =torch.from_numpy(state).float()
			next_state =torch.from_numpy(next_state).float()
			gamma =torch.tensor(self.gamma)
			reward =torch.tensor(reward)
			target = torch.tensor(reward)
			if not done:
				target = reward + gamma*torch.max(self.model(next_state)[0])
			target_f = self.model(state)
			target_f[0][action] = target 
			# Filtering out states and targets for training
			states.append(state)
			targets_f.append(target_f)
		states = torch.cat(states, dim=0)
		targets_f = torch.cat(targets_f, dim=0)
		targets_f  = torch.tensor(targets_f, requires_grad=False)
		loss = criterion(self.model(states), targets_f)
		loss.backward()
		optimizer.step()
		# Keeping track of loss
		loss = loss.item()
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		return loss
	def save(self, name):
		torch.save(self.model.state_dict(), name)

	def load(self, name):
		self.model.load_state_dict(torch.load(name))


if __name__ == "__main__":
	EPISODES = 500
	env = gym.make('CartPole-v1')
	Array_E=[];
	Array_t=[];
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)
	#agent.load("save/cartpole-dqn.h5")
	done = False
	batch_size = 32*8
	for e in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
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
				if time % 50 == 0:
					print("episode: {}/{}, time: {}, loss: {:.4f}"
						.format(e, EPISODES, time, loss))  
		if e % 10 == 0:
			 agent.save("save/cartpole-dqn1.h5")
		Array_E.append(e)
		Array_t.append(time)
	Varible={}
	Varible['E']=Array_E
	Varible['t']=Array_t
	with open('CartPole_torch.pkl', 'wb') as fp:
		pickle.dump(Varible,fp, protocol=pickle.HIGHEST_PROTOCOL)    