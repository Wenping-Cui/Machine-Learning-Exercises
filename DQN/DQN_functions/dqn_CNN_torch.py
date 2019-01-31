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
import gym
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pickle
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _Model_CNN(torch.nn.Module):
    #Our batch shape for input x is (3, 32, 32)
    def __init__(self, state_size, action_size):
        h, w = state_size
        super(_Model_CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2)
        #self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, action_size) # 448 or 512
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
class AstraiAgent:
    def __init__(self, INPUT_SHAPE, action_size,WINDOW_LENGTH):
        self.INPUT_SHAPE = INPUT_SHAPE
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.WINDOW_LENGTH=WINDOW_LENGTH
        # parameters are from Extended Data Table 1 at Mnih et al 2015
        self.gamma = 0.99  	# discount rate
        self.epsilon = 1.0  	# exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model = _Model_CNN(self.INPUT_SHAPE, action_size).to(device)
        self.target_model = _Model_CNN(self.INPUT_SHAPE, action_size).to(device)
        self.target_model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
        with torch.no_grad():
                state =torch.from_numpy(state).float()
                state=state.to(device) 
                # t.max(1) will return largest value for column of each row.second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.model(state).max(1)[1].view(1, 1)


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        criterion= nn.MSELoss()

        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            state =torch.from_numpy(state).float()
            state=state.to(device) 
            next_state =torch.from_numpy(next_state).float()
            next_state=next_state.to(device)   
            #action =torch.from_numpy(action).float()
            #action=action.to(device)   
            gamma =torch.tensor(self.gamma).to(device)
            reward =torch.tensor(reward).to(device)
            target = torch.tensor(reward).to(device)
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

    def update(self,):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        return 0


