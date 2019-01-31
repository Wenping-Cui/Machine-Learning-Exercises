
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
from  DQN_functions.dqn_CNN_torch import *
from DQN_functions.pretraining_funs import *
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

env_name='Assault-v0'
Input_shape=(88,88)
WINDOW_LENGTH=4
T=10000
max_length=20000
EPISODES=1000
Input_shape=(88,88)
batch_size = 32*2
if __name__ == "__main__":
    env = gym.make(env_name)
    #pre_memory=pre_replay(env_name, T, max_length,WINDOW_LENGTH)
    action_size = env.action_space.n
    agent = AstraiAgent(Input_shape, action_size,WINDOW_LENGTH)
    #agent.memory=pre_memory
    done = False
    G=0
    Input_States=np.zeros([1,WINDOW_LENGTH, 88,88])
    Next_States=np.zeros([1,WINDOW_LENGTH, 88,88])
    time =0
    N =0
    Array_E=[]
    Array_T=[]
    Array_G=[]
    Array_L=[]
    for e in range(EPISODES):
        G=0
        for i in range(WINDOW_LENGTH):
            state = env.reset()
            Input_States[0,i,:,:]= state_process(state)
        assert Input_States.shape == (1,4, 88, 88)
        for time in range(10000):
            action = agent.act(Input_States).cpu().numpy().item()
            R=0
            for i in range(WINDOW_LENGTH):
                next_state, reward, done, _ = env.step(action)
                R = R + reward
                Next_States[0,i,:,:]= state_process(next_state)
            G=G+reward
            agent.remember(Input_States, action, R, Next_States, done)
            Input_States = Next_States
            assert Next_States.shape == (1,4, 88, 88)
            Input_States = Next_States
            N =N+1
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if time % 100 == 0: # Logging training loss every 100 timesteps
                    print("episode: {}/{},score: {}, Experience:{:.1f}, time:{:.1f} loss: {:.4f}".format(e, EPISODES, G,N, time, loss))
            if done:
                print("episode: {}/{}, score: {}, e: {:.2},time:{:.1f} ".format(e, EPISODES, G , agent.epsilon, time))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
            #if e % 1 == 0:
                agent.update()
        Array_E.append(e)
        Array_T.append(time)
        Array_G.append(G)
        Array_L.append(loss)
        if e % 20 == 0:
            file_name="save/Assault-torch_e"+str(e)+'.h5'
            agent.save(file_name)
        if e % 100 == 0:
            Varible={}
            Varible['e']=Array_E
            Varible['elength']=Array_T
            Varible['R']=Array_G
            Varible['Loss']=Array_L
            with open(env_name+'.pkl', 'wb') as fp:
                pickle.dump(Varible,fp, protocol=pickle.HIGHEST_PROTOCOL) 