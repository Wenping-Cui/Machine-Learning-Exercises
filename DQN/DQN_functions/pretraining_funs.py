
"""
This is the implication of DQN following (Mnih et al 2015)
Created on Dec 13 2018
@author: Wenping Cui
"""
# ==============================================================================
from PIL import Image
import numpy as np
from collections import deque
import gym

# process the input states from gym
def state_process(state):
    assert state.ndim == 3  # (height, width, channel)
    Input_shape=(88,88)
    img = Image.fromarray(state).convert('L')  # convert to grayscale
    area = (0, 0, 160, 230)  # crop image and ignore the unrelated area
    img = img.crop(area) # cropped = img.crop( ( x, y, x + width , y + height ) )
    img = img.resize(Input_shape) # resize 
    processed_observation = np.array(img)
    assert processed_observation.shape == Input_shape
    return processed_observation.astype('uint8')  # saves storage in experience memory

# replay memory without training
def pre_replay(env_name, T, max_length,WINDOW_LENGTH):
    env = gym.make(env_name)
    memory = deque(maxlen=max_length)
    Input_shape=(88,88)
    States = deque(maxlen=WINDOW_LENGTH)
    Input_States = np.zeros([1, WINDOW_LENGTH, 88, 88])
    Next_States = np.zeros([1, WINDOW_LENGTH, 88, 88])
    # initialize
    state = env.reset()
    action=env.action_space.sample()
    for i in range(WINDOW_LENGTH):
        state, reward, done, _ = env.step(action)
        States.append(state_process(state))
    Input_States = np.asarray(States).reshape(1,4, 88, 88)
    # loop through the premomery size
    for time in range(T):
        action=env.action_space.sample()
        for i in range(WINDOW_LENGTH):
            state, reward, done, _ = env.step(action)
            States.append(state_process(state))
        Next_States = np.asarray(States).reshape(1,4, 88, 88)
        memory.append((Input_States, action, reward, Next_States, done))
        Input_States=Next_States
        if done: env.reset()
    return memory

