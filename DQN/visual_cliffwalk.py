# This is based on examples provided by pygame
"""
Created on Dec 12  2018
@author: Wenping Cui
"""
# ==============================================================================
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random,math,os,time
import pygame
from dqn_batch_cliffwalk import DQNAgent

state_size = 48
action_size = 4
agent = DQNAgent(state_size, action_size)
agent.load("save/CliffWalk-dqn_1.h5")
agent.epsilon=0.9

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 128)
 
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 20*2
HEIGHT = 20*2
 
# This sets the margin between each cell
MARGIN = 5
 
# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
for row in range(4):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(12):
        grid[row].append(0)  # Append a cell
 
# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
grid[3][11] = 2
# Initialize pygame
pygame.init()
 
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [45*12, 45*4]
screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("Cliff Walk")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
state0=[3, 0]
stat_s=[3, 0]
# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # User clicks the mouse. Get the position
            pos = pygame.mouse.get_pos()
            # Change the x/y screen coordinates to grid coordinates
            column = pos[0] // (WIDTH + MARGIN)
            row = pos[1] // (HEIGHT + MARGIN)
            # Set that location to one
            print("Click ", pos, "Grid coordinates: ", row, column)

    if state0[0]==3 and state0[1]==11:
       state0 = stat_s
       for y in range(3):
              for x in range(12):
                    grid[y][x] = 0
       grid[3][0] = 1
       grid[3][11] = 2
    else: 
        array=np.zeros(48)
        array[state0[0]*12+state0[1]]=1;
        array = np.reshape(array, [1, 48])
        action =  agent.act(array)
        if action==0:
            state=[state0[0]-1, state0[1]]
        elif action ==1:
            state=[state0[0], state0[1]+1]
        elif action ==2:
            state=[state0[0]+1, state0[1]]
        else: 
            state=[state0[0], state0[1]-1]
        if state[0]<0: state[0]=0;
        if state[1]<0: state[1]=0;
        if state[0]>3: state[0]=3;
        if state[1]>11: state[1]=11;
        if state[0]==3 and state[1]>0 and state[1]<12: 
            state[1]=0;
            for y in range(3):
              for x in range(12):
                    grid[y][x] = 0
            grid[3][0] = 1
            grid[3][11] = 2
        state0=state;
       
    grid[state0[0]][state0[1]] = 1.0
    # Set the screen background
    screen.fill(BLACK)
    # Draw the grid
    for column in range(1,11):
        grid[3][column]=3
    grid[3][0] = 4
    for row in range(4):
        for column in range(12):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            if grid[row][column] == 2:
                color = RED
            if grid[row][column] == 3:
                color = BLUE
            if grid[row][column] == 4:
                color = RED
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH, HEIGHT])
    

    # Limit to 60 frames per second
    clock.tick(10)
 
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
 
# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()
