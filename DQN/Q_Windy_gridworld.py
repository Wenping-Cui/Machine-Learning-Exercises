import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random,math,os,time

start_time = time.time()
epsilon = 0.1
alpha = 0.5
gamma = 1.0
# Define states
N_s = 7*10
States = np.zeros((N_s,2))
n = 0;
for i in np.arange(10):
    for j in np.arange(7):
        States[n] = [i,j];
        n = n + 1
# Define environment
Env = np.array([0,0,0,1,1,1,2,2,1,0]);

# Define action
Actions = np.array([[-1,0],[1,0],[0,1],[0,-1]])

# Define Q
Q_list = {};
for state in States:
    Q_list[(state[0],state[1])] = np.zeros(len(Actions)); 
# Define Policy
def Policy_greedy(state,epsilon):
    global Q_list
    state0 = state;
    s = np.random.uniform(0,1)
    if s<epsilon:
       n_action = random.randint(0, len(Actions)-1)
    else:
       n_action = np.argmax(Q_list[(state0[0],state0[1])])
    state1 = state0 + Actions[n_action] + np.array([0,Env[state0[0]]])
    if state1[0]>9:
       state1[0] = 9
    if state1[0]<0:
       state1[0] = 0 
    if state1[1]>6:
       state1[1] = 6
    if state1[1]<0:
       state1[1] = 0    
    return state1, n_action

N_episode = 0;
N_steps = 100000;
step = 500;
stat_s = np.array([0,3]);
stat_g = np.array([7,3]);
count = 0;
count_list =np.array([[0,0]]);
state0 = stat_s;
while count < N_steps:      
      count = count + 1.0
      if state0[0]==7 and state0[1]==3:
         state0 = stat_s
         N_episode = N_episode + 1.0 
      else: 
         state1,n_action = Policy_greedy(state0,epsilon)   
         Q_list[(state0[0],state0[1])][n_action] = Q_list[(state0[0],state0[1])][n_action] +alpha*(-1.0+gamma*Q_list[(state1[0],state1[1])][Policy_greedy(state1,epsilon)[1]]-Q_list[(state0[0],state0[1])][n_action])
         state0 = state1
      if count % step == 0:
         count_list = np.concatenate((count_list,np.array([[count,N_episode]])), axis=0)
        
print (count_list)
print("--- %s seconds ---" % (time.time() - start_time))

epsilon = 0.01

"""
 Example program to show using an array to back a grid on-screen.
 
 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
 
 Explanation video: http://youtu.be/mdTeqiWyFnc
"""
import pygame
 
# Define some colors
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
for row in range(7):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(10):
        grid[row].append(0)  # Append a cell
 
# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
grid[3][0] = 2
grid[3][7] = 2
# Initialize pygame
pygame.init()
 
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [45*10, 45*7]
screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("Windy Gridworld")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()

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

    if state0[0]==7 and state0[1]==3:
       state0 = stat_s
       for y in range(7):
              for x in range(10):
                    grid[y][x] = 0
       grid[3][0] = 2
       grid[3][7] = 2
    else: 
       state1,n_action = Policy_greedy(state0,epsilon)   
       Q_list[(state0[0],state0[1])][n_action] = Q_list[(state0[0],state0[1])][n_action] +alpha*(-1.0+gamma*Q_list[(state1[0],state1[1])][Policy_greedy(state1,epsilon)[1]]-Q_list[(state0[0],state0[1])][n_action])
       state0 = state1
    grid[int(6-state0[1])][int(state0[0])] = 1.0
    # Set the screen background
    screen.fill(BLACK)
    # Draw the grid
    for row in range(7):
        for column in range(10):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            if grid[row][column] == 2:
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


 
